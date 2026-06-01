"""
scripts/serve.py

Web UI for playing against (or watching) the Chessformer model.
Opens a browser at http://localhost:8000 automatically.

Usage:
    python scripts/serve.py checkpoint=checkpoints/step_020000.pt
    python scripts/serve.py checkpoint=... port=8080
"""

from __future__ import annotations

import os
import sys
import random
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.svg
import torch
import torch.nn.functional as F
import uvicorn
import hydra
import polars as pl
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.model import ChessformerModel
from chessformer.tokenizer import build_vocab, tokenize_position

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOARD_SIZE = 480
SQ_SIZE    = BOARD_SIZE // 8  # 60 px per square

UI_DIR = Path(__file__).parent.parent / "ui"

# ---------------------------------------------------------------------------
# Global state (single-user server)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board:        chess.Board           = field(default_factory=chess.Board)
    history:      list[chess.Move]      = field(default_factory=list)
    history_idx:  int                   = 0
    mode:         str                   = "human_vs_ai"
    human_side:   str                   = "white"
    white_elo:    int                   = 2800
    black_elo:    int                   = 2800
    temperature:  float                 = 1.0
    argmax:       bool                  = True
    debug:        bool                  = False
    # Puzzle state
    puzzle_id:     Optional[str]        = None
    puzzle_moves:  list[str]            = field(default_factory=list)  # full solution UCI list
    puzzle_step:   int                  = 0   # how many moves played so far
    puzzle_rating: int                  = 0
    # Selected square for two-click move input
    selected_sq:  Optional[int]         = None

_state: GameState = GameState()

# Model and vocab (set in main)
_model:  Optional[ChessformerModel] = None
_vocab  = None
_device = None

# Puzzles index: list of dicts {id, fen, moves: [uci...], rating}
_puzzles: list[dict] = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sq_name(sq: int) -> str:
    return chess.square_name(sq)

def _board_at_idx(idx: int) -> chess.Board:
    b = chess.Board()
    for mv in _state.history[:idx]:
        b.push(mv)
    return b

def _active_board() -> chess.Board:
    if _state.history_idx < len(_state.history):
        return _board_at_idx(_state.history_idx)
    return _state.board

def _flipped() -> bool:
    return _state.mode == "human_vs_ai" and _state.human_side == "black"

def _render_svg(
    board: chess.Board,
    last_move: Optional[chess.Move] = None,
    selected_sq: Optional[int] = None,
    probs: Optional[dict] = None,
) -> str:
    arrows: list[chess.svg.Arrow] = []
    fill:   dict[int, str]        = {}

    if selected_sq is not None:
        fill[selected_sq] = "#ffff88"
        for mv in board.legal_moves:
            if mv.from_square == selected_sq:
                fill[mv.to_square] = "#88ff88"

    if probs and _state.debug:
        from_probs = probs["from_probs"]
        to_probs   = probs["to_probs"]
        # Heat-map: top-5 from squares (blue tint)
        top_from = sorted(range(64), key=lambda i: -from_probs[i])[:5]
        for sq in top_from:
            p = from_probs[sq]
            if p > 0.02:
                fill[sq] = f"rgba(30,100,220,{min(p * 2, 0.6):.2f})"
        # Arrows for top-3 moves
        for mv in probs["top_moves"][:3]:
            alpha = min(mv["prob"] * 8, 0.9)
            if alpha > 0.05:
                arrows.append(chess.svg.Arrow(mv["from"], mv["to"],
                                              color=f"rgba(220,50,50,{alpha:.2f})"))

    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board,
        size      = BOARD_SIZE,
        lastmove  = last_move,
        arrows    = arrows,
        fill      = fill,
        check     = check,
        flipped   = _flipped(),
        coordinates = False,
    )

def _board_to_inputs(board: chess.Board) -> dict:
    pos = tokenize_position(
        fen              = board.fen(),
        vocab            = _vocab,
        white_elo        = _state.white_elo,
        black_elo        = _state.black_elo,
        white_clock_seconds = -1.0,
        black_clock_seconds = -1.0,
        game_id          = "serve",
    )
    def t(lst, dtype=torch.long):
        return torch.tensor([lst], dtype=dtype, device=_device)
    return {
        "meta_ids":       t(pos.meta_tokens),
        "color_ids":      t(pos.color_tokens),
        "piece_type_ids": t(pos.piece_type_tokens),
        "file_ids":       t(pos.file_tokens),
        "rank_ids":       t(pos.rank_tokens),
        "white_elo":      torch.tensor([_state.white_elo], dtype=torch.long, device=_device),
        "black_elo":      torch.tensor([_state.black_elo], dtype=torch.long, device=_device),
        "white_clock_s":  torch.tensor([-1.0], dtype=torch.float32, device=_device),
        "black_clock_s":  torch.tensor([-1.0], dtype=torch.float32, device=_device),
    }

def _run_ai(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal:
        return None
    batch = _board_to_inputs(board)
    temp  = 0.0 if _state.argmax else _state.temperature
    for attempt in range(10):
        t = temp if attempt < 5 else 0.0
        from_ids, to_ids, promo_ids = _model.sample_move(
            meta_ids       = batch["meta_ids"],
            color_ids      = batch["color_ids"],
            piece_type_ids = batch["piece_type_ids"],
            file_ids       = batch["file_ids"],
            rank_ids       = batch["rank_ids"],
            white_elo      = batch["white_elo"],
            black_elo      = batch["black_elo"],
            white_clock_s  = batch["white_clock_s"],
            black_clock_s  = batch["black_clock_s"],
            temperature    = t,
        )
        f_sq = int(from_ids[0].item()) - _vocab.from_square_offset
        t_sq = int(to_ids[0].item())   - _vocab.to_square_offset
        p_local = int(promo_ids[0].item()) - _vocab.promo_offset
        promo = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][p_local] if p_local >= 0 else None
        move = chess.Move(f_sq, t_sq, promotion=promo)
        if board.is_legal(move):
            return move
    return random.choice(legal)

def _get_probs() -> Optional[dict]:
    if not _state.debug or _model is None:
        return None
    batch = _board_to_inputs(_state.board)
    return _model.get_move_probs(
        meta_ids       = batch["meta_ids"],
        color_ids      = batch["color_ids"],
        piece_type_ids = batch["piece_type_ids"],
        file_ids       = batch["file_ids"],
        rank_ids       = batch["rank_ids"],
        white_elo      = batch["white_elo"],
        black_elo      = batch["black_elo"],
        white_clock_s  = batch["white_clock_s"],
        black_clock_s  = batch["black_clock_s"],
    )

def _game_status() -> str:
    b = _state.board
    if b.is_checkmate():   return "checkmate"
    if b.is_stalemate():   return "stalemate"
    if b.is_insufficient_material(): return "draw"
    if b.is_seventyfive_moves():     return "draw"
    if b.is_fivefold_repetition():   return "draw"
    if b.is_check():       return "check"
    return "playing"

def _full_state_json(probs: Optional[dict] = None) -> dict:
    view_board = _active_board()
    last_move  = _state.history[_state.history_idx - 1] if _state.history_idx > 0 else None
    svg = _render_svg(view_board, last_move=last_move,
                      selected_sq=_state.selected_sq,
                      probs=probs if _state.history_idx == len(_state.history) else None)
    human_turn = (
        _state.mode == "human_vs_ai" and
        _state.history_idx == len(_state.history) and
        ((_state.board.turn == chess.WHITE) == (_state.human_side == "white"))
    )
    # Puzzle progress
    puzzle_status = None
    if _state.mode == "puzzle" and _state.puzzle_moves:
        solver_moves = [m for i, m in enumerate(_state.puzzle_moves[1:]) if i % 2 == 0]
        done  = _state.puzzle_step
        total = len(solver_moves)
        puzzle_status = {
            "done": done, "total": total, "rating": _state.puzzle_rating,
            "solved": done == total and total > 0,
        }
    return {
        "svg":          svg,
        "turn":         "white" if _state.board.turn == chess.WHITE else "black",
        "status":       _game_status(),
        "mode":         _state.mode,
        "human_side":   _state.human_side,
        "human_turn":   human_turn,
        "white_elo":    _state.white_elo,
        "black_elo":    _state.black_elo,
        "temperature":  _state.temperature,
        "argmax":       _state.argmax,
        "debug":        _state.debug,
        "history":      [m.uci() for m in _state.history],
        "history_idx":  _state.history_idx,
        "at_latest":    _state.history_idx == len(_state.history),
        "debug_info":   probs if _state.debug else None,
        "puzzle":       puzzle_status,
        "selected_sq":  _state.selected_sq,
        "sq_size":      SQ_SIZE,
        "flipped":      _flipped(),
    }

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

# Serve static files if directory exists
if (UI_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = UI_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="ui/index.html not found")
    return html_path.read_text()

@app.get("/state")
def get_state():
    probs = _get_probs()
    return JSONResponse(_full_state_json(probs))

# --- Move endpoints ----------------------------------------------------------

class MoveRequest(BaseModel):
    uci: str

@app.post("/move")
def post_move(req: MoveRequest):
    if _state.history_idx < len(_state.history):
        raise HTTPException(status_code=400, detail="Navigate to latest before playing")
    board = _state.board
    try:
        move = board.parse_uci(req.uci)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid UCI: {req.uci}")
    if not board.is_legal(move):
        raise HTTPException(status_code=400, detail=f"Illegal move: {req.uci}")

    board.push(move)
    _state.history.append(move)
    _state.history_idx = len(_state.history)
    _state.selected_sq = None

    # Puzzle mode: validate against solution
    # puzzle_moves = [setup, solver1, opp1, solver2, opp2, ...]
    # solver moves are at indices 1, 3, 5, ... = 1 + step*2
    if _state.mode == "puzzle" and _state.puzzle_moves:
        solver_idx = 1 + _state.puzzle_step * 2
        if solver_idx < len(_state.puzzle_moves):
            expected = _state.puzzle_moves[solver_idx]
            if req.uci == expected:
                _state.puzzle_step += 1
                # Auto-play opponent response at index 2 + (step-1)*2 = 2*step
                opp_idx = 2 * _state.puzzle_step
                if opp_idx < len(_state.puzzle_moves):
                    opp_move = board.parse_uci(_state.puzzle_moves[opp_idx])
                    board.push(opp_move)
                    _state.history.append(opp_move)
                    _state.history_idx = len(_state.history)

    # Human vs AI: auto-play AI response
    elif _state.mode == "human_vs_ai" and not board.is_game_over():
        human_color = chess.WHITE if _state.human_side == "white" else chess.BLACK
        if board.turn != human_color:
            ai_mv = _run_ai(board)
            if ai_mv:
                board.push(ai_mv)
                _state.history.append(ai_mv)
                _state.history_idx = len(_state.history)

    probs = _get_probs()
    return JSONResponse(_full_state_json(probs))

@app.post("/ai_move")
def post_ai_move():
    if _state.history_idx < len(_state.history):
        raise HTTPException(status_code=400, detail="Navigate to latest first")
    if _state.board.is_game_over():
        return JSONResponse(_full_state_json())
    move = _run_ai(_state.board)
    if move:
        _state.board.push(move)
        _state.history.append(move)
        _state.history_idx = len(_state.history)
    probs = _get_probs()
    return JSONResponse(_full_state_json(probs))

class HighlightRequest(BaseModel):
    sq: int  # 0-63 square index

@app.post("/select")
def post_select(req: HighlightRequest):
    if _state.history_idx < len(_state.history):
        return JSONResponse(_full_state_json())
    sq = req.sq
    if _state.selected_sq == sq:
        _state.selected_sq = None
    elif _state.selected_sq is not None:
        # Try to submit the move
        promo_needed = (
            _state.board.piece_at(_state.selected_sq) is not None and
            _state.board.piece_at(_state.selected_sq).piece_type == chess.PAWN and
            chess.square_rank(sq) in (0, 7)
        )
        if promo_needed:
            # Signal to frontend that promotion choice is needed
            return JSONResponse({"need_promotion": True, "from_sq": _state.selected_sq, "to_sq": sq})
        uci = chess.square_name(_state.selected_sq) + chess.square_name(sq)
        mv = chess.Move.from_uci(uci)
        if _state.board.is_legal(mv):
            _state.selected_sq = None
            # Delegate to post_move
            return post_move(MoveRequest(uci=uci))
        else:
            # Re-select new square
            _state.selected_sq = sq
    else:
        _state.selected_sq = sq
    probs = _get_probs()
    return JSONResponse(_full_state_json(probs))

# --- Navigation --------------------------------------------------------------

@app.get("/history/{idx}")
def get_history(idx: int):
    n = len(_state.history)
    idx = max(0, min(n, idx))
    _state.history_idx = idx
    return JSONResponse(_full_state_json())

@app.post("/reset")
def post_reset():
    _state.board = chess.Board()
    _state.history.clear()
    _state.history_idx = 0
    _state.selected_sq = None
    _state.mode = "human_vs_ai"
    _state.puzzle_id = None
    _state.puzzle_moves = []
    _state.puzzle_step = 0
    return JSONResponse(_full_state_json())

# --- Settings ----------------------------------------------------------------

class SettingsRequest(BaseModel):
    mode:        Optional[str]   = None
    human_side:  Optional[str]   = None
    white_elo:   Optional[int]   = None
    black_elo:   Optional[int]   = None
    temperature: Optional[float] = None
    argmax:      Optional[bool]  = None
    debug:       Optional[bool]  = None

@app.post("/settings")
def post_settings(req: SettingsRequest):
    if req.mode        is not None: _state.mode        = req.mode
    if req.human_side  is not None: _state.human_side  = req.human_side
    if req.white_elo   is not None: _state.white_elo   = req.white_elo
    if req.black_elo   is not None: _state.black_elo   = req.black_elo
    if req.temperature is not None: _state.temperature = req.temperature
    if req.argmax      is not None: _state.argmax      = req.argmax
    if req.debug       is not None: _state.debug       = req.debug
    probs = _get_probs()
    return JSONResponse(_full_state_json(probs))

# --- Puzzles -----------------------------------------------------------------

@app.get("/puzzles")
def get_puzzles():
    sample = random.sample(_puzzles, min(12, len(_puzzles)))
    return JSONResponse(sample)

@app.post("/puzzles/{puzzle_id}")
def load_puzzle(puzzle_id: str):
    puzzle = next((p for p in _puzzles if p["id"] == puzzle_id), None)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Puzzle not found")

    # Reset board to puzzle FEN and apply opponent's first move
    board = chess.Board(puzzle["fen"])
    moves = puzzle["moves"]  # e.g. ["e2e4", "e7e5", ...]

    _state.board       = board
    _state.history     = []
    _state.history_idx = 0
    _state.selected_sq = None
    _state.mode        = "puzzle"
    _state.puzzle_id   = puzzle_id
    _state.puzzle_moves = moves
    _state.puzzle_step  = 0
    _state.puzzle_rating = puzzle["rating"]

    # Auto-play opponent's setup move (first move in list)
    if moves:
        setup_move = board.parse_uci(moves[0])
        board.push(setup_move)
        _state.history.append(setup_move)
        _state.history_idx = 1

    return JSONResponse(_full_state_json())

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_puzzles(cfg: DictConfig) -> list[dict]:
    puzzle_path = Path(cfg.data.val_puzzles_file)
    if not puzzle_path.exists():
        return []
    df = pl.read_parquet(str(puzzle_path))
    puzzle_ids = df.select("puzzle_id").unique()["puzzle_id"].to_list()
    puzzle_ids = random.sample(puzzle_ids, min(200, len(puzzle_ids)))

    # Load FEN and moves from the raw CSV
    csv_path = Path(cfg.data.get("puzzles_csv", "data/raw_puzzles/lichess_db_puzzle.csv"))
    if not csv_path.exists():
        return []
    raw = pl.read_csv(str(csv_path), columns=["PuzzleId", "FEN", "Moves", "Rating"])
    raw = raw.filter(pl.col("PuzzleId").is_in(puzzle_ids))
    return [
        {
            "id":     row["PuzzleId"],
            "fen":    row["FEN"],
            "moves":  row["Moves"].split(),
            "rating": int(row["Rating"]),
        }
        for row in raw.iter_rows(named=True)
    ]


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    global _model, _vocab, _device, _puzzles

    ckpt_path = cfg.checkpoint
    assert ckpt_path, "Pass checkpoint=<path>"

    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {_device}  |  Checkpoint: {ckpt_path}")

    ckpt      = torch.load(ckpt_path, map_location=_device)
    saved_cfg = OmegaConf.create(ckpt["cfg"])
    _vocab    = build_vocab()
    _model    = ChessformerModel(
        vocab    = _vocab,
        d_model  = saved_cfg.model.d_model,
        n_heads  = saved_cfg.model.n_heads,
        n_layers = saved_cfg.model.n_layers,
        ffn_mult = saved_cfg.model.ffn_mult,
        dropout  = saved_cfg.model.dropout,
    ).to(_device)
    _model.load_state_dict(ckpt["model"], strict=False)
    _model.eval()
    print(f"Model loaded (step {ckpt.get('step', 0)}, "
          f"{sum(p.numel() for p in _model.parameters()):,} params)")

    _puzzles = _load_puzzles(cfg)
    print(f"Puzzles loaded: {len(_puzzles)}")

    port = int(cfg.get("port", 8000))
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
