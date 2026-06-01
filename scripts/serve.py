"""
scripts/serve.py

Web UI for playing against (or watching) the Chessformer model.
Opens a browser at http://localhost:5174 automatically.

Usage:
    python scripts/serve.py checkpoint=checkpoints/step_020000.pt
    python scripts/serve.py checkpoint=... port=5175
"""

from __future__ import annotations

import io
import os
import sys
import random
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
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

BOARD_SIZE = 600
SQ_SIZE    = BOARD_SIZE // 8   # 75 px per square
UI_DIR     = Path(__file__).parent.parent / "ui"

# ---------------------------------------------------------------------------
# Global state (single-user server)
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board:          chess.Board      = field(default_factory=chess.Board)
    history:        list[chess.Move] = field(default_factory=list)
    history_idx:    int              = 0
    mode:           str              = "human_vs_ai"    # human_vs_ai | ai_vs_ai | human_vs_human | puzzle
    human_side:     str              = "white"          # white | black (human_vs_ai only)
    white_elo:      int              = 2800
    black_elo:      int              = 2800
    white_clock_s:  float            = 120.0
    black_clock_s:  float            = 120.0
    temperature:    float            = 1.0
    argmax:         bool             = True
    engine:         bool             = False            # show engine analysis arrows/probs
    selected_sq:    Optional[int]    = None
    # Puzzle state
    puzzle_id:      Optional[str]    = None
    puzzle_moves:   list[str]        = field(default_factory=list)
    puzzle_step:    int              = 0
    puzzle_rating:  int              = 0

_state: GameState = GameState()

_model:   Optional[ChessformerModel] = None
_vocab    = None
_device   = None
_puzzles: list[dict] = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_board() -> chess.Board:
    if _state.history_idx == len(_state.history):
        return _state.board
    b = chess.Board()
    for mv in _state.history[:_state.history_idx]:
        b.push(mv)
    return b

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
        fill[selected_sq] = "#ffff4488"
        for mv in board.legal_moves:
            if mv.from_square == selected_sq:
                fill[mv.to_square] = "#88ff8844"

    if probs and _state.engine:
        from_probs = probs["from_probs"]
        top_from   = sorted(range(64), key=lambda i: -from_probs[i])[:5]
        for sq in top_from:
            p = from_probs[sq]
            if p > 0.02:
                opacity = min(p * 2.0, 0.55)
                fill[sq] = f"rgba(30,100,220,{opacity:.2f})"
        for mv in probs["top_moves"][:3]:
            alpha = min(mv["prob"] * 8.0, 0.9)
            if alpha > 0.05:
                arrows.append(chess.svg.Arrow(
                    mv["from"], mv["to"],
                    color=f"rgba(220,60,60,{alpha:.2f})",
                ))

    check = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board,
        size        = BOARD_SIZE,
        lastmove    = last_move,
        arrows      = arrows,
        fill        = fill,
        check       = check,
        flipped     = _flipped(),
        coordinates = False,
    )

def _board_to_inputs(board: chess.Board) -> dict:
    pos = tokenize_position(
        fen                 = board.fen(),
        vocab               = _vocab,
        white_elo           = _state.white_elo,
        black_elo           = _state.black_elo,
        white_clock_seconds = _state.white_clock_s,
        black_clock_seconds = _state.black_clock_s,
        game_id             = "serve",
    )
    def t(lst, dtype=torch.long):
        return torch.tensor([lst], dtype=dtype, device=_device)
    return {
        "meta_ids":       t(pos.meta_tokens),
        "color_ids":      t(pos.color_tokens),
        "piece_type_ids": t(pos.piece_type_tokens),
        "file_ids":       t(pos.file_tokens),
        "rank_ids":       t(pos.rank_tokens),
        "white_elo":      torch.tensor([_state.white_elo],    dtype=torch.long,    device=_device),
        "black_elo":      torch.tensor([_state.black_elo],    dtype=torch.long,    device=_device),
        "white_clock_s":  torch.tensor([_state.white_clock_s],dtype=torch.float32, device=_device),
        "black_clock_s":  torch.tensor([_state.black_clock_s],dtype=torch.float32, device=_device),
    }

def _analyze_board(board: chess.Board, top_from_k: int = 4) -> tuple[dict, Optional[chess.Move]]:
    """
    Scan top-K from-squares, compute to-probs for each, find best legal move.

    Returns (probs_dict, best_legal_move) where:
      - probs_dict.top_moves contains only legal moves, sorted by joint prob
      - best_legal_move is top_moves[0] (or random fallback)
    Using top_from_k from-squares costs top_from_k+1 forward passes.
    """
    legal_all = list(board.legal_moves)
    if not legal_all:
        return {"from_probs": [0.0]*64, "to_probs": [0.0]*64, "top_moves": []}, None

    batch = _board_to_inputs(board)

    # Build legal (from, to) → best move mapping (queen promo preferred)
    legal_map: dict[tuple[int,int], chess.Move] = {}
    for m in legal_all:
        key = (m.from_square, m.to_square)
        if key not in legal_map or m.promotion == chess.QUEEN:
            legal_map[key] = m

    # Pass 1: from-square probabilities
    dummy = torch.zeros(1, 3, dtype=torch.long, device=_device)
    with torch.no_grad():
        from_logits, _, _ = _model(
            meta_ids=batch["meta_ids"], color_ids=batch["color_ids"],
            piece_type_ids=batch["piece_type_ids"], file_ids=batch["file_ids"],
            rank_ids=batch["rank_ids"], white_elo=batch["white_elo"],
            black_elo=batch["black_elo"], white_clock_s=batch["white_clock_s"],
            black_clock_s=batch["black_clock_s"], move_ids=dummy,
        )
    from_probs = F.softmax(from_logits[0], dim=-1).cpu()

    top_from = from_probs.topk(min(top_from_k, 64)).indices.tolist()

    candidates: list[dict] = []
    to_probs_primary: list[float] = [0.0] * 64

    for i, from_sq in enumerate(top_from):
        dummy2 = dummy.clone()
        dummy2[0, 0] = from_sq + _vocab.from_square_offset
        with torch.no_grad():
            _, to_logits, _ = _model(
                meta_ids=batch["meta_ids"], color_ids=batch["color_ids"],
                piece_type_ids=batch["piece_type_ids"], file_ids=batch["file_ids"],
                rank_ids=batch["rank_ids"], white_elo=batch["white_elo"],
                black_elo=batch["black_elo"], white_clock_s=batch["white_clock_s"],
                black_clock_s=batch["black_clock_s"], move_ids=dummy2,
            )
        to_probs = F.softmax(to_logits[0], dim=-1).cpu()
        if i == 0:
            to_probs_primary = to_probs.tolist()
        fp = float(from_probs[from_sq])
        for to_sq in range(64):
            candidates.append({"from": from_sq, "to": to_sq,
                                "prob": fp * float(to_probs[to_sq])})

    candidates.sort(key=lambda x: -x["prob"])

    top_legal = [c for c in candidates if (c["from"], c["to"]) in legal_map][:5]

    best_legal: Optional[chess.Move] = None
    if top_legal:
        best_legal = legal_map[(top_legal[0]["from"], top_legal[0]["to"])]
    else:
        best_legal = random.choice(legal_all)

    return {
        "from_probs": from_probs.tolist(),
        "to_probs":   to_probs_primary,
        "top_moves":  top_legal,
    }, best_legal


def _run_ai(board: chess.Board) -> Optional[chess.Move]:
    legal = list(board.legal_moves)
    if not legal:
        return None
    if _state.argmax:
        _, best = _analyze_board(board)
        return best or random.choice(legal)
    # Sampling: retry up to 10 times, greedy fallback
    batch = _board_to_inputs(board)
    for attempt in range(10):
        t = _state.temperature if attempt < 5 else 0.0
        from_ids, to_ids, promo_ids = _model.sample_move(
            meta_ids=batch["meta_ids"], color_ids=batch["color_ids"],
            piece_type_ids=batch["piece_type_ids"], file_ids=batch["file_ids"],
            rank_ids=batch["rank_ids"], white_elo=batch["white_elo"],
            black_elo=batch["black_elo"], white_clock_s=batch["white_clock_s"],
            black_clock_s=batch["black_clock_s"], temperature=t,
        )
        f_sq    = int(from_ids[0].item())  - _vocab.from_square_offset
        t_sq    = int(to_ids[0].item())    - _vocab.to_square_offset
        p_local = int(promo_ids[0].item()) - _vocab.promo_offset
        promo   = [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][p_local] if p_local >= 0 else None
        move    = chess.Move(f_sq, t_sq, promotion=promo)
        if board.is_legal(move):
            return move
    return random.choice(legal)


def _get_probs(board: chess.Board) -> Optional[dict]:
    if not _state.engine or _model is None or board.is_game_over():
        return None
    probs, _ = _analyze_board(board)
    return probs

def _game_status(board: Optional[chess.Board] = None) -> str:
    b = board or _state.board
    if b.is_checkmate():             return "checkmate"
    if b.is_stalemate():             return "stalemate"
    if b.is_insufficient_material(): return "draw"
    if b.is_seventyfive_moves():     return "draw"
    if b.is_fivefold_repetition():   return "draw"
    if b.is_check():                 return "check"
    return "playing"

def _full_state_json(probs: Optional[dict] = None) -> dict:
    view_board = _active_board()
    last_move  = _state.history[_state.history_idx - 1] if _state.history_idx > 0 else None
    svg        = _render_svg(
        view_board, last_move=last_move,
        selected_sq = _state.selected_sq if _state.history_idx == len(_state.history) else None,
        probs       = probs,
    )
    live_board  = _state.board
    at_latest   = _state.history_idx == len(_state.history)
    human_color = chess.WHITE if _state.human_side == "white" else chess.BLACK
    human_turn  = (
        at_latest and not live_board.is_game_over() and
        _state.mode in ("human_vs_ai", "human_vs_human") and
        (_state.mode == "human_vs_human" or live_board.turn == human_color)
    )
    puzzle_status = None
    if _state.mode == "puzzle" and _state.puzzle_moves:
        solver_total  = (len(_state.puzzle_moves) - 1 + 1) // 2   # ceil((len-1)/2)
        puzzle_status = {
            "done":   _state.puzzle_step,
            "total":  solver_total,
            "rating": _state.puzzle_rating,
            "solved": _state.puzzle_step >= solver_total and solver_total > 0,
        }

    return {
        "svg":          svg,
        "turn":         "white" if view_board.turn == chess.WHITE else "black",
        "status":       _game_status(view_board),
        "live_status":  _game_status(live_board),
        "mode":         _state.mode,
        "human_side":   _state.human_side,
        "human_turn":   human_turn,
        "white_elo":    _state.white_elo,
        "black_elo":    _state.black_elo,
        "white_clock_s": _state.white_clock_s,
        "black_clock_s": _state.black_clock_s,
        "temperature":  _state.temperature,
        "argmax":       _state.argmax,
        "engine":       _state.engine,
        "history":      [m.uci() for m in _state.history],
        "history_idx":  _state.history_idx,
        "at_latest":    at_latest,
        "debug_info":   probs,
        "puzzle":       puzzle_status,
        "selected_sq":  _state.selected_sq,
        "sq_size":      SQ_SIZE,
        "flipped":      _flipped(),
    }

def _state_with_probs() -> dict:
    view = _active_board()
    probs = _get_probs(view)
    return _full_state_json(probs)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI()

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
    return JSONResponse(_state_with_probs())

# ---------------------------------------------------------------------------
# Move endpoints
# ---------------------------------------------------------------------------

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

    # Puzzle: validate against solution
    # puzzle_moves = [setup_by_opp, solver1, opp1, solver2, opp2, ...]
    if _state.mode == "puzzle" and _state.puzzle_moves:
        solver_idx = 1 + _state.puzzle_step * 2
        if solver_idx < len(_state.puzzle_moves):
            if req.uci == _state.puzzle_moves[solver_idx]:
                _state.puzzle_step += 1
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

    return JSONResponse(_state_with_probs())

@app.post("/ai_move")
def post_ai_move():
    if _state.history_idx < len(_state.history):
        raise HTTPException(status_code=400, detail="Navigate to latest first")
    if _state.board.is_game_over():
        return JSONResponse(_state_with_probs())
    move = _run_ai(_state.board)
    if move:
        _state.board.push(move)
        _state.history.append(move)
        _state.history_idx = len(_state.history)
    return JSONResponse(_state_with_probs())

class SelectRequest(BaseModel):
    sq: int

@app.post("/select")
def post_select(req: SelectRequest):
    if _state.history_idx < len(_state.history):
        return JSONResponse(_state_with_probs())
    sq = req.sq
    if _state.selected_sq == sq:
        _state.selected_sq = None
    elif _state.selected_sq is not None:
        piece = _state.board.piece_at(_state.selected_sq)
        promo_needed = (
            piece is not None and
            piece.piece_type == chess.PAWN and
            chess.square_rank(sq) in (0, 7)
        )
        if promo_needed:
            return JSONResponse({
                "need_promotion": True,
                "from_sq": _state.selected_sq,
                "to_sq":   sq,
            })
        uci = chess.square_name(_state.selected_sq) + chess.square_name(sq)
        mv  = chess.Move.from_uci(uci)
        if _state.board.is_legal(mv):
            _state.selected_sq = None
            return post_move(MoveRequest(uci=uci))
        else:
            _state.selected_sq = sq
    else:
        _state.selected_sq = sq
    return JSONResponse(_state_with_probs())

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

@app.get("/history/{idx}")
def get_history(idx: int):
    n   = len(_state.history)
    idx = max(0, min(n, idx))
    _state.history_idx  = idx
    _state.selected_sq  = None
    return JSONResponse(_state_with_probs())

@app.post("/reset")
def post_reset():
    _state.board        = chess.Board()
    _state.history      = []
    _state.history_idx  = 0
    _state.selected_sq  = None
    _state.puzzle_id    = None
    _state.puzzle_moves = []
    _state.puzzle_step  = 0
    if _state.mode == "puzzle":
        _state.mode = "human_vs_ai"
    return JSONResponse(_state_with_probs())

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class SettingsRequest(BaseModel):
    mode:          Optional[str]   = None
    human_side:    Optional[str]   = None
    white_elo:     Optional[int]   = None
    black_elo:     Optional[int]   = None
    white_clock_s: Optional[float] = None
    black_clock_s: Optional[float] = None
    temperature:   Optional[float] = None
    argmax:        Optional[bool]  = None
    engine:        Optional[bool]  = None

@app.post("/settings")
def post_settings(req: SettingsRequest):
    if req.mode          is not None: _state.mode          = req.mode
    if req.human_side    is not None: _state.human_side    = req.human_side
    if req.white_elo     is not None: _state.white_elo     = req.white_elo
    if req.black_elo     is not None: _state.black_elo     = req.black_elo
    if req.white_clock_s is not None: _state.white_clock_s = req.white_clock_s
    if req.black_clock_s is not None: _state.black_clock_s = req.black_clock_s
    if req.temperature   is not None: _state.temperature   = req.temperature
    if req.argmax        is not None: _state.argmax        = req.argmax
    if req.engine        is not None: _state.engine        = req.engine
    return JSONResponse(_state_with_probs())

# ---------------------------------------------------------------------------
# Import (FEN / PGN)
# ---------------------------------------------------------------------------

class ImportRequest(BaseModel):
    text: str

@app.post("/import")
def post_import(req: ImportRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty input")

    # Try PGN first (starts with [ or contains move numbers like 1.)
    if text.startswith("[") or "1." in text:
        try:
            game = chess.pgn.read_game(io.StringIO(text))
            if game is None:
                raise ValueError("Could not parse PGN")
            board = game.board()
            moves = list(game.mainline_moves())
            _state.board        = board
            _state.history      = []
            _state.history_idx  = 0
            _state.selected_sq  = None
            _state.mode         = "human_vs_ai"
            _state.puzzle_id    = None
            _state.puzzle_moves = []
            for mv in moves:
                board.push(mv)
                _state.history.append(mv)
            _state.history_idx = len(_state.history)
            # Rewind to start for review
            _state.history_idx = 0
            return JSONResponse(_state_with_probs())
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"PGN parse error: {e}")

    # Try FEN
    try:
        board = chess.Board(text)
        _state.board        = board
        _state.history      = []
        _state.history_idx  = 0
        _state.selected_sq  = None
        _state.mode         = "human_vs_ai"
        _state.puzzle_id    = None
        _state.puzzle_moves = []
        return JSONResponse(_state_with_probs())
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"FEN parse error: {e}")

# ---------------------------------------------------------------------------
# Puzzles
# ---------------------------------------------------------------------------

@app.get("/puzzles")
def get_puzzles():
    sample = random.sample(_puzzles, min(12, len(_puzzles)))
    return JSONResponse(sample)

@app.post("/puzzles/{puzzle_id}")
def load_puzzle(puzzle_id: str):
    puzzle = next((p for p in _puzzles if p["id"] == puzzle_id), None)
    if puzzle is None:
        raise HTTPException(status_code=404, detail="Puzzle not found")
    board = chess.Board(puzzle["fen"])
    moves = puzzle["moves"]

    _state.board        = board
    _state.history      = []
    _state.history_idx  = 0
    _state.selected_sq  = None
    _state.mode         = "puzzle"
    _state.puzzle_id    = puzzle_id
    _state.puzzle_moves = moves
    _state.puzzle_step  = 0
    _state.puzzle_rating = puzzle["rating"]

    # Auto-play opponent's setup move (first in list)
    if moves:
        setup_move = board.parse_uci(moves[0])
        board.push(setup_move)
        _state.history.append(setup_move)
        _state.history_idx = 1

    return JSONResponse(_state_with_probs())

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_puzzles(cfg: DictConfig) -> list[dict]:
    puzzles_csv = getattr(cfg.data, "puzzles_csv",
                          "data/raw_puzzles/lichess_db_puzzle.csv")
    val_parquet = getattr(cfg.data, "val_puzzles_file", None)

    csv_path = Path(puzzles_csv)
    if not csv_path.exists():
        print(f"  Puzzle CSV not found: {csv_path}")
        return []

    # Sample puzzle IDs from the val parquet if available
    puzzle_ids = None
    if val_parquet and Path(val_parquet).exists():
        try:
            df  = pl.read_parquet(str(val_parquet))
            ids = df.select("puzzle_id").unique()["puzzle_id"].to_list()
            puzzle_ids = set(random.sample(ids, min(500, len(ids))))
        except Exception as e:
            print(f"  Could not read val parquet: {e}")

    try:
        raw = pl.read_csv(str(csv_path), columns=["PuzzleId", "FEN", "Moves", "Rating"])
        if puzzle_ids is not None:
            raw = raw.filter(pl.col("PuzzleId").is_in(list(puzzle_ids)))
        else:
            raw = raw.sample(min(200, len(raw)))
        return [
            {
                "id":     row["PuzzleId"],
                "fen":    row["FEN"],
                "moves":  row["Moves"].split(),
                "rating": int(row["Rating"]),
            }
            for row in raw.iter_rows(named=True)
        ]
    except Exception as e:
        print(f"  Could not load puzzles: {e}")
        return []


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
    print(f"Model loaded  step={ckpt.get('step', 0)}  "
          f"params={sum(p.numel() for p in _model.parameters()):,}")

    _puzzles = _load_puzzles(cfg)
    print(f"Puzzles loaded: {len(_puzzles)}")

    port = int(getattr(cfg, "port", 5174))
    print(f"Serving at http://localhost:{port}")
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
