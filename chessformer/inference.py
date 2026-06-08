"""
chessformer/inference.py

Model-facing inference utilities: tokenize a board position, run the model,
and return moves or probability distributions.

All functions are pure (no global state) — callers pass model, vocab, device,
and conditioning parameters explicitly. This makes them reusable from serve.py,
play.py, eval scripts, and notebooks without importing any web layer.
"""

from __future__ import annotations

import random
from typing import Optional

import chess
import torch
import torch.nn.functional as F

from chessformer.model import ChessformerModel
from chessformer.tokenizer import Vocab, tokenize_position


# ---------------------------------------------------------------------------
# Board → model inputs
# ---------------------------------------------------------------------------

def sq_to_file_rank_ids(sq: int, vocab: Vocab) -> tuple[int, int]:
    """Convert a 0-63 chess square index to (file_token_id, rank_token_id)."""
    return (
        vocab.token_to_id["file_a"] + (sq % 8),
        vocab.token_to_id["rank_1"] + (sq // 8),
    )


def board_to_inputs(
    board:         chess.Board,
    vocab:         Vocab,
    device:        torch.device,
    white_elo:     int   = -1,
    black_elo:     int   = -1,
    white_clock_s: float = -1.0,
    black_clock_s: float = -1.0,
    increment_s:   float = -1.0,
) -> dict[str, torch.Tensor]:
    """
    Tokenize a chess position and return a dict of batched tensors (batch size 1)
    ready to pass directly to ChessformerModel.forward().
    """
    pos = tokenize_position(
        fen                 = board.fen(),
        vocab               = vocab,
        white_elo           = white_elo,
        black_elo           = black_elo,
        white_clock_seconds = white_clock_s,
        black_clock_seconds = black_clock_s,
        increment_seconds   = increment_s,
    )

    def t(lst: list, dtype=torch.long) -> torch.Tensor:
        return torch.tensor([lst], dtype=dtype, device=device)

    return {
        "meta_ids":       t(pos.meta_tokens),
        "color_ids":      t(pos.color_tokens),
        "piece_type_ids": t(pos.piece_type_tokens),
        "file_ids":       t(pos.file_tokens),
        "rank_ids":       t(pos.rank_tokens),
        "white_elo":      torch.tensor([white_elo],     dtype=torch.long,    device=device),
        "black_elo":      torch.tensor([black_elo],     dtype=torch.long,    device=device),
        "white_clock_s":  torch.tensor([white_clock_s], dtype=torch.float32, device=device),
        "black_clock_s":  torch.tensor([black_clock_s], dtype=torch.float32, device=device),
        "increment_s":    torch.tensor([increment_s],   dtype=torch.float32, device=device),
    }


# ---------------------------------------------------------------------------
# Position analysis
# ---------------------------------------------------------------------------

def analyze_position(
    model:       ChessformerModel,
    vocab:       Vocab,
    device:      torch.device,
    board:       chess.Board,
    white_elo:   int   = -1,
    black_elo:   int   = -1,
    white_clock_s: float = -1.0,
    black_clock_s: float = -1.0,
    increment_s: float = -1.0,
    top_from_k:  int   = 8,
) -> tuple[dict, Optional[chess.Move]]:
    """
    Scan from-squares to compute joint from×to probabilities and find the best legal move.

    Always includes all legal from-squares in the scan (in addition to the model's
    top-k predictions), so the result is never empty when legal moves exist.
    Costs len(candidate_from_sqs) + 1 forward passes.

    Returns:
        probs: dict with "from_probs" [64], "to_probs" [64], "top_moves" (list of dicts)
        best_move: the highest-probability legal move, or None if no legal moves
    """
    legal_all = list(board.legal_moves)
    if not legal_all:
        return {"from_probs": [0.0] * 64, "to_probs": [0.0] * 64, "top_moves": []}, None

    batch = board_to_inputs(board, vocab, device,
                            white_elo, black_elo, white_clock_s, black_clock_s, increment_s)

    # (from, to) → best move, preferring queen promotion
    legal_map: dict[tuple[int, int], chess.Move] = {}
    for m in legal_all:
        key = (m.from_square, m.to_square)
        if key not in legal_map or m.promotion == chess.QUEEN:
            legal_map[key] = m

    # Pass 1: from-square distribution
    dummy = torch.zeros(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        from_logits, _, _ = _forward(model, batch, dummy)
    from_probs = F.softmax(from_logits[0], dim=-1).cpu()

    # Candidate from-squares: model top-k + all legal from-squares
    top_model   = from_probs.topk(min(top_from_k, 64)).indices.tolist()
    legal_from  = list({m.from_square for m in legal_all})
    seen: set[int] = set(top_model)
    candidates_from = top_model + [sq for sq in legal_from if sq not in seen]

    # Pass 2+: to-square distribution for each candidate from-square
    all_candidates: list[dict] = []
    to_probs_primary: list[float] = [0.0] * 64

    for i, from_sq in enumerate(candidates_from):
        file_id, rank_id = sq_to_file_rank_ids(from_sq, vocab)
        dummy2 = dummy.clone()
        dummy2[0, 0] = file_id
        dummy2[0, 1] = rank_id
        with torch.no_grad():
            _, to_logits, _ = _forward(model, batch, dummy2)
        to_probs = F.softmax(to_logits[0], dim=-1).cpu()
        if i == 0:
            to_probs_primary = to_probs.tolist()
        fp = float(from_probs[from_sq])
        for to_sq in range(64):
            all_candidates.append({"from": from_sq, "to": to_sq,
                                   "prob": fp * float(to_probs[to_sq])})

    all_candidates.sort(key=lambda x: -x["prob"])
    top_legal = [c for c in all_candidates if (c["from"], c["to"]) in legal_map][:5]

    best_move: Optional[chess.Move] = None
    if top_legal:
        best_move = legal_map[(top_legal[0]["from"], top_legal[0]["to"])]
    else:
        best_move = random.choice(legal_all)

    return {
        "from_probs": from_probs.tolist(),
        "to_probs":   to_probs_primary,
        "top_moves":  top_legal,
    }, best_move


# ---------------------------------------------------------------------------
# Move selection
# ---------------------------------------------------------------------------

def pick_move(
    model:         ChessformerModel,
    vocab:         Vocab,
    device:        torch.device,
    board:         chess.Board,
    white_elo:     int   = -1,
    black_elo:     int   = -1,
    white_clock_s: float = -1.0,
    black_clock_s: float = -1.0,
    increment_s:   float = -1.0,
    argmax:        bool  = True,
    temperature:   float = 1.0,
) -> Optional[chess.Move]:
    """
    Select a move for the current position.

    argmax=True  → full from×to scan, returns highest-probability legal move.
    argmax=False → sample from the model's distribution; retries up to 10 times
                   with greedy fallback if sampled moves are illegal.
    Returns None if there are no legal moves.
    """
    legal = list(board.legal_moves)
    if not legal:
        return None

    if argmax:
        _, best = analyze_position(model, vocab, device, board,
                                   white_elo, black_elo, white_clock_s, black_clock_s, increment_s)
        return best or random.choice(legal)

    # Sampling path
    batch = board_to_inputs(board, vocab, device,
                            white_elo, black_elo, white_clock_s, black_clock_s, increment_s)
    for attempt in range(10):
        temp = temperature if attempt < 5 else 0.0
        from_ids, to_ids, promo_ids = model.sample_move(
            meta_ids       = batch["meta_ids"],
            color_ids      = batch["color_ids"],
            piece_type_ids = batch["piece_type_ids"],
            file_ids       = batch["file_ids"],
            rank_ids       = batch["rank_ids"],
            white_elo      = batch["white_elo"],
            black_elo      = batch["black_elo"],
            white_clock_s  = batch["white_clock_s"],
            black_clock_s  = batch["black_clock_s"],
            increment_s    = batch["increment_s"],
            temperature    = temp,
        )
        f_sq    = int(from_ids[0].item())   # 0-63
        t_sq    = int(to_ids[0].item())     # 0-63
        p_local = int(promo_ids[0].item())  # 0-3
        piece   = board.piece_at(f_sq)
        promo   = (
            [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][p_local]
            if piece and piece.piece_type == chess.PAWN and chess.square_rank(t_sq) in (0, 7)
            else None
        )
        move = chess.Move(f_sq, t_sq, promotion=promo)
        if board.is_legal(move):
            return move

    return random.choice(legal)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _forward(
    model: ChessformerModel,
    batch: dict[str, torch.Tensor],
    move_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single forward pass with a given move_ids tensor."""
    return model(
        meta_ids       = batch["meta_ids"],
        color_ids      = batch["color_ids"],
        piece_type_ids = batch["piece_type_ids"],
        file_ids       = batch["file_ids"],
        rank_ids       = batch["rank_ids"],
        white_elo      = batch["white_elo"],
        black_elo      = batch["black_elo"],
        white_clock_s  = batch["white_clock_s"],
        black_clock_s  = batch["black_clock_s"],
        increment_s    = batch["increment_s"],
        move_ids       = move_ids,
    )
