"""
chessformer/eval.py

Evaluation functions for games and puzzles.
Called by scripts/eval_offline.py (and optionally by the Trainer for quick val).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chessformer.tokenizer import Vocab


# ---------------------------------------------------------------------------
# Games eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_games(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab: Vocab,
    max_batches: Optional[int] = None,
) -> dict:
    """
    Evaluate on game positions.

    Returns:
        loss, top1_acc, top5_acc, legal_rate
        per_elo: {bucket_str: {loss, top1_acc, top5_acc, legal_rate, n}}
    """
    model.eval()
    from_off = vocab.from_square_offset
    to_off   = vocab.to_square_offset

    totals = defaultdict(lambda: {"loss": 0.0, "top1": 0, "plausible": 0, "n": 0})

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        elo_bkts   = batch["elo_bucket"]
        move_ids   = torch.stack([batch["from_sq"], batch["to_sq"], batch["promo"]], dim=1).to(device)

        from_logits, to_logits, _ = model(
            meta_ids       = batch["meta_ids"].to(device),
            color_ids      = batch["color_ids"].to(device),
            piece_type_ids = batch["piece_type_ids"].to(device),
            file_ids       = batch["file_ids"].to(device),
            rank_ids       = batch["rank_ids"].to(device),
            white_elo      = batch["white_elo"].to(device),
            black_elo      = batch["black_elo"].to(device),
            white_clock_s  = batch["white_clock_s"].to(device),
            black_clock_s  = batch["black_clock_s"].to(device),
            move_ids       = move_ids,
        )

        from_target = batch["from_sq"].to(device) - from_off
        to_target   = batch["to_sq"].to(device)   - to_off

        loss = (F.cross_entropy(from_logits, from_target) + F.cross_entropy(to_logits, to_target)).item()

        from_pred = from_logits.argmax(-1)
        to_pred   = to_logits.argmax(-1)
        top1      = ((from_pred == from_target) & (to_pred == to_target))

        # Plausible rate: correct move has >= 20% probability under each marginal
        from_probs  = F.softmax(from_logits, dim=-1)                        # [B, 64]
        to_probs    = F.softmax(to_logits,   dim=-1)                        # [B, 64]
        from_p_gold = from_probs.gather(1, from_target.unsqueeze(1)).squeeze(1)  # [B]
        to_p_gold   = to_probs.gather(1,   to_target.unsqueeze(1)).squeeze(1)    # [B]
        plausible   = (from_p_gold >= 0.20) & (to_p_gold >= 0.20)

        B = len(elo_bkts)
        for b in range(B):
            bkt = elo_bkts[b]
            for key in ("all", bkt):
                totals[key]["loss"]      += loss / B
                totals[key]["top1"]      += int(top1[b].item())
                totals[key]["plausible"] += int(plausible[b].item())
                totals[key]["n"]         += 1

    def _agg(d: dict) -> dict:
        n = d["n"]
        return {
            "loss":           d["loss"]      / n,
            "top1_acc":       d["top1"]      / n,
            "plausible_rate": d["plausible"] / n,
            "n":              n,
        }

    result = _agg(totals["all"])
    result["per_elo"] = {k: _agg(v) for k, v in totals.items() if k != "all"}
    return result



# ---------------------------------------------------------------------------
# Puzzles eval
# ---------------------------------------------------------------------------

def eval_puzzles(
    model: nn.Module,
    parquet_path: str,
    device: torch.device,
    vocab: Vocab,
) -> dict:
    """
    Evaluate on puzzles.

    Metrics:
      loss        — cross-entropy on solver moves only
      accuracy    — 1.0 if ALL solver moves in a puzzle are correct (macro over puzzles)
      advancement — fraction of consecutive correct moves from start (macro over puzzles)
                    e.g. 3/4 correct → 0.75

    Processes one puzzle at a time (puzzles vary in length and board state).
    """
    model.eval()
    from_off = vocab.from_square_offset
    to_off   = vocab.to_square_offset

    df = pl.read_parquet(parquet_path)

    puzzle_results = []   # list of (n_solver_moves, n_consecutive_correct)
    total_loss   = 0.0
    total_loss_n = 0

    for _, puzzle_df in df.group_by("puzzle_id"):
        puzzle_df   = puzzle_df.sort("seq_index")
        solver_rows = puzzle_df.filter(pl.col("is_solver_move"))
        if len(solver_rows) == 0:
            continue

        n_solver   = len(solver_rows)
        consecutive = 0
        found_mistake = False

        for row in solver_rows.iter_rows(named=True):
            meta  = torch.tensor([row["meta_tokens"]],         dtype=torch.long,    device=device)
            color = torch.tensor([row["color_tokens"]],        dtype=torch.long,    device=device)
            ptype = torch.tensor([row["piece_type_tokens"]],   dtype=torch.long,    device=device)
            filei = torch.tensor([row["file_tokens"]],         dtype=torch.long,    device=device)
            ranki = torch.tensor([row["rank_tokens"]],         dtype=torch.long,    device=device)
            w_elo = torch.tensor([row["white_elo"]],           dtype=torch.long,    device=device)
            b_elo = torch.tensor([row["black_elo"]],           dtype=torch.long,    device=device)
            w_clk = torch.tensor([row["white_clock_seconds"]], dtype=torch.float32, device=device)
            b_clk = torch.tensor([row["black_clock_seconds"]], dtype=torch.float32, device=device)

            from_id  = row["from_square_id"]
            to_id    = row["to_square_id"]
            move_in  = torch.tensor([[from_id, to_id, row["promo_id"]]], dtype=torch.long, device=device)

            with torch.no_grad():
                from_logits, to_logits, _ = model(
                    meta, color, ptype, filei, ranki, w_elo, b_elo, w_clk, b_clk, move_in
                )

            from_target = torch.tensor([from_id - from_off], dtype=torch.long, device=device)
            to_target   = torch.tensor([to_id   - to_off  ], dtype=torch.long, device=device)

            loss = (F.cross_entropy(from_logits, from_target) + F.cross_entropy(to_logits, to_target)).item()
            total_loss   += loss
            total_loss_n += 1

            correct = (
                from_logits.argmax(-1).item() == from_target.item() and
                to_logits.argmax(-1).item()   == to_target.item()
            )
            if correct and not found_mistake:
                consecutive += 1
            else:
                found_mistake = True

        puzzle_results.append((n_solver, consecutive))

    if not puzzle_results:
        return {"loss": float("nan"), "accuracy": float("nan"), "advancement": float("nan"), "n_puzzles": 0}

    n_puzzles   = len(puzzle_results)
    accuracy    = sum(1 for n, k in puzzle_results if k == n) / n_puzzles
    advancement = sum(k / n for n, k in puzzle_results) / n_puzzles
    avg_loss    = total_loss / total_loss_n if total_loss_n > 0 else float("nan")

    return {
        "loss":        avg_loss,
        "accuracy":    accuracy,
        "advancement": advancement,
        "n_puzzles":   n_puzzles,
    }
