"""
chessformer/eval.py

Core evaluation functions for games and puzzles.
Called by scripts/eval.py — not intended to be run directly.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Optional

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    if dtype is None or device.type not in ("cuda", "cpu"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_games(
    model:          nn.Module,
    loader:         DataLoader,
    device:         torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
    max_batches:    Optional[int]          = None,
    progress_cb     = None,   # callable() called after each batch
) -> dict:
    """
    Returns:
        loss, top1_acc, plausible_rate, n
        per_elo: {bucket_str: {loss, top1_acc, plausible_rate, n}}
    """
    model.eval()
    totals: dict     = defaultdict(lambda: {"top1": 0, "plausible 20%": 0, "n": 0})
    loss_sum: float  = 0.0
    n_batches: int   = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        elo_bkts = batch["elo_bucket"]
        move_ids = torch.stack([
            batch["from_file"], batch["from_rank"],
            batch["to_file"],   batch["to_rank"],
        ], dim=1).to(device)

        with _autocast(device, autocast_dtype):
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
                increment_s    = batch["increment_s"].to(device),
                move_ids       = move_ids,
            )

        from_logits = from_logits.float()
        to_logits   = to_logits.float()
        from_target = batch["from_sq"].to(device)
        to_target   = batch["to_sq"].to(device)

        # cross_entropy already averages over the batch — accumulate as batch mean
        loss_sum  += (F.cross_entropy(from_logits, from_target)
                      + F.cross_entropy(to_logits, to_target)).item()
        n_batches += 1

        from_pred = from_logits.argmax(-1)
        to_pred   = to_logits.argmax(-1)
        top1      = (from_pred == from_target) & (to_pred == to_target)

        from_p = F.softmax(from_logits, dim=-1).gather(1, from_target.unsqueeze(1)).squeeze(1)
        to_p   = F.softmax(to_logits,   dim=-1).gather(1, to_target.unsqueeze(1)).squeeze(1)
        plaus  = (from_p >= 0.20) & (to_p >= 0.20)

        B = len(elo_bkts)
        for b in range(B):
            bkt = elo_bkts[b]
            for key in ("all", bkt):
                totals[key]["top1"]      += int(top1[b].item())
                totals[key]["plausible 20%"] += int(plaus[b].item())
                totals[key]["n"]         += 1

        if progress_cb is not None:
            progress_cb()

    avg_loss = loss_sum / n_batches if n_batches else float("nan")

    def _agg(d: dict) -> dict:
        n = d["n"]
        return {"top1_acc": d["top1"]/n, "plausible 20%": d["plausible 20%"]/n, "n": n}

    result              = _agg(totals["all"])
    result["loss"]      = avg_loss
    result["per_elo"]   = {k: _agg(v) for k, v in totals.items() if k != "all"}
    return result


# ---------------------------------------------------------------------------
# Puzzles
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_puzzles(
    model:          nn.Module,
    parquet_path:   str,
    device:         torch.device,
    autocast_dtype: Optional[torch.dtype] = None,
    progress_cb     = None,   # callable() called after each puzzle
) -> dict:
    """
    Returns:
        loss, accuracy (all moves correct), advancement (avg fraction solved), n_puzzles
    """
    model.eval()
    df = pl.read_parquet(parquet_path)

    puzzle_results: list[tuple[int, int]] = []
    total_loss   = 0.0
    total_loss_n = 0

    for _, puzzle_df in df.group_by("puzzle_id"):
        puzzle_df   = puzzle_df.sort("seq_index")
        solver_rows = puzzle_df.filter(pl.col("is_solver_move"))
        if len(solver_rows) == 0:
            continue

        n_solver      = len(solver_rows)
        consecutive   = 0
        found_mistake = False

        for row in solver_rows.iter_rows(named=True):
            meta  = torch.tensor([row["meta_tokens"]],       dtype=torch.long,    device=device)
            color = torch.tensor([row["color_tokens"]],      dtype=torch.long,    device=device)
            ptype = torch.tensor([row["piece_type_tokens"]], dtype=torch.long,    device=device)
            filei = torch.tensor([row["file_tokens"]],       dtype=torch.long,    device=device)
            ranki = torch.tensor([row["rank_tokens"]],       dtype=torch.long,    device=device)
            w_elo = torch.tensor([row["white_elo"]],         dtype=torch.long,    device=device)
            b_elo = torch.tensor([row["black_elo"]],         dtype=torch.long,    device=device)
            w_clk = torch.tensor([row["white_clock_seconds"]], dtype=torch.float32, device=device)
            b_clk = torch.tensor([row["black_clock_seconds"]], dtype=torch.float32, device=device)
            incr  = torch.tensor([row["increment_seconds"]], dtype=torch.float32, device=device)
            move_in = torch.tensor(
                [[row["from_file_id"], row["from_rank_id"],
                  row["to_file_id"],   row["to_rank_id"]]],
                dtype=torch.long, device=device,
            )

            with _autocast(device, autocast_dtype):
                from_logits, to_logits, _ = model(
                    meta_ids=meta, color_ids=color, piece_type_ids=ptype,
                    file_ids=filei, rank_ids=ranki,
                    white_elo=w_elo, black_elo=b_elo,
                    white_clock_s=w_clk, black_clock_s=b_clk,
                    increment_s=incr, move_ids=move_in,
                )

            from_logits = from_logits.float()
            to_logits   = to_logits.float()
            from_tgt = torch.tensor([row["from_square_id"]], dtype=torch.long, device=device)
            to_tgt   = torch.tensor([row["to_square_id"]],   dtype=torch.long, device=device)

            loss = (F.cross_entropy(from_logits, from_tgt)
                    + F.cross_entropy(to_logits, to_tgt)).item()
            total_loss   += loss
            total_loss_n += 1

            correct = (from_logits.argmax(-1).item() == from_tgt.item() and
                       to_logits.argmax(-1).item()   == to_tgt.item())
            if correct and not found_mistake:
                consecutive += 1
            else:
                found_mistake = True

        puzzle_results.append((n_solver, consecutive))
        if progress_cb is not None:
            progress_cb()

    if not puzzle_results:
        return {"loss": float("nan"), "accuracy": float("nan"),
                "advancement": float("nan"), "n_puzzles": 0}

    n_puzzles   = len(puzzle_results)
    accuracy    = sum(1 for n, k in puzzle_results if k == n) / n_puzzles
    advancement = sum(k / n for n, k in puzzle_results) / n_puzzles
    avg_loss    = total_loss / total_loss_n if total_loss_n else float("nan")

    return {"loss": avg_loss, "accuracy": accuracy,
            "advancement": advancement, "n_puzzles": n_puzzles}
