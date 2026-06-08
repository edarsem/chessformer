"""
scripts/eval.py

Offline evaluation: Elo-balanced games, unfiltered games, puzzles.
Adapts to available hardware (1 or 2 GPUs, MPS, CPU). Logs to W&B.

Usage (local):
    python scripts/eval.py checkpoint=checkpoints/chessformer_v0.pt split=test

Override batch size:
    python scripts/eval.py checkpoint=... eval_batch_size=1024

Kaggle / cluster:
    python scripts/eval.py checkpoint=... data=kaggle split=test eval_batch_size=2048
"""

from __future__ import annotations

import os
import sys
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.dataset import make_loader
from chessformer.eval import eval_games, eval_puzzles
from chessformer.model import ChessformerModel, unwrap_state_dict
from chessformer.tokenizer import build_vocab


def _load_model(ckpt_path: str, device: torch.device) -> tuple[nn.Module, object, int]:
    ckpt      = torch.load(ckpt_path, map_location=device)
    saved_cfg = OmegaConf.create(ckpt["cfg"])
    vocab     = build_vocab()
    model     = ChessformerModel(
        vocab    = vocab,
        d_model  = saved_cfg.model.d_model,
        n_heads  = saved_cfg.model.n_heads,
        n_layers = saved_cfg.model.n_layers,
        ffn_mult = saved_cfg.model.ffn_mult,
        dropout  = saved_cfg.model.dropout,
    ).to(device)
    state = unwrap_state_dict(ckpt["model"])
    if device.type != "cuda":
        state = {k: v.float() for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, saved_cfg, ckpt.get("step", 0)


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16


def _print_games(label: str, m: dict) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {label}")
    print(sep)
    print(f"  loss      {m['loss']:.4f}")
    print(f"  top-1     {m['top1_acc']*100:.2f}%")
    print(f"  plausible {m['plausible_rate']*100:.2f}%   (p(correct move) ≥ 20% on each margin)")
    print(f"  n         {m['n']:,}")
    if m.get("per_elo"):
        print(f"\n  {'Elo bucket':14s}  {'top-1':>7}  {'plausible':>9}  {'n':>6}")
        for bkt in sorted(m["per_elo"]):
            b = m["per_elo"][bkt]
            print(f"  {bkt:14s}  {b['top1_acc']*100:6.1f}%  {b['plausible_rate']*100:8.1f}%  {b['n']:>6,}")
    print(flush=True)


def _log_wandb(wandb, split: str, games: dict, games_elo: dict | None,
               puzzles: dict, step: int) -> None:
    log = {
        f"{split}/games/loss":           games["loss"],
        f"{split}/games/top1_acc":       games["top1_acc"],
        f"{split}/games/plausible_rate": games["plausible_rate"],
        f"{split}/puzzles/loss":         puzzles["loss"],
        f"{split}/puzzles/accuracy":     puzzles["accuracy"],
        f"{split}/puzzles/advancement":  puzzles["advancement"],
    }
    if games_elo is not None:
        log.update({
            f"{split}/games_plain/loss":           games_elo["loss"],
            f"{split}/games_plain/top1_acc":       games_elo["top1_acc"],
            f"{split}/games_plain/plausible_rate": games_elo["plausible_rate"],
        })
    if games.get("per_elo"):
        rows = [[bkt, round(b["top1_acc"]*100, 2), round(b["plausible_rate"]*100, 2), b["n"]]
                for bkt, b in sorted(games["per_elo"].items())]
        log[f"{split}/per_elo"] = wandb.Table(
            columns=["elo_bucket", "top1_pct", "plausible_pct", "n"], data=rows
        )
    wandb.log(log, step=step)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    split     = cfg.get("split", "test")
    ckpt_path = cfg.checkpoint
    assert ckpt_path, "Pass checkpoint=<path>"
    assert split in ("val", "test"), "split must be val or test"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    acd        = _autocast_dtype(device)
    batch_size = int(cfg.get("eval_batch_size", 512))
    n_gpus     = torch.cuda.device_count() if device.type == "cuda" else 0
    workers    = 4 if device.type == "cuda" else 0
    pin_mem    = device.type == "cuda"

    model, saved_cfg, step = _load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())

    # Wrap with DataParallel if multiple GPUs available
    # Scales batch size so each GPU sees the requested batch_size
    if n_gpus > 1:
        model      = nn.DataParallel(model)
        batch_size = batch_size * n_gpus
        print(f"DataParallel: {n_gpus} GPUs → effective batch size {batch_size}", flush=True)

    print(f"\nCheckpoint : {ckpt_path}  (step {step})", flush=True)
    print(f"Device     : {device}  n_gpus={max(n_gpus,1)}  "
          f"autocast={'bf16' if acd==torch.bfloat16 else 'fp16' if acd else 'off'}", flush=True)
    print(f"Model      : {n_params/1e6:.1f}M params  "
          f"d={saved_cfg.model.d_model}  L={saved_cfg.model.n_layers}  "
          f"H={saved_cfg.model.n_heads}  (dropout inactive in eval)", flush=True)
    print(f"Batch size : {batch_size}  split={split}\n", flush=True)

    def games_loader(path: str):
        return make_loader(path, batch_size=batch_size, shuffle=False,
                           num_workers=workers, pin_memory=pin_mem)

    if split == "val":
        games_path     = cfg.data.val_games_file
        games_elo_path = None
        puzzles_path   = cfg.data.val_puzzles_file
    else:
        games_path     = cfg.data.test_games_file
        games_elo_path = cfg.data.get("test_games_elo_file", None)
        puzzles_path   = cfg.data.test_puzzles_file

    # ── Elo-balanced games ───────────────────────────────────────────────
    t0     = time.perf_counter()
    loader = games_loader(games_path)
    bar    = tqdm(total=len(loader), desc="games (balanced)", unit="batch", leave=True)
    games_m = eval_games(model, loader, device, acd, progress_cb=bar.update)
    bar.close()
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
    _print_games(f"{split} / games (Elo-balanced)", games_m)

    # ── Unfiltered games ─────────────────────────────────────────────────
    games_elo_m = None
    if games_elo_path and os.path.exists(games_elo_path):
        t0     = time.perf_counter()
        loader = games_loader(games_elo_path)
        bar    = tqdm(total=len(loader), desc="games (plain)  ", unit="batch", leave=True)
        games_elo_m = eval_games(model, loader, device, acd, progress_cb=bar.update)
        bar.close()
        print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)
        _print_games(f"{split} / games (unfiltered)", games_elo_m)

    # ── Puzzles ──────────────────────────────────────────────────────────
    # Puzzles are evaluated one at a time; use the base model to avoid DataParallel overhead
    import polars as pl
    base_model       = model.module if isinstance(model, nn.DataParallel) else model
    n_puzzles_total  = pl.read_parquet(puzzles_path).select("puzzle_id").n_unique()
    t0  = time.perf_counter()
    bar = tqdm(total=n_puzzles_total, desc="puzzles        ", unit="puzzle", leave=True)
    puz_m = eval_puzzles(base_model, puzzles_path, device, acd, progress_cb=bar.update)
    bar.close()
    print(f"  done in {time.perf_counter()-t0:.1f}s", flush=True)

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  {split} / puzzles")
    print(sep)
    print(f"  loss        {puz_m['loss']:.4f}")
    print(f"  solved      {puz_m['accuracy']*100:.2f}%   (all moves correct)")
    print(f"  advancement {puz_m['advancement']*100:.2f}%   (avg fraction of puzzle solved)")
    print(f"  n           {puz_m['n_puzzles']:,}", flush=True)

    # ── W&B ─────────────────────────────────────────────────────────────
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project = cfg.wandb.project,
                entity  = cfg.wandb.get("entity", None),
                name    = f"eval_{split}_step{step}",
                config  = OmegaConf.to_container(saved_cfg, resolve=True),
            )
            _log_wandb(wandb, split, games_m, games_elo_m, puz_m, step)
            wandb.finish()
            print("\nLogged to W&B.", flush=True)
        except ImportError:
            print("wandb not installed — skipping.")


if __name__ == "__main__":
    main()
