"""
scripts/eval.py

Offline evaluation: Elo-balanced games, unfiltered games, puzzles.
Adapts to available hardware. Logs to W&B.

Usage (local):
    python scripts/eval.py checkpoint=checkpoints/chessformer_v0.pt split=test

Override batch size:
    python scripts/eval.py checkpoint=... eval_batch_size=1024

Kaggle / cluster:
    python scripts/eval.py checkpoint=... data=kaggle split=test eval_batch_size=2048

Val instead of test:
    python scripts/eval.py checkpoint=... split=val wandb.enabled=false
"""

from __future__ import annotations

import os
import sys
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.dataset import make_loader
from chessformer.eval import eval_games, eval_puzzles
from chessformer.model import ChessformerModel, unwrap_state_dict
from chessformer.tokenizer import build_vocab

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: str, device: torch.device):
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
        state = {k: v.float() for k, v in state.items()}  # fp16 weights → fp32 for MPS/CPU
    model.load_state_dict(state, strict=True)
    model.eval()  # disables dropout
    return model, saved_cfg, ckpt.get("step", 0)


def _autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _print_games(label: str, m: dict) -> None:
    console.rule(f"[bold cyan]{label}[/]")
    console.print(
        f"  loss [yellow]{m['loss']:.4f}[/]  "
        f"top-1 [green]{m['top1_acc']*100:.1f}%[/]  "
        f"plausible [blue]{m['plausible_rate']*100:.1f}%[/]  "
        f"n={m['n']:,}"
    )
    if m.get("per_elo"):
        t = Table("Elo bucket", "top-1 %", "plausible %", "n",
                  show_header=True, header_style="bold dim", box=None)
        for bkt in sorted(m["per_elo"]):
            b = m["per_elo"][bkt]
            t.add_row(bkt, f"{b['top1_acc']*100:.1f}", f"{b['plausible_rate']*100:.1f}", str(b["n"]))
        console.print(t)


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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
    # On CUDA: more workers + pin memory for faster data loading
    workers    = 4 if device.type == "cuda" else 0
    pin_mem    = device.type == "cuda"

    model, saved_cfg, step = _load_model(ckpt_path, device)
    n_params = sum(p.numel() for p in model.parameters())

    console.print(f"\n[bold]Checkpoint:[/] {ckpt_path}  step={step}")
    console.print(f"[bold]Device:[/] {device}  autocast={'bf16' if acd==torch.bfloat16 else 'fp16' if acd else 'off'}")
    console.print(f"[bold]Model:[/] {n_params/1e6:.1f}M params  d={saved_cfg.model.d_model}  "
                  f"L={saved_cfg.model.n_layers}  H={saved_cfg.model.n_heads}  "
                  f"(dropout={saved_cfg.model.dropout} — [dim]inactive in eval[/])")
    console.print(f"[bold]Batch size:[/] {batch_size}  split={split}\n")

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

    prog_cols = [TextColumn("[progress.description]{task.description}"),
                 BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()]

    # ── Elo-balanced games ───────────────────────────────────────────────
    t0     = time.perf_counter()
    loader = games_loader(games_path)
    console.print(f"Games (Elo-balanced)  {len(loader)} batches …")
    with Progress(*prog_cols, console=console) as prog:
        task   = prog.add_task("  eval", total=len(loader))
        games_m = eval_games(model, loader, device, acd, progress_cb=lambda: prog.advance(task))
    console.print(f"  → {time.perf_counter()-t0:.1f}s")
    _print_games(f"{split} / games (Elo-balanced)", games_m)

    # ── Unfiltered games ─────────────────────────────────────────────────
    games_elo_m = None
    if games_elo_path and os.path.exists(games_elo_path):
        t0     = time.perf_counter()
        loader = games_loader(games_elo_path)
        console.print(f"\nGames (unfiltered)  {len(loader)} batches …")
        with Progress(*prog_cols, console=console) as prog:
            task        = prog.add_task("  eval", total=len(loader))
            games_elo_m = eval_games(model, loader, device, acd, progress_cb=lambda: prog.advance(task))
        console.print(f"  → {time.perf_counter()-t0:.1f}s")
        _print_games(f"{split} / games (unfiltered)", games_elo_m)

    # ── Puzzles ──────────────────────────────────────────────────────────
    import polars as pl
    n_puzzles_total = pl.read_parquet(puzzles_path).select("puzzle_id").n_unique()
    t0 = time.perf_counter()
    console.print(f"\nPuzzles  {n_puzzles_total} puzzles …")
    with Progress(*prog_cols, console=console) as prog:
        task  = prog.add_task("  eval", total=n_puzzles_total)
        puz_m = eval_puzzles(model, puzzles_path, device, acd, progress_cb=lambda: prog.advance(task))
    console.print(f"  → {time.perf_counter()-t0:.1f}s")
    console.rule("[bold cyan]Puzzles[/]")
    console.print(
        f"  loss [yellow]{puz_m['loss']:.4f}[/]  "
        f"solved [green]{puz_m['accuracy']*100:.1f}%[/]  "
        f"advancement [blue]{puz_m['advancement']*100:.1f}%[/]  "
        f"n={puz_m['n_puzzles']:,}"
    )

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
            console.print("\n[green]Logged to W&B.[/]")
        except ImportError:
            console.print("[yellow]wandb not installed — skipping.[/]")


if __name__ == "__main__":
    main()
