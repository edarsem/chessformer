"""
scripts/eval_offline.py

Offline evaluation against val or test splits.

Usage:
    python scripts/eval_offline.py checkpoint=checkpoints/step_XXXXXXX.pt
    python scripts/eval_offline.py checkpoint=... split=test data=local

Results are printed to stdout. Optionally logged to W&B if cfg.wandb.enabled.
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.dataset import make_loader
from chessformer.eval import eval_games, eval_puzzles
from chessformer.model import ChessformerModel
from chessformer.tokenizer import build_vocab


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.checkpoint
    split     = cfg.split
    assert ckpt_path, "Pass checkpoint=<path> on the command line."
    assert split in ("val", "test"), "split must be 'val' or 'test'"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}  |  Checkpoint: {ckpt_path}  |  Split: {split}")

    ckpt = torch.load(ckpt_path, map_location=device)
    saved_cfg = OmegaConf.create(ckpt["cfg"])

    vocab = build_vocab()
    model = ChessformerModel(
        vocab    = vocab,
        d_model  = saved_cfg.model.d_model,
        n_heads  = saved_cfg.model.n_heads,
        n_layers = saved_cfg.model.n_layers,
        ffn_mult = saved_cfg.model.ffn_mult,
        dropout  = saved_cfg.model.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    step = ckpt.get("step", 0)
    print(f"Loaded model at step {step}  ({sum(p.numel() for p in model.parameters()):,} params)")

    if split == "val":
        games_path   = cfg.data.val_games_file
        puzzles_path = cfg.data.val_puzzles_file
    else:
        games_path   = cfg.data.test_games_file
        puzzles_path = cfg.data.test_puzzles_file

    # --- Games eval ----------------------------------------------------------
    print(f"\n{'='*60}\nGames ({split}): {games_path}")
    loader  = make_loader(games_path, batch_size=cfg.train.batch_size, shuffle=False)
    metrics = eval_games(model, loader, device, vocab)

    print(f"\n  loss           {metrics['loss']:.4f}")
    print(f"  top-1 acc      {metrics['top1_acc']:.3f}")
    print(f"  plausible rate {metrics['plausible_rate']:.3f}  (p(correct move) >= 20%)")
    print(f"  positions      {metrics['n']:,}")

    if metrics.get("per_elo"):
        print(f"\n  Per-Elo breakdown:")
        for bucket in sorted(metrics["per_elo"].keys()):
            m = metrics["per_elo"][bucket]
            print(f"    {bucket:12s}  top1={m['top1_acc']:.3f}"
                  f"  plausible={m['plausible_rate']:.3f}  n={m['n']:,}")

    # --- Puzzles eval --------------------------------------------------------
    print(f"\n{'='*60}\nPuzzles ({split}): {puzzles_path}")
    pm = eval_puzzles(model, puzzles_path, device, vocab)

    print(f"\n  loss        {pm['loss']:.4f}")
    print(f"  accuracy    {pm['accuracy']:.3f}  (all moves correct)")
    print(f"  advancement {pm['advancement']:.3f}  (macro avg fraction solved)")
    print(f"  puzzles     {pm['n_puzzles']:,}")

    # --- W&B -----------------------------------------------------------------
    if cfg.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project = cfg.wandb.project,
                entity  = cfg.wandb.get("entity", None),
                name    = f"eval_{split}_step{step}",
                config  = OmegaConf.to_container(saved_cfg, resolve=True),
            )
            wandb.log({
                f"{split}/games/loss":            metrics["loss"],
                f"{split}/games/top1_acc":        metrics["top1_acc"],
                f"{split}/games/plausible_rate":  metrics["plausible_rate"],
                f"{split}/puzzles/loss":        pm["loss"],
                f"{split}/puzzles/accuracy":    pm["accuracy"],
                f"{split}/puzzles/advancement": pm["advancement"],
            }, step=step)
            wandb.finish()
        except ImportError:
            print("wandb not installed — skipping W&B logging")


if __name__ == "__main__":
    main()
