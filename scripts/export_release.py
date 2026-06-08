"""
scripts/export_release.py

Export a training checkpoint to a release-ready weights file.

A training checkpoint is ~600MB because it stores optimizer state, scheduler
state, and torch.compile / DDP key prefixes. A release file strips all of
that down to just the model weights + config, and converts to fp16 — typically
~100MB, suitable for uploading to Hugging Face.

Usage:
    python scripts/export_release.py checkpoints/chessformer_v0_wrapped.pt checkpoints/chessformer_v0.pt

    # Keep fp32 (larger but lossless)
    python scripts/export_release.py input.pt output.pt --fp32
"""

from __future__ import annotations

import argparse
import os
import sys

import chess
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.model import ChessformerModel, unwrap_state_dict
from chessformer.tokenizer import build_vocab
from chessformer.inference import analyze_position


def main() -> None:
    parser = argparse.ArgumentParser(description="Export training checkpoint to release weights")
    parser.add_argument("input",         help="Path to training checkpoint (.pt)")
    parser.add_argument("output",        help="Output path for release file (.pt)")
    parser.add_argument("--fp32", action="store_true", help="Keep fp32 instead of converting to fp16")
    args = parser.parse_args()

    use_fp16 = not args.fp32

    print(f"Loading:  {args.input}")
    ckpt = torch.load(args.input, map_location="cpu")

    step      = ckpt.get("step", 0)
    saved_cfg = OmegaConf.create(ckpt["cfg"])
    print(f"  step={step}  model={dict(saved_cfg.model)}")

    clean_state = unwrap_state_dict(ckpt["model"])

    if use_fp16:
        clean_state = {
            k: v.half() if v.dtype == torch.float32 else v
            for k, v in clean_state.items()
        }
        print("  Converted weights to fp16")

    release = {
        "step":  step,
        "model": clean_state,
        "cfg":   OmegaConf.to_container(saved_cfg, resolve=True),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(release, args.output)
    in_mb  = os.path.getsize(args.input)  / 1e6
    out_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved:    {args.output}  ({out_mb:.1f} MB)  [was {in_mb:.1f} MB]")

    # Smoke-test: reload and run on the starting position
    print("\nSmoke test …")
    vocab = build_vocab()
    model = ChessformerModel(
        vocab    = vocab,
        d_model  = saved_cfg.model.d_model,
        n_heads  = saved_cfg.model.n_heads,
        n_layers = saved_cfg.model.n_layers,
        ffn_mult = saved_cfg.model.ffn_mult,
        dropout  = saved_cfg.model.dropout,
    ).float()  # fp32 inference on CPU regardless of stored dtype
    rel = torch.load(args.output, map_location="cpu")
    model.load_state_dict(unwrap_state_dict(rel["model"]), strict=True)
    model.eval()

    probs, best = analyze_position(
        model, vocab, torch.device("cpu"), chess.Board(),
        white_elo=2500, black_elo=2500,
    )
    print(f"  Best opening move: {best}")
    for m in probs["top_moves"][:3]:
        print(f"    {chess.square_name(m['from'])}{chess.square_name(m['to'])}  p={m['prob']:.3f}")
    print("Export OK.")


if __name__ == "__main__":
    main()
