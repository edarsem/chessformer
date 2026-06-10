"""
scripts/train_eval_head.py

Train an EvalHead on top of a frozen ChessformerModel backbone.

Only EvalHead parameters are updated. Everything in the backbone is frozen
(requires_grad = False). The head learns to predict tanh(cp / 400) where cp
is the Stockfish centipawn evaluation stored in the parquet dataset.

Usage:
    python scripts/train_eval_head.py checkpoint=checkpoints/step_50000.pt \\
        data.eval_train_file=data/processed/train_eval.parquet \\
        data.eval_val_file=data/processed/val_eval.parquet

The parquet files must have all standard position columns plus a numeric
`cp_eval` column (centipawn, white's perspective, float).

Config: defaults to model=small train=eval_head. Override at CLI.
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chessformer.dataset import make_eval_head_loader
from chessformer.model import ChessformerModel, EvalHead, unwrap_state_dict
from chessformer.tokenizer import build_vocab
from chessformer.trainer import build_lr_schedule


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _forward_batch(backbone: ChessformerModel, batch: dict, device: torch.device) -> torch.Tensor:
    return backbone.forward_board(
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
    )


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = _get_device()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.set_float32_matmul_precision("high")

    # --- Checkpoint (required) -----------------------------------------------
    checkpoint_path = cfg.get("checkpoint")
    if not checkpoint_path:
        raise ValueError("Provide checkpoint=<path> on the CLI")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use the model arch stored in the checkpoint, not the Hydra CLI config.
    # This avoids a mismatch when the checkpoint was trained with a different size.
    ckpt_model_cfg = (ckpt.get("cfg") or {}).get("model") or {}
    d_model  = ckpt_model_cfg.get("d_model",  cfg.model.d_model)
    n_heads  = ckpt_model_cfg.get("n_heads",  cfg.model.n_heads)
    n_layers = ckpt_model_cfg.get("n_layers", cfg.model.n_layers)
    ffn_mult = ckpt_model_cfg.get("ffn_mult", cfg.model.ffn_mult)
    print(f"Backbone arch from checkpoint: d_model={d_model} n_heads={n_heads} n_layers={n_layers} ffn_mult={ffn_mult}")

    # --- Vocab & backbone -----------------------------------------------------
    vocab    = build_vocab()
    backbone = ChessformerModel(
        vocab    = vocab,
        d_model  = d_model,
        n_heads  = n_heads,
        n_layers = n_layers,
        ffn_mult = ffn_mult,
        dropout  = 0.0,
    ).to(device)

    state = ckpt.get("model", ckpt)
    backbone.load_state_dict(unwrap_state_dict(state), strict=True)
    print(f"Loaded backbone from {checkpoint_path}")

    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # --- Eval head ------------------------------------------------------------
    eval_head = EvalHead(d_model).to(device)
    n_head = sum(p.numel() for p in eval_head.parameters())
    print(f"EvalHead: {n_head:,} trainable parameters")

    # --- Data -----------------------------------------------------------------
    cp_clip = cfg.train.get("cp_clip", 3000.0)

    train_path = cfg.data.get("eval_train_file")
    val_path   = cfg.data.get("eval_val_file")
    if not train_path or not val_path:
        raise ValueError(
            "Set data.eval_train_file and data.eval_val_file in config or CLI"
        )

    num_workers = cfg.train.get("num_workers", 0)
    train_loader = make_eval_head_loader(
        train_path,
        batch_size   = cfg.train.batch_size,
        shuffle      = True,
        num_workers  = num_workers,
        pin_memory   = (device.type == "cuda"),
        cp_clip      = cp_clip,
    )
    val_loader = make_eval_head_loader(
        val_path,
        batch_size   = cfg.train.batch_size,
        shuffle      = False,
        num_workers  = 0,
        pin_memory   = (device.type == "cuda"),
        cp_clip      = cp_clip,
    )
    print(f"Train: {len(train_loader.dataset):,} positions  |  Val: {len(val_loader.dataset):,} positions")

    # --- Optimizer & scheduler ------------------------------------------------
    optimizer = torch.optim.AdamW(
        eval_head.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.get("weight_decay", 0.0),
    )
    scheduler = build_lr_schedule(optimizer, cfg.train.warmup_steps, cfg.train.max_steps)

    use_amp = cfg.train.get("mixed_precision", True) and device.type == "cuda"
    amp_ctx = torch.amp.autocast(device_type="cuda") if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- W&B ------------------------------------------------------------------
    run = None
    if cfg.wandb.get("enabled", True):
        try:
            import wandb
            wandb.login(anonymous="never", timeout=10)
            run = wandb.init(
                project = cfg.wandb.project,
                entity  = cfg.wandb.get("entity") or os.environ.get("WANDB_ENTITY"),
                name    = "eval_head",
                config  = {
                    "backbone_ckpt": checkpoint_path,
                    "d_model":       d_model,
                    "n_eval_params": n_head,
                    **dict(cfg.train),
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}")

    # --- Training loop --------------------------------------------------------
    checkpoints_dir = cfg.paths.get("checkpoints_dir", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    step       = 0
    train_iter = iter(train_loader)

    def _val_loss() -> float:
        eval_head.eval()
        total, count = 0.0, 0
        max_batches = cfg.train.get("val_max_batches", None)
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if max_batches and i >= max_batches:
                    break
                with amp_ctx:
                    board_repr = _forward_batch(backbone, batch, device)
                    target     = batch["cp_target"].to(device)
                    pred       = eval_head(board_repr)
                    loss       = F.mse_loss(pred, target)
                total += loss.item() * len(target)
                count += len(target)
        eval_head.train()
        return total / count if count else float("nan")

    eval_head.train()
    while step < cfg.train.max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            board_repr = _forward_batch(backbone, batch, device)
            target     = batch["cp_target"].to(device)
            pred       = eval_head(board_repr)
            loss       = F.mse_loss(pred, target)

        scaler.scale(loss).backward()
        grad_clip = cfg.train.get("grad_clip", 1.0)
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(eval_head.parameters(), grad_clip).item()
        else:
            grad_norm = sum(p.grad.norm().item() ** 2 for p in eval_head.parameters() if p.grad is not None) ** 0.5
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        if step % 100 == 0:
            print(f"step {step:>6}  train_loss={loss.item():.4f}  grad_norm={grad_norm:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if step % cfg.train.val_every == 0:
            val_loss = _val_loss()
            print(f"step {step:>6}  val_loss={val_loss:.4f}")
            if run is not None:
                run.log({"eval_head/train_loss": loss.item(), "eval_head/val_loss": val_loss, "eval_head/grad_norm": grad_norm, "step": step})

        elif run is not None and step % 100 == 0:
            run.log({"eval_head/train_loss": loss.item(), "eval_head/grad_norm": grad_norm, "step": step})

        if step % cfg.train.checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"eval_head_step_{step:06d}.pt")
            torch.save({"step": step, "eval_head": eval_head.state_dict()}, ckpt_path)
            print(f"Saved eval head checkpoint: {ckpt_path}")

    # Final checkpoint
    ckpt_path = os.path.join(checkpoints_dir, f"eval_head_step_{step:06d}.pt")
    torch.save({"step": step, "eval_head": eval_head.state_dict()}, ckpt_path)
    print(f"Training complete. Final checkpoint: {ckpt_path}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
