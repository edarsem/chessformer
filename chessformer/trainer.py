"""
chessformer/trainer.py

Trainer class. Owns the training loop, val loop, checkpointing, and W&B logging.

Precision strategy:
  MPS:          float32 only (MPS float16 still has op coverage gaps)
  CUDA Ampere+: bfloat16 autocast, no GradScaler needed
  CUDA older:   float16 autocast + GradScaler

Multi-GPU:
  Caller wraps model in DDP before passing it here.
  Logging and checkpointing gated on rank == 0.
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


def _autocast_ctx(device: torch.device, dtype: Optional[torch.dtype]):
    """Return the right autocast context manager for this device.
    MPS and CPU don't support autocast — use a no-op cuda context (enabled=False)."""
    if dtype is None:
        return torch.amp.autocast(device_type="cuda", enabled=False)
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        device: torch.device,
        vocab_offsets: dict,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model        = model
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device       = device
        self.offsets      = vocab_offsets
        self.rank         = rank
        self.world_size   = world_size
        self.is_main      = (rank == 0)

        # Precision: None = no autocast (MPS / CPU)
        #            bfloat16 for CUDA Ampere+ (sm >= 8.0)
        #            float16  for older CUDA
        self._autocast_dtype: Optional[torch.dtype] = None
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None

        if cfg.train.mixed_precision and device.type == "cuda":
            major = torch.cuda.get_device_capability(device)[0]
            if major >= 8:
                self._autocast_dtype = torch.bfloat16   # Ampere+: no scaler needed
            else:
                self._autocast_dtype = torch.float16
                self._scaler = torch.cuda.amp.GradScaler()

        self._wandb = None
        if self.is_main and cfg.wandb.enabled:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                print("wandb not installed — skipping W&B logging")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def fit(self, start_step: int = 0) -> None:
        self.model.train()
        step      = start_step
        it        = iter(self.train_loader)
        cfg_t     = self.cfg.train
        log_every = 50
        t_wall    = time.perf_counter()
        # skip the first log window from throughput reporting — torch.compile runs there
        _first_log = True

        while step < cfg_t.max_steps:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(step)

            metrics = self.train_step(batch)
            step += 1

            if self.is_main and step % log_every == 0:
                lr      = self.scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_wall
                sps     = log_every / elapsed
                pos_per_s = sps * cfg_t.batch_size * self.world_size
                if _first_log:
                    # first window includes torch.compile — report time but flag it
                    sps_str = f"{sps:.1f} steps/s (incl. compile)"
                    _first_log = False
                else:
                    sps_str = f"{sps:.1f} steps/s | {pos_per_s:,.0f} pos/s"
                print(
                    f"step {step:6d} | loss {metrics['loss']:.4f} "
                    f"| gnorm {metrics['grad_norm']:.3f} | lr {lr:.2e} "
                    f"| {sps_str}",
                    flush=True,
                )
                if self._wandb and not _first_log:
                    self._wandb.log({
                        "train/loss":         metrics["loss"],
                        "train/grad_norm":    metrics["grad_norm"],
                        "train/lr":           lr,
                        "perf/steps_per_s":   sps,
                        "perf/positions_per_s": pos_per_s,
                    }, step=step)
                t_wall = time.perf_counter()

            if step % cfg_t.val_every == 0:
                t_val0      = time.perf_counter()
                val_metrics = self.run_val()
                t_val       = time.perf_counter() - t_val0
                self.model.train()
                if self.is_main:
                    val_metrics.pop("t_data"); val_metrics.pop("t_fwd")
                    print(
                        f"  [val] loss {val_metrics['loss']:.4f} "
                        f"| acc {val_metrics['joint_acc']:.3f} "
                        f"| {t_val:.1f}s",
                        flush=True,
                    )
                    if self._wandb:
                        self._wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
                    self.save_checkpoint(step, tag="latest")
                t_wall = time.perf_counter()

            if self.is_main and step % cfg_t.checkpoint_every == 0:
                self.save_checkpoint(step)

        if self.is_main:
            self.save_checkpoint(step)
            print(f"Training complete at step {step}.")

    # -----------------------------------------------------------------------
    # Train / val steps
    # -----------------------------------------------------------------------

    def train_step(self, batch: dict) -> dict:
        batch = _to_device(batch, self.device)
        batch = _conditioning_dropout(batch, self.cfg.train, self.device)
        self.optimizer.zero_grad(set_to_none=True)

        with _autocast_ctx(self.device, self._autocast_dtype):
            from_logits, to_logits, promo_logits = self.model(
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
                move_ids       = _make_move_ids(batch, self.device),
            )
            from_target = batch["from_sq"]   # 0-63
            to_target   = batch["to_sq"]     # 0-63
            loss        = F.cross_entropy(from_logits, from_target) + F.cross_entropy(to_logits, to_target)

            promo_mask = batch["promo"] >= 0
            if promo_mask.any():
                promo_target = batch["promo"][promo_mask] - self.offsets["promo"]
                loss = loss + F.cross_entropy(promo_logits[promo_mask], promo_target)

        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()

        self.scheduler.step()

        return {"loss": loss.item(), "grad_norm": float(grad_norm)}

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        batch = _to_device(batch, self.device)
        with _autocast_ctx(self.device, self._autocast_dtype):
            from_logits, to_logits, _ = self.model(
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
                move_ids       = _make_move_ids(batch, self.device),
            )
            from_target = batch["from_sq"]   # 0-63
            to_target   = batch["to_sq"]     # 0-63
            from_loss   = F.cross_entropy(from_logits, from_target)
            to_loss     = F.cross_entropy(to_logits,   to_target)

        from_pred = from_logits.argmax(-1)
        to_pred   = to_logits.argmax(-1)
        joint_acc = ((from_pred == from_target) & (to_pred == to_target)).float().mean().item()

        return {
            "loss":      (from_loss + to_loss).item(),
            "from_acc":  (from_pred == from_target).float().mean().item(),
            "to_acc":    (to_pred   == to_target  ).float().mean().item(),
            "joint_acc": joint_acc,
            "n":         len(from_target),
        }

    def run_val(self) -> dict:
        self.model.eval()
        totals    = {"loss": 0.0, "from_acc": 0.0, "to_acc": 0.0, "joint_acc": 0.0, "n": 0}
        t_data    = t_fwd = 0.0
        it        = iter(self.val_loader)
        max_batches = getattr(self.cfg.train, "val_max_batches", None)
        n_batches = 0

        while True:
            if max_batches is not None and n_batches >= max_batches:
                break
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                break
            t_data += time.perf_counter() - t0
            n_batches += 1

            t0 = time.perf_counter()
            m  = self.val_step(batch)
            t_fwd += time.perf_counter() - t0

            w = m["n"]
            for k in ("loss", "from_acc", "to_acc", "joint_acc"):
                totals[k] += m[k] * w
            totals["n"] += w

        n = totals["n"]
        return {
            **{k: totals[k] / n for k in ("loss", "from_acc", "to_acc", "joint_acc")},
            "t_data": t_data,
            "t_fwd":  t_fwd,
        }

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(self, step: int, tag: str | None = None) -> None:
        os.makedirs(self.cfg.paths.checkpoints_dir, exist_ok=True)
        fname = f"{tag}.pt" if tag else f"step_{step:07d}.pt"
        path  = os.path.join(self.cfg.paths.checkpoints_dir, fname)
        model_state = (
            self.model.module.state_dict()
            if hasattr(self.model, "module") else self.model.state_dict()
        )
        torch.save({
            "step":      step,
            "model":     model_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "cfg":       OmegaConf.to_container(self.cfg, resolve=True),
        }, path)
        print(f"  checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> int:
        ckpt  = torch.load(path, map_location=self.device)
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        step = ckpt["step"]
        print(f"  resumed from {path} (step {step})")
        return step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _conditioning_dropout(batch: dict, cfg_t, device: torch.device) -> dict:
    """
    Randomly mask conditioning scalars to their "unknown" sentinel values (-1 / -1.0).

    Two-stage: first flag each position with probability cond_dropout_rate, then
    independently drop each of the 5 conditioning fields with cond_field_drop_rate.
    Only applied during training (not val).
    """
    rate = float(getattr(cfg_t, "cond_dropout_rate", 0.0))
    if rate <= 0.0:
        return batch

    field_rate = float(getattr(cfg_t, "cond_field_drop_rate", 0.5))
    B = batch["white_elo"].shape[0]

    flagged   = torch.rand(B, device=device) < rate          # [B] — positions to affect
    field_drop = torch.rand(B, 5, device=device) < field_rate # [B, 5] — which fields

    # Only drop fields in flagged positions
    drop = flagged.unsqueeze(1) & field_drop  # [B, 5]

    _FIELDS_INT   = ["white_elo",    "black_elo"]
    _FIELDS_FLOAT = ["white_clock_s", "black_clock_s", "increment_s"]

    for i, key in enumerate(_FIELDS_INT):
        batch[key] = torch.where(drop[:, i], torch.full_like(batch[key], -1), batch[key])
    for j, key in enumerate(_FIELDS_FLOAT):
        batch[key] = torch.where(drop[:, 2 + j], torch.full_like(batch[key], -1.0), batch[key])

    return batch


def _make_move_ids(batch: dict, device: torch.device) -> torch.Tensor:
    # [B, 4]: from_file_id, from_rank_id, to_file_id, to_rank_id
    return torch.stack(
        [batch["from_file"], batch["from_rank"], batch["to_file"], batch["to_rank"]], dim=1
    ).to(device)


def build_lr_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
