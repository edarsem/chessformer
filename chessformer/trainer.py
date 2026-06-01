"""
chessformer/trainer.py

Trainer class. Owns the training loop, val loop, checkpointing, and W&B logging.

Loss:
  from_loss + to_loss  (cross-entropy over 64 squares each)
  + promo_loss only on positions where a promotion occurred (promo_id >= 0)

Metrics logged every step:  loss, from_loss, to_loss, grad_norm, lr
Metrics logged every val:   val_loss, val_from_acc, val_to_acc, val_joint_acc

Device handling:
  autocast:   enabled when cfg.train.mixed_precision and device is cuda or mps
  GradScaler: only on CUDA (MPS doesn't support it reliably)

Multi-GPU:
  Logging and checkpointing gated on rank == 0.
  Caller wraps model in DDP before passing it here.
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
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader


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
        vocab_offsets: dict,   # {"from": int, "to": int, "promo": int}
        rank: int = 0,
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
        self.is_main      = (rank == 0)

        self._use_autocast = (cfg.train.mixed_precision and device.type == "cuda")
        self._scaler = GradScaler() if self._use_autocast else None

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
        step  = start_step
        it    = iter(self.train_loader)
        cfg_t = self.cfg.train

        # Rolling timing accumulators (reset every log_every steps)
        log_every = 50
        t_data = t_forward = t_backward = t_opt = 0.0
        t_wall = time.perf_counter()

        while step < cfg_t.max_steps:
            t0 = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                batch = next(it)
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(step)
            t_data += time.perf_counter() - t0

            metrics = self.train_step(batch)
            t_forward  += metrics.pop("t_forward")
            t_backward += metrics.pop("t_backward")
            t_opt      += metrics.pop("t_opt")
            step += 1

            if self.is_main and step % log_every == 0:
                lr      = self.scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t_wall
                sps     = log_every / elapsed  # steps per second
                print(
                    f"step {step:6d} | loss {metrics['loss']:.4f} "
                    f"| from {metrics['from_loss']:.4f} | to {metrics['to_loss']:.4f} "
                    f"| gnorm {metrics['grad_norm']:.3f} | lr {lr:.2e} "
                    f"| {sps:.1f} steps/s  "
                    f"[data {t_data*1000/log_every:.1f}ms  "
                    f"fwd {t_forward*1000/log_every:.1f}ms  "
                    f"bwd {t_backward*1000/log_every:.1f}ms  "
                    f"opt {t_opt*1000/log_every:.1f}ms]",
                    flush=True,
                )
                if self._wandb:
                    self._wandb.log({
                        "train/loss":       metrics["loss"],
                        "train/from_loss":  metrics["from_loss"],
                        "train/to_loss":    metrics["to_loss"],
                        "train/grad_norm":  metrics["grad_norm"],
                        "train/lr":         lr,
                        "perf/steps_per_s": sps,
                        "perf/ms_data":     t_data     * 1000 / log_every,
                        "perf/ms_forward":  t_forward  * 1000 / log_every,
                        "perf/ms_backward": t_backward * 1000 / log_every,
                        "perf/ms_opt":      t_opt      * 1000 / log_every,
                    }, step=step)
                t_data = t_forward = t_backward = t_opt = 0.0
                t_wall = time.perf_counter()

            if step % cfg_t.val_every == 0:
                t_val0      = time.perf_counter()
                val_metrics = self.run_val()
                t_val       = time.perf_counter() - t_val0
                self.model.train()
                if self.is_main:
                    print(
                        f"  [val] loss {val_metrics['loss']:.4f} "
                        f"| from_acc {val_metrics['from_acc']:.3f} "
                        f"| to_acc {val_metrics['to_acc']:.3f} "
                        f"| joint_acc {val_metrics['joint_acc']:.3f} "
                        f"| val_time {t_val:.1f}s "
                        f"[data {val_metrics.pop('t_data'):.1f}s  fwd {val_metrics.pop('t_fwd'):.1f}s]",
                        flush=True,
                    )
                    if self._wandb:
                        self._wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)
                # reset wall clock so val time doesn't pollute train step timing
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
        self.optimizer.zero_grad()

        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", enabled=self._use_autocast):
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
                move_ids       = _make_move_ids(batch, self.device),
            )
            from_target = batch["from_sq"] - self.offsets["from"]
            to_target   = batch["to_sq"]   - self.offsets["to"]

            from_loss = F.cross_entropy(from_logits, from_target)
            to_loss   = F.cross_entropy(to_logits,   to_target)
            loss      = from_loss + to_loss

            promo_mask = batch["promo"] >= 0
            if promo_mask.any():
                promo_target = batch["promo"][promo_mask] - self.offsets["promo"]
                loss = loss + F.cross_entropy(promo_logits[promo_mask], promo_target)
        _sync(self.device)
        t_fwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        else:
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
        _sync(self.device)
        t_bwd = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self._scaler is not None:
            self._scaler.step(self.optimizer)
            self._scaler.update()
        else:
            self.optimizer.step()
        self.scheduler.step()
        _sync(self.device)
        t_opt = time.perf_counter() - t0

        return {
            "loss":       loss.item(),
            "from_loss":  from_loss.item(),
            "to_loss":    to_loss.item(),
            "grad_norm":  float(grad_norm),
            "t_forward":  t_fwd,
            "t_backward": t_bwd,
            "t_opt":      t_opt,
        }

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        batch = _to_device(batch, self.device)
        with torch.amp.autocast(device_type="cuda", enabled=self._use_autocast):
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
                move_ids       = _make_move_ids(batch, self.device),
            )
            from_target = batch["from_sq"] - self.offsets["from"]
            to_target   = batch["to_sq"]   - self.offsets["to"]

            from_loss = F.cross_entropy(from_logits, from_target)
            to_loss   = F.cross_entropy(to_logits,   to_target)

        from_pred = from_logits.argmax(-1)
        to_pred   = to_logits.argmax(-1)
        from_acc  = (from_pred == from_target).float().mean().item()
        to_acc    = (to_pred   == to_target  ).float().mean().item()
        joint_acc = ((from_pred == from_target) & (to_pred == to_target)).float().mean().item()

        return {
            "loss":      (from_loss + to_loss).item(),
            "from_acc":  from_acc,
            "to_acc":    to_acc,
            "joint_acc": joint_acc,
            "n":         len(from_target),
        }

    def run_val(self) -> dict:
        self.model.eval()
        totals = {"loss": 0.0, "from_acc": 0.0, "to_acc": 0.0, "joint_acc": 0.0, "n": 0}
        t_data = t_fwd = 0.0
        it = iter(self.val_loader)
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
            m = self.val_step(batch)
            _sync(self.device)
            t_fwd += time.perf_counter() - t0

            w = m["n"]
            totals["loss"]      += m["loss"]      * w
            totals["from_acc"]  += m["from_acc"]  * w
            totals["to_acc"]    += m["to_acc"]    * w
            totals["joint_acc"] += m["joint_acc"] * w
            totals["n"]         += w
        n = totals["n"]
        return {
            **{k: totals[k] / n for k in ("loss", "from_acc", "to_acc", "joint_acc")},
            "t_data": t_data,
            "t_fwd":  t_fwd,
        }

    # -----------------------------------------------------------------------
    # Checkpointing
    # -----------------------------------------------------------------------

    def save_checkpoint(self, step: int) -> None:
        os.makedirs(self.cfg.paths.checkpoints_dir, exist_ok=True)
        path = os.path.join(self.cfg.paths.checkpoints_dir, f"step_{step:07d}.pt")
        # Unwrap DDP if needed
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
        ckpt = torch.load(path, map_location=self.device)
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

def _sync(device: torch.device) -> None:
    """Synchronize device so perf_counter captures actual GPU/MPS work."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _make_move_ids(batch: dict, device: torch.device) -> torch.Tensor:
    """Build the [B, 3] move_ids tensor for teacher-forced input to the model."""
    return torch.stack([batch["from_sq"], batch["to_sq"], batch["promo"]], dim=1).to(device)


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
