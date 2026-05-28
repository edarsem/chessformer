"""
chessformer/trainer.py

Trainer class. Owns the training loop, val loop, checkpointing, and logging.
Instantiated from scripts/train.py.

Designed to work on CPU, MPS, and CUDA without branching in calling code.
GradScaler is only active on CUDA (MPS doesn't support it reliably).
"""

# TODO Phase 4: implement
#
# class Trainer:
#     def __init__(self, model, optimizer, scheduler, train_loader, val_loader,
#                  cfg, device, rank=0):
#         ...
#
#     def fit(self) -> None:
#         """Run training loop for cfg.train.max_steps steps."""
#
#     def train_step(self, batch) -> dict:
#         """Single forward+backward+step. Returns loss dict."""
#
#     def val_step(self, batch) -> dict:
#         """No-grad forward. Returns loss + accuracy dict."""
#
#     def run_val(self) -> dict:
#         """Full val loop over val_loader. Returns aggregated metrics."""
#
#     def save_checkpoint(self, step: int) -> None:
#         """Save model, optimizer, scheduler state + config to disk."""
#
#     def load_checkpoint(self, path: str) -> int:
#         """Load checkpoint, return step to resume from."""
#
# Notes:
#   - Log grad norm every step (early warning for instability)
#   - W&B logging gated on rank == 0
#   - autocast: device_type=device.type, enabled=cfg.train.mixed_precision
#   - GradScaler: only if device.type == "cuda"
#   - Checkpoint format: {"step": int, "model": state_dict, "optimizer": ...,
#                          "scheduler": ..., "cfg": OmegaConf.to_container(cfg)}
