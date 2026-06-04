"""
scripts/train.py

Main training entry point. Uses Hydra for config management.

Usage (local, single device):
    python scripts/train.py
    python scripts/train.py model=small train.lr=3e-4
    python scripts/train.py train.train_file=train_games_tiny_file  # smoke test

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 scripts/train.py model=medium

Override any config value at the CLI with dot notation.
Hydra logs the full config to outputs/<date>/<time>/.hydra/ automatically.
"""

from __future__ import annotations

import os
import sys

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chessformer.dataset import make_loader
from chessformer.model import ChessformerModel
from chessformer.tokenizer import build_vocab
from chessformer.trainer import Trainer, build_lr_schedule


def _setup_distributed() -> tuple[int, int]:
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size()


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, world_size = _setup_distributed()
    device           = _get_device()

    # CUDA-only global settings — no-ops on CPU/MPS
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        torch.set_float32_matmul_precision("high")

    if rank == 0:
        print(f"Device: {device}  |  World size: {world_size}")

    # --- Vocab & data --------------------------------------------------------
    vocab = build_vocab()

    train_file_key = cfg.train.get("train_file", "train_games_small_file")
    train_path     = cfg.data[train_file_key]
    val_path       = cfg.data.val_games_file

    if rank == 0:
        print(f"Train: {train_path}")
        print(f"Val:   {val_path}")

    num_workers = cfg.train.get("num_workers", 0)
    train_loader = make_loader(
        train_path,
        batch_size       = cfg.train.batch_size,
        shuffle          = True,
        num_workers      = num_workers,
        rank             = rank,
        world_size       = world_size,
        pin_memory       = (device.type == "cuda"),
        persistent_workers = (num_workers > 0),
    )
    val_loader = make_loader(
        val_path,
        batch_size       = cfg.train.batch_size,
        shuffle          = False,
        num_workers      = 0,  # val runs infrequently; workers waste RAM
        rank             = rank,
        world_size       = world_size,
        pin_memory       = (device.type == "cuda"),
        persistent_workers = False,
    )

    # --- Model ---------------------------------------------------------------
    model = ChessformerModel(
        vocab                  = vocab,
        d_model                = cfg.model.d_model,
        n_heads                = cfg.model.n_heads,
        n_layers               = cfg.model.n_layers,
        ffn_mult               = cfg.model.ffn_mult,
        dropout                = cfg.model.dropout,
        gradient_checkpointing = cfg.train.get("gradient_checkpointing", False),
    ).to(device)

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{'='*60}")
        print(f"  Model:  ChessformerModel  |  {n_params/1e6:.1f}M params")
        print(f"  Config: d_model={cfg.model.d_model}  n_heads={cfg.model.n_heads}  n_layers={cfg.model.n_layers}")
        print(f"  Batch:  {cfg.train.batch_size} × {world_size} GPUs = {cfg.train.batch_size * world_size} pos/step")
        print(f"{'='*60}")

    if cfg.train.get("compile", False) and device.type == "cuda":
        compile_mode = cfg.train.get("compile_mode", "default")
        model = torch.compile(model, mode=compile_mode)
        if rank == 0:
            print("Model compiled with torch.compile")

    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["LOCAL_RANK"])],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # --- Optimizer & scheduler -----------------------------------------------
    _adamw_doc = torch.optim.AdamW.__init__.__doc__ or ""
    use_fused = device.type == "cuda" and "fused" in _adamw_doc
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.weight_decay,
        fused        = use_fused,
    )
    scheduler = build_lr_schedule(optimizer, cfg.train.warmup_steps, cfg.train.max_steps)

    # --- W&B -----------------------------------------------------------------
    if rank == 0 and cfg.wandb.enabled:
        try:
            import wandb
            wandb.init(
                project = cfg.wandb.project,
                entity  = cfg.wandb.get("entity") or None,
                config  = {
                    "model":  dict(cfg.model),
                    "train":  dict(cfg.train),
                    "n_params": sum(p.numel() for p in model.parameters()),
                },
            )
        except ImportError:
            print("wandb not installed — skipping")

    # --- Trainer -------------------------------------------------------------
    vocab_offsets = {
        "promo": vocab.promo_offset,
    }
    trainer = Trainer(
        model        = model,
        optimizer    = optimizer,
        scheduler    = scheduler,
        train_loader = train_loader,
        val_loader   = val_loader,
        cfg          = cfg,
        device       = device,
        vocab_offsets= vocab_offsets,
        rank         = rank,
        world_size   = world_size,
    )

    # --- Resume --------------------------------------------------------------
    start_step = 0
    if cfg.train.get("resume_from"):
        start_step = trainer.load_checkpoint(cfg.train.resume_from)

    # --- Train ---------------------------------------------------------------
    trainer.fit(start_step=start_step)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
