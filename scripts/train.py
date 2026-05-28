"""
scripts/train.py

Main training entry point. Uses Hydra for config management.

Usage (local, single device):
    python scripts/train.py
    python scripts/train.py model=small train.lr=3e-4
    python scripts/train.py data=full model=medium

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 scripts/train.py data=full model=medium

Override any config value at the CLI with dot notation.
Hydra logs the full config to outputs/<date>/<time>/.hydra/ automatically.
"""

import os
import torch
import torch.distributed as dist
import hydra
from omegaconf import DictConfig


def setup_distributed() -> tuple[int, int]:
    """Initialize process group if launched with torchrun. Returns (rank, world_size)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    rank, world_size = setup_distributed()
    device = get_device()

    if rank == 0:
        print(f"Device: {device}  |  World size: {world_size}")

    # TODO Phase 3-4: implement
    # Steps:
    #   1. Build vocab: chessformer.tokenizer.build_vocab()
    #   2. Load datasets: chessformer.dataset.ChessDataset(cfg.paths.train_games_file)
    #      Use DistributedSampler if world_size > 1
    #   3. Build model: chessformer.model.ChessTransformer(cfg.model)
    #      Wrap in DDP if world_size > 1
    #   4. Build optimizer + scheduler
    #   5. Optionally resume: cfg.train.resume_from
    #   6. Init W&B on rank 0 if cfg.wandb.enabled
    #   7. trainer = chessformer.trainer.Trainer(...)
    #   8. trainer.fit()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
