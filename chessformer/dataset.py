"""
chessformer/dataset.py

Dataset and DataLoader utilities for reading tokenized chess positions from parquet files.

Each parquet row is one position with a known move (from_square_id always set for game rows).
Data is loaded fully into memory at init time for fast random access during shuffled training.

Scalar fields (elo, clock, from/to/promo IDs) are stored as numpy arrays so __getitem__
can use torch.from_numpy (zero-copy) instead of torch.tensor (allocates + copies).
Variable-length lists (meta, piece tokens) must stay as Python lists since lengths differ.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import polars as pl


class PositionDataset(Dataset):
    """
    Reads a games or puzzles parquet file. Each item is one chess position.
    Only rows with a valid from_square_id (>= 0) are included — this automatically
    skips opponent moves in puzzle sequences.
    """

    def __init__(self, parquet_path: str):
        df = pl.read_parquet(parquet_path)
        df = df.filter(pl.col("from_square_id") >= 0)

        # Variable-length lists: must stay as Python lists (ragged rows)
        self.meta_tokens       = df["meta_tokens"].to_list()
        self.color_tokens      = df["color_tokens"].to_list()
        self.piece_type_tokens = df["piece_type_tokens"].to_list()
        self.file_tokens       = df["file_tokens"].to_list()
        self.rank_tokens       = df["rank_tokens"].to_list()

        # Fixed-size scalars: numpy arrays → torch.from_numpy in __getitem__ (zero-copy)
        self.white_elo    = df["white_elo"].to_numpy().astype(np.int64)
        self.black_elo    = df["black_elo"].to_numpy().astype(np.int64)
        self.white_clock_s = df["white_clock_seconds"].to_numpy().astype(np.float32)
        self.black_clock_s = df["black_clock_seconds"].to_numpy().astype(np.float32)
        self.from_sq      = df["from_square_id"].to_numpy().astype(np.int64)
        self.to_sq        = df["to_square_id"].to_numpy().astype(np.int64)
        self.promo        = df["promo_id"].to_numpy().astype(np.int64)
        self.elo_bucket   = df["elo_bucket"].to_list() if "elo_bucket" in df.columns else [""] * len(df)

    def __len__(self) -> int:
        return len(self.from_sq)

    def __getitem__(self, idx: int) -> dict:
        return {
            "meta_ids":       torch.tensor(self.meta_tokens[idx],       dtype=torch.long),
            "color_ids":      torch.tensor(self.color_tokens[idx],      dtype=torch.long),
            "piece_type_ids": torch.tensor(self.piece_type_tokens[idx], dtype=torch.long),
            "file_ids":       torch.tensor(self.file_tokens[idx],       dtype=torch.long),
            "rank_ids":       torch.tensor(self.rank_tokens[idx],       dtype=torch.long),
            "white_elo":      torch.from_numpy(self.white_elo[idx:idx+1]).squeeze(0),
            "black_elo":      torch.from_numpy(self.black_elo[idx:idx+1]).squeeze(0),
            "white_clock_s":  torch.from_numpy(self.white_clock_s[idx:idx+1]).squeeze(0),
            "black_clock_s":  torch.from_numpy(self.black_clock_s[idx:idx+1]).squeeze(0),
            "from_sq":        torch.from_numpy(self.from_sq[idx:idx+1]).squeeze(0),
            "to_sq":          torch.from_numpy(self.to_sq[idx:idx+1]).squeeze(0),
            "promo":          torch.from_numpy(self.promo[idx:idx+1]).squeeze(0),
            "elo_bucket":     self.elo_bucket[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    def _pad(seqs: list[torch.Tensor], pad_val: int = -1) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out

    return {
        "meta_ids":       _pad([b["meta_ids"]       for b in batch]),
        "color_ids":      _pad([b["color_ids"]      for b in batch]),
        "piece_type_ids": _pad([b["piece_type_ids"] for b in batch]),
        "file_ids":       _pad([b["file_ids"]       for b in batch]),
        "rank_ids":       _pad([b["rank_ids"]       for b in batch]),
        "white_elo":      torch.stack([b["white_elo"]     for b in batch]),
        "black_elo":      torch.stack([b["black_elo"]     for b in batch]),
        "white_clock_s":  torch.stack([b["white_clock_s"] for b in batch]),
        "black_clock_s":  torch.stack([b["black_clock_s"] for b in batch]),
        "from_sq":        torch.stack([b["from_sq"]  for b in batch]),
        "to_sq":          torch.stack([b["to_sq"]    for b in batch]),
        "promo":          torch.stack([b["promo"]    for b in batch]),
        "elo_bucket":     [b["elo_bucket"] for b in batch],
    }


def make_loader(
    parquet_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = False,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = PositionDataset(parquet_path)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = shuffle if sampler is None else False,
        sampler            = sampler,
        num_workers        = num_workers,
        collate_fn         = collate_fn,
        pin_memory         = pin_memory,
        persistent_workers = persistent_workers and num_workers > 0,
        drop_last          = True,
    )
