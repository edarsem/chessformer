"""
chessformer/dataset.py

Dataset and DataLoader utilities for reading tokenized chess positions from parquet files.

Each parquet row is one position with a known move (from_square_id always set for game rows).
Data is loaded fully into memory at init time (polars → Python lists/arrays) for fast
random access during shuffled training.

The collate function pads variable-length meta and piece token lists to the batch maximum.
"""

from __future__ import annotations

from typing import Optional

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
        # Keep only positions with a known move (from_square_id >= 0)
        df = df.filter(pl.col("from_square_id") >= 0)

        # Store each column as a Python list for O(1) random access
        self.meta_tokens       = df["meta_tokens"].to_list()
        self.color_tokens      = df["color_tokens"].to_list()
        self.piece_type_tokens = df["piece_type_tokens"].to_list()
        self.file_tokens       = df["file_tokens"].to_list()
        self.rank_tokens       = df["rank_tokens"].to_list()
        self.white_elo         = df["white_elo"].to_list()
        self.black_elo         = df["black_elo"].to_list()
        self.white_clock_s     = df["white_clock_seconds"].to_list()
        self.black_clock_s     = df["black_clock_seconds"].to_list()
        self.from_sq           = df["from_square_id"].to_list()
        self.to_sq             = df["to_square_id"].to_list()
        self.promo             = df["promo_id"].to_list()
        self.elo_bucket        = df["elo_bucket"].to_list() if "elo_bucket" in df.columns else [""] * len(df)

    def __len__(self) -> int:
        return len(self.from_sq)

    def __getitem__(self, idx: int) -> dict:
        return {
            "meta_ids":       torch.tensor(self.meta_tokens[idx],       dtype=torch.long),
            "color_ids":      torch.tensor(self.color_tokens[idx],      dtype=torch.long),
            "piece_type_ids": torch.tensor(self.piece_type_tokens[idx], dtype=torch.long),
            "file_ids":       torch.tensor(self.file_tokens[idx],       dtype=torch.long),
            "rank_ids":       torch.tensor(self.rank_tokens[idx],       dtype=torch.long),
            "white_elo":      torch.tensor(self.white_elo[idx],         dtype=torch.long),
            "black_elo":      torch.tensor(self.black_elo[idx],         dtype=torch.long),
            "white_clock_s":  torch.tensor(self.white_clock_s[idx],     dtype=torch.float32),
            "black_clock_s":  torch.tensor(self.black_clock_s[idx],     dtype=torch.float32),
            "from_sq":        torch.tensor(self.from_sq[idx],           dtype=torch.long),
            "to_sq":          torch.tensor(self.to_sq[idx],             dtype=torch.long),
            "promo":          torch.tensor(self.promo[idx],             dtype=torch.long),
            "elo_bucket":     self.elo_bucket[idx],
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Pad variable-length meta and piece token lists to the batch maximum.
    Scalars are stacked directly. elo_bucket stays as a list of strings.
    """
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
) -> DataLoader:
    dataset = PositionDataset(parquet_path)
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False  # DistributedSampler handles shuffling
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
