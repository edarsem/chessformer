"""
chessformer/dataset.py

Dataset and DataLoader utilities for reading tokenized chess positions from parquet files.

Each parquet row is one position with a known move (from_square_id always set for game rows).
Data is loaded fully into memory at init time for fast random access during shuffled training.

All variable-length columns are stored as padded int16 numpy arrays (not Python lists) so
that forked DataLoader workers share memory via copy-on-write. Python lists break CoW because
Python's reference counting writes to every object on access, duplicating pages across workers.
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

        n = len(df)

        # Variable-length columns: pad into int16 numpy arrays so forked workers
        # share memory via CoW (Python lists break CoW via refcount writes).
        meta_list       = df["meta_tokens"].to_list()
        color_list      = df["color_tokens"].to_list()
        piece_type_list = df["piece_type_tokens"].to_list()
        file_list       = df["file_tokens"].to_list()
        rank_list       = df["rank_tokens"].to_list()

        meta_max  = max(len(x) for x in meta_list)
        piece_max = max(len(x) for x in color_list)

        self.meta_len  = np.array([len(x) for x in meta_list],  dtype=np.int16)
        self.piece_len = np.array([len(x) for x in color_list], dtype=np.int16)

        self.meta_tokens       = np.zeros((n, meta_max),  dtype=np.int16)
        self.color_tokens      = np.zeros((n, piece_max), dtype=np.int16)
        self.piece_type_tokens = np.zeros((n, piece_max), dtype=np.int16)
        self.file_tokens       = np.zeros((n, piece_max), dtype=np.int16)
        self.rank_tokens       = np.zeros((n, piece_max), dtype=np.int16)

        for i, (m, c, pt, f, r) in enumerate(
            zip(meta_list, color_list, piece_type_list, file_list, rank_list)
        ):
            self.meta_tokens[i, :len(m)]  = m
            self.color_tokens[i, :len(c)] = c
            self.piece_type_tokens[i, :len(pt)] = pt
            self.file_tokens[i, :len(f)]  = f
            self.rank_tokens[i, :len(r)]  = r

        # Fixed-size scalars: numpy arrays → torch.from_numpy in __getitem__ (zero-copy)
        self.white_elo    = df["white_elo"].to_numpy().astype(np.int64)
        self.black_elo    = df["black_elo"].to_numpy().astype(np.int64)
        self.white_clock_s = df["white_clock_seconds"].to_numpy().astype(np.float32)
        self.black_clock_s = df["black_clock_seconds"].to_numpy().astype(np.float32)
        self.from_sq      = df["from_square_id"].to_numpy().astype(np.int64)   # 0-63
        self.to_sq        = df["to_square_id"].to_numpy().astype(np.int64)     # 0-63
        self.from_file    = df["from_file_id"].to_numpy().astype(np.int64)
        self.from_rank    = df["from_rank_id"].to_numpy().astype(np.int64)
        self.to_file      = df["to_file_id"].to_numpy().astype(np.int64)
        self.to_rank      = df["to_rank_id"].to_numpy().astype(np.int64)
        self.promo        = df["promo_id"].to_numpy().astype(np.int64)
        self.elo_bucket   = df["elo_bucket"].to_list() if "elo_bucket" in df.columns else [""] * n

    def __len__(self) -> int:
        return len(self.from_sq)

    def __getitem__(self, idx: int) -> dict:
        ml = int(self.meta_len[idx])
        pl = int(self.piece_len[idx])
        return {
            "meta_ids":       torch.tensor(self.meta_tokens[idx, :ml].astype(np.int64),       dtype=torch.long),
            "color_ids":      torch.tensor(self.color_tokens[idx, :pl].astype(np.int64),      dtype=torch.long),
            "piece_type_ids": torch.tensor(self.piece_type_tokens[idx, :pl].astype(np.int64), dtype=torch.long),
            "file_ids":       torch.tensor(self.file_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "rank_ids":       torch.tensor(self.rank_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "white_elo":      torch.from_numpy(self.white_elo[idx:idx+1]).squeeze(0),
            "black_elo":      torch.from_numpy(self.black_elo[idx:idx+1]).squeeze(0),
            "white_clock_s":  torch.from_numpy(self.white_clock_s[idx:idx+1]).squeeze(0),
            "black_clock_s":  torch.from_numpy(self.black_clock_s[idx:idx+1]).squeeze(0),
            "from_sq":        torch.from_numpy(self.from_sq[idx:idx+1]).squeeze(0),
            "to_sq":          torch.from_numpy(self.to_sq[idx:idx+1]).squeeze(0),
            "from_file":      torch.from_numpy(self.from_file[idx:idx+1]).squeeze(0),
            "from_rank":      torch.from_numpy(self.from_rank[idx:idx+1]).squeeze(0),
            "to_file":        torch.from_numpy(self.to_file[idx:idx+1]).squeeze(0),
            "to_rank":        torch.from_numpy(self.to_rank[idx:idx+1]).squeeze(0),
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
        "from_sq":        torch.stack([b["from_sq"]   for b in batch]),
        "to_sq":          torch.stack([b["to_sq"]     for b in batch]),
        "from_file":      torch.stack([b["from_file"] for b in batch]),
        "from_rank":      torch.stack([b["from_rank"] for b in batch]),
        "to_file":        torch.stack([b["to_file"]   for b in batch]),
        "to_rank":        torch.stack([b["to_rank"]   for b in batch]),
        "promo":          torch.stack([b["promo"]     for b in batch]),
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
