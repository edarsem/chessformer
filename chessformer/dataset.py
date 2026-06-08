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

        from chessformer.tokenizer import META_SLOT_SIZE

        # Meta slot: fixed-size [N, META_SLOT_SIZE], padded with -1 (not 0, since 0 is a valid token).
        # All valid token embeddings are SUMMED into one sequence position by the model.
        meta_list = df["meta_tokens"].to_list()
        self.meta_tokens = np.full((n, META_SLOT_SIZE), -1, dtype=np.int16)
        for i, m in enumerate(meta_list):
            self.meta_tokens[i, :len(m)] = m

        # Piece tokens: variable-length, padded with 0 in numpy (collate_fn pads to -1 in batch).
        color_list      = df["color_tokens"].to_list()
        piece_type_list = df["piece_type_tokens"].to_list()
        file_list       = df["file_tokens"].to_list()
        rank_list       = df["rank_tokens"].to_list()

        piece_max      = max(len(x) for x in color_list)
        self.piece_len = np.array([len(x) for x in color_list], dtype=np.int16)

        self.color_tokens      = np.zeros((n, piece_max), dtype=np.int16)
        self.piece_type_tokens = np.zeros((n, piece_max), dtype=np.int16)
        self.file_tokens       = np.zeros((n, piece_max), dtype=np.int16)
        self.rank_tokens       = np.zeros((n, piece_max), dtype=np.int16)

        for i, (c, pt, f, r) in enumerate(
            zip(color_list, piece_type_list, file_list, rank_list)
        ):
            self.color_tokens[i, :len(c)]      = c
            self.piece_type_tokens[i, :len(pt)] = pt
            self.file_tokens[i, :len(f)]        = f
            self.rank_tokens[i, :len(r)]        = r

        # Fixed-size scalars: numpy arrays → torch.from_numpy in __getitem__ (zero-copy)
        self.white_elo    = df["white_elo"].to_numpy().astype(np.int64)
        self.black_elo    = df["black_elo"].to_numpy().astype(np.int64)
        self.white_clock_s = df["white_clock_seconds"].to_numpy().astype(np.float32)
        self.black_clock_s = df["black_clock_seconds"].to_numpy().astype(np.float32)
        self.increment_s  = df["increment_seconds"].to_numpy().astype(np.float32)
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
        pl = int(self.piece_len[idx])
        return {
            "meta_ids":       torch.tensor(self.meta_tokens[idx].astype(np.int64),            dtype=torch.long),
            "color_ids":      torch.tensor(self.color_tokens[idx, :pl].astype(np.int64),      dtype=torch.long),
            "piece_type_ids": torch.tensor(self.piece_type_tokens[idx, :pl].astype(np.int64), dtype=torch.long),
            "file_ids":       torch.tensor(self.file_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "rank_ids":       torch.tensor(self.rank_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "white_elo":      torch.from_numpy(self.white_elo[idx:idx+1]).squeeze(0),
            "black_elo":      torch.from_numpy(self.black_elo[idx:idx+1]).squeeze(0),
            "white_clock_s":  torch.from_numpy(self.white_clock_s[idx:idx+1]).squeeze(0),
            "black_clock_s":  torch.from_numpy(self.black_clock_s[idx:idx+1]).squeeze(0),
            "increment_s":    torch.from_numpy(self.increment_s[idx:idx+1]).squeeze(0),
            "from_sq":        torch.from_numpy(self.from_sq[idx:idx+1]).squeeze(0),
            "to_sq":          torch.from_numpy(self.to_sq[idx:idx+1]).squeeze(0),
            "from_file":      torch.from_numpy(self.from_file[idx:idx+1]).squeeze(0),
            "from_rank":      torch.from_numpy(self.from_rank[idx:idx+1]).squeeze(0),
            "to_file":        torch.from_numpy(self.to_file[idx:idx+1]).squeeze(0),
            "to_rank":        torch.from_numpy(self.to_rank[idx:idx+1]).squeeze(0),
            "promo":          torch.from_numpy(self.promo[idx:idx+1]).squeeze(0),
            "elo_bucket":     self.elo_bucket[idx],
        }


class EvalPositionDataset(Dataset):
    """
    Like PositionDataset but requires a `cp_eval` column (Stockfish centipawn score).
    Targets are stored as tanh(cp / 400), clipped to [-1, 1].
    Rows without a valid cp_eval (NaN or extreme outliers) are dropped.
    """

    CP_SCALE = 400.0

    def __init__(self, parquet_path: str, cp_clip: float = 12_000.0):
        df = pl.read_parquet(parquet_path)
        if "cp_eval" not in df.columns:
            raise ValueError(f"{parquet_path} has no 'cp_eval' column")
        df = df.filter(pl.col("cp_eval").is_not_null() & pl.col("cp_eval").is_finite())
        df = df.filter(pl.col("cp_eval").abs() <= cp_clip)

        n = len(df)

        from chessformer.tokenizer import META_SLOT_SIZE

        meta_list = df["meta_tokens"].to_list()
        self.meta_tokens = np.full((n, META_SLOT_SIZE), -1, dtype=np.int16)
        for i, m in enumerate(meta_list):
            self.meta_tokens[i, :len(m)] = m

        color_list      = df["color_tokens"].to_list()
        piece_type_list = df["piece_type_tokens"].to_list()
        file_list       = df["file_tokens"].to_list()
        rank_list       = df["rank_tokens"].to_list()

        piece_max      = max(len(x) for x in color_list)
        self.piece_len = np.array([len(x) for x in color_list], dtype=np.int16)

        self.color_tokens      = np.zeros((n, piece_max), dtype=np.int16)
        self.piece_type_tokens = np.zeros((n, piece_max), dtype=np.int16)
        self.file_tokens       = np.zeros((n, piece_max), dtype=np.int16)
        self.rank_tokens       = np.zeros((n, piece_max), dtype=np.int16)

        for i, (c, pt, f, r) in enumerate(
            zip(color_list, piece_type_list, file_list, rank_list)
        ):
            self.color_tokens[i, :len(c)]       = c
            self.piece_type_tokens[i, :len(pt)]  = pt
            self.file_tokens[i, :len(f)]         = f
            self.rank_tokens[i, :len(r)]         = r

        self.white_elo    = df["white_elo"].to_numpy().astype(np.int64)
        self.black_elo    = df["black_elo"].to_numpy().astype(np.int64)
        self.white_clock_s = df["white_clock_seconds"].to_numpy().astype(np.float32)
        self.black_clock_s = df["black_clock_seconds"].to_numpy().astype(np.float32)
        self.increment_s  = df["increment_seconds"].to_numpy().astype(np.float32)

        cp = df["cp_eval"].to_numpy().astype(np.float32)
        self.cp_target = np.tanh(cp / self.CP_SCALE).astype(np.float32)

    def __len__(self) -> int:
        return len(self.cp_target)

    def __getitem__(self, idx: int) -> dict:
        pl = int(self.piece_len[idx])
        return {
            "meta_ids":       torch.tensor(self.meta_tokens[idx].astype(np.int64),            dtype=torch.long),
            "color_ids":      torch.tensor(self.color_tokens[idx, :pl].astype(np.int64),      dtype=torch.long),
            "piece_type_ids": torch.tensor(self.piece_type_tokens[idx, :pl].astype(np.int64), dtype=torch.long),
            "file_ids":       torch.tensor(self.file_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "rank_ids":       torch.tensor(self.rank_tokens[idx, :pl].astype(np.int64),       dtype=torch.long),
            "white_elo":      torch.from_numpy(self.white_elo[idx:idx+1]).squeeze(0),
            "black_elo":      torch.from_numpy(self.black_elo[idx:idx+1]).squeeze(0),
            "white_clock_s":  torch.from_numpy(self.white_clock_s[idx:idx+1]).squeeze(0),
            "black_clock_s":  torch.from_numpy(self.black_clock_s[idx:idx+1]).squeeze(0),
            "increment_s":    torch.from_numpy(self.increment_s[idx:idx+1]).squeeze(0),
            "cp_target":      torch.tensor(float(self.cp_target[idx]), dtype=torch.float32),
        }


def eval_collate_fn(batch: list[dict]) -> dict:
    def _pad(seqs: list[torch.Tensor], pad_val: int = -1) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out

    return {
        "meta_ids":       torch.stack([b["meta_ids"]  for b in batch]),
        "color_ids":      _pad([b["color_ids"]      for b in batch]),
        "piece_type_ids": _pad([b["piece_type_ids"] for b in batch]),
        "file_ids":       _pad([b["file_ids"]       for b in batch]),
        "rank_ids":       _pad([b["rank_ids"]       for b in batch]),
        "white_elo":      torch.stack([b["white_elo"]     for b in batch]),
        "black_elo":      torch.stack([b["black_elo"]     for b in batch]),
        "white_clock_s":  torch.stack([b["white_clock_s"] for b in batch]),
        "black_clock_s":  torch.stack([b["black_clock_s"] for b in batch]),
        "increment_s":    torch.stack([b["increment_s"]   for b in batch]),
        "cp_target":      torch.stack([b["cp_target"]     for b in batch]),
    }


def make_eval_head_loader(
    parquet_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    rank: int = 0,
    world_size: int = 1,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    cp_clip: float = 3000.0,
) -> DataLoader:
    dataset = EvalPositionDataset(parquet_path, cp_clip=cp_clip)
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
        collate_fn         = eval_collate_fn,
        pin_memory         = pin_memory,
        persistent_workers = persistent_workers and num_workers > 0,
        drop_last          = True,
    )


def collate_fn(batch: list[dict]) -> dict:
    def _pad(seqs: list[torch.Tensor], pad_val: int = -1) -> torch.Tensor:
        max_len = max(s.shape[0] for s in seqs)
        out = torch.full((len(seqs), max_len), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            out[i, : s.shape[0]] = s
        return out

    return {
        "meta_ids":       torch.stack([b["meta_ids"]  for b in batch]),  # [B, META_SLOT_SIZE]
        "color_ids":      _pad([b["color_ids"]      for b in batch]),
        "piece_type_ids": _pad([b["piece_type_ids"] for b in batch]),
        "file_ids":       _pad([b["file_ids"]       for b in batch]),
        "rank_ids":       _pad([b["rank_ids"]       for b in batch]),
        "white_elo":      torch.stack([b["white_elo"]     for b in batch]),
        "black_elo":      torch.stack([b["black_elo"]     for b in batch]),
        "white_clock_s":  torch.stack([b["white_clock_s"] for b in batch]),
        "black_clock_s":  torch.stack([b["black_clock_s"] for b in batch]),
        "increment_s":    torch.stack([b["increment_s"]   for b in batch]),
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
