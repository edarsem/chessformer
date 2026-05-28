"""
scripts/make_splits.py

Produce final train/val/test parquets from the outputs of preprocess.py.

Inputs:
  processed/all_games_train.parquet   → sampled → train_games_{size}.parquet
  processed/all_games_valtest.parquet → split by game_id hash → val_games.parquet
                                                                  test_games.parquet
  processed/all_puzzles.parquet       → split by puzzle_id hash → val_puzzles.parquet
                                                                    test_puzzles.parquet

Train subsets:
  Two train files are always produced from all_games_train.parquet:
    train_games_tiny.parquet  — cfg.data.train_games_tiny games (smoke tests)
    train_games_small.parquet — cfg.data.train_games_small games (local runs)

  These are produced by random game sampling (with cfg.data.split_seed).
  All positions from a sampled game are included.

Val/test split (games):
  hash(game_id, seed) % 2: 0 → val, 1 → test.
  The valtest parquet is already Elo-balanced (from preprocess.py), so the
  50/50 hash split naturally produces balanced val and test sets.

Val/test split (puzzles):
  hash(puzzle_id, seed) % 2: 0 → val, 1 → test.
  Then subsampled to cfg.data.val_puzzles and cfg.data.test_puzzles total.

Usage:
    python scripts/make_splits.py          # local config
    python scripts/make_splits.py data=full
"""

from __future__ import annotations

import hashlib
import os
import sys

import hydra
import polars as pl
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _hash_mod(id_str: str, seed: int, modulus: int) -> int:
    h = hashlib.sha256(f"{id_str}:{seed}".encode()).digest()
    return int.from_bytes(h[:4], "big") % modulus


def _sample_games(df: pl.DataFrame, n_games: int, seed: int) -> pl.DataFrame:
    """Sample n_games complete games (all their positions) from df."""
    unique_ids = df["game_id"].unique().sort()
    if len(unique_ids) <= n_games:
        return df
    sampled = unique_ids.sample(n=n_games, seed=seed)
    return df.join(sampled.to_frame("game_id"), on="game_id", how="inner")


def _hash_split_games(
    df: pl.DataFrame, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split game positions into val/test by hashing game_id."""
    unique_ids = df["game_id"].unique().to_list()
    val_ids = set(gid for gid in unique_ids if _hash_mod(gid, seed, 2) == 0)

    split_col = df["game_id"].map_elements(
        lambda gid: "val" if gid in val_ids else "test",
        return_dtype=pl.Utf8,
    )
    df = df.with_columns(split_col.alias("_split"))
    val  = df.filter(pl.col("_split") == "val").drop("_split")
    test = df.filter(pl.col("_split") == "test").drop("_split")
    return val, test


def _hash_split_puzzles(
    df: pl.DataFrame, seed: int, n_val: int, n_test: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split puzzle positions into val/test by hashing puzzle_id, then subsample."""
    unique_ids = df["puzzle_id"].unique().to_list()
    val_ids = set(pid for pid in unique_ids if _hash_mod(pid, seed, 2) == 0)

    split_col = df["puzzle_id"].map_elements(
        lambda pid: "val" if pid in val_ids else "test",
        return_dtype=pl.Utf8,
    )
    df = df.with_columns(split_col.alias("_split"))
    val_pool  = df.filter(pl.col("_split") == "val").drop("_split")
    test_pool = df.filter(pl.col("_split") == "test").drop("_split")

    val  = _subsample_puzzles(val_pool,  n_val,  seed)
    test = _subsample_puzzles(test_pool, n_test, seed)
    return val, test


def _subsample_puzzles(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """Keep n complete puzzles from df."""
    unique_ids = df["puzzle_id"].unique().sort()
    if len(unique_ids) <= n:
        return df
    sampled = unique_ids.sample(n=n, seed=seed)
    return df.join(sampled.to_frame("puzzle_id"), on="puzzle_id", how="inner")


def _print_elo_distribution(df: pl.DataFrame, id_col: str, label: str) -> None:
    dist = (
        df.select(["elo_bucket", id_col])
        .unique()
        .group_by("elo_bucket")
        .agg(pl.len().alias("games"))
        .sort("elo_bucket")
    )
    print(f"  {label} Elo distribution:")
    for row in dist.iter_rows(named=True):
        print(f"    {row['elo_bucket']}: {row['games']}")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = cfg.paths.processed_dir
    seed = cfg.data.split_seed

    train_src    = os.path.join(processed_dir, "all_games_train.parquet")
    valtest_src  = os.path.join(processed_dir, "all_games_valtest.parquet")
    puzzles_src  = os.path.join(processed_dir, "all_puzzles.parquet")

    for path in [train_src, valtest_src, puzzles_src]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found — run scripts/preprocess.py first.")

    # --- Train -----------------------------------------------------------
    print("Reading all_games_train.parquet …")
    train_all = pl.read_parquet(train_src)
    n_all = train_all["game_id"].n_unique()
    print(f"  {n_all:,} games / {len(train_all):,} positions")

    tiny_games  = cfg.data.train_games_tiny
    small_games = cfg.data.train_games_small

    for label, n_games, path_key in [
        ("tiny",  tiny_games,  "train_games_tiny_file"),
        ("small", small_games, "train_games_small_file"),
    ]:
        out = cfg.data[path_key]
        df = _sample_games(train_all, n_games, seed)
        df.write_parquet(out, compression="snappy")
        print(f"  train_{label}: {df['game_id'].n_unique():,} games / {len(df):,} positions → {out}")

    # --- Val / Test (games) ----------------------------------------------
    print("\nReading all_games_valtest.parquet …")
    valtest = pl.read_parquet(valtest_src)
    print(f"  {valtest['game_id'].n_unique():,} games / {len(valtest):,} positions")

    val_g, test_g = _hash_split_games(valtest, seed)

    for label, df, path_key in [
        ("val",  val_g,  "val_games_file"),
        ("test", test_g, "test_games_file"),
    ]:
        out = cfg.data[path_key]
        df.write_parquet(out, compression="snappy")
        n_games = df["game_id"].n_unique()
        print(f"  {label}_games: {n_games:,} games / {len(df):,} positions → {out}")

    _print_elo_distribution(val_g,  "game_id", "val")
    _print_elo_distribution(test_g, "game_id", "test")

    # --- Val / Test (puzzles) --------------------------------------------
    print("\nReading all_puzzles.parquet …")
    puzzles = pl.read_parquet(puzzles_src)
    print(f"  {puzzles['puzzle_id'].n_unique():,} puzzles / {len(puzzles):,} positions")

    val_p, test_p = _hash_split_puzzles(
        puzzles,
        seed=seed,
        n_val=cfg.data.val_puzzles,
        n_test=cfg.data.test_puzzles,
    )

    for label, df, path_key in [
        ("val",  val_p,  "val_puzzles_file"),
        ("test", test_p, "test_puzzles_file"),
    ]:
        out = cfg.data[path_key]
        df.write_parquet(out, compression="snappy")
        print(f"  {label}_puzzles: {df['puzzle_id'].n_unique():,} puzzles / {len(df):,} positions → {out}")

    # --- Summary ---------------------------------------------------------
    print("\n=== Split summary ===")
    print(f"  train_tiny  : {_sample_games(train_all, tiny_games, seed)['game_id'].n_unique():,} games")
    print(f"  train_small : {_sample_games(train_all, small_games, seed)['game_id'].n_unique():,} games")
    print(f"  val_games   : {val_g['game_id'].n_unique():,} games")
    print(f"  test_games  : {test_g['game_id'].n_unique():,} games")
    print(f"  val_puzzles : {val_p['puzzle_id'].n_unique():,} puzzles")
    print(f"  test_puzzles: {test_p['puzzle_id'].n_unique():,} puzzles")


if __name__ == "__main__":
    main()
