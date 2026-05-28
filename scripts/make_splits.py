"""
scripts/make_splits.py

Assign preprocessed positions to train / val / test splits and write
one parquet per split.

Usage:
    python scripts/make_splits.py              # local config
    python scripts/make_splits.py data=full

Split strategy:
  - Games: hash(game_id, seed) % 100
      < 90  → train
      90–94 → val
      95–99 → test
    All positions from one game stay in the same split (no leakage).
  - Puzzles: hash(puzzle_id, seed) % 100
      < 90  → val pool
      >= 90 → test pool
    Then subsample to val_puzzles / test_puzzles counts.
  - Val and test games are further subsampled to
    val_games_per_elo_bucket / test_games_per_elo_bucket per Elo category
    so all Elo levels are equally represented at eval time.

The seed is in conf/data/*.yaml (split_seed). Document it; never change it
after the first training run or your val/test sets move.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Phase 2: implement
    # Steps:
    #   1. Load the full preprocessed parquet (output of preprocess.py)
    #   2. Hash game_id / puzzle_id with cfg.data.split_seed
    #   3. Assign splits
    #   4. Subsample val/test to the configured per-bucket counts
    #   5. Write train_games.parquet, val_games.parquet, val_puzzles.parquet,
    #      test_games.parquet, test_puzzles.parquet to cfg.paths.processed_dir
    #   6. Print a summary: N positions per split, per Elo bucket breakdown
    raise NotImplementedError("Phase 2: make_splits.py not yet implemented")


if __name__ == "__main__":
    main()
