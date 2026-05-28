"""
scripts/preprocess.py

Convert raw data sources into tokenized parquet files ready for training.

Usage:
    python scripts/preprocess.py              # uses conf/data/local.yaml
    python scripts/preprocess.py data=full    # uses conf/data/full.yaml

What this does:
  1. Reads PGN (raw_pgn/) and puzzle CSV (raw_puzzles/).
  2. Extracts positions with metadata (elo, clock, time control, game_id).
  3. Tokenizes each position using chessformer.tokenizer.
  4. Writes one parquet per split to data/processed/.

Run this once before training. Re-run if you change the tokenizer.
The output filenames include a short hash of the tokenizer vocab so stale
files are easy to detect.

Splits are done by scripts/make_splits.py (run separately, or pass --split).
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Phase 2: implement
    # Steps:
    #   1. Build/load vocab from chessformer.tokenizer.build_vocab()
    #   2. Stream PGN with python-chess, extract positions via xfen logic
    #      - Include %clk annotations for per-move clock remaining
    #      - Include Elo from PGN headers (both players)
    #      - Skip games with missing Elo or time control
    #   3. For each position, call tokenizer.tokenize_position() → dict of arrays
    #   4. Write to parquet in batches (don't accumulate in RAM)
    #   5. Repeat for puzzles (CSV format, different schema)
    raise NotImplementedError("Phase 2: preprocess.py not yet implemented")


if __name__ == "__main__":
    main()
