"""
scripts/eval_offline.py

Run offline evaluation on a trained checkpoint against val or test sets.

Usage:
    python scripts/eval_offline.py checkpoint=checkpoints/step_10000.pt
    python scripts/eval_offline.py checkpoint=... split=test

Metrics reported:

  Games (per Elo bucket + aggregate):
    - top1_accuracy: exact (from, to) match
    - top5_accuracy: correct move in top-5
    - legal_rate: % of argmax moves that are legal

  Puzzles (always conditioned on GM Elo):
    - move1_accuracy: first solution move correct
    - full_solve_rate: entire solution correct
    - loss: cross-entropy averaged over solution positions

Results are printed to stdout and optionally logged to W&B.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Phase 5: implement
    # Steps:
    #   1. Load checkpoint + rebuild model with saved config
    #   2. Load val or test parquet (games + puzzles)
    #   3. Run chessformer.eval.eval_games() → per-bucket metrics dict
    #   4. Run chessformer.eval.eval_puzzles() → puzzle metrics dict
    #   5. Pretty-print results with rich.table
    #   6. Optionally log to W&B
    raise NotImplementedError("Phase 5: eval_offline.py not yet implemented")


if __name__ == "__main__":
    main()
