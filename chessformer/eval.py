"""
chessformer/eval.py

Evaluation functions called by both the Trainer (during training) and
scripts/eval_offline.py (standalone eval).

All functions take a model in eval mode and a DataLoader, return a metrics dict.
"""

# TODO Phase 5: implement
#
# def eval_games(model, loader, device, vocab) -> dict:
#     """
#     Compute game metrics over a DataLoader of game positions.
#
#     Returns:
#         {
#           "loss": float,
#           "top1_accuracy": float,
#           "top5_accuracy": float,
#           "legal_rate": float,
#           "per_elo": {
#               "elo_800": {"loss": ..., "top1": ..., "top5": ..., "legal_rate": ...},
#               ...
#           }
#         }
#
#     Notes:
#     - top1: argmax of (from_logits, to_logits) must match gold (from, to)
#     - top5: gold move in top-5 of from_logits AND top-5 of to_logits
#     - legal_rate: apply argmax move to board, check chess.Board.is_legal()
#     - per_elo breakdown: group by elo_bucket column in batch metadata
#     """
#
# def eval_puzzles(model, loader, device, vocab) -> dict:
#     """
#     Compute puzzle metrics.
#     Always called with model conditioned on GM Elo (set in the puzzle dataset).
#
#     Returns:
#         {
#           "loss": float,
#           "move1_accuracy": float,  # first move of solution correct
#           "full_solve_rate": float,  # all solution moves correct
#         }
#
#     Notes:
#     - Each puzzle has N solution moves (N >= 1). Opponent moves are given
#       (teacher-forced in the sequence). Only the player-to-move's moves
#       are evaluated.
#     - full_solve_rate: 1 only if every player-to-move step is correct.
#     - Loss: cross-entropy averaged over player-to-move steps only.
#     """
