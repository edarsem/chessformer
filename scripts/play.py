"""
scripts/play.py

Play against a trained model or watch AI vs AI. No Jupyter required.

Usage:
    # Human (white) vs AI (black)
    python scripts/play.py checkpoint=checkpoints/step_10000.pt mode=human_vs_ai

    # AI vs AI (watch the model play itself)
    python scripts/play.py checkpoint=... mode=ai_vs_ai

    # Condition AI on a specific Elo level
    python scripts/play.py checkpoint=... mode=human_vs_ai ai_elo=2800

    # Adjust sampling temperature
    python scripts/play.py checkpoint=... temperature=0.5 top_k=5

Display: SVG board written to /tmp/chessformer_board.svg and opened in the
default browser each move. Falls back to ASCII board if --ascii flag is set.

Human moves: enter UCI notation (e.g. e2e4, g1f3, e7e8q for promotion).
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # TODO Phase 6: implement
    # Steps:
    #   1. Load checkpoint
    #   2. Initialize board (chess.Board)
    #   3. Game loop:
    #      - Display board (SVG or ASCII)
    #      - If human turn: read UCI input, validate, apply
    #      - If AI turn: call chessformer.model.predict_move(board, elo=cfg.ai_elo,
    #          temperature=cfg.temperature, top_k=cfg.top_k)
    #        Retry up to 5 times if sampled move is illegal; fallback to random legal move
    #      - Check for game over (checkmate, stalemate, draw)
    raise NotImplementedError("Phase 6: play.py not yet implemented")


if __name__ == "__main__":
    main()
