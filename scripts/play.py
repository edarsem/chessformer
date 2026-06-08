"""
scripts/play.py

Play against a trained model or watch AI vs AI in the terminal.

Usage:
    python scripts/play.py checkpoint=checkpoints/chessformer_v0.pt
    python scripts/play.py checkpoint=... mode=ai_vs_ai
    python scripts/play.py checkpoint=... mode=human_vs_ai human_side=black
    python scripts/play.py checkpoint=... temperature=0.8 ai_elo=2800

Human moves: UCI notation — e2e4, g1f3, e7e8q (promotion to queen).
Type 'quit' to resign.
"""

from __future__ import annotations

import os
import sys
import subprocess

import hydra
import chess
import chess.svg
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chessformer.model import ChessformerModel, unwrap_state_dict
from chessformer.tokenizer import build_vocab
from chessformer.inference import pick_move


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_board(board: chess.Board) -> None:
    print()
    print(board)
    print()


def show_svg(board: chess.Board, last_move: chess.Move | None = None) -> None:
    svg  = chess.svg.board(board, lastmove=last_move, size=400)
    path = "/tmp/chessformer_board.svg"
    with open(path, "w") as f:
        f.write(svg)
    try:
        subprocess.Popen(["open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def display(board: chess.Board, last_move: chess.Move | None, use_svg: bool) -> None:
    if use_svg:
        show_svg(board, last_move)
    print_board(board)


def game_result_str(board: chess.Board) -> str:
    result = board.result()
    if result == "1-0": return "White wins!"
    if result == "0-1": return "Black wins!"
    return "Draw."


# ---------------------------------------------------------------------------
# Game loops
# ---------------------------------------------------------------------------

def human_vs_ai(
    model: ChessformerModel,
    vocab,
    device: torch.device,
    human_side: str,
    white_elo: int,
    black_elo: int,
    temperature: float,
    use_svg: bool,
) -> None:
    board       = chess.Board()
    human_color = chess.WHITE if human_side == "white" else chess.BLACK
    last_move   = None

    print(f"\nYou are {'White' if human_color == chess.WHITE else 'Black'}. "
          f"Elo: white={white_elo}  black={black_elo}")
    print("Enter moves in UCI notation (e2e4). Type 'quit' to resign.\n")

    while not board.is_game_over():
        display(board, last_move, use_svg)
        print(f"{'White' if board.turn == chess.WHITE else 'Black'} to move  "
              f"(move {board.fullmove_number})")

        if board.turn == human_color:
            while True:
                raw = input("Your move: ").strip().lower()
                if raw in ("quit", "resign"):
                    print("You resigned.")
                    return
                try:
                    move = board.parse_uci(raw)
                    if board.is_legal(move):
                        break
                    print("  Illegal move, try again.")
                except Exception:
                    print("  Invalid notation, try again.")
        else:
            move = pick_move(model, vocab, device, board,
                             white_elo=white_elo, black_elo=black_elo,
                             argmax=False, temperature=temperature)
            if move is None:
                print("AI has no legal moves.")
                break
            print(f"AI plays: {move.uci()}")

        last_move = move
        board.push(move)

    display(board, last_move, use_svg)
    print(game_result_str(board))


def ai_vs_ai(
    model: ChessformerModel,
    vocab,
    device: torch.device,
    white_elo: int,
    black_elo: int,
    temperature: float,
    use_svg: bool,
    max_moves: int = 200,
) -> None:
    board     = chess.Board()
    last_move = None
    print(f"\nAI vs AI  white_elo={white_elo}  black_elo={black_elo}  temp={temperature}\n")

    while not board.is_game_over() and board.fullmove_number <= max_moves:
        display(board, last_move, use_svg)
        side = "White" if board.turn == chess.WHITE else "Black"
        move = pick_move(model, vocab, device, board,
                         white_elo=white_elo, black_elo=black_elo,
                         argmax=False, temperature=temperature)
        if move is None:
            break
        print(f"{side} plays: {move.uci()}")
        last_move = move
        board.push(move)
        if use_svg:
            input("  [Enter for next move]")

    display(board, last_move, use_svg)
    print(game_result_str(board) if board.is_game_over() else "Max moves reached.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    ckpt_path    = cfg.checkpoint
    mode         = cfg.mode
    human_side   = cfg.human_side
    ai_elo       = cfg.ai_elo
    white_elo    = cfg.white_elo if cfg.white_elo is not None else ai_elo
    black_elo    = cfg.black_elo if cfg.black_elo is not None else ai_elo
    temperature  = float(cfg.temperature)
    display_mode = cfg.display

    assert ckpt_path, "Pass checkpoint=<path>"
    assert mode in ("human_vs_ai", "ai_vs_ai"), "mode must be human_vs_ai or ai_vs_ai"

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ckpt      = torch.load(ckpt_path, map_location=device)
    saved_cfg = OmegaConf.create(ckpt["cfg"])
    vocab     = build_vocab()
    model     = ChessformerModel(
        vocab    = vocab,
        d_model  = saved_cfg.model.d_model,
        n_heads  = saved_cfg.model.n_heads,
        n_layers = saved_cfg.model.n_layers,
        ffn_mult = saved_cfg.model.ffn_mult,
        dropout  = saved_cfg.model.dropout,
    ).to(device)
    model.load_state_dict(unwrap_state_dict(ckpt["model"]), strict=True)
    model.eval()
    print(f"Loaded checkpoint at step {ckpt.get('step', 0)}  (device: {device})")

    use_svg = display_mode == "svg"

    if mode == "human_vs_ai":
        human_vs_ai(model, vocab, device, human_side, white_elo, black_elo, temperature, use_svg)
    else:
        ai_vs_ai(model, vocab, device, white_elo, black_elo, temperature, use_svg)


if __name__ == "__main__":
    main()
