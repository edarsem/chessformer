"""
chessformer/pgn.py

Streaming utilities for converting raw PGN and puzzle CSV files into
PositionTokens. These are used by scripts/preprocess.py.

PGN clock handling
------------------
Lichess PGN files include [%clk H:MM:SS] annotations in move comments.
python-chess exposes this as node.clock() → float seconds remaining after the
move, or None if absent. We use this as the clock value for that move:
"time remaining after this move" closely tracks "time pressure under which
this move was made" and is the natural signal for Elo-conditioned behavior.

Puzzle CSV format (Lichess)
---------------------------
Columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
Themes, GameUrl, OpeningTags
- FEN is the position BEFORE the opponent's first move (which is also in Moves).
- Moves is space-separated UCI. First move is the opponent's; remaining moves
  are the puzzle solution (player-to-move alternates opponent/solver).
- We skip the first move (opponent's), then tokenize each solver move as a
  PositionTokens with uci_move set; opponent moves are tokenized with
  uci_move=None to mark them as "given" (teacher-forced at eval time).
"""

from __future__ import annotations

import csv
from typing import Iterator, Optional

import chess
import chess.pgn

from chessformer.tokenizer import (
    PositionTokens,
    Vocab,
    classify_time_control,
    tokenize_position,
)


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------

def iter_pgn_positions(pgn_path: str, vocab: Vocab) -> Iterator[PositionTokens]:
    """
    Stream every position from a PGN file as PositionTokens.

    Yields one PositionTokens per move in the game. Positions from games with
    missing or unparseable Elo are yielded with elo=-1.

    Args:
        pgn_path: Path to a PGN file (may contain multiple games).
        vocab: Vocab from build_vocab().
    """
    with open(pgn_path, errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            yield from _game_positions(game, vocab)


def _game_positions(game: chess.pgn.Game, vocab: Vocab) -> Iterator[PositionTokens]:
    headers = game.headers
    white_elo = _parse_elo(headers.get("WhiteElo", "?"))
    black_elo = _parse_elo(headers.get("BlackElo", "?"))
    tc_token = classify_time_control(headers.get("TimeControl", "-"))
    game_id = headers.get("Site", "") or (
        headers.get("Date", "") + headers.get("White", "") + headers.get("Black", "")
    )

    board = game.board()
    for node in game.mainline():
        move = node.move
        fen = board.fen()
        elo = white_elo if board.turn == chess.WHITE else black_elo
        clock_s = node.clock()  # seconds remaining after this move; None if absent

        yield tokenize_position(
            fen=fen,
            vocab=vocab,
            uci_move=move.uci(),
            elo=elo,
            clock_seconds=clock_s if clock_s is not None else -1.0,
            time_control=tc_token,
            game_id=game_id,
        )
        board.push(move)


def _parse_elo(elo_str: str) -> int:
    try:
        return int(elo_str)
    except (ValueError, TypeError):
        return -1


# ---------------------------------------------------------------------------
# Puzzle CSV streaming
# ---------------------------------------------------------------------------

def iter_puzzle_positions(
    puzzle_csv_path: str,
    vocab: Vocab,
    rating_min: Optional[int] = None,
    rating_max: Optional[int] = None,
) -> Iterator[list[PositionTokens]]:
    """
    Stream puzzles from a Lichess puzzle CSV.

    Yields one list[PositionTokens] per puzzle. Each list represents the full
    puzzle sequence:
      - Solver moves have uci_move set (these are predicted / evaluated).
      - Opponent moves have uci_move=None (teacher-forced at eval time).

    The first move in the CSV is the opponent's trigger move; it is applied to
    reach the actual puzzle starting position, not yielded as a solver move.

    Args:
        puzzle_csv_path: Path to lichess_db_puzzle.csv.
        vocab: Vocab from build_vocab().
        rating_min: Skip puzzles with Rating below this (inclusive). None = no filter.
        rating_max: Skip puzzles with Rating above this (inclusive). None = no filter.
    """
    with open(puzzle_csv_path, newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            puzzle_id = row["PuzzleId"]
            fen = row["FEN"]
            moves_uci = row["Moves"].split()
            try:
                rating = int(row["Rating"])
            except (ValueError, KeyError):
                rating = -1

            if rating_min is not None and rating >= 0 and rating < rating_min:
                continue
            if rating_max is not None and rating >= 0 and rating > rating_max:
                continue

            positions = _puzzle_positions(
                puzzle_id=puzzle_id,
                fen=fen,
                moves_uci=moves_uci,
                vocab=vocab,
                puzzle_rating=rating,
            )
            if positions:
                yield positions


def _puzzle_positions(
    puzzle_id: str,
    fen: str,
    moves_uci: list[str],
    vocab: Vocab,
    puzzle_rating: int,
) -> list[PositionTokens]:
    """
    Convert a raw puzzle into a list of PositionTokens.

    The Lichess CSV FEN is the position before the opponent's first move.
    moves_uci[0] is the opponent's move (apply it to reach the puzzle position).
    Subsequent moves alternate: solver, opponent, solver, ...

    We condition all puzzle positions on GM Elo (3400) at eval time, as agreed.
    The model is never trained on puzzles — elo here is stored for reference only.
    """
    board = chess.Board(fen)

    # Apply the opponent's trigger move to reach the actual puzzle starting FEN.
    try:
        board.push_uci(moves_uci[0])
    except (ValueError, chess.IllegalMoveError):
        return []

    solution_moves = moves_uci[1:]  # alternating: solver, opponent, solver, ...
    positions: list[PositionTokens] = []
    solver_color = board.turn  # the side that needs to solve the puzzle

    for i, uci in enumerate(solution_moves):
        is_solver_move = (board.turn == solver_color)
        pos = tokenize_position(
            fen=board.fen(),
            vocab=vocab,
            uci_move=uci if is_solver_move else None,
            elo=3400,          # condition on GM Elo for all puzzle positions
            clock_seconds=-1.0,
            time_control="unknown_time",
            game_id=puzzle_id,
        )
        positions.append(pos)
        try:
            board.push_uci(uci)
        except (ValueError, chess.IllegalMoveError):
            break  # truncate on bad data rather than crash

    return positions
