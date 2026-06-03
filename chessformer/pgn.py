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
import io
import os
from typing import Iterator, Optional

import chess
import chess.pgn

from chessformer.tokenizer import (
    PositionTokens,
    Vocab,
    parse_increment_seconds,
    tokenize_position,
)


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------

def _open_pgn(path: str):
    """Open a PGN file, transparently decompressing .zst if needed."""
    if path.endswith(".zst"):
        import zstandard as zstd
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, errors="replace")


def iter_pgn_games(
    pgn_path: str,
    vocab: Vocab,
    max_games: int | None = None,
) -> Iterator[list[PositionTokens]]:
    """
    Stream a PGN file game by game.

    Yields one list[PositionTokens] per game (one entry per move).
    Games with zero moves (e.g. forfeits before move 1) are skipped.

    Args:
        pgn_path: Path to a PGN file.
        vocab: Vocab from build_vocab().
        max_games: Stop after this many games. None = read entire file.
    """
    n = 0
    with _open_pgn(pgn_path) as f:
        while max_games is None or n < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            positions = list(_game_positions(game, vocab))
            if positions:
                yield positions
                n += 1


def iter_pgn_games_with_elos(
    pgn_path: str,
    vocab: Vocab,
    skip_games: int = 0,
) -> Iterator[tuple[list[PositionTokens], int, int]]:
    """
    Like iter_pgn_games but also yields (white_elo, black_elo) per game.
    Used by preprocess.py for Elo-based filtering.
    Yields (positions, white_elo, black_elo). Games with zero moves are skipped.

    Args:
        skip_games: Skip this many games at the start of the file (zero-move
                    games count toward the skip). Useful for non-overlapping
                    subsets of the same source file.
    """
    skipped = 0
    with _open_pgn(pgn_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            if skipped < skip_games:
                skipped += 1
                continue
            w_elo, b_elo = game_elos(game)
            positions = list(_game_positions(game, vocab))
            if positions:
                yield positions, w_elo, b_elo


def iter_pgn_positions(pgn_path: str, vocab: Vocab) -> Iterator[PositionTokens]:
    """
    Stream every position from a PGN file as a flat sequence of PositionTokens.
    Convenience wrapper around iter_pgn_games for when game boundaries don't matter.
    """
    for game_positions in iter_pgn_games(pgn_path, vocab):
        yield from game_positions


def game_elos(game: chess.pgn.Game) -> tuple[int, int]:
    """Return (white_elo, black_elo) from game headers. -1 if missing/unparseable."""
    return _parse_elo(game.headers.get("WhiteElo", "?")), _parse_elo(game.headers.get("BlackElo", "?"))


def _game_positions(game: chess.pgn.Game, vocab: Vocab) -> Iterator[PositionTokens]:
    headers = game.headers
    white_elo = _parse_elo(headers.get("WhiteElo", "?"))
    black_elo = _parse_elo(headers.get("BlackElo", "?"))
    increment_s = parse_increment_seconds(headers.get("TimeControl", "-"))
    game_id = headers.get("Site", "") or (
        headers.get("Date", "") + headers.get("White", "") + headers.get("Black", "")
    )

    # Track each player's last known clock. node.clock() = time remaining for the
    # player who just moved, recorded AFTER their move. We yield BEFORE the move,
    # so each player's clock reflects their time at the start of their thinking.
    w_clock = -1.0
    b_clock = -1.0

    board = game.board()
    for node in game.mainline():
        move = node.move

        yield tokenize_position(
            fen=board.fen(),
            vocab=vocab,
            uci_move=move.uci(),
            white_elo=white_elo,
            black_elo=black_elo,
            white_clock_seconds=w_clock,
            black_clock_seconds=b_clock,
            increment_seconds=increment_s,
            game_id=game_id,
        )

        # Update clock for the player who just moved
        clock_s = node.clock()
        if clock_s is not None:
            if board.turn == chess.WHITE:
                w_clock = clock_s
            else:
                b_clock = clock_s

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
) -> Iterator[tuple[list[PositionTokens], int]]:
    """
    Stream puzzles from a Lichess puzzle CSV.

    Yields (list[PositionTokens], puzzle_rating) per puzzle. Each list represents the full
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
                yield positions, rating


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
            white_elo=3400,
            black_elo=3400,
            white_clock_seconds=-1.0,
            black_clock_seconds=-1.0,
            increment_seconds=-1.0,
            game_id=puzzle_id,
        )
        positions.append(pos)
        try:
            board.push_uci(uci)
        except (ValueError, chess.IllegalMoveError):
            break  # truncate on bad data rather than crash

    return positions
