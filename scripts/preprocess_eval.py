"""
scripts/preprocess_eval.py

Extract positions with Stockfish evaluations from PGN files that contain
[%eval] annotations (e.g. lichess_db_standard_rated_2014-09.pgn).

Output columns: all standard GAMES_SCHEMA columns + `cp_eval` (float32,
centipawns from white's perspective).

The eval is the evaluation of the position BEFORE the move (i.e. the eval
annotation of the previous node). The first move of each game is skipped
because there is no prior annotation.

Mate scores (#N) are mapped to ±MATE_CP (default 3000) preserving sign
from white's perspective.

A small train/val/test split is made by hashing the game_id:
  - val:   game_id hash % 10 == 0  (~10%)
  - test:  game_id hash % 10 == 1  (~10%)
  - train: remainder               (~80%)

Usage:
    python scripts/preprocess_eval.py
    python scripts/preprocess_eval.py max_games=2000 pgn_path=data/raw_pgn/lichess_db_standard_rated_2014-09.pgn
"""

from __future__ import annotations

import hashlib
import os
import sys
import time

import hydra
import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chessformer.pgn import _open_pgn, game_elos
from chessformer.tokenizer import (
    build_vocab,
    parse_base_seconds,
    parse_increment_seconds,
    tokenize_position,
)


MATE_CP = 10_000  # centipawns for forced mate (100 pawns; tanh(10000/400) ≈ 1.0)


EVAL_SCHEMA = pa.schema([
    pa.field("game_id",              pa.string()),
    pa.field("elo_bucket",           pa.string()),
    pa.field("white_elo",            pa.int16()),
    pa.field("black_elo",            pa.int16()),
    pa.field("avg_elo",              pa.int16()),
    pa.field("elo_spread",           pa.int16()),
    pa.field("white_clock_seconds",  pa.float32()),
    pa.field("black_clock_seconds",  pa.float32()),
    pa.field("increment_seconds",    pa.float32()),
    pa.field("meta_tokens",          pa.list_(pa.int16())),
    pa.field("color_tokens",         pa.list_(pa.int8())),
    pa.field("piece_type_tokens",    pa.list_(pa.int8())),
    pa.field("file_tokens",          pa.list_(pa.int8())),
    pa.field("rank_tokens",          pa.list_(pa.int8())),
    pa.field("from_square_id",       pa.int8()),
    pa.field("to_square_id",         pa.int8()),
    pa.field("from_file_id",         pa.int8()),
    pa.field("from_rank_id",         pa.int8()),
    pa.field("to_file_id",           pa.int8()),
    pa.field("to_rank_id",           pa.int8()),
    pa.field("promo_id",             pa.int16()),
    pa.field("cp_eval",              pa.float32()),
])


class _ParquetWriter:
    def __init__(self, path: str, batch_size: int = 10_000):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._writer = pq.ParquetWriter(path, EVAL_SCHEMA, compression="snappy")
        self._buf: list[dict] = []
        self._batch_size = batch_size
        self._total = 0

    def write(self, row: dict) -> None:
        self._buf.append(row)
        if len(self._buf) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        arrays = {f.name: [r[f.name] for r in self._buf] for f in EVAL_SCHEMA}
        self._writer.write_table(pa.table(arrays, schema=EVAL_SCHEMA))
        self._total += len(self._buf)
        self._buf = []

    def close(self) -> int:
        self._flush()
        self._writer.close()
        return self._total


def _split(game_id: str) -> str:
    """Deterministic train/val/test split by game_id hash."""
    h = int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 10
    if h == 0:
        return "val"
    if h == 1:
        return "test"
    return "train"


def _iter_eval_games(pgn_path: str, vocab, max_games: int | None):
    """Yield (positions_with_eval, w_elo, b_elo) for games that have %eval annotations."""
    n_scanned = n_yielded = 0
    with _open_pgn(pgn_path) as f:
        while max_games is None or n_yielded < max_games:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            n_scanned += 1

            w_elo, b_elo = game_elos(game)
            if w_elo < 0 or b_elo < 0:
                continue

            tc_str      = game.headers.get("TimeControl", "-")
            increment_s = parse_increment_seconds(tc_str)
            base_s      = parse_base_seconds(tc_str)
            game_id     = game.headers.get("Site", "") or (
                game.headers.get("Date", "") + game.headers.get("White", "") + game.headers.get("Black", "")
            )

            w_clock = base_s
            b_clock = base_s
            board = game.board()
            rows: list[dict] = []
            has_eval = False

            for node in game.mainline():
                move      = node.move
                # Eval of the current position = eval annotation of the previous node
                parent_ev = node.parent.eval()

                if parent_ev is not None:
                    cp = parent_ev.white().score(mate_score=MATE_CP)
                    has_eval = True
                else:
                    cp = None

                pos = tokenize_position(
                    fen=board.fen(),
                    vocab=vocab,
                    uci_move=move.uci(),
                    white_elo=w_elo,
                    black_elo=b_elo,
                    white_clock_seconds=w_clock,
                    black_clock_seconds=b_clock,
                    increment_seconds=increment_s,
                    game_id=game_id,
                )

                if cp is not None:
                    avg_elo = (w_elo + b_elo) // 2
                    spread  = abs(w_elo - b_elo)
                    rows.append({
                        "game_id":             pos.game_id,
                        "elo_bucket":          pos.elo_bucket or "",
                        "white_elo":           pos.white_elo,
                        "black_elo":           pos.black_elo,
                        "avg_elo":             avg_elo,
                        "elo_spread":          spread,
                        "white_clock_seconds": pos.white_clock_seconds,
                        "black_clock_seconds": pos.black_clock_seconds,
                        "increment_seconds":   pos.increment_seconds,
                        "meta_tokens":         pos.meta_tokens,
                        "color_tokens":        pos.color_tokens,
                        "piece_type_tokens":   pos.piece_type_tokens,
                        "file_tokens":         pos.file_tokens,
                        "rank_tokens":         pos.rank_tokens,
                        "from_square_id":      pos.from_square_id if pos.from_square_id is not None else -1,
                        "to_square_id":        pos.to_square_id   if pos.to_square_id   is not None else -1,
                        "from_file_id":        pos.from_file_id   if pos.from_file_id   is not None else -1,
                        "from_rank_id":        pos.from_rank_id   if pos.from_rank_id   is not None else -1,
                        "to_file_id":          pos.to_file_id     if pos.to_file_id     is not None else -1,
                        "to_rank_id":          pos.to_rank_id     if pos.to_rank_id     is not None else -1,
                        "promo_id":            pos.promo_id if pos.promo_id is not None else -1,
                        "cp_eval":             float(cp),
                    })

                # Update clock for the player who just moved
                clock_s = node.clock()
                if clock_s is not None:
                    if board.turn == chess.WHITE:
                        w_clock = clock_s
                    else:
                        b_clock = clock_s

                board.push(move)

            if has_eval and rows:
                yield rows, game_id
                n_yielded += 1
                if n_yielded % 500 == 0:
                    print(f"  {n_yielded} games with evals (scanned {n_scanned})", flush=True)

    print(f"Finished scanning {n_scanned} games, found {n_yielded} with eval annotations")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = cfg.paths.processed_dir
    os.makedirs(processed_dir, exist_ok=True)

    pgn_path  = cfg.get("eval_pgn_path",  "data/raw_pgn/lichess_db_standard_rated_2014-09.pgn")
    max_games = cfg.get("eval_max_games", 5000)

    print(f"Source  : {pgn_path}")
    print(f"Max eval games: {max_games}")

    vocab = build_vocab()
    writers = {
        "train": _ParquetWriter(os.path.join(processed_dir, "train_eval.parquet")),
        "val":   _ParquetWriter(os.path.join(processed_dir, "val_eval.parquet")),
        "test":  _ParquetWriter(os.path.join(processed_dir, "test_eval.parquet")),
    }

    t0 = time.time()
    counts = {"train": 0, "val": 0, "test": 0}

    for rows, game_id in _iter_eval_games(pgn_path, vocab, max_games):
        split = _split(game_id)
        for row in rows:
            writers[split].write(row)
        counts[split] += 1

    totals = {k: w.close() for k, w in writers.items()}
    elapsed = time.time() - t0
    total_pos = sum(totals.values())
    total_games = sum(counts.values())

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Games : train={counts['train']}  val={counts['val']}  test={counts['test']}  total={total_games}")
    print(f"Positions: train={totals['train']:,}  val={totals['val']:,}  test={totals['test']:,}  total={total_pos:,}")
    print(f"Output: {processed_dir}/{{train,val,test}}_eval.parquet")


if __name__ == "__main__":
    main()
