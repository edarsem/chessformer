"""
scripts/preprocess.py

Convert raw PGN and puzzle CSV into parquet files. Run before make_splits.py.

Two modes, controlled by cfg.data.mode:

  train   — Stream games from cfg.data.train_pgn_source (2018-01).
            Stop at cfg.data.train_games games (None = all).
            Output: processed/all_games_train.parquet

  valtest — Stream games from cfg.data.valtest_pgn_source (2017-12).
            Select exactly cfg.data.valtest_games_per_elo_bucket games per
            Elo bucket (800..3000 in steps of 100). Stop early once all
            buckets are full, so we never read the whole 18M-game file.
            Eligibility: avg(white_elo, black_elo) in [bucket, bucket+100)
            AND |white_elo - black_elo| <= cfg.data.elo_spread_max.
            Output: processed/all_games_valtest.parquet

  puzzles — Process cfg.data.puzzles_source.
            Output: processed/all_puzzles.parquet

Run each mode separately, or pass mode=all to run all three sequentially.

Usage:
    python scripts/preprocess.py data.mode=train
    python scripts/preprocess.py data.mode=valtest
    python scripts/preprocess.py data.mode=puzzles
    python scripts/preprocess.py data.mode=all
    python scripts/preprocess.py data.mode=train data.train_games=100000
"""

from __future__ import annotations

import os
import sys
import time

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from chessformer.pgn import iter_pgn_games_with_elos, iter_puzzle_positions
from chessformer.tokenizer import PositionTokens, Vocab, build_vocab


# ---------------------------------------------------------------------------
# Parquet schemas
# ---------------------------------------------------------------------------

GAMES_SCHEMA = pa.schema([
    pa.field("game_id",              pa.string()),
    pa.field("elo_bucket",           pa.string()),
    pa.field("white_elo",            pa.int16()),
    pa.field("black_elo",            pa.int16()),
    pa.field("avg_elo",              pa.int16()),
    pa.field("elo_spread",           pa.int16()),
    pa.field("white_clock_seconds",  pa.float32()),
    pa.field("black_clock_seconds",  pa.float32()),
    pa.field("meta_tokens",          pa.list_(pa.int16())),
    pa.field("color_tokens",         pa.list_(pa.int8())),
    pa.field("piece_type_tokens",    pa.list_(pa.int8())),
    pa.field("file_tokens",          pa.list_(pa.int8())),
    pa.field("rank_tokens",          pa.list_(pa.int8())),
    pa.field("from_square_id",       pa.int8()),   # 0-63 chess square index
    pa.field("to_square_id",         pa.int8()),   # 0-63 chess square index
    pa.field("from_file_id",         pa.int8()),   # vocab ID of file_a..file_h
    pa.field("from_rank_id",         pa.int8()),   # vocab ID of rank_1..rank_8
    pa.field("to_file_id",           pa.int8()),
    pa.field("to_rank_id",           pa.int8()),
    pa.field("promo_id",             pa.int16()),  # -1 = no promotion
])

PUZZLES_SCHEMA = pa.schema([
    pa.field("puzzle_id",            pa.string()),
    pa.field("puzzle_rating",        pa.int16()),
    pa.field("seq_index",            pa.int16()),
    pa.field("is_solver_move",       pa.bool_()),
    pa.field("white_elo",            pa.int16()),     # always 3400 (GM conditioning)
    pa.field("black_elo",            pa.int16()),     # always 3400
    pa.field("white_clock_seconds",  pa.float32()),   # always -1.0 (unknown)
    pa.field("black_clock_seconds",  pa.float32()),   # always -1.0
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
])


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def pos_to_game_row(pos: PositionTokens, avg_elo: int, elo_spread: int) -> dict:
    return {
        "game_id":             pos.game_id,
        "elo_bucket":          pos.elo_bucket,
        "white_elo":           pos.white_elo,
        "black_elo":           pos.black_elo,
        "avg_elo":             avg_elo,
        "elo_spread":          elo_spread,
        "white_clock_seconds": pos.white_clock_seconds,
        "black_clock_seconds": pos.black_clock_seconds,
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
    }


def pos_to_puzzle_row(pos: PositionTokens, puzzle_rating: int, seq_index: int) -> dict:
    is_solver = pos.has_move
    return {
        "puzzle_id":             pos.game_id,
        "puzzle_rating":         puzzle_rating,
        "seq_index":             seq_index,
        "is_solver_move":        is_solver,
        "white_elo":             pos.white_elo,
        "black_elo":             pos.black_elo,
        "white_clock_seconds":   pos.white_clock_seconds,
        "black_clock_seconds":   pos.black_clock_seconds,
        "meta_tokens":           pos.meta_tokens,
        "color_tokens":          pos.color_tokens,
        "piece_type_tokens":     pos.piece_type_tokens,
        "file_tokens":           pos.file_tokens,
        "rank_tokens":           pos.rank_tokens,
        "from_square_id":        (pos.from_square_id if pos.from_square_id is not None else -1) if is_solver else -1,
        "to_square_id":          (pos.to_square_id   if pos.to_square_id   is not None else -1) if is_solver else -1,
        "from_file_id":          (pos.from_file_id   if pos.from_file_id   is not None else -1) if is_solver else -1,
        "from_rank_id":          (pos.from_rank_id   if pos.from_rank_id   is not None else -1) if is_solver else -1,
        "to_file_id":            (pos.to_file_id     if pos.to_file_id     is not None else -1) if is_solver else -1,
        "to_rank_id":            (pos.to_rank_id     if pos.to_rank_id     is not None else -1) if is_solver else -1,
        "promo_id":              (pos.promo_id if pos.promo_id is not None else -1) if is_solver else -1,
    }


# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------

class ParquetBatchWriter:
    def __init__(self, path: str, schema: pa.Schema, batch_size: int = 50_000):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._writer = pq.ParquetWriter(path, schema, compression="snappy")
        self._schema = schema
        self._batch_size = batch_size
        self._buffer: list[dict] = []
        self._total = 0

    def write(self, row: dict) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self._batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        arrays = {field.name: [row[field.name] for row in self._buffer]
                  for field in self._schema}
        self._writer.write_table(pa.table(arrays, schema=self._schema))
        self._total += len(self._buffer)
        self._buffer = []

    def close(self) -> int:
        self._flush()
        self._writer.close()
        return self._total


# ---------------------------------------------------------------------------
# Elo bucket helpers
# ---------------------------------------------------------------------------

ELO_BUCKET_RANGE = list(range(1000, 2901, 100))  # [1000, 1100, ..., 2900] — 20 buckets


def _elo_bucket_for_game(avg_elo: int) -> int | None:
    """Return the bucket lower bound if avg_elo falls within ELO_BUCKET_RANGE, else None."""
    bucket = (avg_elo // 100) * 100
    if bucket not in _BUCKET_SET:
        return None
    return bucket

_BUCKET_SET = set(ELO_BUCKET_RANGE)


# ---------------------------------------------------------------------------
# Processing modes
# ---------------------------------------------------------------------------

def process_train(pgn_path: str, out_path: str, vocab: Vocab, max_games: int | None) -> None:
    print(f"[train] source : {pgn_path}")
    print(f"[train] max    : {'all' if max_games is None else f'{max_games:,}'} games")
    print(f"[train] output : {out_path}")

    writer = ParquetBatchWriter(out_path, GAMES_SCHEMA)
    t0 = time.time()
    n_games = n_skipped = 0

    for positions, w_elo, b_elo in iter_pgn_games_with_elos(pgn_path, vocab):
        if max_games is not None and n_games >= max_games:
            break

        # Skip games where either Elo is unknown — not useful for train conditioning
        if w_elo < 0 or b_elo < 0:
            n_skipped += 1
            continue

        avg_elo = (w_elo + b_elo) // 2
        spread = abs(w_elo - b_elo)
        for pos in positions:
            writer.write(pos_to_game_row(pos, avg_elo, spread))
        n_games += 1

        if n_games % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"  {n_games:,} games  ({n_games/elapsed:.0f} games/s)", flush=True)

    total = writer.close()
    elapsed = time.time() - t0
    print(f"[train] done: {n_games:,} games, {total:,} positions in {elapsed:.1f}s  (skipped {n_skipped} no-elo)")


def process_valtest(
    pgn_path: str,
    out_path: str,
    vocab: Vocab,
    max_games: int,
    elo_spread_max: int,
) -> None:
    """
    Collect up to max_games qualifying games from pgn_path and write them.
    Qualifying = avg(white_elo, black_elo) in ELO_BUCKET_RANGE and
    |white_elo - black_elo| <= elo_spread_max.

    No attempt to balance across buckets — just take the first max_games
    qualifying games. The natural Elo distribution of the source file is
    preserved. Per-bucket eval breakdown is done at eval time on whatever
    distribution we get.
    """
    print(f"[valtest] source     : {pgn_path}")
    print(f"[valtest] max games  : {max_games:,}")
    print(f"[valtest] elo range  : {ELO_BUCKET_RANGE[0]}–{ELO_BUCKET_RANGE[-1]}")
    print(f"[valtest] spread max : ±{elo_spread_max} Elo")
    print(f"[valtest] output     : {out_path}")

    writer = ParquetBatchWriter(out_path, GAMES_SCHEMA)
    t0 = time.time()
    n_seen = n_written = 0

    for positions, w_elo, b_elo in iter_pgn_games_with_elos(pgn_path, vocab):
        n_seen += 1
        if n_written >= max_games:
            break

        if w_elo < 0 or b_elo < 0:
            continue
        if abs(w_elo - b_elo) > elo_spread_max:
            continue

        avg_elo = (w_elo + b_elo) // 2
        if _elo_bucket_for_game(avg_elo) is None:
            continue

        spread = abs(w_elo - b_elo)
        for pos in positions:
            writer.write(pos_to_game_row(pos, avg_elo, spread))
        n_written += 1

        if n_written % 1_000 == 0:
            elapsed = time.time() - t0
            print(f"  {n_written:,} / {max_games:,} games  (seen {n_seen:,}, {n_written/elapsed:.0f} games/s)", flush=True)

    total = writer.close()
    elapsed = time.time() - t0
    print(f"[valtest] done: {n_written:,} games, {total:,} positions in {elapsed:.1f}s  (scanned {n_seen:,})")


def process_test_plain(
    pgn_path: str,
    out_path: str,
    vocab: Vocab,
    max_games: int,
) -> None:
    """First max_games games with valid Elo from pgn_path. No Elo range restriction."""
    print(f"[test_plain] source   : {pgn_path}")
    print(f"[test_plain] max games: {max_games:,}")
    print(f"[test_plain] output   : {out_path}")

    writer = ParquetBatchWriter(out_path, GAMES_SCHEMA)
    t0 = time.time()
    n_seen = n_written = 0

    for positions, w_elo, b_elo in iter_pgn_games_with_elos(pgn_path, vocab):
        n_seen += 1
        if n_written >= max_games:
            break
        if w_elo < 0 or b_elo < 0:
            continue
        avg_elo = (w_elo + b_elo) // 2
        spread = abs(w_elo - b_elo)
        for pos in positions:
            writer.write(pos_to_game_row(pos, avg_elo, spread))
        n_written += 1
        if n_written % 1_000 == 0:
            elapsed = time.time() - t0
            print(f"  {n_written:,} / {max_games:,} games  (seen {n_seen:,}, {n_written/elapsed:.0f} games/s)", flush=True)

    total = writer.close()
    elapsed = time.time() - t0
    print(f"[test_plain] done: {n_written:,} games, {total:,} positions in {elapsed:.1f}s  (scanned {n_seen:,})")


def process_test_elo(
    pgn_path: str,
    out_path: str,
    vocab: Vocab,
    skip_games: int,
    scan_games: int,
    games_per_bucket: int,
    elo_spread_max: int,
) -> None:
    """
    Collect up to games_per_bucket qualifying games per Elo bucket.
    Skips the first skip_games games to avoid overlap with test_plain.
    Stops when all buckets are full OR scan_games have been scanned.
    """
    print(f"[test_elo] source         : {pgn_path}")
    print(f"[test_elo] skip games     : {skip_games:,}")
    print(f"[test_elo] scan limit     : {scan_games:,}")
    print(f"[test_elo] games/bucket   : {games_per_bucket}")
    print(f"[test_elo] elo range      : {ELO_BUCKET_RANGE[0]}–{ELO_BUCKET_RANGE[-1]}")
    print(f"[test_elo] spread max     : ±{elo_spread_max} Elo")
    print(f"[test_elo] output         : {out_path}")

    writer = ParquetBatchWriter(out_path, GAMES_SCHEMA)
    t0 = time.time()
    n_scanned = n_written = 0
    bucket_counts: dict[int, int] = {b: 0 for b in ELO_BUCKET_RANGE}
    n_buckets = len(ELO_BUCKET_RANGE)

    for positions, w_elo, b_elo in iter_pgn_games_with_elos(pgn_path, vocab, skip_games=skip_games):
        n_scanned += 1
        if n_scanned > scan_games:
            break
        if sum(c >= games_per_bucket for c in bucket_counts.values()) == n_buckets:
            break

        if w_elo < 0 or b_elo < 0:
            continue
        if abs(w_elo - b_elo) > elo_spread_max:
            continue

        avg_elo = (w_elo + b_elo) // 2
        bucket = _elo_bucket_for_game(avg_elo)
        if bucket is None:
            continue
        if bucket_counts[bucket] >= games_per_bucket:
            continue

        spread = abs(w_elo - b_elo)
        for pos in positions:
            writer.write(pos_to_game_row(pos, avg_elo, spread))
        bucket_counts[bucket] += 1
        n_written += 1

        if n_written % 500 == 0:
            elapsed = time.time() - t0
            full = sum(c >= games_per_bucket for c in bucket_counts.values())
            print(f"  {n_written:,} games written  ({full}/{n_buckets} buckets full, scanned {n_scanned:,}, {n_written/elapsed:.0f} games/s)", flush=True)

    total = writer.close()
    elapsed = time.time() - t0
    full = sum(c >= games_per_bucket for c in bucket_counts.values())
    print(f"[test_elo] done: {n_written:,} games, {total:,} positions in {elapsed:.1f}s  (scanned {n_scanned:,}, {full}/{n_buckets} buckets full)")


def process_puzzles(
    puzzle_path: str,
    out_path: str,
    vocab: Vocab,
    rating_min: int | None,
    rating_max: int | None,
) -> None:
    print(f"[puzzles] source : {puzzle_path}")
    print(f"[puzzles] rating : {rating_min or 'any'}..{rating_max or 'any'}")
    print(f"[puzzles] output : {out_path}")

    writer = ParquetBatchWriter(out_path, PUZZLES_SCHEMA)
    t0 = time.time()
    n_puzzles = 0

    for puzzle_seq, puzzle_rating in iter_puzzle_positions(
        puzzle_path, vocab, rating_min=rating_min, rating_max=rating_max
    ):
        for i, pos in enumerate(puzzle_seq):
            writer.write(pos_to_puzzle_row(pos, puzzle_rating, i))
        n_puzzles += 1
        if n_puzzles % 100_000 == 0:
            print(f"  {n_puzzles:,} puzzles", flush=True)

    total = writer.close()
    elapsed = time.time() - t0
    print(f"[puzzles] done: {n_puzzles:,} puzzles, {total:,} positions in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    processed_dir = cfg.paths.processed_dir
    os.makedirs(processed_dir, exist_ok=True)

    vocab = build_vocab()
    mode = cfg.data.get("mode", "all")

    if mode in ("train", "all"):
        process_train(
            pgn_path=cfg.data.train_pgn_source,
            out_path=os.path.join(processed_dir, "all_games_train.parquet"),
            vocab=vocab,
            max_games=cfg.data.get("train_games", None),
        )

    if mode in ("valtest", "all"):
        process_valtest(
            pgn_path=cfg.data.valtest_pgn_source,
            out_path=os.path.join(processed_dir, "all_games_valtest.parquet"),
            vocab=vocab,
            max_games=cfg.data.valtest_games,
            elo_spread_max=cfg.data.get("elo_spread_max", 200),
        )

    if mode in ("test_plain", "all"):
        process_test_plain(
            pgn_path=cfg.data.valtest_pgn_source,
            out_path=os.path.join(processed_dir, "test_games.parquet"),
            vocab=vocab,
            max_games=cfg.data.get("test_plain_games", 10000),
        )

    if mode in ("test_elo", "all"):
        process_test_elo(
            pgn_path=cfg.data.valtest_pgn_source,
            out_path=os.path.join(processed_dir, "test_games_elo.parquet"),
            vocab=vocab,
            skip_games=cfg.data.get("test_elo_skip_games", 10000),
            scan_games=cfg.data.get("test_elo_scan_games", 200000),
            games_per_bucket=cfg.data.get("test_elo_games_per_bucket", 500),
            elo_spread_max=cfg.data.get("elo_spread_max", 200),
        )

    if mode in ("puzzles", "all"):
        process_puzzles(
            puzzle_path=cfg.data.puzzles_source,
            out_path=os.path.join(processed_dir, "all_puzzles.parquet"),
            vocab=vocab,
            rating_min=cfg.data.get("puzzle_rating_min", None),
            rating_max=cfg.data.get("puzzle_rating_max", None),
        )


if __name__ == "__main__":
    main()
