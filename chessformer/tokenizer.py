"""
chessformer/tokenizer.py

Vocabulary and position tokenization.

The vocab is defined entirely in code — no JSON file. Call build_vocab() once at
startup and pass the resulting Vocab everywhere.

Board representation
--------------------
Each piece gets a single sequence slot. Its embedding in the model is the sum of
four independent lookups:
    emb[color] + emb[piece_type] + emb[file] + emb[rank]

Sequence layout
---------------
1. Meta slot (1 position): additive sum of side-to-move + castling rights + en-passant.
   All applicable token embeddings are summed into a single sequence position.
2. Conditioning slots (5 positions): w_elo, b_elo, w_clock, b_clock, increment.
   Each uses soft interpolation between the two nearest bracket embeddings.
3. Piece slots (up to 32 positions): one per piece on the board.
4. Move suffix (3 positions): MOVE_BOS → from_square → to_square.

Elo, clock, and increment conditioning
---------------------------------------
Raw scalars stored in PositionTokens. The model soft-blends between adjacent
bracket embeddings:
    alpha * emb[lower_bracket] + (1 - alpha) * emb[upper_bracket]

Move representation
-------------------
Moves are autoregressive: from-square → to-square → promotion (if any).
At training time this is done in one forward pass with causal masking.
from/to squares reuse shared file_*/rank_* token embeddings (like pieces).
tokenize_position() returns from/to/promo as separate targets.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Optional

import chess


# ---------------------------------------------------------------------------
# Vocab constants
# ---------------------------------------------------------------------------

# Elo brackets: 0, 100, ..., 3400. Store raw int for soft blending at runtime.
ELO_BRACKETS: list[int] = list(range(0, 3401, 100))  # 35 values

# Clock brackets in seconds. Finer resolution under 30s where time pressure
# most affects move quality. Max 10800s (3h) covers classical time controls.
CLOCK_BRACKETS_S: list[float] = [
    0, 5, 10, 20, 30, 60, 120, 180, 300, 600, 1200, 3600, 10800,
]

# Increment brackets in seconds. 60 acts as "60+" (clamped). -1 = unknown.
INCREMENT_BRACKETS_S: list[int] = [0, 1, 3, 5, 10, 30, 60]

# Maximum additive tokens in the single meta slot:
# 1 (side to move) + 4 (castling rights) + 1 (en passant) = 6
META_SLOT_SIZE: int = 6

_FILES = list("abcdefgh")
_RANKS = list("12345678")
_PIECE_TYPES = ["pawn", "knight", "bishop", "rook", "queen", "king"]
_COLORS = ["w_color", "b_color"]
_PROMOTIONS = ["promote_n", "promote_b", "promote_r", "promote_q"]

_CHESS_PIECE_TO_TYPE = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_PROMO_CHAR_TO_TOKEN = {"n": "promote_n", "b": "promote_b", "r": "promote_r", "q": "promote_q"}


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: dict[int, str]
    vocab_size: int

    # Role tokens for the move suffix (distinguish from-square from to-square)
    from_tok_id: int   # "from_tok"
    to_tok_id: int     # "to_tok"

    # Role token for the increment conditioning slot
    incr_tok_id: int   # "incr_tok"

    # First token ID of the promotion group (model output head)
    promo_offset: int  # promote_n .. promote_q

    # Ordered bracket IDs for soft blending
    elo_bracket_ids:  tuple[int, ...]
    clock_bracket_ids: tuple[int, ...]
    incr_bracket_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        assert len(self.elo_bracket_ids)  == len(ELO_BRACKETS)
        assert len(self.clock_bracket_ids) == len(CLOCK_BRACKETS_S)
        assert len(self.incr_bracket_ids) == len(INCREMENT_BRACKETS_S)


def build_vocab() -> Vocab:
    """Build and return the full token vocabulary. Call once; reuse everywhere."""
    tokens: list[str] = []

    # Meta slot: side to move, castling rights, en passant (summed additively in model)
    tokens += ["play_w", "play_b"]
    tokens += ["w_k", "w_q", "b_k", "b_q"]
    tokens += [f"ep_{f}" for f in _FILES]

    # Elo conditioning (soft blend)
    tokens += [f"elo_{e}" for e in ELO_BRACKETS] + ["elo_unknown"]

    # Clock conditioning (soft blend)
    tokens += [f"clock_{int(s) if s == int(s) else s}" for s in CLOCK_BRACKETS_S]

    # Increment conditioning (soft blend)
    incr_tok_id = len(tokens); tokens += ["incr_tok"]
    tokens += [f"increment_{s}" for s in INCREMENT_BRACKETS_S] + ["increment_unknown"]

    # Piece embeddings (color + piece_type + file + rank, summed per piece)
    tokens += _COLORS
    tokens += _PIECE_TYPES
    tokens += [f"file_{f}" for f in _FILES]
    tokens += [f"rank_{r}" for r in _RANKS]

    # Move suffix role tokens (from/to squares reuse file_*/rank_* embeddings)
    from_tok_id = len(tokens); tokens += ["from_tok"]
    to_tok_id   = len(tokens); tokens += ["to_tok"]

    promo_offset = len(tokens)
    tokens += _PROMOTIONS

    t2i = {tok: i for i, tok in enumerate(tokens)}
    i2t = {i: tok for i, tok in enumerate(tokens)}

    return Vocab(
        token_to_id=t2i,
        id_to_token=i2t,
        vocab_size=len(tokens),
        from_tok_id=from_tok_id,
        to_tok_id=to_tok_id,
        incr_tok_id=incr_tok_id,
        promo_offset=promo_offset,
        elo_bracket_ids=tuple(t2i[f"elo_{e}"] for e in ELO_BRACKETS),
        clock_bracket_ids=tuple(
            t2i[f"clock_{int(s) if s == int(s) else s}"] for s in CLOCK_BRACKETS_S
        ),
        incr_bracket_ids=tuple(t2i[f"increment_{s}"] for s in INCREMENT_BRACKETS_S),
    )


# ---------------------------------------------------------------------------
# Position tokens
# ---------------------------------------------------------------------------

@dataclass
class PositionTokens:
    """
    Tokenized representation of a chess position and (optionally) a move target.

    Board pieces are stored as four parallel lists — one entry per piece on the
    board, in chess.SQUARES order (a1, b1, ..., h8). The model embeds each piece
    as emb[color] + emb[piece_type] + emb[file] + emb[rank].

    Elo and clock are stored as raw scalars. The model handles soft blending via
    elo_blend_weights() / clock_blend_weights(). Use -1 / -1.0 for unknown.

    Move fields are None when tokenizing for inference (no move known yet).
    """
    # Meta slot: additive token IDs (side_to_move + castling + ep), length ≤ META_SLOT_SIZE.
    # These are SUMMED into a single sequence position in the model (not sequenced).
    meta_tokens: list[int]

    # Board — parallel lists, one entry per piece (up to 32)
    color_tokens: list[int]
    piece_type_tokens: list[int]
    file_tokens: list[int]
    rank_tokens: list[int]

    # Continuous conditioning — raw values, model handles soft blending. -1/-1.0 = unknown.
    white_elo: int              # white player's Elo
    black_elo: int              # black player's Elo
    white_clock_seconds: float  # white's remaining clock at this position
    black_clock_seconds: float  # black's remaining clock at this position
    increment_seconds: float    # time increment per move; -1.0 = unknown

    # Move targets (None during inference).
    # from/to square as a plain 0-63 chess square index (chess.SQUARES order).
    # from/to file/rank are vocab token IDs reusing the shared file_*/rank_* tokens.
    from_square_id: Optional[int] = None   # 0-63
    to_square_id:   Optional[int] = None   # 0-63
    from_file_id:   Optional[int] = None   # vocab ID of file_a..file_h
    from_rank_id:   Optional[int] = None   # vocab ID of rank_1..rank_8
    to_file_id:     Optional[int] = None
    to_rank_id:     Optional[int] = None
    promo_id:       Optional[int] = None   # None also when no promotion

    # Metadata (not fed to model; used for splits and per-bucket eval)
    game_id: str = ""
    elo_bucket: str = ""  # side-to-move's Elo bucket, e.g. "elo_1500"

    @property
    def n_pieces(self) -> int:
        return len(self.color_tokens)

    @property
    def has_move(self) -> bool:
        return self.from_square_id is not None


# ---------------------------------------------------------------------------
# Core tokenization
# ---------------------------------------------------------------------------

def tokenize_position(
    fen: str,
    vocab: Vocab,
    uci_move: Optional[str] = None,
    white_elo: int = -1,
    black_elo: int = -1,
    white_clock_seconds: float = -1.0,
    black_clock_seconds: float = -1.0,
    increment_seconds: float = -1.0,
    game_id: str = "",
) -> PositionTokens:
    """
    Tokenize a single chess position.

    Args:
        fen: Standard FEN string.
        vocab: Vocab from build_vocab().
        uci_move: Move in UCI notation ("e2e4", "e7e8q"). None during inference.
        white_elo: White player's Elo. -1 if unknown.
        black_elo: Black player's Elo. -1 if unknown.
        white_clock_seconds: White's remaining clock seconds. -1.0 if unknown.
        black_clock_seconds: Black's remaining clock seconds. -1.0 if unknown.
        increment_seconds: Time increment per move in seconds. -1.0 if unknown.
        game_id: Source game identifier (for reproducible splits).

    Returns:
        PositionTokens. Move fields are None when uci_move is None.
    """
    t = vocab.token_to_id
    board = chess.Board(fen)
    turn_is_white = board.turn == chess.WHITE

    # --- Meta slot (additive — all embeddings summed into one sequence position) ---
    meta: list[int] = [t["play_w" if turn_is_white else "play_b"]]

    if board.has_kingside_castling_rights(chess.WHITE):
        meta.append(t["w_k"])
    if board.has_queenside_castling_rights(chess.WHITE):
        meta.append(t["w_q"])
    if board.has_kingside_castling_rights(chess.BLACK):
        meta.append(t["b_k"])
    if board.has_queenside_castling_rights(chess.BLACK):
        meta.append(t["b_q"])

    if board.ep_square is not None:
        meta.append(t[f"ep_{_FILES[chess.square_file(board.ep_square)]}"])

    # --- Board pieces -------------------------------------------------------
    color_toks: list[int] = []
    ptype_toks: list[int] = []
    file_toks: list[int] = []
    rank_toks: list[int] = []

    for sq in chess.SQUARES:  # a1=0 .. h8=63
        piece = board.piece_at(sq)
        if piece is None:
            continue
        color_toks.append(t["w_color" if piece.color == chess.WHITE else "b_color"])
        ptype_toks.append(t[_CHESS_PIECE_TO_TYPE[piece.piece_type]])
        file_toks.append(t[f"file_{_FILES[chess.square_file(sq)]}"])
        rank_toks.append(t[f"rank_{_RANKS[chess.square_rank(sq)]}"])

    # --- Move targets -------------------------------------------------------
    from_sq = to_sq = from_file = from_rank = to_file = to_rank = promo_id = None
    if uci_move is not None:
        from_sq_name, to_sq_name = uci_move[:2], uci_move[2:4]
        from_sq = chess.parse_square(from_sq_name)   # 0-63
        to_sq   = chess.parse_square(to_sq_name)
        from_file = t[f"file_{_FILES[chess.square_file(from_sq)]}"]
        from_rank = t[f"rank_{_RANKS[chess.square_rank(from_sq)]}"]
        to_file   = t[f"file_{_FILES[chess.square_file(to_sq)]}"]
        to_rank   = t[f"rank_{_RANKS[chess.square_rank(to_sq)]}"]
        if len(uci_move) > 4:
            promo_id = t[_PROMO_CHAR_TO_TOKEN[uci_move[4].lower()]]

    side_elo = white_elo if turn_is_white else black_elo
    elo_bucket = _elo_bucket_str(side_elo)

    return PositionTokens(
        meta_tokens=meta,
        color_tokens=color_toks,
        piece_type_tokens=ptype_toks,
        file_tokens=file_toks,
        rank_tokens=rank_toks,
        white_elo=white_elo,
        black_elo=black_elo,
        white_clock_seconds=white_clock_seconds,
        black_clock_seconds=black_clock_seconds,
        increment_seconds=increment_seconds,
        from_square_id=from_sq,
        to_square_id=to_sq,
        from_file_id=from_file,
        from_rank_id=from_rank,
        to_file_id=to_file,
        to_rank_id=to_rank,
        promo_id=promo_id,
        game_id=game_id,
        elo_bucket=elo_bucket,
    )


# ---------------------------------------------------------------------------
# Soft-blend weight helpers  (called by model embedding layer)
# ---------------------------------------------------------------------------

def elo_blend_weights(elo: int, vocab: Vocab) -> tuple[int, int, float]:
    """
    Map a raw Elo value to (lower_id, upper_id, alpha) for soft blending:
        embedding = alpha * emb[lower_id] + (1 - alpha) * emb[upper_id]
    alpha=1.0 means exactly at the lower bracket (upper_id is ignored).
    """
    return _blend_weights(elo, ELO_BRACKETS, vocab.elo_bracket_ids)


def clock_blend_weights(clock_s: float, vocab: Vocab) -> tuple[int, int, float]:
    """
    Map clock time (seconds) to (lower_id, upper_id, alpha) for soft blending.
    """
    return _blend_weights(clock_s, CLOCK_BRACKETS_S, vocab.clock_bracket_ids)


def _blend_weights(
    value: float,
    brackets: list,
    bracket_ids: tuple[int, ...],
) -> tuple[int, int, float]:
    if value <= brackets[0]:
        return bracket_ids[0], bracket_ids[0], 1.0
    if value >= brackets[-1]:
        return bracket_ids[-1], bracket_ids[-1], 1.0

    # bisect_right gives the index of the first bracket strictly greater than value
    i = bisect.bisect_right(brackets, value) - 1
    lo, hi = brackets[i], brackets[i + 1]
    alpha = (hi - value) / (hi - lo)  # 1.0 at lo, 0.0 at hi
    return bracket_ids[i], bracket_ids[i + 1], float(alpha)


# ---------------------------------------------------------------------------
# Time-control helpers
# ---------------------------------------------------------------------------

def parse_increment_seconds(tc_str: str) -> float:
    """
    Extract the per-move increment from a Lichess time-control string.

    "300+3"  → 3.0    (3 seconds per move)
    "600+0"  → 0.0    (no increment)
    "600"    → 0.0    (no increment specified)
    "-"      → -1.0   (unknown / no time control)
    """
    if not tc_str or tc_str == "-":
        return -1.0
    try:
        parts = tc_str.split("+")
        return float(int(parts[1])) if len(parts) > 1 else 0.0
    except (ValueError, IndexError):
        return -1.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _elo_bucket_str(elo: int) -> str:
    """Return the nearest lower Elo bracket string, e.g. 'elo_1500'."""
    if elo < 0:
        return "elo_unknown"
    clamped = max(0, min(3400, (elo // 100) * 100))
    return f"elo_{clamped}"
