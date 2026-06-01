"""
chessformer/model.py

Sequence layout fed to the transformer (all as embeddings):
  [w_elo] [b_elo] [w_clock] [b_clock] [meta_0..meta_n] [piece_0..piece_k] [MOVE_BOS] [from_in] [to_in]

  Conditioning slots (4 total; analogous to piece embeddings):
    w_elo:   emb[w_color] + blend(white_elo)
    b_elo:   emb[b_color] + blend(black_elo)
    w_clock: emb[w_color] + blend(white_clock_s)   — zero clock part if unknown
    b_clock: emb[b_color] + blend(black_clock_s)   — zero clock part if unknown

  meta:   play_w/b, time_control, castling rights, en passant — standard lookup
  piece:  emb[color] + emb[piece_type] + emb[file] + emb[rank] (additive)
  MOVE_BOS: learned "start of move" parameter (not in vocab)
  from_in / to_in: teacher-forced move tokens at training time

Attention:
  - Board prefix [w_elo .. pieces]: fully bidirectional
  - Move suffix  [MOVE_BOS .. to_in]: causal (sees all board + earlier move tokens)
  - Padded positions: masked out

Max sequence length: 4 + 7 + 32 + 3 = 46
  (4 conditioning + max 7 meta + max 32 pieces + 3 move suffix)

Output heads at move positions:
  MOVE_BOS → from_logits  [B, 64]
  from_in  → to_logits    [B, 64]
  to_in    → promo_logits [B, 4]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer.tokenizer import Vocab, elo_blend_weights, clock_blend_weights


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = self.d_k ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask  # additive bias: 0 or -inf
        attn = self.drop(F.softmax(scores, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, C))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = _Attention(d_model, n_heads, dropout)
        d_ff = d_model * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ChessformerModel(nn.Module):
    N_SQUARES    = 64
    N_PROMO      = 4
    N_MOVE_STEPS = 3  # MOVE_BOS, from_in, to_in

    def __init__(
        self,
        vocab: Vocab,
        d_model: int   = 128,
        n_heads: int   = 4,
        n_layers: int  = 4,
        ffn_mult: int  = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab   = vocab
        self.d_model = d_model

        self.emb      = nn.Embedding(vocab.vocab_size, d_model)
        self.move_bos = nn.Parameter(torch.empty(d_model))  # learned MOVE_BOS
        self.move_pos = nn.Embedding(self.N_MOVE_STEPS, d_model)

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head_from  = nn.Linear(d_model, self.N_SQUARES, bias=False)
        self.head_to    = nn.Linear(d_model, self.N_SQUARES, bias=False)
        self.head_promo = nn.Linear(d_model, self.N_PROMO,   bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.move_bos, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Soft-blend embeddings
    # -----------------------------------------------------------------------

    def _elo_part(self, elo: torch.Tensor) -> torch.Tensor:
        """Soft-blended Elo embedding (without color). [B] int → [B, d_model]"""
        W = self.emb.weight
        out = torch.zeros(len(elo), self.d_model, device=elo.device, dtype=W.dtype)
        unk_id = self.vocab.token_to_id["elo_unknown"]
        for i, e in enumerate(elo.tolist()):
            if e < 0:
                out[i] = W[unk_id]
            else:
                lo, hi, alpha = elo_blend_weights(int(e), self.vocab)
                out[i] = alpha * W[lo] + (1.0 - alpha) * W[hi]
        return out

    def _clock_part(self, clock_s: torch.Tensor) -> torch.Tensor:
        """Soft-blended clock embedding (without color). [B] float → [B, d_model]; unknown → zero."""
        W = self.emb.weight
        out = torch.zeros(len(clock_s), self.d_model, device=clock_s.device, dtype=W.dtype)
        for i, c in enumerate(clock_s.tolist()):
            if c >= 0:
                lo, hi, alpha = clock_blend_weights(float(c), self.vocab)
                out[i] = alpha * W[lo] + (1.0 - alpha) * W[hi]
        return out

    def _color_emb(self, color_token: str, B: int) -> torch.Tensor:
        """[B, d_model] of a single color token, broadcast across batch."""
        cid = self.vocab.token_to_id[color_token]
        return self.emb.weight[cid].unsqueeze(0).expand(B, -1)

    # -----------------------------------------------------------------------
    # Sequence builder
    # -----------------------------------------------------------------------

    def _build_sequence(
        self,
        meta_ids:       torch.Tensor,  # [B, M]
        color_ids:      torch.Tensor,  # [B, P]
        piece_type_ids: torch.Tensor,  # [B, P]
        file_ids:       torch.Tensor,  # [B, P]
        rank_ids:       torch.Tensor,  # [B, P]
        white_elo:      torch.Tensor,  # [B]
        black_elo:      torch.Tensor,  # [B]
        white_clock_s:  torch.Tensor,  # [B]
        black_clock_s:  torch.Tensor,  # [B]
        move_ids:       torch.Tensor,  # [B, 3]  — from_sq_id, to_sq_id, (ignored promo)
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            x:         [B, T, d_model]
            mask:      [B, 1, T, T]   additive (0 or -inf)
            board_len: int
        """
        B      = meta_ids.shape[0]
        device = meta_ids.device
        dtype  = self.emb.weight.dtype

        # Conditioning slots: emb[color] + blend(elo/clock)  [B, 1, d] each
        w_col = self._color_emb("w_color", B)
        b_col = self._color_emb("b_color", B)
        w_elo_emb   = (w_col + self._elo_part(white_elo)).unsqueeze(1)
        b_elo_emb   = (b_col + self._elo_part(black_elo)).unsqueeze(1)
        w_clock_emb = (w_col + self._clock_part(white_clock_s)).unsqueeze(1)
        b_clock_emb = (b_col + self._clock_part(black_clock_s)).unsqueeze(1)

        # Meta tokens  [B, M, d]
        meta_valid = meta_ids >= 0
        meta_emb   = torch.zeros(B, meta_ids.shape[1], self.d_model, device=device, dtype=dtype)
        if meta_valid.any():
            safe = meta_ids.clamp(min=0)
            meta_emb[meta_valid] = self.emb(safe)[meta_valid]

        # Piece tokens (additive)  [B, P, d]
        piece_valid = color_ids >= 0
        piece_emb   = torch.zeros(B, color_ids.shape[1], self.d_model, device=device, dtype=dtype)
        if piece_valid.any():
            p = (self.emb(color_ids.clamp(min=0))
                 + self.emb(piece_type_ids.clamp(min=0))
                 + self.emb(file_ids.clamp(min=0))
                 + self.emb(rank_ids.clamp(min=0)))
            piece_emb[piece_valid] = p[piece_valid]

        board     = torch.cat([w_elo_emb, b_elo_emb, w_clock_emb, b_clock_emb, meta_emb, piece_emb], dim=1)
        board_len = board.shape[1]

        # Padding flags per token  [B, board_len]
        board_pad = torch.cat([
            torch.zeros(B, 4, dtype=torch.bool, device=device),  # 4 conditioning slots never pad
            ~meta_valid,
            ~piece_valid,
        ], dim=1)

        # Move suffix  [B, 3, d]
        pos0 = self.move_pos.weight[0].to(dtype)
        pos1 = self.move_pos.weight[1].to(dtype)
        pos2 = self.move_pos.weight[2].to(dtype)
        bos_emb  = (self.move_bos.to(dtype) + pos0).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        from_emb = (self.emb(move_ids[:, 0].clamp(min=0)) + pos1).unsqueeze(1)
        to_emb   = (self.emb(move_ids[:, 1].clamp(min=0)) + pos2).unsqueeze(1)
        move_seq = torch.cat([bos_emb, from_emb, to_emb], dim=1)

        x   = torch.cat([board, move_seq], dim=1)                               # [B, T, d]
        pad = torch.cat([board_pad, torch.zeros(B, 3, dtype=torch.bool, device=device)], dim=1)
        T   = x.shape[1]

        # Build mask on CPU (vectorized) to avoid MPS -inf arithmetic bugs,
        # then move to device. All ops are torch vectorized — no Python loops.
        mask_cpu = torch.zeros(B, 1, T, T, dtype=dtype)

        # Causal upper triangle for move suffix — broadcasts over B and the single head dim
        mask_cpu[:, :, board_len:, board_len:] = torch.triu(
            torch.full((self.N_MOVE_STEPS, self.N_MOVE_STEPS), float('-inf')), diagonal=1,
        )

        # Key-padding: convert bool pad [B, T] → float [B, 1, 1, T] (0 or -inf),
        # then add — broadcasts over all query positions. On CPU: 0+-inf=-inf, -inf+-inf=-inf, no nans.
        pad_float = torch.where(pad.cpu(),
                                torch.full((B, T), float('-inf'), dtype=dtype),
                                torch.zeros(B, T, dtype=dtype))
        mask_cpu = mask_cpu + pad_float.unsqueeze(1).unsqueeze(1)  # [B,1,T,T] + [B,1,1,T]

        mask = mask_cpu.to(device)
        return x, mask, board_len

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        meta_ids:       torch.Tensor,
        color_ids:      torch.Tensor,
        piece_type_ids: torch.Tensor,
        file_ids:       torch.Tensor,
        rank_ids:       torch.Tensor,
        white_elo:      torch.Tensor,
        black_elo:      torch.Tensor,
        white_clock_s:  torch.Tensor,
        black_clock_s:  torch.Tensor,
        move_ids:       torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            meta_ids:       [B, max_meta]    int, padded -1
            color_ids:      [B, max_pieces]  int, padded -1
            piece_type_ids: [B, max_pieces]  int, padded -1
            file_ids:       [B, max_pieces]  int, padded -1
            rank_ids:       [B, max_pieces]  int, padded -1
            white_elo:      [B]  int, -1 = unknown
            black_elo:      [B]  int, -1 = unknown
            white_clock_s:  [B]  float, -1 = unknown
            black_clock_s:  [B]  float, -1 = unknown
            move_ids:       [B, 3]  (from_sq_id, to_sq_id, dummy) padded -1

        Returns:
            from_logits:  [B, 64]
            to_logits:    [B, 64]
            promo_logits: [B, 4]
        """
        x, mask, board_len = self._build_sequence(
            meta_ids, color_ids, piece_type_ids, file_ids, rank_ids,
            white_elo, black_elo, white_clock_s, black_clock_s, move_ids,
        )
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)

        return (
            self.head_from(x[:, board_len]),
            self.head_to(x[:, board_len + 1]),
            self.head_promo(x[:, board_len + 2]),
        )

    # -----------------------------------------------------------------------
    # Inference helper
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sample_move(
        self,
        meta_ids:       torch.Tensor,
        color_ids:      torch.Tensor,
        piece_type_ids: torch.Tensor,
        file_ids:       torch.Tensor,
        rank_ids:       torch.Tensor,
        white_elo:      torch.Tensor,
        black_elo:      torch.Tensor,
        white_clock_s:  torch.Tensor,
        black_clock_s:  torch.Tensor,
        temperature:    float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample from-sq, to-sq, promo autoregressively (3 forward passes).
        Returns (from_sq_ids, to_sq_ids, promo_ids) each [B], IDs in vocab space.
        """
        B      = meta_ids.shape[0]
        device = meta_ids.device
        vocab  = self.vocab
        board  = (meta_ids, color_ids, piece_type_ids, file_ids, rank_ids,
                  white_elo, black_elo, white_clock_s, black_clock_s)
        dummy  = torch.zeros(B, 3, dtype=torch.long, device=device)

        from_logits, _, _ = self.forward(*board, dummy)
        from_local = _sample_logits(from_logits, temperature)
        from_ids   = from_local + vocab.from_square_offset

        dummy2 = dummy.clone(); dummy2[:, 0] = from_ids
        _, to_logits, _ = self.forward(*board, dummy2)
        to_local = _sample_logits(to_logits, temperature)
        to_ids   = to_local + vocab.to_square_offset

        dummy3 = dummy2.clone(); dummy3[:, 1] = to_ids
        _, _, promo_logits = self.forward(*board, dummy3)
        promo_local = _sample_logits(promo_logits, temperature)
        promo_ids   = promo_local + vocab.promo_offset

        return from_ids, to_ids, promo_ids


def _sample_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(1)
