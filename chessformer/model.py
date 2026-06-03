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
  - Board positions cannot attend to move suffix (prevents reading ground-truth labels)
  - Padded positions: masked out

Max sequence length: 4 + 7 + 32 + 3 = 46
  (4 conditioning + max 7 meta + max 32 pieces + 3 move suffix)

Output heads at move positions:
  MOVE_BOS → from_logits  [B, 64]
  from_in  → to_logits    [B, 64]
  to_in    → promo_logits [B, 4]

Device notes:
  MPS:  mask built on CPU then moved to avoid MPS -inf arithmetic bugs.
  CUDA: mask built directly on device.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from chessformer.tokenizer import Vocab, ELO_BRACKETS, CLOCK_BRACKETS_S


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class _Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.dropout  = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Flash Attention via SDPA — mask is [B, 1, T, T] float with -inf for blocked positions
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = mask,
            dropout_p  = self.dropout if self.training else 0.0,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


class _SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_mult: int, dropout: float):
        super().__init__()
        d_ff = d_model * ffn_mult
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class _Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int, dropout: float):
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.norm2 = _RMSNorm(d_model)
        self.attn  = _Attention(d_model, n_heads, dropout)
        self.ffn   = _SwiGLU(d_model, ffn_mult, dropout)

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
        self.move_bos = nn.Parameter(torch.empty(d_model))
        self.move_pos = nn.Embedding(self.N_MOVE_STEPS, d_model)

        self.blocks = nn.ModuleList([
            _Block(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)
        ])
        self.norm = _RMSNorm(d_model)

        self.head_from  = nn.Linear(d_model, self.N_SQUARES, bias=False)
        self.head_to    = nn.Linear(d_model, self.N_SQUARES, bias=False)
        self.head_promo = nn.Linear(d_model, self.N_PROMO,   bias=False)

        # Bracket boundaries and token IDs for vectorized soft-blend embedding.
        # Registered as buffers so they move with the model to the right device.
        self.register_buffer("elo_bounds",   torch.tensor(ELO_BRACKETS,    dtype=torch.float32))
        self.register_buffer("clock_bounds", torch.tensor(CLOCK_BRACKETS_S, dtype=torch.float32))
        self.register_buffer("elo_ids",   torch.tensor(list(vocab.elo_bracket_ids),   dtype=torch.long))
        self.register_buffer("clock_ids", torch.tensor(list(vocab.clock_bracket_ids), dtype=torch.long))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.move_bos, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------------
    # Vectorized soft-blend embeddings
    # -----------------------------------------------------------------------

    def _blend(
        self,
        values:  torch.Tensor,   # [B]  raw scalar (elo int or clock float)
        bounds:  torch.Tensor,   # [N]  bracket boundaries on device
        ids:     torch.Tensor,   # [N]  vocab token IDs on device
        unknown: torch.Tensor,   # [B]  bool mask for unknown/missing values
        unk_emb: torch.Tensor,   # [d]  embedding to use for unknown
    ) -> torch.Tensor:           # [B, d]
        """Fully vectorized soft-blend between two adjacent bracket embeddings."""
        W = self.emb.weight                                         # [V, d]
        values_f = values.float().clamp(bounds[0], bounds[-1])

        # idx = last bracket index <= value  (shape [B])
        idx = torch.bucketize(values_f, bounds, right=True) - 1
        idx = idx.clamp(0, len(bounds) - 2)

        lo_val = bounds[idx]        # [B]
        hi_val = bounds[idx + 1]    # [B]
        span   = (hi_val - lo_val).clamp(min=1e-6)
        alpha  = ((hi_val - values_f) / span).clamp(0.0, 1.0).to(W.dtype)  # [B]

        lo_emb = W[ids[idx    ]]    # [B, d]
        hi_emb = W[ids[idx + 1]]    # [B, d]
        out    = alpha.unsqueeze(1) * lo_emb + (1.0 - alpha).unsqueeze(1) * hi_emb

        if unknown.any():
            out = out.clone()
            out[unknown] = unk_emb
        return out

    def _elo_part(self, elo: torch.Tensor) -> torch.Tensor:
        """[B] int → [B, d_model]. -1 = unknown → elo_unknown embedding."""
        unk_emb = self.emb.weight[self.vocab.token_to_id["elo_unknown"]]
        return self._blend(elo, self.elo_bounds, self.elo_ids, elo < 0, unk_emb)

    def _clock_part(self, clock_s: torch.Tensor) -> torch.Tensor:
        """[B] float → [B, d_model]. Negative = unknown → zero vector."""
        unknown = clock_s < 0
        zero    = torch.zeros(self.d_model, device=clock_s.device, dtype=self.emb.weight.dtype)
        return self._blend(clock_s, self.clock_bounds, self.clock_ids, unknown, zero)

    def _color_emb(self, color_token: str, B: int) -> torch.Tensor:
        cid = self.vocab.token_to_id[color_token]
        return self.emb.weight[cid].unsqueeze(0).expand(B, -1)

    # -----------------------------------------------------------------------
    # Sequence builder
    # -----------------------------------------------------------------------

    def _build_sequence(
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
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        B      = meta_ids.shape[0]
        device = meta_ids.device
        dtype  = self.emb.weight.dtype

        w_col = self._color_emb("w_color", B)
        b_col = self._color_emb("b_color", B)
        w_elo_emb   = (w_col + self._elo_part(white_elo)).unsqueeze(1)
        b_elo_emb   = (b_col + self._elo_part(black_elo)).unsqueeze(1)
        w_clock_emb = (w_col + self._clock_part(white_clock_s)).unsqueeze(1)
        b_clock_emb = (b_col + self._clock_part(black_clock_s)).unsqueeze(1)

        meta_valid = meta_ids >= 0
        meta_emb   = torch.zeros(B, meta_ids.shape[1], self.d_model, device=device, dtype=dtype)
        if meta_valid.any():
            safe = meta_ids.clamp(min=0)
            meta_emb[meta_valid] = self.emb(safe)[meta_valid]

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

        board_pad = torch.cat([
            torch.zeros(B, 4, dtype=torch.bool, device=device),
            ~meta_valid,
            ~piece_valid,
        ], dim=1)

        pos0 = self.move_pos.weight[0].to(dtype)
        pos1 = self.move_pos.weight[1].to(dtype)
        pos2 = self.move_pos.weight[2].to(dtype)
        bos_emb  = (self.move_bos.to(dtype) + pos0).unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
        from_emb = (self.emb(move_ids[:, 0].clamp(min=0)) + pos1).unsqueeze(1)
        to_emb   = (self.emb(move_ids[:, 1].clamp(min=0)) + pos2).unsqueeze(1)
        move_seq = torch.cat([bos_emb, from_emb, to_emb], dim=1)

        x   = torch.cat([board, move_seq], dim=1)
        pad = torch.cat([board_pad, torch.zeros(B, 3, dtype=torch.bool, device=device)], dim=1)
        T   = x.shape[1]

        # MPS has a bug with -inf arithmetic in broadcast ops — build mask on CPU,
        # then move to device. On CUDA, build directly on device (avoids H2D copy).
        if device.type == "mps":
            mask = self._build_mask_cpu(B, T, board_len, pad, dtype).to(device)
        else:
            mask = self._build_mask_device(B, T, board_len, pad, dtype, device)

        return x, mask, board_len

    @staticmethod
    def _build_mask_cpu(
        B: int, T: int, board_len: int,
        pad: torch.Tensor, dtype: torch.dtype,
    ) -> torch.Tensor:
        mask = torch.zeros(B, 1, T, T, dtype=dtype)
        mask[:, :, :board_len, board_len:] = float("-inf")
        mask[:, :, board_len:, board_len:] = torch.triu(
            torch.full((3, 3), float("-inf")), diagonal=1
        )
        pad_f = torch.where(pad.cpu(),
                            torch.full((B, T), float("-inf"), dtype=dtype),
                            torch.zeros(B, T, dtype=dtype))
        return mask + pad_f.unsqueeze(1).unsqueeze(1)

    @staticmethod
    def _build_mask_device(
        B: int, T: int, board_len: int,
        pad: torch.Tensor, dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
        mask[:, :, :board_len, board_len:] = float("-inf")
        mask[:, :, board_len:, board_len:] = torch.triu(
            torch.full((3, 3), float("-inf"), device=device), diagonal=1
        )
        pad_f = torch.where(pad,
                            torch.full((B, T), float("-inf"), dtype=dtype, device=device),
                            torch.zeros(B, T, dtype=dtype, device=device))
        return mask + pad_f.unsqueeze(1).unsqueeze(1)

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
    # Inference
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
        """Autoregressively sample from-sq → to-sq → promo (3 forward passes)."""
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

    @torch.no_grad()
    def get_move_probs(
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
    ) -> dict:
        """Return from/to probability distributions for debug display.

        Uses greedy from-square for the to-distribution (single forward pass each).
        Returns CPU tensors.
        """
        device = meta_ids.device
        board  = (meta_ids, color_ids, piece_type_ids, file_ids, rank_ids,
                  white_elo, black_elo, white_clock_s, black_clock_s)
        dummy = torch.zeros(1, 3, dtype=torch.long, device=device)

        from_logits, _, _ = self.forward(*board, dummy)
        from_probs = F.softmax(from_logits[0], dim=-1).cpu()  # [64]

        best_from = int(from_probs.argmax())
        dummy2 = dummy.clone()
        dummy2[0, 0] = best_from + self.vocab.from_square_offset
        _, to_logits, _ = self.forward(*board, dummy2)
        to_probs = F.softmax(to_logits[0], dim=-1).cpu()  # [64]

        # Top-5 joint moves (from_prob[i] * to_prob_greedy[j])
        joint = (from_probs.unsqueeze(1) * to_probs.unsqueeze(0)).flatten()
        top5 = joint.topk(5)
        top_moves = [
            {"from": int(idx // 64), "to": int(idx % 64), "prob": float(joint[idx])}
            for idx in top5.indices
        ]

        return {
            "from_probs": from_probs.tolist(),
            "to_probs":   to_probs.tolist(),
            "top_moves":  top_moves,
        }


def _sample_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    return torch.multinomial(F.softmax(logits / temperature, dim=-1), num_samples=1).squeeze(1)
