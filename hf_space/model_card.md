---
license: apache-2.0
tags:
  - chess
  - transformer
  - pytorch
  - imitation-learning
language:
  - en
pipeline_tag: text-generation
library_name: pytorch
---

# Chessformer v0

A ~50M parameter transformer trained to **imitate human chess play across the full Elo spectrum**, conditioned on player Elo, remaining clock time, and time increment.

> **44.4% top-1 joint move accuracy** on a held-out, Elo-balanced validation set (predicting the exact human move — from-square and to-square).

## What it does

- **Strength is a dial.** Pass `white_elo=1200` and it plays like a 1200. Pass `white_elo=2800` and it plays like a 2800.
- **No rules, no search.** Legal-move understanding is emergent from training data only.
- **Clock-aware.** Conditioning on remaining time lets the model represent time-pressure play.
- **One forward pass per move.** From-square and to-square are predicted autoregressively in a single masked forward, not via tree search.

## Usage

```python
import chess, torch
from chessformer.model import ChessformerModel, unwrap_state_dict
from chessformer.tokenizer import build_vocab
from chessformer.inference import pick_move

vocab = build_vocab()
ckpt  = torch.load("chessformer_v0.pt", map_location="cpu")
model = ChessformerModel(vocab=vocab, **ckpt["cfg"]["model"])
model.load_state_dict(unwrap_state_dict(ckpt["model"]), strict=True)
model.eval()

board = chess.Board()
move  = pick_move(model, vocab, torch.device("cpu"), board,
                  white_elo=2500, black_elo=2500,
                  white_clock_s=120.0, black_clock_s=120.0,
                  increment_s=5.0)
print(move)  # e.g. e2e4
```

## Architecture

| Param | Value |
| --- | --- |
| Parameters | ~50M |
| d_model | 512 |
| n_heads | 8 |
| n_layers | 12 |
| FFN | SwiGLU, 4× expansion |
| Norm | RMSNorm (pre-norm) |
| Attention | Flash Attention (SDPA) |
| Max sequence length | 41 tokens |

Board pieces use **additive embeddings**: `emb[color] + emb[piece_type] + emb[file] + emb[rank]`. No positional encoding needed. Elo, clock, and increment use soft bracket interpolation.

## Training

- **Data:** Lichess standard rated games, January 2018 (~100k games, ~6.6M positions)
- **Val/Test:** December 2017 (different month — no game-level leakage), Elo-balanced across 1000–2900
- **Steps:** 20,000
- **Batch size:** 1024 (2× T4 GPU, DDP)
- **LR:** 3e-4, cosine decay with 1000-step warmup
- **Optimizer:** AdamW, weight decay 0.01

## Metrics (val_games, Elo-balanced)

| Metric | Value |
| --- | --- |
| Top-1 joint move accuracy | **44.4%** |
| Validation loss | **1.785** |

## Limitations

- Trained on one month of Lichess — tactical blindspots exist, especially in quiet positions.
- Puzzle-solving is emergent and not explicitly optimized.
- Clock conditioning is only as good as the training data's clock annotations.

## License

Apache 2.0. Training data from [database.lichess.org](https://database.lichess.org/) (CC0).

## Citation

```
@misc{chessformer2024,
  author = {Albert-Roulhac, Edouard},
  title  = {Chessformer: Elo-conditioned human chess imitation via transformer},
  year   = {2024},
  url    = {https://github.com/edarsem/chessformer},
}
```
