# Chessformer

A transformer that learns the human way to play chess by **imitating human Lichess games**.

[Demo](https://huggingface.co/spaces/edarsem/chessformer-demo)

Instead of training one "best move" engine, Chessformer is conditioned on the real game position: player's **Elo, remaining clock time, and time increment**. Ask it to play like a 1200, and it plays like a 1200. Ask it to play like a 2800, and it plays like a 2800. The same model spans the whole spectrum of human strength.

> **Status:** first public model. ~50M parameters, trained on ~6.6M positions from Lichess. Reaches **44.4% top-1 move accuracy** on a held-out, Elo-balanced validation set (predicting the *exact* human move, from-square and to-square).

---

## Why this is interesting

- **Human-like, not optimal.** Engines like Stockfish tell you the best move and they don't understand intuition or difficulty besides depth and compute budget. Chessformer tells you the intuitive moves *at a given level*. Useful for coaching ("what should a 1500 spot here?"), style imitation, and human-aware analysis.
- **Strength is a dial.** Elo is a conditioning input, not a separate model.
- **Clock-aware.** It sees how much time is left and the increment, so it can model time-pressure mistakes.
- **No rules, no search.** It only ever sees board positions and the moves humans played. Legal-move understanding is emergent. It predicts moves in one forward pass, no tree search.
- **Puzzles are a test, not a training signal.** Tactical puzzle-solving is measured as an emergent capability — the model is never trained on puzzles.

## Play against it

The repo ships a web UI (board, engine arrows, Elo / clock / increment sliders, puzzle mode):

```bash
pip install -r requirements.txt
python scripts/serve.py checkpoint=checkpoints/chessformer_v0.pt
```

This opens `http://localhost:5174`. Set both sides' Elo, drag pieces, and watch the engine's top moves as arrows. Turn on "engine" to see the model's move distribution for any position.

---

## How it works

### Board as a set of pieces

Each piece on the board is one token whose embedding is the **sum of four independent lookups**:

```text
piece_embedding = emb[color] + emb[piece_type] + emb[file] + emb[rank]
```

This factorization lets the model generalize across files, ranks, and piece types independently — a knight on f3 and a knight on c3 share structure automatically, and no explicit positional encoding is needed.

### Conditioning by soft-blended brackets

Elo, clock, and increment are continuous, so they're encoded by **interpolating between the two nearest bracket embeddings**:

```text
emb = α · emb[lower_bracket] + (1 − α) · emb[upper_bracket]
```

At inference you can dial in any value (e.g. Elo 1873, 47 s on the clock) and the model blends smoothly. During training, conditioning is randomly dropped so the model also works when Elo/clock are unknown.

### Sequence layout

```text
[meta] [w_elo] [b_elo] [w_clock] [b_clock] [increment] [piece_0 … piece_k] [BOS] [from] [to]
└── 1 ──┘└──────────── 5 conditioning ──────────────┘└── up to 32 ──────┘└── move suffix ──┘
```

The **meta slot** is a single position whose embedding is the *sum* of the side-to-move, castling-rights, and en-passant tokens (additive, not sequential — castling rights don't deserve four sequence slots).

### Autoregressive move head

Moves are predicted in order: **from-square → to-square → promotion**. A single shared head predicts a square over 64 outputs (used for both from and to); a small separate head handles promotion. At training time this is one forward pass with a causal mask over the move suffix; the board prefix is fully bidirectional and cannot peek at the move tokens.

### Architecture

An autoregressive transformer block:

- **Pre-norm RMSNorm**
- **SwiGLU** feed-forward (4× expansion)
- **Flash Attention** via `scaled_dot_product_attention`
- **No biases** anywhere

Default (`medium`) config: `d_model=512`, `n_heads=8`, `n_layers=12`, `ffn_mult=4` → **~50M parameters**. Max sequence length is just **41** tokens — the position encoding keeps sequences short, which makes training fast.

---

## Results

The released checkpoint (`chessformer_v0.pt`, 20k steps) on the Elo-balanced `val_games` split:

| Metric | Value |
| --- | --- |
| Top-1 joint move accuracy (exact from+to) | **44.4%** |
| Validation loss (from + to cross-entropy) | **1.785** |

Validation games are balanced across 20 Elo buckets (1000–2900) and drawn from a *different month* than the training data, so there's no game-level leakage. Run the full breakdown (per-Elo-bucket accuracy, legal-move rate, puzzle solve rate) with:

```bash
python scripts/eval_offline.py checkpoint=checkpoints/chessformer_v0.pt
```

---

## Quickstart

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Play / analyze (with a trained checkpoint)

```bash
python scripts/serve.py checkpoint=checkpoints/chessformer_v0.pt   # web UI
python scripts/play.py   checkpoint=checkpoints/chessformer_v0.pt   # terminal
```

### 3. Train from scratch

```bash
# Preprocess PGN + puzzle data → parquet
python scripts/preprocess.py
python scripts/make_splits.py

# Smoke test (minutes, tiny subset)
python scripts/train.py model=small train.train_file=train_games_tiny_file train.max_steps=50

# Full run (single GPU)
python scripts/train.py model=medium train.train_file=train_games_small_file

# Multi-GPU
torchrun --nproc_per_node=2 scripts/train.py model=medium
```

Config is [Hydra](https://hydra.cc/); override anything from the CLI (`train.lr=1e-4`, `model.n_layers=16`, …). A Kaggle 2×T4 training notebook is in [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb).

---

## Data & splits

Train/val/test boundaries are set by **source file**, so no position can leak across splits:

| Split | Source | Use |
| --- | --- | --- |
| Train | Lichess `2018-01` | `train_games_small` ≈ 100k games / ~6.6M positions |
| Val / Test | Lichess `2017-12` | ~2k Elo-balanced games each (1000–2900) |
| Puzzles | Lichess puzzle CSV | val/test only — **never trained on** |

Lichess monthly dumps are at [database.lichess.org](https://database.lichess.org/).

## Repo structure

```text
chessformer/
├── conf/            # Hydra configs (model / data / train)
├── chessformer/     # importable library
│   ├── tokenizer.py # vocab + FEN → tokens
│   ├── model.py     # transformer
│   ├── dataset.py   # parquet → batches
│   ├── trainer.py   # training loop
│   ├── inference.py # board → move / move distribution
│   └── eval.py      # metrics
├── scripts/         # preprocess, make_splits, train, eval_offline, serve, play
├── ui/              # web UI
└── notebooks/
```

## Roadmap

The v0 model is a proof of concept. It trained well on 100k games. Planned next steps:

**Bigger model, bigger data.** The current model saw ~6.6M positions from a single month of Lichess. We plan to train on multiple years Lichess history (billions of positions) and add a dedicated GM-level database to sharpen the top-Elo end to improve the overall model's understanding while keeping human games only. Larger model (100M–300M params) with a longer training run on proper GPU hardware.

**Style impersonation.** Add a "player ID" token to the conditioning, so the model can learn to imitate specific players' styles. This is a natural extension of Elo conditioning, and the model can learn a person's favorite openings and aggressivity etc. I plan on training from scratch with conditioning on a few players, and then probably only do prompt-tuning for new players to avoid retraining the whole model every time or biasing the model.

**Position encoder as a shared backbone.** Chessformer is designed so the board-encoding layers — the set-of-pieces representation with additive piece embeddings — form a reusable *position encoder*, analogous to a vision encoder in a VLM. This encoder can be plugged into other tasks without retraining from scratch:

- **Coach / multimodal LLM.** A language model that can *see* the board by attending to the position encoder's output, the same way a VLM attends to image patch embeddings. "Explain the plan for White in this position", "at 1500 Elo, is that tactic findable?" becomes a natural-language generation task conditioned on the encoded board state.

- **Guess the Elo / Cheat detection.** Human-move likelihood from a model like this is a principled signal for detection: top moves that ranks in the top 1% of the model's distribution for a 1600-rated player and are played by a 1600 are suspicious in a way pure engine-centipawn analysis misses. The position encoder can feed into a dedicated anomaly-detection head.

## Modeling

As of now, the model has no notion of game history and objective.

- **History.** means that it can't learn about threefold repetition and 50-move rule, but I think that is not too important. It does have casting rights information. It also means that it doesn't know what the last move played was or on which move time was spent, which is quite important for a chess player in real life. For now, I don't have a satisfying way to represent history and to train the model with that. It may come in a subsequent model when starting to train actual text-based LLM with this model as a position encoder.

- **Objective.** The model is trained to predict the next move, but it doesn't have a built-in notion of winning or losing nor an objective evaluation head. It can learn to imitate human moves, but it is not built to learn to win. I might train an evaluation head with the rest of the model frozen in the future. It may also be possible to fine-tune the model with AlphaZero (reinforcement learning from self-play with MCTS) but we need to make sure that this doesn't break the model's human intuition.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Training data (Lichess monthly dumps) is released under [CC0](https://database.lichess.org/#standard_games) — no restrictions on use.
