import torch
import numpy as np

import model
import tokenizer
import preprocessing
import download_games

model_args = {
    'block_size': 42,
    'vocab_size': 1024,
    'n_layer': 4,
    'n_head': 3,
    'n_embd': 48,
    'dropout': 0.0,
    'bias': True
}

optimizer_args = {
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-08,
    'weight_decay': 0.0,
}

user_name = 'FairChess_on_YouTube'
batch_size = 8
block_size = model_args['block_size']
all_tokens_path = 'all_tokens.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = model.GPTConfig(**model_args)
chessformer = model.GPT(config)
tokenizer = tokenizer.Tokenizer(all_tokens_path)
optimizer = chessformer.configure_optimizers(
    weight_decay=optimizer_args['weight_decay'], 
    learning_rate=optimizer_args['lr'],
    betas=optimizer_args['betas'],
    device_type=device
    )

games = download_games.download_games(user_name, max_games=3, days_back=10)
fen_list = preprocessing.pgn_to_fen(games)

pretokens_list = preprocessing.fen_list_to_pretokens_list(fen_list, training_mode=True)

for fen in fen_list:
    pretokens = preprocessing.fen_to_pretokens(fen[0], fen[1])
    pretokens_list.append(pretokens)

tokens_batches = tokenizer.encode_in_batches(pretokens_list, batch_size)
print(len(tokens_batches))

# Training loop
chessformer.train(mode=True)
all_losses = []
for tokens_batch in tokens_batches:
    input_batch, output_batch = tokenizer.batch_to_tensor(tokens_batch, device, separate_last_token=True)
    logits, loss = chessformer(input_batch, output_batch)
    all_losses.append(loss.item())
    loss.backward()
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

print(np.array(all_losses))

print(np.mean(all_losses))