import torch
import numpy as np

import model
import tokenizer
import preprocessing
import download_games

model_args = {
    'block_size': 4,
    'vocab_size': 1024,
    'n_layer': 4,
    'n_head': 3,
    'n_embd': 48,
    'dropout': 0.0,
    'bias': True
}

config = model.GPTConfig(**model_args)
chessformer = model.GPT(config)

tokenizer = tokenizer.Tokenizer('all_tokens.txt')

user_name = 'FairChess_on_YouTube'
batch_size = 8
block_size = model_args['block_size']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

games = download_games.download_games(user_name, max_games=20, days_back=10)

fen_list = preprocessing.pgn_to_fen(games)

pretokens_list = []
for fen in fen_list:
    pretokens = preprocessing.fen_to_pretokens(fen[0], fen[1])
    pretokens_list.append(pretokens)

tokens_batches = tokenizer.encode_in_batches(pretokens_list, batch_size)

# for pretokens in fen_list:
#     print(pretokens)

# print(np.array(tokens_batches))
# for batch in tokens_batches:
#     print(np.array(batch))
print(np.array(tokens_batches[0]))

def get_batch(data, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.Tensor((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.Tensor((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# print(get_batch(batch_size=batch_size))