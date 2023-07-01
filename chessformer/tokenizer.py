import torch
import random

# write all possible tokens to a all_tokens text file
def create_all_tokens():
    all_tokens = ['<pad>', 'w', 'b', 'k', 'q', 'K', 'Q']
    for piece in ['p', 'n', 'r', 'b', 'q', 'k', 'P', 'N', 'R', 'B', 'Q', 'K']:
        for col in 'abcdefgh':
            for line in range(1, 9):
                if not (piece in ['p', 'P'] and line in [1, 8]):
                    all_tokens.append(f'{piece}{col}{line}')
    for col in 'abcdefgh':
        for line in range(1, 9):
            all_tokens.append(f'f_{col}{line}')
            all_tokens.append(f't_{col}{line}')
    for level in range(10):
        all_tokens.append(f'lvl_{level}')
    # add en passant
    for col in 'abcdefgh':
        all_tokens.append(f'ep_{col}')
    # add promotions
    for piece in ['n', 'r', 'b', 'q']:
        all_tokens.append(f'prom_{piece}')
    with open('all_tokens.txt', 'w') as f:
        for item in all_tokens:
            f.write(item + '\n')

# create_all_tokens()

class Tokenizer_old:
    def __init__(self, all_tokens_file):
        self.token_to_id = {}
        self.id_to_token = {}
        with open(all_tokens_file, 'r') as f:
            for i, line in enumerate(f):
                self.token_to_id[line.strip()] = i
                self.id_to_token[i] = line.strip()

    def encode(self, pretokens):
        return [self.token_to_id[token] for token in pretokens]

    def decode(self, tokens):
        return [self.id_to_token[token] for token in tokens]

# evolution of the tokenizer handling attention mask, padding, and batching

class Tokenizer:
    def __init__(self, all_tokens_file):
        self.token_to_id = {}
        self.id_to_token = {}
        with open(all_tokens_file, 'r') as f:
            for i, line in enumerate(f):
                self.token_to_id[line.strip()] = i
                self.id_to_token[i] = line.strip()

    def encode(self, pretokens):
        return [self.token_to_id[token] for token in pretokens]

    def decode(self, tokens):
        return [self.id_to_token[token] for token in tokens]

    def encode_one_batch(self, pretokens_batch):
        max_len = max([len(pretokens) for pretokens in pretokens_batch])
        tokens_batch = []
        for pretokens in pretokens_batch:
            tokens = self.encode(pretokens)
            tokens += [self.token_to_id['<pad>']] * (max_len - len(tokens))
            tokens_batch.append(tokens)
        return tokens_batch

    def decode_one_batch(self, tokens_batch):
        return [self.decode(tokens) for tokens in tokens_batch]
    
    def encode_in_batches(self, pretokens_list, batch_size, random_state=0):
        pretokens_list = pretokens_list.copy()
        random.Random(random_state).shuffle(pretokens_list)
        tokens_batches = []
        for i in range(0, len(pretokens_list), batch_size):
            tokens_batches.append(self.encode_one_batch(pretokens_list[i:i+batch_size]))
        return tokens_batches
    
    def decode_in_batches(self, tokens_batches):
        return [self.decode_one_batch(tokens_batch) for tokens_batch in tokens_batches]

    def attention_mask(self, tokens_batch):
        return [[1 if token != self.token_to_id['<pad>'] else 0 for token in tokens] for tokens in tokens_batch]

    def batch_to_tensor(self, tokens_batch):
        return torch.tensor(tokens_batch, dtype=torch.long)

    def batch_to_attention_mask(self, tokens_batch):
        return torch.tensor(self.attention_mask(tokens_batch), dtype=torch.long)

    def batch_to_device(self, tokens_batch, device):
        return self.batch_to_tensor(tokens_batch).to(device), self.batch_to_attention_mask(tokens_batch).to(device)