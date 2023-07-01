import json

# write all possible tokens to a all_tokens text file
def all_tokens():
    all_tokens = ['w', 'b', 'k', 'q', 'K', 'Q']
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

all_tokens()
# write a tokenizer object from the all_tokens text file using the number of the line as the token id. It can encode pretokens and decode to pretokens

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