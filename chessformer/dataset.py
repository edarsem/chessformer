import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class ChessDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta_tokens, pieces_tokens, squares_tokens, next_move = self.data[idx]
        seq_length = len(meta_tokens) + len(pieces_tokens) # square are added as positional embeddings
        return {
            'meta_tokens': torch.tensor(meta_tokens, dtype=torch.long),
            'pieces_tokens': torch.tensor(pieces_tokens, dtype=torch.long),
            'squares_tokens': torch.tensor(squares_tokens, dtype=torch.long),
            'next_move': torch.tensor(next_move, dtype=torch.long),
            'seq_length': seq_length
        }

def collate_fn(batch):
    # Extract all items and find max sequence length
    meta_tokens = [item['meta_tokens'] for item in batch]
    pieces_tokens = [item['pieces_tokens'] for item in batch]
    squares_tokens = [item['squares_tokens'] for item in batch]
    next_moves = [item['next_move'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]

    max_seq_length = max(seq_lengths)

    # Pad sequences to max length found in the batch
    meta_tokens_padded = pad_sequence(meta_tokens, batch_first=True, padding_value=0)
    pieces_tokens_padded = pad_sequence(pieces_tokens, batch_first=True, padding_value=0)
    squares_tokens_padded = pad_sequence(squares_tokens, batch_first=True, padding_value=0)

    next_moves = torch.stack(next_moves)

    return {
        'meta_tokens': meta_tokens_padded,
        'pieces_tokens': pieces_tokens_padded,
        'squares_tokens': squares_tokens_padded,
        'next_move': next_moves
    }
