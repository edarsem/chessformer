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
        sequence_tokens = torch.tensor(meta_tokens + pieces_tokens, dtype=torch.long)
        square_tokens = torch.tensor(squares_tokens, dtype=torch.long)
        return {
            'sequence_tokens': sequence_tokens,
            'square_tokens': square_tokens,
            'next_move': torch.tensor(next_move, dtype=torch.long),
            'meta_length': len(meta_tokens)
        }

def collate_fn(batch):
    sequence_tokens = pad_sequence([item['sequence_tokens'] for item in batch], batch_first=True, padding_value=-1)
    square_tokens = pad_sequence([item['square_tokens'] for item in batch], batch_first=True, padding_value=-1)
    next_moves = torch.stack([item['next_move'] for item in batch])
    meta_lengths = torch.tensor([item['meta_length'] for item in batch], dtype=torch.long)

    return {
        'sequence_tokens': sequence_tokens,
        'square_tokens': square_tokens,
        'next_move': next_moves,
        'meta_lengths': meta_lengths
    }
