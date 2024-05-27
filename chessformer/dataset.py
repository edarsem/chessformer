import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json

class ChessDataset(Dataset):
    """
    Chess dataset for loading and processing chess game data.

    Args:
        file_path (str): Path to the file containing tokenized data.
    """
    def __init__(self, file_path):
        with open(file_path, 'r') as file:
            self.data = [json.loads(line) for line in file]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assume each line in the file is a JSON with the required tokens
        meta_tokens, pieces_tokens, squares_tokens, next_move = self.data[idx]
        sequence_tokens = meta_tokens + pieces_tokens  # Concatenate meta and pieces tokens
        squares_tokens = [-1] * len(meta_tokens) + squares_tokens  # Pad meta positional tokens
        return {
            'sequence_tokens': torch.tensor(sequence_tokens, dtype=torch.long),
            'squares_tokens': torch.tensor(squares_tokens, dtype=torch.long),
            'next_move': torch.tensor(next_move, dtype=torch.long),
        }

def collate_fn(batch):
    """
    Collate function for the DataLoader to handle variable-length sequences.

    Args:
        batch (list): A batch of data.

    Returns:
        dict: A dictionary containing padded sequences and next moves.
    """
    sequence_tokens = [item['sequence_tokens'] for item in batch]
    squares_tokens = [item['squares_tokens'] for item in batch]
    next_moves = [item['next_move'] for item in batch]

    sequence_padded = pad_sequence(sequence_tokens, batch_first=True, padding_value=-1)
    squares_padded = pad_sequence(squares_tokens, batch_first=True, padding_value=-1)
    next_moves = torch.stack(next_moves)

    return {
        'sequence_tokens': sequence_padded,
        'squares_tokens': squares_padded,
        'next_move': next_moves
    }
