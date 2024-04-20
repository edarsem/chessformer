import torch

# Path configuration
DATA_DIR = 'data/'
RAW_PGN_DIR = DATA_DIR + 'raw_pgn/'
RAW_PUZZLES_DIR = DATA_DIR + 'raw_puzzles/'
PROCESSED_FEN_DIR = DATA_DIR + 'fen_processed/'
TOKEN_TO_ID_PATH = 'tmp/token_to_id.json'
XFEN_OUTPUT_FILE = 'tmp/output_xfen.csv'
TOKENIZED_XFEN_FILE = 'tmp/tokenized_xfen.txt'

# Model configuration
MODEL_PARAMS = {
    'num_tokens': 257,
    'num_classes': 132,
    'd_model': 64,
    'nhead': 4,
    'num_layers': 4,
    'dropout': 0.1,
}

# Training configuration
TRAINING_PARAMS = {
    'batch_size': 1024,
    'max_steps': 10000,
    'learning_rate': 1e-3
}

# Hardware configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Logging configuration
LOGGING_LEVEL = 'INFO'
