import torch
import torch.nn as nn
import torch.nn.functional as F

class ChessModel(nn.Module):
    """
    A transformer based model for chess."""
    def __init__(self, num_tokens, d_model=64, nhead=4, num_layers=4, seq_length=42, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(num_tokens, d_model)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.next_move = nn.Linear(d_model, num_tokens)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_decoder(x)
        x = self.next_move(x[0])
        return x

