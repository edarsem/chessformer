import torch
import torch.nn as nn

class ChessModel(nn.Module):
    """
    A transformer based model for chess."""
    def __init__(self, num_classes=132, d_model=64, nhead=4, num_layers=4, seq_length=42, dropout=0.1):
        super().__init__()
        self.first_class_token = 125
        self.num_classes = num_classes
        self.d_model = d_model
        self.seq_length = seq_length
        
        self.embedding = nn.Embedding(num_classes, d_model)
        
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)

        self.next_move = nn.Linear(d_model, num_classes)
    
    def forward(self, meta_tokens, pieces_tokens, squares_tokens):
        # Embeddings for meta, pieces, and squares
        meta_embeddings = self.embedding(meta_tokens)
        piece_embeddings = self.embedding(pieces_tokens)
        square_embeddings = self.embedding(squares_tokens)

        # Combine piece and square embeddings
        position_embeddings = piece_embeddings + square_embeddings

        # Concatenate all embeddings
        x = torch.cat((meta_embeddings, position_embeddings), dim=1)
        x = self.transformer_decoder(x)

        # Pass only the first token's embedding to the linear layer
        first_token_output = self.next_move(x[:, 0, :])

        return first_token_output

