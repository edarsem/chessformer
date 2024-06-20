import torch.nn as nn

class ChessModel(nn.Module):
    """
    Transformer-based model for predicting chess moves.

    Args:
        num_tokens (int): Number of unique tokens.
        num_classes (int): Number of output classes.
        d_model (int): Dimensionality of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dropout (float): Dropout rate.
    """
    def __init__(self, num_tokens=257, num_classes=132, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.first_class_token = 125
        self.d_model = d_model
        
        self.embedding = nn.Embedding(1 + num_tokens, d_model, padding_idx=-1)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
        self.next_move = nn.Linear(d_model, num_classes)

    def forward(self, sequence_tokens, squares_tokens):
        """
        Forward pass of the model.

        Args:
            sequence_tokens (Tensor): Tensor containing the sequence tokens.
            squares_tokens (Tensor): Tensor containing the squares tokens.

        Returns:
            Tensor: The output logits for the next move prediction.
        """
        # Combine sequence and squares embeddings
        x = self.embedding(sequence_tokens) + self.embedding(squares_tokens)

        # Compute mask for padding
        pad_mask = (sequence_tokens == -1)
        
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        first_token_output = self.next_move(x[:, 0, :]) # play_w or play_b token used for next move prediction

        return first_token_output