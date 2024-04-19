import torch.nn as nn

class ChessModel(nn.Module):
    def __init__(self, num_classes=132, d_model=64, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.d_model = d_model
        
        self.embedding = nn.Embedding(num_classes, d_model, padding_idx=-1)
        
        self.decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(self.decoder_layer, num_layers=num_layers)
        
        self.next_move = nn.Linear(d_model, num_classes)
    
    def forward(self, sequence_tokens, square_tokens, meta_lengths):
        # Embeddings for combined and squares
        combined_embeddings = self.embedding(sequence_tokens)
        square_embeddings = self.embedding(square_tokens)

        # Adjust square embeddings by meta_lengths
        for i, length in enumerate(meta_lengths):
            if length > 0:
                square_embeddings[i, :length] = square_embeddings[i, :length] + combined_embeddings[i, :length]
        
        # Compute mask for padding
        pad_mask = (sequence_tokens == -1)
        
        x = self.transformer_decoder(combined_embeddings, src_key_padding_mask=pad_mask)
        first_token_output = self.next_move(x[:, 0, :]) # play_w or play_b token used for next move prediction

        return first_token_output