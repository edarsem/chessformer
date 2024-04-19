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

    def forward(self, sequence_tokens, squares_tokens):
        # Embeddings
        sequence_embeddings = self.embedding(sequence_tokens)
        squares_embeddings = self.embedding(squares_tokens)
        
        # Combine sequence and squares embeddings
        sequence_embeddings = sequence_embeddings + squares_embeddings


        # Compute mask for padding
        pad_mask = (sequence_tokens == -1)
        
        x = self.transformer_decoder(sequence_embeddings, src_key_padding_mask=pad_mask)
        first_token_output = self.next_move(x[:, 0, :]) # play_w or play_b token used for next move prediction

        return first_token_output