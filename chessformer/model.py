import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention implementation optimized for chess token prediction.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention: [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output

class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerLayer(nn.Module):
    """
    A single transformer encoder layer with pre-layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Pre-layer normalization
        attn_output = self.attn(self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout2(ff_output)
        
        return x

class ChessModel(nn.Module):
    """
    Transformer-based model for predicting chess moves without positional embeddings.
    Position information is provided through the squares_tokens which are added to the
    piece embeddings.

    Args:
        num_tokens (int): Number of unique tokens.
        num_classes (int): Number of output classes.
        d_model (int): Dimensionality of the model.
        nhead (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dropout (float): Dropout rate.
        d_ff (int): Dimension of the feed-forward network.
    """
    def __init__(
            self, 
            num_tokens=257, 
            num_classes=132, 
            d_model=256, 
            nhead=8, 
            num_layers=6, 
            dropout=0.1
            ):
        super().__init__()
        d_ff = d_model * 4
        self.num_tokens = num_tokens
        self.num_classes = num_classes
        self.first_class_token = 125
        self.d_model = d_model
        
        # Token embeddings
        self.embedding = nn.Embedding(num_tokens + 1, d_model, padding_idx=-1)
        
        # Special token type embedding to give different weight to meta tokens
        self.meta_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        self.token_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )
        
    def create_attention_mask(self, pad_mask):
        """Create attention mask from padding mask"""
        seq_len = pad_mask.size(1)
        # Create a mask of shape [batch_size, seq_len, seq_len]
        attn_mask = pad_mask.unsqueeze(1).expand(-1, seq_len, -1)
        return attn_mask
        
    def forward(self, sequence_tokens, squares_tokens):
        """
        Forward pass of the model.

        Args:
            sequence_tokens (Tensor): Tensor containing the sequence tokens. [batch_size, seq_len]
            squares_tokens (Tensor): Tensor containing the squares tokens. [batch_size, seq_len]

        Returns:
            Tensor: The output logits for the next move prediction.
        """
        batch_size, seq_len = sequence_tokens.shape
        
        # Embed tokens
        seq_embeddings = self.embedding(sequence_tokens)  # [batch_size, seq_len, d_model]
        
        # Add square embeddings (when available - when not -1)
        square_mask = squares_tokens != -1
        x = seq_embeddings.clone()
        if square_mask.any():
            square_embeddings = torch.zeros_like(seq_embeddings)
            square_embeddings[square_mask] = self.embedding(squares_tokens[square_mask])
            x = x + square_embeddings
        
        # Create token type embeddings (differentiate between meta tokens and piece tokens)
        # Meta tokens are identified by -1 in squares_tokens
        meta_mask = squares_tokens == -1
        token_type_embeddings = torch.zeros_like(x)
        token_type_embeddings[meta_mask] = self.meta_embedding
        token_type_embeddings[~meta_mask] = self.token_embedding
        x = x + token_type_embeddings
        
        # Create padding mask (1 means padding)
        padding_mask = sequence_tokens == -1  # [batch_size, seq_len]
        attention_mask = self.create_attention_mask(padding_mask)  # [batch_size, seq_len, seq_len]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Apply final normalization
        x = self.norm(x)
        
        # Use the first token (player token) for prediction
        first_token = x[:, 0]
        
        # Project to class predictions
        logits = self.output_projection(first_token)
        
        return logits
    
    def predict_move(self, sequence_tokens, squares_tokens):
        """
        Predict the next move given a chess position.
        
        Args:
            sequence_tokens (Tensor): Tensor containing the sequence tokens.
            squares_tokens (Tensor): Tensor containing the squares tokens.
            
        Returns:
            int: The predicted move token.
        """
        with torch.no_grad():
            logits = self(sequence_tokens, squares_tokens)
            prediction = logits.argmax(dim=-1) + self.first_class_token
            return prediction