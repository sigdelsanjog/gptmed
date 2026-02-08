"""
Embeddings (From Scratch)

Implements token embeddings and positional embeddings for the transformer.
Combines both into a single embedding layer.
"""

import torch
import torch.nn as nn
import math
from .quantized_layers import QuantizedEmbedding


class PositionalEncoding(nn.Module):
    """
    Positional Encoding using sine and cosine functions
    
    Allows the model to learn the position of tokens in the sequence.
    Uses the formula from the original Transformer paper.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        
        # Position indices [0, 1, 2, ..., max_seq_len-1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a trainable parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.shape[1]
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TokenPositionalEmbedding(nn.Module):
    """
    Combined Token and Positional Embedding
    
    Converts token IDs to embeddings and adds positional information.
    """
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension (embedding size)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        # Token embedding layer (quantized to int8 for memory efficiency)
        self.token_embedding = QuantizedEmbedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.d_model = d_model
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: Tensor of token indices [batch_size, seq_len]
            
        Returns:
            Embedded tensor with positional encoding [batch_size, seq_len, d_model]
        """
        # Embed tokens
        embeddings = self.token_embedding(token_ids)  # [batch_size, seq_len, d_model]
        
        # Scale embeddings
        embeddings = embeddings * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        return embeddings
