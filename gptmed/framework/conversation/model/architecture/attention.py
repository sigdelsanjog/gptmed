"""
Multi-Head Causal Self-Attention (From Scratch)

Implements multi-head causal self-attention mechanism for transformer decoder.
No external dependencies on transformers or pre-built attention layers.

Key Features:
- Causal masking (prevents attending to future tokens)
- Multi-head mechanism for diverse feature learning
- Scaled dot-product attention
- Dropout for regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention Layer
    
    Allows tokens to attend to previous tokens (causal) with multiple attention heads.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask as buffer (won't be trained)
        self.register_buffer("causal_mask", None)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask: upper triangular matrix set to -inf
        Position i can only attend to positions 0...i (not future positions)
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'),
            diagonal=1
        )
        return mask
    
    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional mask for padding tokens
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = self.W_q(x)  # [batch_size, seq_len, d_model]
        K = self.W_k(x)  # [batch_size, seq_len, d_model]
        V = self.W_v(x)  # [batch_size, seq_len, d_model]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # [batch_size, n_heads, seq_len, d_head]
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        # [batch_size, n_heads, seq_len, seq_len]
        
        # Apply causal mask (prevent attending to future tokens)
        if self.causal_mask is None or self.causal_mask.shape[0] < seq_len:
            self.causal_mask = self._create_causal_mask(seq_len, x.device)
        
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores + causal_mask
        
        # Apply attention mask (for padding)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        # [batch_size, n_heads, seq_len, d_head]
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous()
        # [batch_size, seq_len, n_heads, d_head]
        context = context.view(batch_size, seq_len, d_model)
        # [batch_size, seq_len, d_model]
        
        # Output projection
        output = self.W_o(context)
        
        return output
