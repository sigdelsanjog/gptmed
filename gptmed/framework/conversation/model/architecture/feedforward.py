"""
Feed-Forward Network (From Scratch)

Implements the feed-forward network component of transformer decoder blocks.
Simple 2-layer network with intermediate expansion.
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Consists of two linear layers with an activation function (ReLU) in between.
    Applied to each position separately and identically.
    """
    
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            d_ff: Hidden layer dimension (default: 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # First linear layer: expand dimensions
        self.fc1 = nn.Linear(d_model, d_ff)
        
        # ReLU activation
        self.activation = nn.ReLU()
        
        # Dropout after activation
        self.dropout = nn.Dropout(dropout)
        
        # Second linear layer: project back to d_model
        self.fc2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Expand
        x = self.fc1(x)  # [batch_size, seq_len, d_ff]
        
        # Activate
        x = self.activation(x)
        
        # Dropout
        x = self.dropout(x)
        
        # Project back
        x = self.fc2(x)  # [batch_size, seq_len, d_model]
        
        return x
