"""
Quantized Layers for Memory Efficiency

Implements int8 quantization for embedding layers to reduce memory usage
while maintaining reasonable training quality.

Reduces memory from float32 (4 bytes) to int8 (1 byte) = 4x reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class QuantizedEmbedding(nn.Module):
    """
    Quantized Token Embedding Layer using int8 storage
    
    Stores embeddings in int8 (1 byte per weight) instead of float32 (4 bytes)
    Provides ~4x memory reduction while maintaining training quality.
    
    During forward pass: int8 weight → float32 computation
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        """
        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embedding vectors
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Create embedding in float32 first
        self.register_buffer(
            'weight',
            torch.randn(num_embeddings, embedding_dim, dtype=torch.float32)
        )
        
        # Initialize with proper scaling
        nn.init.normal_(self.weight, std=0.02)
        
        # Quantize to int8
        self._quantize_weights()
    
    def _quantize_weights(self):
        """Quantize float32 weights to int8 for storage"""
        # Find min/max for quantization range
        weight_min = self.weight.min().item()
        weight_max = self.weight.max().item()
        
        # Store scale and zero point for dequantization
        self.register_buffer('scale', torch.tensor([(weight_max - weight_min) / 255.0], dtype=torch.float32))
        self.register_buffer('zero_point', torch.tensor([int(-weight_min / self.scale.item())], dtype=torch.int32))
        
        # Quantize: float32 → int8
        quantized = ((self.weight / self.scale) + self.zero_point).clamp(0, 255).to(torch.uint8)
        
        # Replace weight with quantized version
        self.register_buffer('weight_quantized', quantized)
        
        # Delete the full precision weight to save memory
        del self.weight
    
    def _dequantize_weights(self) -> torch.Tensor:
        """Dequantize int8 weights back to float32 for computation"""
        return (self.weight_quantized.float() - self.zero_point.float()) * self.scale
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token indices [batch_size, seq_len]
            
        Returns:
            Embedded tokens [batch_size, seq_len, embedding_dim]
        """
        # Dequantize weights on-the-fly
        weight = self._dequantize_weights()
        
        # Standard embedding lookup
        return F.embedding(input_ids, weight)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer using int8 storage
    
    Reduces memory usage from float32 to int8.
    Useful for the output projection layer (to vocab_size)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Initialize weight in float32
        weight = torch.randn(out_features, in_features, dtype=torch.float32) * 0.02
        
        # Quantize
        weight_min = weight.min().item()
        weight_max = weight.max().item()
        scale = (weight_max - weight_min) / 255.0
        zero_point = int(-weight_min / scale)
        
        quantized = ((weight / scale) + zero_point).clamp(0, 255).to(torch.uint8)
        
        self.register_buffer('weight_quantized', quantized)
        self.register_buffer('scale', torch.tensor([scale], dtype=torch.float32))
        self.register_buffer('zero_point', torch.tensor([zero_point], dtype=torch.int32))
        
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.zeros(out_features)))
        else:
            self.register_parameter('bias', None)
    
    def _dequantize_weight(self) -> torch.Tensor:
        """Dequantize weight for computation"""
        return (self.weight_quantized.float() - self.zero_point.float()) * self.scale
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: [batch_size, ..., in_features]
            
        Returns:
            Output [batch_size, ..., out_features]
        """
        weight = self._dequantize_weight()
        return F.linear(input, weight, self.bias)
