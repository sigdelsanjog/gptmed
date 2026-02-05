"""
Conversation Language Model (From Scratch)

Full GPT-style causal language model assembled from components:
- Token and Positional Embeddings
- Stack of Decoder Blocks
- Output projection to vocabulary
"""

import torch
import torch.nn as nn
from .embeddings import TokenPositionalEmbedding
from .decoder_block import TransformerDecoderBlock


class ConversationLanguageModel(nn.Module):
    """
    Conversation Language Model for next-token prediction
    
    Architecture:
    1. Token + Positional Embeddings
    2. Stack of N Decoder Blocks (with multi-head attention + FFN)
    3. Final Layer Normalization
    4. Output Projection to vocabulary size
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding and hidden size)
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layer
        self.embedding = TokenPositionalEmbedding(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
        # Stack of decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff or 4 * d_model,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for the model
        
        Args:
            token_ids: Input token indices [batch_size, seq_len]
            attention_mask: Optional mask for padding tokens [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Get embeddings
        x = self.embedding(token_ids)  # [batch_size, seq_len, d_model]
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, attention_mask)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(
        self,
        initial_tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively
        
        Args:
            initial_tokens: Starting tokens [1, seq_len]
            max_length: Maximum generation length
            temperature: Softmax temperature (higher = more random)
            top_k: If set, only sample from top-k tokens
            device: Device to use
            
        Returns:
            Generated token sequence
        """
        if device is None:
            device = next(self.parameters()).device
        
        initial_tokens = initial_tokens.to(device)
        generated = initial_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Keep only last max_seq_len tokens to avoid out of memory
                input_tokens = generated[:, -self.max_seq_len:]
                
                # Get logits
                logits = self.forward(input_tokens)  # [1, seq_len, vocab_size]
                
                # Get logits of last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k sampling if specified
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        return generated
