"""
Inference Utilities for Conversation Model

Model loading, generation, and inference helpers.
"""

import torch
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json

from ..model.architecture import ConversationLanguageModel
from ..model.configs.model_config import ConversationModelConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Utility for loading and managing model checkpoints"""
    
    @staticmethod
    def load_model(
        checkpoint_path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> Tuple[ConversationLanguageModel, ConversationModelConfig]:
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on
            
        Returns:
            Tuple of (model, config)
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load config
        config = ConversationModelConfig(**checkpoint['config'])
        
        # Create and load model
        model = ConversationLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        ).to(device)
        
        # Filter out causal_mask keys which are buffers, not state
        model_state = checkpoint['model_state']
        keys_to_remove = [k for k in model_state.keys() if 'causal_mask' in k]
        for key in keys_to_remove:
            del model_state[key]
        
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
        
        return model, config
    
    @staticmethod
    def find_best_checkpoint(checkpoint_dir: str) -> str:
        """
        Find best checkpoint (best_*.pt) in directory
        
        Args:
            checkpoint_dir: Directory containing checkpoints
            
        Returns:
            Path to best checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Look for best checkpoint
        best_checkpoints = list(checkpoint_dir.glob('checkpoint_best*.pt'))
        
        if best_checkpoints:
            return str(best_checkpoints[0])
        
        # Fallback to latest checkpoint
        all_checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
        
        if all_checkpoints:
            return str(sorted(all_checkpoints)[-1])
        
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")


class ConversationGenerator:
    """Generate text from conversation model"""
    
    def __init__(
        self,
        model: ConversationLanguageModel,
        config: ConversationModelConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize generator
        
        Args:
            model: Language model
            config: Model configuration
            device: Device to use for generation
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
    
    def generate(
        self,
        prompt_tokens: List[int],
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 10,
    ) -> List[int]:
        """
        Generate tokens from prompt
        
        Args:
            prompt_tokens: List of token IDs to start generation from
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Use top-k sampling
            
        Returns:
            Generated token sequence including prompt
        """
        # Clamp tokens to vocab_size to prevent out-of-bounds errors
        vocab_size = self.config.vocab_size
        prompt_tokens = [min(int(t), vocab_size - 1) for t in prompt_tokens]
        
        prompt_tokens = torch.tensor(
            prompt_tokens,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)  # Add batch dimension
        
        generated = self.model.generate(
            prompt_tokens,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
        
        return generated.squeeze(0).cpu().tolist()
    
    def beam_search(
        self,
        prompt_tokens: List[int],
        max_length: int = 50,
        beam_width: int = 3,
    ) -> List[Tuple[List[int], float]]:
        """
        Generate using beam search
        
        Args:
            prompt_tokens: Initial token sequence
            max_length: Maximum sequence length
            beam_width: Number of beams to maintain
            
        Returns:
            List of (token_sequence, score) tuples
        """
        # Simple beam search implementation
        with torch.no_grad():
            sequences = [(prompt_tokens, 0.0)]
            
            for _ in range(max_length):
                # Generate next token for each sequence
                new_sequences = []
                
                for seq, score in sequences:
                    if len(seq) >= self.config.max_seq_len:
                        new_sequences.append((seq, score))
                        continue
                    
                    # Forward pass
                    input_ids = torch.tensor(
                        [seq],
                        dtype=torch.long,
                        device=self.device
                    )
                    
                    with torch.no_grad():
                        logits = self.model(input_ids)
                    
                    # Get last token logits
                    last_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Get top-k candidates
                    top_logits, top_indices = torch.topk(last_logits, beam_width)
                    
                    # Convert to probabilities
                    probs = torch.softmax(top_logits, dim=-1)
                    
                    # Add new candidates
                    for idx, prob in zip(top_indices, probs):
                        new_seq = seq + [idx.item()]
                        new_score = score + torch.log(prob).item()
                        new_sequences.append((new_seq, new_score))
                
                # Keep top-k sequences by score
                new_sequences.sort(key=lambda x: x[1], reverse=True)
                sequences = new_sequences[:beam_width]
        
        return sequences


class TokenizerHelper:
    """Simple tokenizer helper (placeholder for full tokenizer integration)"""
    
    def __init__(self, vocab_size: int = 10000):
        """Initialize tokenizer helper"""
        self.vocab_size = vocab_size
        # In practice, would load tokenizer from file
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to tokens (placeholder)
        
        In production, would use actual tokenizer (sentencepiece, etc.)
        
        Args:
            text: Text to encode
            
        Returns:
            Token IDs
        """
        # Placeholder: just return dummy tokens
        # In practice, integrate with actual tokenizer
        logger.warning("Using placeholder tokenizer - integrate with real tokenizer")
        
        # Simple character-level encoding for demo
        tokens = [ord(c) % self.vocab_size for c in text[:100]]
        return tokens if tokens else [1]  # Return at least one token
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode tokens to text (placeholder)
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Placeholder: just return dummy text
        logger.warning("Using placeholder tokenizer - integrate with real tokenizer")
        
        # Simple character-level decoding for demo
        text = ''.join(chr(t) for t in token_ids if t < 256)
        return text if text else "[UNK]"


class InferenceConfig:
    """Configuration for inference"""
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_length: int = 50,
        temperature: float = 0.7,
        top_k: int = 10,
    ):
        """
        Initialize inference config
        
        Args:
            checkpoint_path: Direct path to checkpoint
            checkpoint_dir: Directory to find best checkpoint in
            device: Device to use
            max_length: Default generation length
            temperature: Default temperature
            top_k: Default top-k value
        """
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
    
    def get_checkpoint_path(self) -> str:
        """Get path to checkpoint to load"""
        if self.checkpoint_path:
            return self.checkpoint_path
        
        if self.checkpoint_dir:
            return ModelLoader.find_best_checkpoint(self.checkpoint_dir)
        
        raise ValueError("Must specify checkpoint_path or checkpoint_dir")


class ConversationInference:
    """High-level interface for conversation inference"""
    
    def __init__(self, config: InferenceConfig):
        """
        Initialize inference interface
        
        Args:
            config: Inference configuration
        """
        self.config = config
        self.tokenizer = TokenizerHelper()
        
        # Load model
        checkpoint_path = config.get_checkpoint_path()
        self.model, self.model_config = ModelLoader.load_model(
            checkpoint_path,
            device=config.device
        )
        
        self.generator = ConversationGenerator(
            self.model,
            self.model_config,
            device=config.device
        )
    
    def chat(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """
        Simple chat interface
        
        Args:
            prompt: Input prompt text
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)
        
        # Generate
        generated_tokens = self.generator.generate(
            prompt_tokens,
            max_length=max_tokens,
            temperature=temperature,
            top_k=self.config.top_k,
        )
        
        # Decode
        response = self.tokenizer.decode(generated_tokens)
        
        return response
