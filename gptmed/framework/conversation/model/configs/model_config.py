"""
Conversation Model Configuration

Hyperparameters and settings for training the conversation language model.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


def get_default_checkpoint_dir() -> str:
    """
    Get the default checkpoint directory.
    
    Returns the string "./checkpoints" which will be resolved relative to
    the current working directory when the training script is run.
    
    When user installs gptmed and runs training from their directory,
    checkpoints will be saved in ./checkpoints/ in their current working directory.
    """
    return "./checkpoints"


@dataclass
class ConversationModelConfig:
    """Configuration for Conversation Language Model
    
    GTX 1060 FINAL OPTIMIZED (6GB VRAM):
    - max_seq_len: 256 (attention O(n²) is main memory consumer)
    - d_model: 96 (ultra-ultra-compact)
    - n_layers: 3 (minimal depth)
    - batch_size: 1, gradient_accumulation: 4
    - ~150K parameters (minimal model)
    - Expected memory: ~800MB per step (SAFE)
    """
    
    # Model Architecture (GTX 1060 FINAL OPTIMIZED)
    vocab_size: int = 50256  # From tokenizer
    d_model: int = 96  # GTX 1060: ULTRA-REDUCED to 96 (was 128)
    n_layers: int = 3  # GTX 1060: 3 layers (was 4)
    n_heads: int = 3  # 3 heads, d_model/n_heads = 32 dims/head
    d_ff: Optional[int] = None  # Feed-forward (default: 4 * d_model = 384)
    max_seq_len: int = 256  # GTX 1060: 256 (attention memory critical)
    dropout: float = 0.3  # Strong regularization
    attention_dropout: float = 0.2  # Attention dropout
    use_gradient_checkpointing: bool = False  # NOT AVAILABLE in this PyTorch version
    
    # Training (GTX 1060 FINAL)
    batch_size: int = 1  # CRITICAL
    learning_rate: float = 1e-3  # For small batches
    weight_decay: float = 0.01  # L2 regularization
    num_epochs: int = 5  # Training epochs
    warmup_steps: int = 100  # Shorter warmup (was 200)
    gradient_clip: float = 1.0  # Max gradient norm
    early_stopping_patience: int = 3  # Early stopping
    early_stopping_threshold: float = 0.0001
    eval_steps: int = 300  # Evaluate every N steps
    save_steps: int = 300  # Save checkpoint every N steps
    enable_amp: bool = True  # Mixed precision ENABLED
    
    # Data
    train_ratio: float = 0.9  # Train/validation split
    num_workers: int = 0  # No multiprocessing
    
    # Checkpointing
    checkpoint_dir: str = field(default_factory=get_default_checkpoint_dir)
    log_interval: int = 50  # Log every N steps
    save_interval: int = 300  # Save checkpoint every N steps
    gradient_accumulation_steps: int = 4  # Effective batch = 4
    clear_cache_interval: int = 10  # Clear GPU cache every N steps (NEW)
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        """Validate configuration"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
        }
