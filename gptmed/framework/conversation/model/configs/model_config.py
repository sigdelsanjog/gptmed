"""
Conversation Model Configuration

Hyperparameters and settings for training the conversation language model.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ConversationModelConfig:
    """Configuration for Conversation Language Model"""
    
    # Model Architecture
    vocab_size: int = 10000  # From tokenizer
    d_model: int = 256  # Embedding and hidden dimension
    n_layers: int = 4  # Number of decoder blocks
    n_heads: int = 8  # Number of attention heads
    d_ff: Optional[int] = None  # Feed-forward dimension (default: 4 * d_model)
    max_seq_len: int = 256  # Maximum sequence length (reduced for GTX 1060)
    dropout: float = 0.1  # Dropout probability
    
    # Training
    batch_size: int = 8  # GTX 1060: 6GB VRAM, reduced from 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    
    # Data
    train_ratio: float = 0.8  # Train/validation split
    num_workers: int = 8
    
    # Checkpointing
    checkpoint_dir: str = "./model/checkpoints"
    log_interval: int = 50  # Log every N steps
    save_interval: int = 200  # Save checkpoint every N steps
    gradient_accumulation_steps: int = 1  # For effective larger batches
    
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
