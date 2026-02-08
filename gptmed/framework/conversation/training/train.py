"""
Training Script for Conversation Language Model

Complete training loop with validation, checkpointing, and progress tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast, GradScaler
import logging
import time
from pathlib import Path
from typing import Tuple, Optional
import json

from ..model.architecture import ConversationLanguageModel
from ..model.configs.model_config import ConversationModelConfig
from .data_loader import get_data_loaders
from ...logging_utils import setup_training_logger


logger = setup_training_logger(__name__, model_type="conversation")


class Trainer:
    """Training orchestrator for conversation model"""
    
    def __init__(self, config: ConversationModelConfig):
        """
        Initialize trainer
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Smart checkpoint directory resolution (backward compatibility)
        # If old location exists, use it; otherwise use default
        checkpoint_dir = Path(config.checkpoint_dir)
        old_checkpoint_paths = [
            Path("./model/checkpoints"),
            Path("model/checkpoints"),
        ]
        
        for old_path in old_checkpoint_paths:
            if old_path.exists():
                logger.info(f"Found existing checkpoints in old location: {old_path.resolve()}")
                self.config.checkpoint_dir = str(old_path.resolve())
                checkpoint_dir = Path(self.config.checkpoint_dir)
                break
        
        # Create torch device object (handles both 'cuda' and 'cuda:0')
        self.device = torch.device(config.device)
        
        # Validate device availability
        if self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available!")
            # Don't call torch.cuda.init() - let PyTorch handle it automatically
        
        # Log device info
        logger.info(f"Device: {self.device}")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"CUDA currently using: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            if self.device.type == 'cuda':
                logger.info("✓ GPU TRAINING ENABLED")
            else:
                logger.warning("⚠ Using CPU even though CUDA is available!")
        else:
            logger.warning("CUDA not available - using CPU (this will be slow!)")
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = ConversationLanguageModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        ).to(self.device)
        
        # Verify model is on correct device
        model_device = next(self.model.parameters()).device
        logger.info(f"Model successfully moved to: {model_device}")
        if model_device.type != self.device.type:
            raise RuntimeError(f"Model device mismatch! Expected {self.device}, got {model_device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=1000,
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Mixed precision (for GPU memory efficiency)
        self.use_amp = self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Using automatic mixed precision (AMP) for memory efficiency")
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_epoch = 0
        self.last_checkpoint_step = 0  # Track last checkpoint step
        self.checkpoint_history = {'latest': None, 'previous': None, 'best': None}  # Checkpoint tracking
    
    def train_epoch(self, train_loader, val_loader) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Data loader for training
            val_loader: Data loader for validation
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        logger.info("Starting training epoch...")
        logger.info(f"Total batches to process: {len(train_loader)}")
        
        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            try:
                # Log first batch for debugging
                if batch_idx == 0:
                    logger.info(f"First batch shapes - input_ids: {input_ids.shape}, target_ids: {target_ids.shape}")
                    logger.info(f"Before device transfer - input_ids device: {input_ids.device}")
                
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Verify tensors are on correct device
                if batch_idx == 0:
                    logger.info(f"After device transfer - input_ids device: {input_ids.device}")
                    logger.info(f"Model device: {next(self.model.parameters()).device}")
                    if torch.cuda.is_available():
                        logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                
                # Verify shapes before forward pass
                if input_ids.dim() != 2 or target_ids.dim() != 2:
                    logger.error(
                        f"Batch {batch_idx}: Invalid tensor shapes - "
                        f"input_ids: {input_ids.shape}, target_ids: {target_ids.shape}"
                    )
                    continue
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        logits = self.model(input_ids)  # [batch_size, seq_len, vocab_size]
                        logits_reshaped = logits.view(-1, self.config.vocab_size)
                        targets_reshaped = target_ids.view(-1)
                        loss = self.criterion(logits_reshaped, targets_reshaped)
                else:
                    logits = self.model(input_ids)
                    logits_reshaped = logits.view(-1, self.config.vocab_size)
                    targets_reshaped = target_ids.view(-1)
                    loss = self.criterion(logits_reshaped, targets_reshaped)
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    avg_loss = total_loss / num_batches
                    logger.info(
                        f"Epoch [{batch_idx}/{len(train_loader)}] "
                        f"Step [{self.global_step}] "
                    f"Loss: {avg_loss:.4f}"
                )
                
                # Validate and save checkpoint every save_interval steps
                if self.global_step % self.config.save_interval == 0 and self.global_step > self.last_checkpoint_step:
                    logger.info(f"\n💾 Checkpoint interval reached (step {self.global_step})")
                    # Perform validation
                    intra_val_loss = self.validate(val_loader)
                    self.last_checkpoint_step = self.global_step
                    
                    # Save checkpoint with rotation (latest, previous, best)
                    if intra_val_loss < self.best_val_loss:
                        # Found a better model
                        self.best_val_loss = intra_val_loss
                        self.save_checkpoint_rotated(tag="best", step=self.global_step)
                    else:
                        # Regular checkpoint
                        self.save_checkpoint_rotated(tag="latest", step=self.global_step)
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                continue
        
        if num_batches == 0:
            logger.error("No batches processed successfully!")
            return float('inf')
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch average loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """
        Validate the model
        
        Args:
            val_loader: Data loader for validation
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        logger.info("Starting validation...")
        
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with autocast():
                        logits = self.model(input_ids)
                        logits_reshaped = logits.view(-1, self.config.vocab_size)
                        targets_reshaped = target_ids.view(-1)
                        loss = self.criterion(logits_reshaped, targets_reshaped)
                else:
                    logits = self.model(input_ids)
                    logits_reshaped = logits.view(-1, self.config.vocab_size)
                    targets_reshaped = target_ids.view(-1)
                    loss = self.criterion(logits_reshaped, targets_reshaped)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        logger.info(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint_rotated(self, tag: str = "latest", step: int = 0):
        """
        Save checkpoint with rotation (keep only latest, previous, and best)
        
        Args:
            tag: Checkpoint type ("latest" or "best")
            step: Current step number
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if tag == "best":
            checkpoint_path = checkpoint_dir / "checkpoint_best.pt"
        elif tag == "latest":
            checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_{tag}.pt"
        
        checkpoint = {
            'step': self.global_step,
            'epoch': 0,  # Not using epochs with step-based saving
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved {tag} checkpoint: {checkpoint_path.name} (step {self.global_step})")
        logger.debug(f"  Full path: {checkpoint_path.resolve()}")
        
        # Rotation logic: move latest to previous when saving new latest
        if tag == "latest":
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            previous_path = checkpoint_dir / "checkpoint_previous.pt"
            
            # If latest exists, move it to previous
            if latest_path.exists() and latest_path != checkpoint_path:
                try:
                    if previous_path.exists():
                        previous_path.unlink()
                    latest_path.rename(previous_path)
                    logger.debug(f"Rotated checkpoint: latest → previous")
                except Exception as e:
                    logger.warning(f"Failed to rotate checkpoint: {e}")
            
            # Save new latest
            torch.save(checkpoint, latest_path)
    
    
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            epoch: The epoch at which checkpoint was saved
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Load scheduler state if available
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.global_step = checkpoint['step']
        self.start_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.start_epoch}, step {self.global_step}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")
        
        return self.start_epoch
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the most recent checkpoint for resuming training.
        
        Priority order:
        1. checkpoint_latest.pt (most recent)
        2. checkpoint_previous.pt (previous checkpoint)
        3. checkpoint_best.pt (best validation loss)
        4. checkpoint_step_*.pt files (old format)
        
        Returns:
            Path to checkpoint to resume from, or None if no checkpoint exists
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        # Try latest checkpoint first (most recent step)
        latest = checkpoint_dir / "checkpoint_latest.pt"
        if latest.exists():
            return latest
        
        # Then try previous checkpoint
        previous = checkpoint_dir / "checkpoint_previous.pt"
        if previous.exists():
            return previous
        
        # Try best checkpoint
        best = checkpoint_dir / "checkpoint_best.pt"
        if best.exists():
            return best
        
        # Check for old checkpoint_step_*.pt files
        step_checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if step_checkpoints:
            return step_checkpoints[0]
        
        return None
    
    def train(self, train_data_file: str, resume: bool = True):
        """
        Full training loop with automatic checkpoint resuming
        
        Args:
            train_data_file: Path to merged_tokens.jsonl
            resume: Whether to resume from latest checkpoint (default: True)
        """
        logger.info("="*70)
        logger.info("Starting Conversation Model Training")
        logger.info("="*70)
        logger.info(f"Checkpoint directory: {Path(self.config.checkpoint_dir).resolve()}")
        logger.info(f"Config: {json.dumps(self.config.to_dict(), indent=2)}")
        
        # Try to resume from checkpoint if available
        if resume:
            latest_checkpoint = self.find_latest_checkpoint()
            if latest_checkpoint:
                logger.info(f"\n🔄 Found checkpoint: {latest_checkpoint.name}")
                logger.info(f"   Path: {latest_checkpoint.resolve()}")
                self.load_checkpoint(str(latest_checkpoint))
                logger.info(f"✓ Resuming training from epoch {self.start_epoch + 1}\n")
            else:
                logger.info(f"\n✨ No checkpoint found in {Path(self.config.checkpoint_dir).resolve()} - starting fresh training\n")
        
        # Load data
        train_loader, val_loader = get_data_loaders(
            data_file=train_data_file,
            batch_size=self.config.batch_size,
            max_seq_len=self.config.max_seq_len,
            train_ratio=self.config.train_ratio,
            num_workers=self.config.num_workers,
        )
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            logger.info(f"\n{'='*70}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*70}")
            
            # Train
            train_loss = self.train_epoch(train_loader, val_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Save checkpoint if validation loss improved (epoch-based backup)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint_rotated(tag="best", step=self.global_step)
            
            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Best Val Loss: {self.best_val_loss:.4f}"
            )
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Checkpoints saved in: {self.config.checkpoint_dir}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train conversation model')
    parser.add_argument(
        '--data-file',
        required=True,
        help='Path to merged_tokens.jsonl file'
    )
    parser.add_argument(
        '--d-model',
        type=int,
        default=256,
        help='Model dimension'
    )
    parser.add_argument(
        '--n-layers',
        type=int,
        default=4,
        help='Number of decoder layers'
    )
    parser.add_argument(
        '--n-heads',
        type=int,
        default=8,
        help='Number of attention heads'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ConversationModelConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        device=args.device,
    )
    
    # Train
    trainer = Trainer(config)
    trainer.train(args.data_file)


if __name__ == '__main__':
    main()
