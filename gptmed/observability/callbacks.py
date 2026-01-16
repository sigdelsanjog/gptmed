"""
Training Callbacks

PURPOSE:
Concrete observer implementations for common training needs.

CALLBACKS INCLUDED:
- ConsoleCallback: Print progress to console
- JSONLoggerCallback: Log to JSONL file
- EarlyStoppingCallback: Stop training if no improvement
- (Future) TensorBoardCallback: Log to TensorBoard
- (Future) WandBCallback: Log to Weights & Biases
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

from gptmed.observability.base import (
    TrainingObserver,
    StepMetrics,
    ValidationMetrics,
    GradientMetrics,
)


class ConsoleCallback(TrainingObserver):
    """
    Prints training progress to console.
    
    Features:
    - Colored output for warnings
    - Progress bar style step counter
    - Issue detection (NaN, high loss, etc.)
    
    Example:
        >>> trainer.add_observer(ConsoleCallback(log_interval=100))
    """
    
    def __init__(
        self,
        log_interval: int = 100,
        show_progress_bar: bool = True,
        name: str = "ConsoleCallback",
    ):
        """
        Initialize ConsoleCallback.
        
        Args:
            log_interval: Print every N steps
            show_progress_bar: Whether to show progress indicators
            name: Callback name
        """
        super().__init__(name=name)
        self.log_interval = log_interval
        self.show_progress_bar = show_progress_bar
        self.total_steps: int = 0
        self.start_time: Optional[float] = None
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Called when training begins."""
        self.start_time = time.time()
        self.total_steps = config.get('max_steps', 0) or config.get('total_steps', 0)
        
        print("\n" + "=" * 70)
        print("🚀 Training Started")
        print("=" * 70)
        print(f"  Model: {config.get('model_size', 'unknown')}")
        print(f"  Device: {config.get('device', 'unknown')}")
        print(f"  Batch size: {config.get('batch_size', 'unknown')}")
        print(f"  Learning rate: {config.get('learning_rate', 'unknown')}")
        print(f"  Total steps: {self.total_steps}")
        print("=" * 70 + "\n")
    
    def on_step(self, metrics: StepMetrics) -> None:
        """Called after each training step."""
        if metrics.step % self.log_interval != 0:
            return
        
        # Calculate progress
        progress = ""
        if self.total_steps > 0:
            pct = (metrics.step / self.total_steps) * 100
            progress = f"[{pct:5.1f}%] "
        
        # Calculate speed
        elapsed = time.time() - self.start_time if self.start_time else 0
        steps_per_sec = metrics.step / elapsed if elapsed > 0 else 0
        
        # Build message
        msg = f"{progress}Step {metrics.step:6d} | "
        msg += f"Loss: {metrics.loss:.4f} | "
        msg += f"PPL: {metrics.perplexity:.2f} | "
        msg += f"LR: {metrics.learning_rate:.2e} | "
        msg += f"Grad: {metrics.grad_norm:.3f} | "
        msg += f"{steps_per_sec:.1f} steps/s"
        
        # Check for issues
        warnings = []
        if metrics.loss != metrics.loss:  # NaN check
            warnings.append("🔥 NaN LOSS!")
        elif metrics.loss > 100:
            warnings.append("⚠️ High loss")
        
        if metrics.grad_norm > 10:
            warnings.append("⚠️ Large grads")
        
        if warnings:
            msg += " | " + " ".join(warnings)
        
        print(msg)
    
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """Called after validation."""
        print(f"\n{'─' * 50}")
        print(f"📊 Validation @ Step {metrics.step}")
        print(f"   Val Loss: {metrics.val_loss:.4f}")
        print(f"   Val PPL:  {metrics.val_perplexity:.2f}")
        print(f"{'─' * 50}\n")
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training completes."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        print("\n" + "=" * 70)
        print("✅ Training Completed")
        print("=" * 70)
        print(f"  Total time: {elapsed/60:.2f} minutes")
        print(f"  Best val loss: {final_metrics.get('best_val_loss', 'N/A')}")
        print(f"  Best checkpoint: {final_metrics.get('best_checkpoint', 'N/A')}")
        print("=" * 70 + "\n")
    
    def on_epoch_start(self, epoch: int) -> None:
        """Called at start of each epoch."""
        print(f"\n📅 Epoch {epoch + 1} starting...")
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Called at end of each epoch."""
        print(f"📅 Epoch {epoch + 1} completed")
        if 'train_loss' in metrics:
            print(f"   Avg train loss: {metrics['train_loss']:.4f}")
        if 'val_loss' in metrics:
            print(f"   Val loss: {metrics['val_loss']:.4f}")


class JSONLoggerCallback(TrainingObserver):
    """
    Logs metrics to JSONL file.
    
    Format: One JSON object per line (JSONL)
    Easy to parse and analyze with Python/pandas.
    
    Example:
        >>> callback = JSONLoggerCallback(log_dir='logs', experiment_name='exp1')
        >>> trainer.add_observer(callback)
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "training",
        log_interval: int = 1,
        name: str = "JSONLoggerCallback",
    ):
        """
        Initialize JSONLoggerCallback.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name for log files
            log_interval: Log every N steps
            name: Callback name
        """
        super().__init__(name=name)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        
        self.log_file = self.log_dir / f"{experiment_name}_log.jsonl"
        self.start_time: Optional[float] = None
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Called when training begins."""
        self.start_time = time.time()
        
        # Log config
        self._write_log({
            "event": "train_start",
            "timestamp": 0,
            "config": config,
        })
    
    def on_step(self, metrics: StepMetrics) -> None:
        """Called after each training step."""
        if metrics.step % self.log_interval != 0:
            return
        
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self._write_log({
            "event": "step",
            "timestamp": timestamp,
            **metrics.to_dict(),
        })
    
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """Called after validation."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self._write_log({
            "event": "validation",
            "timestamp": timestamp,
            **metrics.to_dict(),
        })
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Called when training completes."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self._write_log({
            "event": "train_end",
            "timestamp": timestamp,
            **final_metrics,
        })
    
    def on_checkpoint(self, step: int, checkpoint_path: str) -> None:
        """Called when a checkpoint is saved."""
        timestamp = time.time() - self.start_time if self.start_time else 0
        
        self._write_log({
            "event": "checkpoint",
            "timestamp": timestamp,
            "step": step,
            "checkpoint_path": checkpoint_path,
        })
    
    def _write_log(self, data: Dict[str, Any]) -> None:
        """Write log entry to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data, default=str) + '\n')


class EarlyStoppingCallback(TrainingObserver):
    """
    Stops training if validation loss doesn't improve.
    
    Features:
    - Patience: Number of validations to wait
    - Min delta: Minimum improvement to count as progress
    - Restore best: Flag to restore best weights (handled by trainer)
    
    Example:
        >>> callback = EarlyStoppingCallback(patience=5, min_delta=0.01)
        >>> trainer.add_observer(callback)
        >>> # During training, check: callback.should_stop
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        name: str = "EarlyStoppingCallback",
    ):
        """
        Initialize EarlyStoppingCallback.
        
        Args:
            patience: Number of validations without improvement before stopping
            min_delta: Minimum change to qualify as improvement
            name: Callback name
        """
        super().__init__(name=name)
        
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_loss: float = float('inf')
        self.best_step: int = 0
        self.wait_count: int = 0
        self.should_stop: bool = False
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Reset state at training start."""
        self.best_loss = float('inf')
        self.best_step = 0
        self.wait_count = 0
        self.should_stop = False
    
    def on_step(self, metrics: StepMetrics) -> None:
        """Not used - we check on validation."""
        pass
    
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """Check if we should stop training."""
        current_loss = metrics.val_loss
        
        if current_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = current_loss
            self.best_step = metrics.step
            self.wait_count = 0
        else:
            # No improvement
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                self.should_stop = True
                print(f"\n⏹️ Early stopping triggered!")
                print(f"   No improvement for {self.patience} validations")
                print(f"   Best val loss: {self.best_loss:.4f} at step {self.best_step}")
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Report final state."""
        if self.should_stop:
            print(f"   Training stopped early at step {final_metrics.get('step', 'unknown')}")


class LRSchedulerCallback(TrainingObserver):
    """
    Monitors and can adjust learning rate based on training progress.
    
    Features:
    - Reduce LR on plateau
    - Warmup monitoring
    - LR range test (experimental)
    """
    
    def __init__(
        self,
        mode: str = 'monitor',  # 'monitor' or 'plateau'
        factor: float = 0.5,
        patience: int = 3,
        min_lr: float = 1e-7,
        name: str = "LRSchedulerCallback",
    ):
        """
        Initialize LRSchedulerCallback.
        
        Args:
            mode: 'monitor' (just watch) or 'plateau' (reduce on plateau)
            factor: Factor to reduce LR by
            patience: Validations without improvement before reducing
            min_lr: Minimum learning rate
            name: Callback name
        """
        super().__init__(name=name)
        
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        
        self.best_loss: float = float('inf')
        self.wait_count: int = 0
        self.lr_history: list = []
        self.suggested_lr: Optional[float] = None
    
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """Reset state."""
        self.best_loss = float('inf')
        self.wait_count = 0
        self.lr_history = []
        self.suggested_lr = None
    
    def on_step(self, metrics: StepMetrics) -> None:
        """Track learning rate."""
        self.lr_history.append((metrics.step, metrics.learning_rate))
    
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """Check for plateau."""
        if self.mode != 'plateau':
            return
        
        if metrics.val_loss < self.best_loss:
            self.best_loss = metrics.val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                if self.lr_history:
                    current_lr = self.lr_history[-1][1]
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    self.suggested_lr = new_lr
                    print(f"\n📉 Suggest reducing LR: {current_lr:.2e} → {new_lr:.2e}")
                self.wait_count = 0
    
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Report LR summary."""
        if self.lr_history:
            initial_lr = self.lr_history[0][1]
            final_lr = self.lr_history[-1][1]
            print(f"   LR range: {initial_lr:.2e} → {final_lr:.2e}")
