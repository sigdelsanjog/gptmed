"""
Base Observer Classes

PURPOSE:
Define abstract interfaces for training observation.
Implements Observer Pattern for decoupled monitoring.

DESIGN PATTERNS:
- Observer Pattern: Subjects (Trainer) notify observers of state changes
- Template Method: Base class defines interface, subclasses implement
- Dependency Inversion: Trainer depends on abstraction, not concrete observers

WHY OBSERVER PATTERN:
1. Decoupling: Trainer doesn't know about logging implementations
2. Extensibility: Add new observers without modifying Trainer
3. Single Responsibility: Each observer handles one concern
4. Open/Closed: Open for extension, closed for modification
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum


class TrainingEvent(Enum):
    """Training lifecycle events that observers can subscribe to."""
    TRAIN_START = "on_train_start"
    TRAIN_END = "on_train_end"
    EPOCH_START = "on_epoch_start"
    EPOCH_END = "on_epoch_end"
    STEP = "on_step"
    VALIDATION = "on_validation"
    CHECKPOINT = "on_checkpoint"
    GRADIENT_COMPUTED = "on_gradient_computed"


@dataclass
class StepMetrics:
    """
    Metrics collected at each training step.
    
    Using dataclass for:
    - Type safety
    - Clear documentation of expected metrics
    - Easy serialization
    """
    step: int
    loss: float
    learning_rate: float
    grad_norm: float
    batch_size: int
    seq_len: int
    tokens_per_sec: float = 0.0
    perplexity: float = 0.0
    
    def __post_init__(self):
        """Compute derived metrics."""
        if self.perplexity == 0.0 and self.loss > 0:
            import math
            self.perplexity = math.exp(min(self.loss, 100))  # Cap to avoid overflow
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "grad_norm": self.grad_norm,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "tokens_per_sec": self.tokens_per_sec,
            "perplexity": self.perplexity,
        }


@dataclass
class ValidationMetrics:
    """Metrics collected during validation."""
    step: int
    val_loss: float
    val_perplexity: float = 0.0
    
    def __post_init__(self):
        """Compute derived metrics."""
        if self.val_perplexity == 0.0 and self.val_loss > 0:
            import math
            self.val_perplexity = math.exp(min(self.val_loss, 100))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "val_loss": self.val_loss,
            "val_perplexity": self.val_perplexity,
        }


@dataclass
class GradientMetrics:
    """
    Gradient statistics for observability.
    
    Used for detecting:
    - Vanishing gradients (norm → 0)
    - Exploding gradients (norm → ∞)
    - Dead neurons (high zero fraction)
    """
    step: int
    total_norm: float
    layer_norms: Dict[str, float] = field(default_factory=dict)
    max_grad: float = 0.0
    min_grad: float = 0.0
    zero_fraction: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            "total_norm": self.total_norm,
            "layer_norms": self.layer_norms,
            "max_grad": self.max_grad,
            "min_grad": self.min_grad,
            "zero_fraction": self.zero_fraction,
        }


class TrainingObserver(ABC):
    """
    Abstract base class for training observers.
    
    Implements Observer Pattern - receives notifications from Trainer
    without Trainer knowing the concrete implementation.
    
    Lifecycle:
        on_train_start → [on_epoch_start → [on_step]* → on_validation → on_epoch_end]* → on_train_end
    
    Example:
        >>> class MyObserver(TrainingObserver):
        ...     def on_step(self, metrics: StepMetrics) -> None:
        ...         print(f"Loss: {metrics.loss}")
        ...
        >>> trainer.add_observer(MyObserver())
    """
    
    def __init__(self, name: str = None):
        """
        Initialize observer.
        
        Args:
            name: Human-readable name for this observer
        """
        self.name = name or self.__class__.__name__
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        """Whether this observer is active."""
        return self._enabled
    
    def enable(self) -> None:
        """Enable this observer."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable this observer."""
        self._enabled = False
    
    # Required methods (must implement)
    
    @abstractmethod
    def on_train_start(self, config: Dict[str, Any]) -> None:
        """
        Called when training begins.
        
        Args:
            config: Training configuration dictionary
        """
        pass
    
    @abstractmethod
    def on_step(self, metrics: StepMetrics) -> None:
        """
        Called after each training step.
        
        Args:
            metrics: Step metrics (loss, grad_norm, etc.)
        """
        pass
    
    @abstractmethod
    def on_validation(self, metrics: ValidationMetrics) -> None:
        """
        Called after validation.
        
        Args:
            metrics: Validation metrics (val_loss, val_perplexity)
        """
        pass
    
    @abstractmethod
    def on_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """
        Called when training completes.
        
        Args:
            final_metrics: Final training summary
        """
        pass
    
    # Optional methods (override if needed)
    
    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at start of each epoch.
        
        Args:
            epoch: Current epoch number (0-indexed)
        """
        pass
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Called at end of each epoch.
        
        Args:
            epoch: Current epoch number
            metrics: Epoch summary metrics
        """
        pass
    
    def on_checkpoint(self, step: int, checkpoint_path: str) -> None:
        """
        Called when a checkpoint is saved.
        
        Args:
            step: Training step
            checkpoint_path: Path to saved checkpoint
        """
        pass
    
    def on_gradient_computed(self, metrics: GradientMetrics) -> None:
        """
        Called after gradients are computed (before optimizer step).
        
        Useful for gradient flow analysis.
        
        Args:
            metrics: Gradient statistics
        """
        pass


class ObserverManager:
    """
    Manages multiple observers and dispatches events.
    
    Implements Composite pattern - treats collection of observers uniformly.
    
    Example:
        >>> manager = ObserverManager()
        >>> manager.add(ConsoleCallback())
        >>> manager.add(MetricsTracker(log_dir='logs'))
        >>> manager.notify_step(step_metrics)
    """
    
    def __init__(self):
        """Initialize empty observer list."""
        self._observers: List[TrainingObserver] = []
    
    def add(self, observer: TrainingObserver) -> 'ObserverManager':
        """
        Add an observer.
        
        Args:
            observer: Observer to add
            
        Returns:
            Self for method chaining
        """
        self._observers.append(observer)
        return self
    
    def remove(self, observer: TrainingObserver) -> bool:
        """
        Remove an observer.
        
        Args:
            observer: Observer to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            self._observers.remove(observer)
            return True
        except ValueError:
            return False
    
    def get_observer(self, name: str) -> Optional[TrainingObserver]:
        """
        Get observer by name.
        
        Args:
            name: Observer name
            
        Returns:
            Observer if found, None otherwise
        """
        for obs in self._observers:
            if obs.name == name:
                return obs
        return None
    
    @property
    def observers(self) -> List[TrainingObserver]:
        """Get list of all observers."""
        return self._observers.copy()
    
    def _notify(self, event: str, *args, **kwargs) -> None:
        """
        Dispatch event to all enabled observers.
        
        Args:
            event: Event method name
            *args, **kwargs: Event arguments
        """
        for observer in self._observers:
            if observer.enabled:
                handler = getattr(observer, event, None)
                if handler and callable(handler):
                    try:
                        handler(*args, **kwargs)
                    except Exception as e:
                        print(f"Warning: Observer {observer.name} failed on {event}: {e}")
    
    # Convenience methods for each event type
    
    def notify_train_start(self, config: Dict[str, Any]) -> None:
        """Notify all observers of training start."""
        self._notify('on_train_start', config)
    
    def notify_train_end(self, final_metrics: Dict[str, Any]) -> None:
        """Notify all observers of training end."""
        self._notify('on_train_end', final_metrics)
    
    def notify_epoch_start(self, epoch: int) -> None:
        """Notify all observers of epoch start."""
        self._notify('on_epoch_start', epoch)
    
    def notify_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Notify all observers of epoch end."""
        self._notify('on_epoch_end', epoch, metrics)
    
    def notify_step(self, metrics: StepMetrics) -> None:
        """Notify all observers of training step."""
        self._notify('on_step', metrics)
    
    def notify_validation(self, metrics: ValidationMetrics) -> None:
        """Notify all observers of validation."""
        self._notify('on_validation', metrics)
    
    def notify_checkpoint(self, step: int, checkpoint_path: str) -> None:
        """Notify all observers of checkpoint save."""
        self._notify('on_checkpoint', step, checkpoint_path)
    
    def notify_gradient(self, metrics: GradientMetrics) -> None:
        """Notify all observers of gradient computation."""
        self._notify('on_gradient_computed', metrics)
