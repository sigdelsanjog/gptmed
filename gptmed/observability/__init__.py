"""
Observability Module for GptMed

PURPOSE:
Provides training observability, metrics tracking, and XAI capabilities.
Implements Observer Pattern for decoupled monitoring.

DESIGN PATTERNS:
- Observer Pattern: Trainer emits events, observers react independently
- Strategy Pattern: Swap between different logging backends
- Open/Closed Principle: Add new observers without modifying Trainer

COMPONENTS:
- TrainingObserver: Abstract base class for all observers
- MetricsTracker: Enhanced loss curves and training metrics
- Callbacks: TensorBoard, W&B, Early Stopping, etc.

FUTURE EXTENSIONS:
- Attention visualization
- Saliency maps / input attribution
- Embedding space analysis
- Gradient flow analysis
"""

from gptmed.observability.base import TrainingObserver, ObserverManager
from gptmed.observability.metrics_tracker import MetricsTracker
from gptmed.observability.callbacks import (
    ConsoleCallback,
    JSONLoggerCallback,
    EarlyStoppingCallback,
)

__all__ = [
    # Base classes
    "TrainingObserver",
    "ObserverManager",
    # Metrics
    "MetricsTracker",
    # Callbacks
    "ConsoleCallback",
    "JSONLoggerCallback",
    "EarlyStoppingCallback",
]
