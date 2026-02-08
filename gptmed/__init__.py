"""
GptMed: A lightweight GPT-based language model framework

A domain-agnostic framework for training custom question-answering models.
Train your own GPT model on any Q&A dataset - medical, technical support,
education, legal, customer service, or any other domain.

Key Features:
    - Simple 3-step training: config → train → generate
    - Built-in training observability with loss curves and metrics
    - Flexible model sizes (tiny, small, medium)
    - Device-agnostic (CPU/CUDA with auto-detection)
    - XAI-ready architecture for model interpretability

Quick Start:
    >>> import gptmed
    >>> 
    >>> # 1. Create a config file
    >>> gptmed.create_config('my_config.yaml')
    >>> 
    >>> # 2. Edit my_config.yaml with your settings
    >>> 
    >>> # 3. Train your model (with automatic metrics tracking)
    >>> results = gptmed.train_from_config('my_config.yaml')
    >>> 
    >>> # 4. Generate answers
    >>> answer = gptmed.generate(
    ...     checkpoint=results['best_checkpoint'],
    ...     tokenizer='tokenizer/my_tokenizer.model',
    ...     prompt='Your question here?'
    ... )

Observability (v0.4.0+):
    >>> from gptmed import MetricsTracker, EarlyStoppingCallback
    >>> 
    >>> # Training automatically tracks metrics and generates reports:
    >>> # - Loss curves (train/val)
    >>> # - Gradient norms
    >>> # - Learning rate schedule
    >>> # - Perplexity
    >>> # - Training health checks

Advanced Usage:
    >>> from gptmed.model.architecture import GPTTransformer
    >>> from gptmed.model.configs.model_config import get_small_config
    >>> from gptmed.inference.generator import TextGenerator
    >>> 
    >>> config = get_small_config()
    >>> model = GPTTransformer(config)
"""

__version__ = "0.4.0"
__author__ = "Sanjog Sigdel"
__email__ = "sigdelsanjog@gmail.com"

# Initialize framework (creates logs folder and other necessary directories)
# This is done lazily to avoid import errors when framework is initialized
def _initialize_framework():
    try:
        from gptmed.framework.logging_utils import get_framework_logs_dir
        get_framework_logs_dir()
    except Exception:
        pass  # Framework initialization is optional for backwards compatibility

_initialize_framework()

# High-level API - Main user interface
from gptmed.api import (
    create_config,
    train_from_config,
    generate,
)

# Expose main components at package level for convenience
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import ModelConfig, get_small_config, get_tiny_config

# Observability module
from gptmed.observability import (
    TrainingObserver,
    ObserverManager,
    MetricsTracker,
    ConsoleCallback,
    JSONLoggerCallback,
    EarlyStoppingCallback,
)

__all__ = [
    # Simple API
    "create_config",
    "train_from_config", 
    "generate",
    # Advanced API
    "GPTTransformer",
    "ModelConfig",
    "get_small_config",
    "get_tiny_config",
    # Observability
    "TrainingObserver",
    "ObserverManager",
    "MetricsTracker",
    "ConsoleCallback",
    "JSONLoggerCallback",
    "EarlyStoppingCallback",
]
