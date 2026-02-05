"""Conversation Language Model Framework

From-scratch PyTorch implementation of a transformer-based language model
designed for medical conversation generation.

No external model libraries (transformers, HuggingFace) - pure PyTorch.
"""

from . import model, training, inference

__all__ = [
    'model',
    'training',
    'inference',
]
