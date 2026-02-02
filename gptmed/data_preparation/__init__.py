"""
Data Preparation Module for GptMed

A comprehensive data preprocessing and cleaning framework for preparing various data types
(text, image, video, and audio) as a preprocessing baseline for the gptmed framework.

Usage:
    >>> from gptmed.data_preparation import TextPreprocessor, ImagePreprocessor
    >>> 
    >>> # Text preprocessing
    >>> text_prep = TextPreprocessor()
    >>> cleaned_text = text_prep.process('raw text data')
    >>> 
    >>> # Image preprocessing
    >>> image_prep = ImagePreprocessor()
    >>> processed_image = image_prep.process('path/to/image.jpg')
"""

from .base import BaseDataPreprocessor, PreprocessingConfig
from .text import TextPreprocessor
from .image import ImagePreprocessor
from .audio import AudioPreprocessor
from .video import VideoPreprocessor

__version__ = "0.1.0"
__all__ = [
    "BaseDataPreprocessor",
    "PreprocessingConfig",
    "TextPreprocessor",
    "ImagePreprocessor",
    "AudioPreprocessor",
    "VideoPreprocessor",
]
