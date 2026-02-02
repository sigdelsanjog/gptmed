"""
Punctuation handling strategy - removes or normalizes punctuation
"""

import re
import string
from typing import Any, Dict

from .base_strategy import TextPreprocessingStrategy


class PunctuationHandler(TextPreprocessingStrategy):
    """
    Handles punctuation removal or normalization
    """
    
    def __init__(self, remove: bool = False, normalize_spacing: bool = True):
        """
        Initialize punctuation handler
        
        Args:
            remove: If True, removes all punctuation; if False, only normalizes spacing
            normalize_spacing: If True, normalizes spacing around punctuation
        """
        self.remove = remove
        self.normalize_spacing = normalize_spacing
        self.stats = {
            'punctuation_removed': 0,
            'spacing_normalized': 0,
        }
    
    def process(self, text: str) -> str:
        """
        Handle punctuation in text
        
        Args:
            text: Text to process
            
        Returns:
            Processed text
        """
        if self.remove:
            # Remove all punctuation
            original_len = len(text)
            text = text.translate(str.maketrans('', '', string.punctuation))
            self.stats['punctuation_removed'] += original_len - len(text)
        else:
            # Normalize spacing around punctuation
            if self.normalize_spacing:
                # Remove space before punctuation
                original = text
                text = re.sub(r'\s+([.!?,;:])', r'\1', text)
                if text != original:
                    self.stats['spacing_normalized'] += 1
        
        return text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get punctuation handling statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats['punctuation_removed'] = 0
        self.stats['spacing_normalized'] = 0
