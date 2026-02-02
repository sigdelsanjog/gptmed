"""
Unicode normalization strategy - handles unicode characters and accents
"""

import unicodedata
from typing import Any, Dict

from .base_strategy import TextPreprocessingStrategy


class UnicodeNormalizer(TextPreprocessingStrategy):
    """
    Normalizes unicode characters, removes diacritics and combining marks
    """
    
    def __init__(self, form: str = 'NFD'):
        """
        Initialize normalizer
        
        Args:
            form: Unicode normalization form (NFD, NFC, NFKD, NFKC)
                  NFD is decomposed form (default)
                  NFC is composed form
        """
        self.form = form
        self.stats = {
            'characters_removed': 0,
            'form_used': form,
        }
    
    def process(self, text: str) -> str:
        """
        Normalize unicode text
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        # Normalize unicode (NFD - decomposed form removes accents)
        normalized = unicodedata.normalize(self.form, text)
        
        # Remove combining marks (accents, diacritics)
        cleaned = ''.join(
            ch for ch in normalized 
            if unicodedata.category(ch) != 'Mn'
        )
        
        chars_removed = len(text) - len(cleaned)
        self.stats['characters_removed'] += chars_removed
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats['characters_removed'] = 0
