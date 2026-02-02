"""
Tokenization strategy - converts text to tokens
"""

import re
from typing import Any, Dict, List

from .base_strategy import TextPreprocessingStrategy


class Tokenizer(TextPreprocessingStrategy):
    """
    Tokenizes text into words or sentences
    """
    
    def __init__(self, mode: str = 'word'):
        """
        Initialize tokenizer
        
        Args:
            mode: Tokenization mode ('word' or 'sentence')
        """
        if mode not in ('word', 'sentence'):
            raise ValueError(f"Invalid mode: {mode}. Use 'word' or 'sentence'")
        
        self.mode = mode
        self.stats = {
            'mode': mode,
            'tokens_created': 0,
        }
        self.last_tokens: List[str] = []
    
    def process(self, text: str) -> str:
        """
        Tokenize text and return as space-separated tokens
        
        Args:
            text: Text to tokenize
            
        Returns:
            Space-separated tokens
        """
        tokens = self.tokenize(text)
        self.stats['tokens_created'] = len(tokens)
        return ' '.join(tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into list of tokens
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        if self.mode == 'word':
            tokens = text.split()
        elif self.mode == 'sentence':
            # Split on sentence-ending punctuation
            tokens = re.split(r'[.!?]+', text)
            tokens = [s.strip() for s in tokens if s.strip()]
        
        self.last_tokens = tokens
        self.stats['tokens_created'] = len(tokens)
        return tokens
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats['tokens_created'] = 0
