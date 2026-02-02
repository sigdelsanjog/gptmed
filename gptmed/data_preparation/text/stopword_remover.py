"""
Stopword removal strategy - removes common words
"""

from typing import Any, Dict, Set, Optional

from .base_strategy import TextPreprocessingStrategy


class StopwordRemover(TextPreprocessingStrategy):
    """
    Removes common English stopwords from text
    """
    
    def __init__(self, stopwords: Optional[Set[str]] = None, enable: bool = True):
        """
        Initialize stopword remover
        
        Args:
            stopwords: Custom set of stopwords; uses default if None
            enable: Whether stopword removal is enabled
        """
        self.enable = enable
        
        if stopwords is None:
            self.stopwords = self._get_default_stopwords()
        else:
            self.stopwords = stopwords
        
        self.stats = {
            'stopwords_removed': 0,
            'enabled': enable,
        }
    
    @staticmethod
    def _get_default_stopwords() -> Set[str]:
        """Get default English stopwords"""
        return {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'is', 'was',
            'are', 'been', 'were', 'or', 'an', 'which', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'them', 'me',
            'my', 'your', 'we', 'us', 'they', 'them', 'these', 'those',
        }
    
    def process(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text: Text to process
            
        Returns:
            Text with stopwords removed (if enabled)
        """
        if not self.enable:
            return text
        
        words = text.split()
        original_count = len(words)
        
        filtered_words = [w for w in words if w not in self.stopwords]
        
        self.stats['stopwords_removed'] += original_count - len(filtered_words)
        
        return ' '.join(filtered_words)
    
    def add_stopword(self, word: str) -> None:
        """Add a custom stopword"""
        self.stopwords.add(word.lower())
    
    def remove_stopword(self, word: str) -> None:
        """Remove a stopword"""
        self.stopwords.discard(word.lower())
    
    def set_stopwords(self, stopwords: Set[str]) -> None:
        """Replace all stopwords with a new set"""
        self.stopwords = stopwords
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stopword removal statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats['stopwords_removed'] = 0
