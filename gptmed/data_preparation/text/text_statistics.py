"""
Text statistics extraction strategy
"""

import re
from typing import Any, Dict

from .base_strategy import TextPreprocessingStrategy


class TextStatistics(TextPreprocessingStrategy):
    """
    Analyzes and extracts statistics from text
    """
    
    def __init__(self):
        """Initialize text statistics analyzer"""
        self.stats = {
            'analyses_performed': 0,
        }
        self.last_analysis: Dict[str, Any] = {}
    
    def process(self, text: str) -> str:
        """
        Analyze text and return original text unchanged
        (statistics are stored in get_stats())
        
        Args:
            text: Text to analyze
            
        Returns:
            Original text unchanged
        """
        self.analyze(text)
        return text
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Extract comprehensive statistics from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of statistics
        """
        if not text:
            self.last_analysis = self._empty_analysis()
            return self.last_analysis
        
        # Count basic metrics
        word_count = len(text.split())
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', ''))
        line_count = len(text.split('\n'))
        
        # Sentence count (approximate)
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        # Average metrics
        avg_word_length = char_count_no_spaces / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Unique words
        words = text.lower().split()
        unique_words = len(set(words))
        
        # Vocabulary richness
        vocabulary_richness = unique_words / word_count if word_count > 0 else 0
        
        # Punctuation counts
        punctuation_count = len([c for c in text if c in '.,!?;:'])
        
        # Number extraction
        numbers = re.findall(r'\d+', text)
        number_count = len(numbers)
        
        # Capitalized words
        capitalized_words = len([w for w in text.split() if w and w[0].isupper()])
        
        self.last_analysis = {
            'word_count': word_count,
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'line_count': line_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'unique_words': unique_words,
            'vocabulary_richness': round(vocabulary_richness, 3),
            'punctuation_count': punctuation_count,
            'number_count': number_count,
            'capitalized_words': capitalized_words,
        }
        
        self.stats['analyses_performed'] += 1
        return self.last_analysis
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis dictionary"""
        return {
            'word_count': 0,
            'char_count': 0,
            'char_count_no_spaces': 0,
            'line_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0.0,
            'avg_sentence_length': 0.0,
            'unique_words': 0,
            'vocabulary_richness': 0.0,
            'punctuation_count': 0,
            'number_count': 0,
            'capitalized_words': 0,
        }
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Convenience method to analyze text and return stats immediately
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of statistics
        """
        return self.analyze(text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        result = self.stats.copy()
        result['last_analysis'] = self.last_analysis.copy()
        return result
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {'analyses_performed': 0}
        self.last_analysis = {}
