"""
Text cleaning strategy - removes HTML, URLs, emails, and special characters
"""

import re
from typing import Any, Dict

from .base_strategy import TextPreprocessingStrategy


class TextCleaner(TextPreprocessingStrategy):
    """
    Removes HTML tags, URLs, email addresses, and normalizes whitespace
    
    This is the first step in the preprocessing pipeline.
    """
    
    def __init__(self):
        self.stats = {
            'html_tags_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0,
            'whitespace_normalized': 0,
        }
    
    def process(self, text: str) -> str:
        """
        Clean text by removing artifacts
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove HTML tags
        html_pattern = r'<[^>]+>'
        html_matches = len(re.findall(html_pattern, text))
        text = re.sub(html_pattern, '', text)
        self.stats['html_tags_removed'] += html_matches
        
        # Remove URLs
        url_pattern = r'http[s]?://\S+|www\.\S+'
        url_matches = len(re.findall(url_pattern, text))
        text = re.sub(url_pattern, '', text)
        self.stats['urls_removed'] += url_matches
        
        # Remove email addresses
        email_pattern = r'\S+@\S+'
        email_matches = len(re.findall(email_pattern, text))
        text = re.sub(email_pattern, '', text)
        self.stats['emails_removed'] += email_matches
        
        # Remove extra whitespace
        original_spaces = len(re.findall(r'\s+', text))
        text = re.sub(r'\s+', ' ', text)
        self.stats['whitespace_normalized'] += original_spaces - len(re.findall(r'\s+', text))
        
        # Remove common control characters
        text = ''.join(ch for ch in text if ch.isprintable() or ch.isspace())
        
        return text.strip()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cleaning statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        for key in self.stats:
            self.stats[key] = 0
