"""
Case normalization strategy - converts text case
"""

from typing import Any, Dict

from .base_strategy import TextPreprocessingStrategy


class CaseNormalizer(TextPreprocessingStrategy):
    """
    Handles text case conversion (lowercase, uppercase, title case)
    """
    
    def __init__(self, mode: str = 'lower'):
        """
        Initialize case normalizer
        
        Args:
            mode: Case conversion mode ('lower', 'upper', 'title', 'sentence')
        """
        if mode not in ('lower', 'upper', 'title', 'sentence'):
            raise ValueError(f"Invalid mode: {mode}. Use 'lower', 'upper', 'title', or 'sentence'")
        
        self.mode = mode
        self.stats = {
            'mode_used': mode,
            'conversions_applied': 0,
        }
    
    def process(self, text: str) -> str:
        """
        Normalize text case
        
        Args:
            text: Text to normalize
            
        Returns:
            Case-normalized text
        """
        if not text:
            return text
        
        original = text
        
        if self.mode == 'lower':
            text = text.lower()
        elif self.mode == 'upper':
            text = text.upper()
        elif self.mode == 'title':
            text = text.title()
        elif self.mode == 'sentence':
            text = text[0].upper() + text[1:].lower() if len(text) > 0 else text
        
        if text != original:
            self.stats['conversions_applied'] += 1
        
        return text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats['conversions_applied'] = 0
