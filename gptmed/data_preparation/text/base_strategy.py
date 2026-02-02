"""
Base strategy interface for text preprocessing
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class TextPreprocessingStrategy(ABC):
    """Abstract base class for all text preprocessing strategies"""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process the text according to the strategy
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing"""
        pass
