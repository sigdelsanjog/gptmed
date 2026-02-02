"""
Text data preprocessing and cleaning module

Handles text normalization, cleaning, tokenization, and validation
"""

import re
import string
import unicodedata
import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import json

from ..base import BaseDataPreprocessor, PreprocessingConfig


logger = logging.getLogger(__name__)


class TextPreprocessor(BaseDataPreprocessor):
    """
    Text preprocessing with cleaning, normalization, and validation
    
    Features:
        - Text cleaning (whitespace, special characters)
        - Case normalization
        - Unicode normalization
        - Stopword removal
        - Punctuation handling
        - Language detection
        - Sentiment preservation
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        lowercase: bool = True,
        min_length: int = 3,
        max_length: Optional[int] = None,
    ):
        """
        Initialize text preprocessor
        
        Args:
            config: PreprocessingConfig instance
            remove_stopwords: Whether to remove common stopwords
            remove_punctuation: Whether to remove punctuation
            lowercase: Whether to convert to lowercase
            min_length: Minimum text length to keep
            max_length: Maximum text length (None for unlimited)
        """
        if config is None:
            config = PreprocessingConfig(
                input_path="./data/raw",
                output_path="./data/processed",
                data_type="text"
            )
        
        super().__init__(config)
        
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_length = min_length
        self.max_length = max_length
        
        # Load stopwords
        self.stopwords = self._load_stopwords() if remove_stopwords else set()
    
    def _load_stopwords(self) -> set:
        """Load common English stopwords"""
        # Basic English stopwords
        stopwords = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'is', 'was',
            'are', 'been', 'were', 'or', 'an', 'which', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'them', 'me',
        }
        return stopwords
    
    def validate(self, data: Any) -> bool:
        """
        Validate text input
        
        Args:
            data: Input text
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, str):
            self.logger.warning(f"Invalid text type: {type(data)}")
            return False
        
        if len(data.strip()) < self.min_length:
            self.logger.debug(f"Text too short: {len(data)}")
            return False
        
        if self.max_length and len(data) > self.max_length:
            self.logger.debug(f"Text too long: {len(data)}")
            return False
        
        return True
    
    def clean(self, text: str) -> str:
        """
        Clean text by removing artifacts and normalizing
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize unicode (NFD - decomposed form)
        text = unicodedata.normalize('NFD', text)
        text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common control characters
        text = ''.join(ch for ch in text if ch.isprintable() or ch.isspace())
        
        return text.strip()
    
    def normalize(self, text: str) -> str:
        """
        Normalize text
        
        Args:
            text: Cleaned text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Optional punctuation removal
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        else:
            # Just normalize spacing around punctuation
            text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            words = text.split()
            words = [w for w in words if w not in self.stopwords]
            text = ' '.join(words)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple word tokenization
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Process text first
        processed = self.process(text)
        if processed is None:
            return []
        
        return processed.split()
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about the text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        processed = self.process(text)
        if processed is None:
            return {}
        
        words = processed.split()
        sentences = re.split(r'[.!?]+', processed)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            'original_length': len(text),
            'cleaned_length': len(processed),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'unique_words': len(set(words)),
            'vocabulary_diversity': len(set(words)) / len(words) if words else 0,
        }
    
    def batch_process_files(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        pattern: str = "*.txt"
    ) -> Dict[str, Any]:
        """
        Process multiple text files from a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path (uses config if None)
            pattern: File pattern to match
            
        Returns:
            Processing results
        """
        output_dir = output_dir or self.config.output_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_dir)
        results = []
        
        for file_path in input_path.glob(pattern):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                processed = self.process(text)
                
                if processed:
                    output_file = Path(output_dir) / file_path.name
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(processed)
                    
                    results.append({
                        'file': str(file_path),
                        'status': 'success',
                        'stats': self.get_text_stats(text)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")
                results.append({
                    'file': str(file_path),
                    'status': 'error',
                    'error': str(e)
                })
        
        self.logger.info(f"Processed {len(results)} files")
        return {'results': results, 'stats': self.get_statistics()}
