"""
Text data preprocessing and cleaning module - Modular Architecture

This module provides strategy-based text preprocessing with support for:
- Text cleaning (whitespace, special characters, HTML, URLs, emails)
- Unicode normalization with multiple forms
- Case conversion (lowercase, uppercase, title, sentence)
- Punctuation handling (removal, spacing normalization)
- Stopword removal with customizable lists
- Tokenization (word and sentence level)
- Text statistics extraction
- PDF text extraction and CSV export
- Batch file processing

Architecture:
Each preprocessing feature is implemented as a separate strategy class following
the Strategy Design Pattern and SOLID principles. This allows:
- Independent composition of preprocessing steps
- Easy addition of new strategies without modifying existing code
- Single Responsibility: Each class has one reason to change
- Dependency Inversion: Code depends on TextPreprocessingStrategy interface

Usage:
    from gptmed.data_preparation.text import TextCleaner, CaseNormalizer, StopwordRemover
    
    # Use individual strategies
    cleaner = TextCleaner()
    normalizer = CaseNormalizer(mode='lower')
    stopwords = StopwordRemover()
    
    # Compose into pipeline
    text = "Your text here..."
    text = cleaner.process(text)
    text = normalizer.process(text)
    text = stopwords.process(text)
    
    # Get statistics
    stats = cleaner.get_stats()
"""

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..base import BaseDataPreprocessor, PreprocessingConfig

# Import all strategy classes
from .base_strategy import TextPreprocessingStrategy
from .text_cleaner import TextCleaner
from .unicode_normalizer import UnicodeNormalizer
from .case_normalizer import CaseNormalizer
from .punctuation_handler import PunctuationHandler
from .stopword_remover import StopwordRemover
from .tokenizer import Tokenizer
from .text_statistics import TextStatistics
from .pdf_processor import PDFProcessor


logger = logging.getLogger(__name__)


class TextPreprocessor(BaseDataPreprocessor):
    """
    Orchestrator for composing text preprocessing strategies
    
    This class provides a unified interface for text preprocessing by composing
    individual strategy classes. It maintains backward compatibility with the
    previous monolithic implementation while enabling flexible strategy composition.
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        strategies: Optional[List[TextPreprocessingStrategy]] = None,
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
            strategies: List of strategy instances to use in order
            remove_stopwords: Whether to remove common stopwords (default False)
            remove_punctuation: Whether to remove punctuation (default False)
            lowercase: Whether to convert to lowercase (default True)
            min_length: Minimum text length to keep (default 3)
            max_length: Maximum text length, None for unlimited (default None)
        """
        if config is None:
            config = PreprocessingConfig(
                input_path="./data/raw",
                output_path="./data/processed",
                data_type="text"
            )
        
        super().__init__(config)
        
        self.min_length = min_length
        self.max_length = max_length
        
        # Initialize strategies
        if strategies is None:
            strategies = self._create_default_pipeline(
                remove_stopwords=remove_stopwords,
                remove_punctuation=remove_punctuation,
                lowercase=lowercase,
            )
        
        self.strategies = strategies
        
        # Initialize individual strategies for backward compatibility
        self.text_cleaner = TextCleaner()
        self.unicode_normalizer = UnicodeNormalizer()
        self.tokenizer = Tokenizer()
        self.text_stats = TextStatistics()
        self.pdf_processor = PDFProcessor()
    
    def _create_default_pipeline(
        self,
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        lowercase: bool = True,
    ) -> List[TextPreprocessingStrategy]:
        """Create default preprocessing pipeline"""
        pipeline = [
            TextCleaner(),
            UnicodeNormalizer(),
        ]
        
        if lowercase:
            pipeline.append(CaseNormalizer(mode='lower'))
        
        if remove_punctuation:
            pipeline.append(PunctuationHandler(remove=True))
        
        if remove_stopwords:
            pipeline.append(StopwordRemover())
        
        return pipeline
    
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
    
    def clean(self, data: Any) -> Optional[str]:
        """
        Clean text data (remove HTML, URLs, emails, etc.)
        
        Args:
            data: Input text
            
        Returns:
            Cleaned text
        """
        if not self.validate(data):
            return None
        
        text = data
        # Apply TextCleaner from strategies
        cleaner = TextCleaner()
        text = cleaner.process(text)
        return text
    
    def normalize(self, data: Any) -> Optional[str]:
        """
        Normalize text (unicode, case, punctuation, stopwords)
        
        Args:
            data: Input text (may be already cleaned)
            
        Returns:
            Normalized text
        """
        if not isinstance(data, str):
            return None
        
        text = data
        # Apply remaining strategies except TextCleaner
        for strategy in self.strategies:
            if not isinstance(strategy, TextCleaner):
                text = strategy.process(text)
                if not text:
                    return None
        
        return text.strip() if text else None
    
    def process(self, data: Any) -> Optional[str]:
        """
        Process text through the pipeline of strategies
        
        Args:
            data: Input text
            
        Returns:
            Processed text or None if validation fails
        """
        if not self.validate(data):
            return None
        
        text = data
        for strategy in self.strategies:
            text = strategy.process(text)
            if not text:
                return None
        
        return text.strip() if text else None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        processed = self.process(text)
        if processed is None:
            return []
        
        return self.tokenizer.tokenize(processed)
    
    def get_text_stats(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about text (before processing)
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        return self.text_stats.get_text_stats(text)
    
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
    
    # PDF-related methods delegated to PDFProcessor
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from a single PDF file"""
        try:
            return self.pdf_processor.extract_text_from_pdf(pdf_path)
        except Exception as e:
            self.logger.error(f"Error extracting PDF: {str(e)}")
            return None
    
    def batch_process_pdfs(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process multiple PDF files from a directory"""
        output_dir = output_dir or self.config.output_path
        return self.pdf_processor.batch_process_pdfs(input_dir, output_dir)
    
    def export_to_csv(
        self,
        pdf_dir: str,
        output_csv: str,
        process_text: bool = True,
    ) -> Dict[str, Any]:
        """Export PDF text to CSV with filename and text content columns"""
        return self.pdf_processor.export_to_csv(pdf_dir, output_csv)
    
    def export_to_csv_detailed(
        self,
        pdf_dir: str,
        output_csv: str,
        process_text: bool = True,
        include_stats: bool = True,
    ) -> Dict[str, Any]:
        """Export PDF text to CSV with additional statistics"""
        return self.pdf_processor.export_to_csv_detailed(pdf_dir, output_csv)


# Export all strategy classes for direct import
__all__ = [
    'TextPreprocessor',
    'TextPreprocessingStrategy',
    'TextCleaner',
    'UnicodeNormalizer',
    'CaseNormalizer',
    'PunctuationHandler',
    'StopwordRemover',
    'Tokenizer',
    'TextStatistics',
    'PDFProcessor',
]
