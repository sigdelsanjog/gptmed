"""
PDF processing strategy - extracts text from PDFs and exports to CSV

Supports parallel processing for batch operations using concurrent.futures
for efficient handling of multiple PDF files.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from .base_strategy import TextPreprocessingStrategy


logger = logging.getLogger(__name__)


class PDFProcessor(TextPreprocessingStrategy):
    """
    Processes PDF files: extracts text and exports to CSV
    
    Supports both sequential and parallel batch processing for efficient
    handling of multiple PDF files using ThreadPoolExecutor.
    """
    
    def __init__(self, max_workers: int = 4, use_threading: bool = True):
        """
        Initialize PDF processor
        
        Args:
            max_workers: Maximum number of parallel workers (default: 4)
            use_threading: Use ThreadPoolExecutor (True) or ProcessPoolExecutor (False)
                          Threading is preferred for I/O-bound PDF extraction
        """
        self.PyPDF2 = None
        self._import_pdf_library()
        
        self.max_workers = max_workers
        self.use_threading = use_threading
        
        self.stats = {
            'pdfs_processed': 0,
            'total_text_extracted': 0,
            'csv_exports': 0,
            'parallel_jobs_completed': 0,
            'processing_errors': 0,
        }
        self.last_extracted_texts: Dict[str, str] = {}
    
    def _import_pdf_library(self) -> None:
        """Import PyPDF2 library, raise error if not available"""
        try:
            import PyPDF2
            self.PyPDF2 = PyPDF2
        except ImportError:
            raise ImportError(
                "PyPDF2 is required for PDF processing. "
                "Install it with: pip install gptmed[data-preparation]"
            )
    
    def process(self, text: str) -> str:
        """
        This strategy is not designed for streaming text processing.
        Use extract_text_from_pdf() or batch_process_pdfs() instead.
        
        Args:
            text: Not used for PDF processing
            
        Returns:
            Empty string (PDFs are processed via dedicated methods)
        """
        return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text from PDF
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF processing fails
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            text_content = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = self.PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content.append(page.extract_text())
            
            full_text = '\n'.join(text_content)
            self.last_extracted_texts[str(pdf_path)] = full_text
            self.stats['pdfs_processed'] += 1
            self.stats['total_text_extracted'] += len(full_text)
            
            return full_text
        
        except Exception as e:
            raise Exception(f"Failed to process PDF {pdf_path}: {str(e)}")
    
    def batch_process_pdfs(
        self, 
        pdf_dir: str, 
        output_format: str = 'dict',
        parallel: bool = True,
        max_workers: Optional[int] = None,
    ) -> Any:
        """
        Process multiple PDFs from a directory
        
        Args:
            pdf_dir: Directory containing PDF files
            output_format: Format to return ('dict' or 'list')
            parallel: Whether to use parallel processing (default: True)
            max_workers: Override default max workers for this batch operation
            
        Returns:
            Dictionary or list of {filename: text} pairs
        """
        pdf_dir = Path(pdf_dir)
        
        if not pdf_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob('*.pdf'))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_dir}")
            return {} if output_format == 'dict' else []
        
        if parallel and len(pdf_files) > 1:
            return self._batch_process_parallel(
                pdf_files, 
                output_format, 
                max_workers or self.max_workers
            )
        else:
            return self._batch_process_sequential(pdf_files, output_format)
    
    def _batch_process_sequential(self, pdf_files: List[Path], output_format: str) -> Any:
        """
        Process PDFs sequentially (original method)
        
        Args:
            pdf_files: List of PDF file paths
            output_format: Format to return ('dict' or 'list')
            
        Returns:
            Dictionary or list of {filename: text} pairs
        """
        results = {}
        
        for pdf_file in pdf_files:
            try:
                text = self.extract_text_from_pdf(str(pdf_file))
                results[pdf_file.name] = text
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                results[pdf_file.name] = error_msg
                self.stats['processing_errors'] += 1
                logger.error(f"Error processing {pdf_file.name}: {str(e)}")
        
        if output_format == 'list':
            return [{'filename': k, 'text': v} for k, v in results.items()]
        
        return results
    
    def _batch_process_parallel(
        self, 
        pdf_files: List[Path], 
        output_format: str,
        max_workers: int,
    ) -> Any:
        """
        Process PDFs in parallel using ThreadPoolExecutor
        
        Args:
            pdf_files: List of PDF file paths
            output_format: Format to return ('dict' or 'list')
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary or list of {filename: text} pairs
        """
        results = {}
        executor_class = ThreadPoolExecutor if self.use_threading else ProcessPoolExecutor
        
        logger.info(f"Starting parallel PDF processing with {max_workers} workers")
        
        with executor_class(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.extract_text_from_pdf, str(pdf_file)): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process completed tasks as they finish
            for i, future in enumerate(as_completed(future_to_file), 1):
                pdf_file = future_to_file[future]
                
                try:
                    text = future.result()
                    results[pdf_file.name] = text
                    self.stats['parallel_jobs_completed'] += 1
                    logger.debug(f"[{i}/{len(pdf_files)}] Completed: {pdf_file.name}")
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    results[pdf_file.name] = error_msg
                    self.stats['processing_errors'] += 1
                    logger.error(f"Error processing {pdf_file.name}: {str(e)}")
        
        logger.info(f"Parallel processing complete: {len(results)} files processed")
        
        if output_format == 'list':
            return [{'filename': k, 'text': v} for k, v in results.items()]
        
        return results
    
    def export_to_csv(self, texts: Dict[str, str], output_file: str) -> None:
        """
        Export text data to CSV with filename and text content
        
        Args:
            texts: Dictionary of {filename: text_content}
            output_file: Path to output CSV file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['filename', 'text_content'])
                
                for filename, text in texts.items():
                    writer.writerow([filename, text])
            
            self.stats['csv_exports'] += 1
        
        except Exception as e:
            raise Exception(f"Failed to export CSV: {str(e)}")
    
    def export_to_csv_detailed(self, texts: Dict[str, str], output_file: str) -> None:
        """
        Export text data to CSV with detailed statistics
        
        Args:
            texts: Dictionary of {filename: text_content}
            output_file: Path to output CSV file
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'filename', 'text_content', 'word_count', 
                    'char_count', 'sentence_count'
                ])
                
                for filename, text in texts.items():
                    word_count = len(text.split())
                    char_count = len(text)
                    sentence_count = len([s for s in text.split('.') if s.strip()])
                    
                    writer.writerow([
                        filename, text, word_count, char_count, sentence_count
                    ])
            
            self.stats['csv_exports'] += 1
        
        except Exception as e:
            raise Exception(f"Failed to export detailed CSV: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get PDF processing statistics"""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics"""
        self.stats = {
            'pdfs_processed': 0,
            'total_text_extracted': 0,
            'csv_exports': 0,
            'parallel_jobs_completed': 0,
            'processing_errors': 0,
        }
