"""
Batch PDF Processing (Intermediate Step)

Parallelly processes all PDFs from a directory and extracts text.
Note: Does not save files - text is processed in-memory for preprocessing pipeline.

Usage:
    python3 batch_pdf_to_jsonl.py [--input-dir ./pdfs] [--output-dir ./output] [--workers 4]
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

# Adjust path if running from workspace
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gptmed.data_preparation.text import PDFProcessor


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PDFRecord:
    """Data class for a single PDF record"""
    filename: str
    text: str
    word_count: int
    char_count: int
    sentence_count: int
    extraction_time: float
    status: str


class PDFBatchProcessor:
    """Process multiple PDFs in parallel and export to JSONL"""
    
    def __init__(
        self,
        input_dir: str = "./pdfs",
        output_dir: str = "./output",
        max_workers: int = 4,
    ):
        """
        Initialize batch processor
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output JSONL files
            max_workers: Number of parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        
        # Create directories if they don't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(max_workers=max_workers, use_threading=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _extract_pdf(self, pdf_file: Path) -> Optional[PDFRecord]:
        """
        Extract text from a single PDF file
        
        Args:
            pdf_file: Path to PDF file
            
        Returns:
            PDFRecord with extracted information or None if failed
        """
        start_time = time.time()
        
        try:
            text = self.pdf_processor.extract_text_from_pdf(str(pdf_file))
            
            # Calculate statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len([s for s in text.split('.') if s.strip()])
            extraction_time = time.time() - start_time
            
            record = PDFRecord(
                filename=pdf_file.name,
                text=text,
                word_count=word_count,
                char_count=char_count,
                sentence_count=sentence_count,
                extraction_time=extraction_time,
                status="success"
            )
            
            self.logger.info(
                f"✓ Extracted: {pdf_file.name} "
                f"({word_count} words, {char_count} chars) in {extraction_time:.2f}s"
            )
            
            return record
            
        except Exception as e:
            self.logger.error(f"✗ Failed to extract {pdf_file.name}: {str(e)}")
            return PDFRecord(
                filename=pdf_file.name,
                text="",
                word_count=0,
                char_count=0,
                sentence_count=0,
                extraction_time=time.time() - start_time,
                status=f"error: {str(e)}"
            )
    
    def _process_pdfs_parallel(self) -> List[PDFRecord]:
        """
        Process all PDFs in parallel
        
        Returns:
            List of PDFRecord objects
        """
        # Find all PDF files
        pdf_files = sorted(self.input_dir.glob('*.pdf'))
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {self.input_dir}")
            return []
        
        self.logger.info(
            f"Processing {len(pdf_files)} PDF files with {self.max_workers} workers..."
        )
        
        records = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all PDF extraction tasks
            future_to_file = {
                executor.submit(self._extract_pdf, pdf_file): pdf_file 
                for pdf_file in pdf_files
            }
            
            # Process completed tasks as they finish
            for i, future in enumerate(as_completed(future_to_file), 1):
                try:
                    record = future.result()
                    if record:
                        records.append(record)
                except Exception as e:
                    self.logger.error(f"Error processing future: {str(e)}")
        
        return records
    

    
    def process(self) -> Dict[str, Any]:
        """
        Main processing pipeline
        
        Returns:
            Dictionary with processing results and statistics
        """
        start_time = time.time()
        
        self.logger.info(f"Input directory: {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        
        # Step 1: Process all PDFs in parallel
        records = self._process_pdfs_parallel()
        
        if not records:
            self.logger.warning("No records to process")
            return {
                'status': 'failure',
                'message': 'No PDFs were successfully processed',
                'total_pdfs': 0,
                'successful_extractions': 0,
                'failed_extractions': 0,
                'individual_jsonl_files': 0,
                'combined_jsonl_created': False,
                'total_time': 0,
            }
        
        # Count successes and failures
        successful = [r for r in records if r.status == "success"]
        failed = [r for r in records if r.status != "success"]
        
        self.logger.info(f"\nExtraction Results:")
        self.logger.info(f"  ✓ Successful: {len(successful)}/{len(records)}")
        self.logger.info(f"  ✗ Failed: {len(failed)}/{len(records)}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        total_words = sum(r.word_count for r in successful)
        total_chars = sum(r.char_count for r in successful)
        
        # Print summary
        self.logger.info(f"\n" + "="*60)
        self.logger.info(f"Processing Summary")
        self.logger.info(f"="*60)
        self.logger.info(f"Total PDFs: {len(records)}")
        self.logger.info(f"Successfully processed: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")
        self.logger.info(f"Total words extracted: {total_words:,}")
        self.logger.info(f"Total characters: {total_chars:,}")
        self.logger.info(f"Note: Records processed in-memory for preprocessing")
        self.logger.info(f"Total processing time: {total_time:.2f}s")
        self.logger.info(f"="*60)
        
        return {
            'status': 'success',
            'records': records,
            'total_pdfs': len(records),
            'successful_extractions': len(successful),
            'failed_extractions': len(failed),
            'total_words': total_words,
            'total_characters': total_chars,
            'total_time': total_time,
        }


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch process PDFs and export to JSONL format'
    )
    parser.add_argument(
        '--input-dir',
        default='./pdfs',
        help='Input directory containing PDFs (default: ./pdfs)'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for JSONL files (default: ./output)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    
    args = parser.parse_args()
    
    # Create and run processor
    processor = PDFBatchProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_workers=args.workers,
    )
    
    result = processor.process()
    
    # Return exit code
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
