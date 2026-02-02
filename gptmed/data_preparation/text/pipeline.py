"""
Complete PDF → Tokens Pipeline

Orchestrates the full preprocessing pipeline:
1. Extract text from PDFs (in-memory)
2. Preprocess text (in-memory)
3. Tokenize (saves merged_tokens.jsonl only)

This is the main entry point for generating training data.

Usage:
    python3 pipeline.py \
        --input-dir ./pdfs \
        --output-dir ./output \
        --tokenizer-method huggingface \
        --tokenizer-model gpt2 \
        --workers 4
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import asdict
import argparse
import importlib.util

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Load batch_pdf_to_jsonl
spec1 = importlib.util.spec_from_file_location("batch_pdf_to_jsonl", Path(__file__).parent / "batch_pdf_to_jsonl.py")
batch_module = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(batch_module)
PDFBatchProcessor = batch_module.PDFBatchProcessor
PDFRecord = batch_module.PDFRecord

# Load preprocess_jsonl
spec2 = importlib.util.spec_from_file_location("preprocess_jsonl", Path(__file__).parent / "preprocess_jsonl.py")
preprocess_module = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(preprocess_module)
ComprehensiveJSONLPreprocessor = preprocess_module.ComprehensiveJSONLPreprocessor
FullPreprocessedRecord = preprocess_module.FullPreprocessedRecord

# Load tokenize_jsonl
spec3 = importlib.util.spec_from_file_location("tokenize_jsonl", Path(__file__).parent / "tokenize_jsonl.py")
tokenize_module = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(tokenize_module)
ParallelJSONLTokenizer = tokenize_module.ParallelJSONLTokenizer
SimplifiedTokenizedRecord = tokenize_module.SimplifiedTokenizedRecord


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Complete PDF to training data pipeline"""
    
    def __init__(
        self,
        input_dir: str = "./pdfs",
        output_dir: str = "./output",
        tokenizer_method: str = "huggingface",
        tokenizer_model: str = "gpt2",
        workers: int = 4,
        case_mode: str = "lower",
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
    ):
        """
        Initialize pipeline
        
        Args:
            input_dir: Directory containing PDFs
            output_dir: Output directory for final results
            tokenizer_method: Tokenization method (huggingface/custom/sentencepiece)
            tokenizer_model: Tokenizer model name
            workers: Number of parallel workers
            case_mode: Case normalization (lower/upper/title/sentence)
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tokenizer_method = tokenizer_method
        self.tokenizer_model = tokenizer_model
        self.workers = workers
        self.case_mode = case_mode
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def step1_extract_pdfs(self) -> List[PDFRecord]:
        """Step 1: Extract text from PDFs in-memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 1: PDF EXTRACTION")
        self.logger.info("="*70)
        
        processor = PDFBatchProcessor(
            input_dir=str(self.input_dir),
            output_dir=str(self.output_dir),
            max_workers=self.workers,
        )
        
        result = processor.process()
        records = result.get('records', [])
        
        self.logger.info(f"\n✓ Extracted {len(records)} PDF records")
        return records
    
    def step2_preprocess_text(self, records: List[PDFRecord]) -> List[FullPreprocessedRecord]:
        """Step 2: Preprocess text in-memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 2: TEXT PREPROCESSING")
        self.logger.info("="*70)
        
        # Create a dummy temp file just to initialize the preprocessor
        # We'll manually preprocess records using its methods
        temp_input = self.output_dir / "_temp_input.jsonl"
        with open(temp_input, 'w') as f:
            f.write("{}\n")
        
        try:
            processor = ComprehensiveJSONLPreprocessor(
                input_file=str(temp_input),
                output_file=None,
                case_mode=self.case_mode,
                remove_stopwords=self.remove_stopwords,
                remove_punctuation=self.remove_punctuation,
            )
        finally:
            temp_input.unlink()
        
        preprocessed_records = []
        
        # Convert PDFRecord to dict format expected by preprocess_record
        for record in records:
            record_dict = {
                'filename': record.filename,
                'text': record.text,
                'word_count': record.word_count,
            }
            
            # Preprocess the record
            preprocessed = processor.preprocess_record(record_dict)
            preprocessed_records.append(preprocessed)
        
        self.logger.info(f"✓ Preprocessed {len(preprocessed_records)} records")
        
        # Save preprocessed output
        output_file = self.output_dir / "full_preprocessed.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in preprocessed_records:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write('\n')
        
        self.logger.info(f"✓ Saved: {output_file.name}")
        
        return preprocessed_records
    
    def step3_tokenize(self, preprocessed_records: List[FullPreprocessedRecord]) -> Dict[str, Any]:
        """Step 3: Tokenize preprocessed text"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 3: TOKENIZATION")
        self.logger.info("="*70)
        
        # Save preprocessed records to temporary JSONL for tokenizer to consume
        temp_file = self.output_dir / "_temp_preprocessed.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for record in preprocessed_records:
                json.dump(asdict(record), f, ensure_ascii=False)
                f.write('\n')
        
        # Initialize tokenizer
        tokens_dir = self.output_dir / "tokens"
        tokenizer = ParallelJSONLTokenizer(
            input_file=str(temp_file),
            output_dir=str(tokens_dir),
            method=self.tokenizer_method,
            model_name=self.tokenizer_model,
            workers=self.workers,
        )
        
        # Tokenize
        result = tokenizer.process()
        
        # Clean up temporary file
        temp_file.unlink()
        
        return result
    
    def run(self) -> Dict[str, Any]:
        """Execute full pipeline"""
        start_time = time.time()
        
        self.logger.info("\n")
        self.logger.info("╔" + "="*68 + "╗")
        self.logger.info("║" + " "*15 + "END-TO-END PDF → TOKENS PIPELINE" + " "*21 + "║")
        self.logger.info("╚" + "="*68 + "╝")
        
        try:
            # Step 1: Extract PDFs
            pdf_records = self.step1_extract_pdfs()
            
            if not pdf_records:
                self.logger.error("No PDF records extracted. Exiting.")
                return {'status': 'failure', 'message': 'No PDFs extracted'}
            
            # Step 2: Preprocess
            preprocessed_records = self.step2_preprocess_text(pdf_records)
            
            if not preprocessed_records:
                self.logger.error("No records preprocessed. Exiting.")
                return {'status': 'failure', 'message': 'Preprocessing failed'}
            
            # Step 3: Tokenize
            tokenization_result = self.step3_tokenize(preprocessed_records)
            
            total_time = time.time() - start_time
            
            # Final summary
            self.logger.info("\n" + "="*70)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"\nFinal Outputs:")
            self.logger.info(f"  1. {self.output_dir}/full_preprocessed.jsonl (cleaned text)")
            self.logger.info(f"  2. {self.output_dir}/tokens/merged_tokens.jsonl (training tokens)")
            self.logger.info(f"\nTotal Time: {total_time:.2f}s")
            self.logger.info(f"="*70 + "\n")
            
            return {
                'status': 'success',
                'total_time': total_time,
                'pdf_extraction': {
                    'records': len(pdf_records),
                },
                'preprocessing': {
                    'records': len(preprocessed_records),
                },
                'tokenization': tokenization_result,
                'output_dir': str(self.output_dir),
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'failure', 'error': str(e)}


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Complete PDF to training tokens pipeline'
    )
    parser.add_argument(
        '--input-dir',
        default='./pdfs',
        help='Input directory containing PDFs (default: ./pdfs)'
    )
    parser.add_argument(
        '--output-dir',
        default='./output',
        help='Output directory for final results (default: ./output)'
    )
    parser.add_argument(
        '--tokenizer-method',
        default='huggingface',
        choices=['huggingface', 'custom', 'sentencepiece'],
        help='Tokenization method (default: huggingface)'
    )
    parser.add_argument(
        '--tokenizer-model',
        default='gpt2',
        help='Tokenizer model name (default: gpt2)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )
    parser.add_argument(
        '--case-mode',
        default='lower',
        choices=['lower', 'upper', 'title', 'sentence'],
        help='Case normalization mode (default: lower)'
    )
    parser.add_argument(
        '--remove-stopwords',
        action='store_true',
        help='Remove common stopwords during preprocessing'
    )
    parser.add_argument(
        '--remove-punctuation',
        action='store_true',
        help='Remove punctuation during preprocessing'
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = EndToEndPipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        tokenizer_method=args.tokenizer_method,
        tokenizer_model=args.tokenizer_model,
        workers=args.workers,
        case_mode=args.case_mode,
        remove_stopwords=args.remove_stopwords,
        remove_punctuation=args.remove_punctuation,
    )
    
    result = pipeline.run()
    
    # Exit with appropriate code
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
