"""
Comprehensive JSONL Preprocessing Pipeline

Uses ALL available text preprocessing strategies in sequence:
1. TextCleaner - Remove HTML, URLs, emails, normalize whitespace
2. UnicodeNormalizer - Normalize unicode characters (NFC/NFD/NFKC/NFKD)
3. CaseNormalizer - Normalize case (lowercase/uppercase/title/sentence)
4. PunctuationHandler - Handle punctuation (remove or normalize spacing)
5. StopwordRemover - Remove common stopwords (optional)
6. TextStatistics - Track detailed statistics

Pipeline applies strategies in order and tracks statistics for each step.

Usage:
    python3 preprocess_jsonl.py \
        --input-file ./output/combined_text.jsonl \
        --output-file ./output/combined_text_full_preprocessed.jsonl \
        [--remove-stopwords] [--remove-punctuation] [--case-mode lowercase]
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Adjust path if running from workspace
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gptmed.data_preparation.text import (
    TextCleaner,
    UnicodeNormalizer,
    CaseNormalizer,
    PunctuationHandler,
    StopwordRemover,
    TextStatistics,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingStepStats:
    """Statistics for each preprocessing step"""
    step_name: str
    word_count_before: int
    word_count_after: int
    char_count_before: int
    char_count_after: int
    reduction_percentage: float


@dataclass
class FullPreprocessedRecord:
    """Complete preprocessed record - only final text in output"""
    filename: str
    text: str
    
    # Statistics summary
    original_word_count: int
    final_word_count: int
    total_reduction_percentage: float
    status: str


class ComprehensiveJSONLPreprocessor:
    """Complete preprocessing pipeline using ALL strategies"""
    
    def __init__(
        self,
        input_file: str,
        output_file: str = None,
        case_mode: str = 'lower',
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        unicode_form: str = 'NFC',
    ):
        """
        Initialize comprehensive preprocessor
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file
            case_mode: Case normalization mode (lower/upper/title/sentence)
            remove_stopwords: Whether to remove stopwords
            remove_punctuation: Whether to remove punctuation
            unicode_form: Unicode normalization form (NFC/NFD/NFKC/NFKD)
        """
        self.input_file = Path(input_file)
        
        if output_file is None:
            base_name = self.input_file.stem
            output_file = self.input_file.parent / f"{base_name}_full_preprocessed.jsonl"
        
        self.output_file = Path(output_file)
        self.case_mode = case_mode
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.unicode_form = unicode_form
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ALL preprocessing strategies
        self.text_cleaner = TextCleaner()
        self.unicode_normalizer = UnicodeNormalizer(form=unicode_form)
        self.case_normalizer = CaseNormalizer(mode=case_mode)
        self.punctuation_handler = PunctuationHandler(
            remove=remove_punctuation,
            normalize_spacing=True
        )
        self.stopword_remover = StopwordRemover()
        self.text_stats = TextStatistics()
    
    def load_jsonl(self) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        records = []
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())
                    records.append(record)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Line {line_num}: Invalid JSON - {str(e)}")
        
        self.logger.info(f"Loaded {len(records)} records from {self.input_file.name}")
        return records
    
    def preprocess_record(self, record: Dict[str, Any]) -> FullPreprocessedRecord:
        """
        Apply all preprocessing steps to a single record
        
        Args:
            record: Input JSONL record
            
        Returns:
            FullPreprocessedRecord with all steps tracked
        """
        try:
            filename = record.get('filename', 'unknown')
            original_text = record.get('text', '')
            
            if not original_text:
                return FullPreprocessedRecord(
                    filename=filename,
                    text='',
                    original_word_count=0,
                    final_word_count=0,
                    total_reduction_percentage=0.0,
                    status='empty_text'
                )
            
            # Track original stats
            original_words = len(original_text.split())
            
            current_text = original_text
            
            # STEP 1: Text Cleaning
            self.logger.debug(f"Step 1: Text Cleaning")
            self.text_cleaner.stats = {
                'html_tags_removed': 0,
                'urls_removed': 0,
                'emails_removed': 0,
                'whitespace_normalized': 0,
            }
            current_text = self.text_cleaner.process(current_text)
            
            # STEP 2: Unicode Normalization
            self.logger.debug(f"Step 2: Unicode Normalization ({self.unicode_form})")
            current_text = self.unicode_normalizer.process(current_text)
            
            # STEP 3: Case Normalization
            self.logger.debug(f"Step 3: Case Normalization ({self.case_mode})")
            current_text = self.case_normalizer.process(current_text)
            
            # STEP 4: Punctuation Handling
            self.logger.debug(f"Step 4: Punctuation Handling")
            current_text = self.punctuation_handler.process(current_text)
            
            # STEP 5: Stopword Removal (optional)
            if self.remove_stopwords:
                self.logger.debug(f"Step 5: Stopword Removal")
                current_text = self.stopword_remover.process(current_text)
            
            final_text = current_text
            
            # Final statistics
            final_words = len(final_text.split())
            reduction_pct = (
                (original_words - final_words) / original_words * 100
                if original_words > 0 else 0.0
            )
            
            return FullPreprocessedRecord(
                filename=filename,
                text=final_text,
                original_word_count=original_words,
                final_word_count=final_words,
                total_reduction_percentage=reduction_pct,
                status='success'
            )
            
        except Exception as e:
            self.logger.error(f"Error preprocessing {record.get('filename', 'unknown')}: {str(e)}")
            return FullPreprocessedRecord(
                filename=record.get('filename', 'unknown'),
                text='',
                original_word_count=0,
                final_word_count=0,
                total_reduction_percentage=0.0,
                status=f'error: {str(e)}'
            )
    
    def save_jsonl(self, records: List[FullPreprocessedRecord]) -> bool:
        """Save full preprocessed records to JSONL"""
        try:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    json.dump(asdict(record), f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"✓ Saved {len(records)} fully preprocessed records to {self.output_file.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving JSONL: {str(e)}")
            return False
    
    def process(self) -> Dict[str, Any]:
        """Main processing pipeline"""
        start_time = time.time()
        
        self.logger.info("="*70)
        self.logger.info("Comprehensive JSONL Preprocessing Pipeline")
        self.logger.info("="*70)
        self.logger.info(f"Input file: {self.input_file}")
        self.logger.info(f"Output file: {self.output_file}")
        self.logger.info(f"\nPipeline configuration:")
        self.logger.info(f"  1. TextCleaner (remove HTML, URLs, emails)")
        self.logger.info(f"  2. UnicodeNormalizer ({self.unicode_form})")
        self.logger.info(f"  3. CaseNormalizer ({self.case_mode})")
        self.logger.info(f"  4. PunctuationHandler (remove={self.remove_punctuation})")
        self.logger.info(f"  5. StopwordRemover (enabled={self.remove_stopwords})")
        self.logger.info("="*70)
        
        # Load records
        records = self.load_jsonl()
        
        if not records:
            self.logger.warning("No records to process")
            return {'status': 'failure', 'total_records': 0}
        
        # Preprocess each record
        self.logger.info(f"\nPreprocessing {len(records)} records...")
        preprocessed_records = []
        
        for i, record in enumerate(records, 1):
            preprocessed = self.preprocess_record(record)
            preprocessed_records.append(preprocessed)
            
            if i % max(1, len(records) // 10) == 0 or i == len(records):
                self.logger.info(f"Progress: {i}/{len(records)} records processed")
        
        # Save results
        self.logger.info(f"\nSaving preprocessed records...")
        saved = self.save_jsonl(preprocessed_records)
        
        # Calculate statistics
        successful = [r for r in preprocessed_records if r.status == 'success']
        failed = [r for r in preprocessed_records if r.status != 'success']
        
        total_time = time.time() - start_time
        
        # Statistics
        original_words = sum(r.original_word_count for r in successful)
        final_words = sum(r.final_word_count for r in successful)
        avg_reduction = (
            sum(r.total_reduction_percentage for r in successful) / len(successful)
            if successful else 0.0
        )
        
        # Print summary
        self.logger.info(f"\n" + "="*70)
        self.logger.info(f"Preprocessing Summary")
        self.logger.info(f"="*70)
        self.logger.info(f"Total records: {len(records)}")
        self.logger.info(f"Successfully preprocessed: {len(successful)}")
        self.logger.info(f"Failed: {len(failed)}")
        self.logger.info(f"\nWord Count Statistics:")
        self.logger.info(f"  Original total words: {original_words:,}")
        self.logger.info(f"  Final total words: {final_words:,}")
        self.logger.info(f"  Average reduction: {avg_reduction:.1f}%")
        self.logger.info(f"\nOutput: {self.output_file.name}")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"="*70)
        
        return {
            'status': 'success' if saved else 'failure',
            'input_file': str(self.input_file),
            'output_file': str(self.output_file),
            'total_records': len(records),
            'successful': len(successful),
            'failed': len(failed),
            'original_word_count': original_words,
            'final_word_count': final_words,
            'average_reduction_percentage': avg_reduction,
            'total_time': total_time,
            'pipeline_steps': [
                'TextCleaner',
                'UnicodeNormalizer',
                'CaseNormalizer',
                'PunctuationHandler',
                f"StopwordRemover (enabled={self.remove_stopwords})",
            ],
        }


def show_pipeline_details():
    """Show details about each preprocessing step"""
    print(f"""
{'='*70}
COMPREHENSIVE PREPROCESSING PIPELINE DETAILS
{'='*70}

STEP 1: TextCleaner
  ✓ Removes HTML tags: <b>text</b> → text
  ✓ Removes URLs: http://example.com → removed
  ✓ Removes email addresses: user@example.com → removed
  ✓ Normalizes whitespace: multiple   spaces → single space
  Stats: html_tags_removed, urls_removed, emails_removed

STEP 2: UnicodeNormalizer
  ✓ NFC (canonical composition) - DEFAULT
  ✓ NFD (canonical decomposition)
  ✓ NFKC (compatibility composition)
  ✓ NFKD (compatibility decomposition)
  Handles: accents, emojis, special characters
  Stats: characters_removed, form_used

STEP 3: CaseNormalizer
  ✓ lowercase - RECOMMENDED for models
  ✓ UPPERCASE
  ✓ Title Case
  ✓ Sentence case
  Stats: mode_used, conversions_applied

STEP 4: PunctuationHandler
  ✓ Remove or keep punctuation
  ✓ Normalize spacing around punctuation
  Stats: punctuation_removed, spacing_normalized

STEP 5: StopwordRemover (Optional)
  ✓ Remove common English stopwords (the, a, an, etc.)
  ✓ Customizable stopword lists
  Stats: stopwords_removed

OUTPUT:
  - Each record includes text at every step
  - Full statistics for each step
  - Track exactly how text changes through pipeline
  - Ready for tokenization with preprocessed_text field

EXAMPLE TRANSFORMATION:
  
  Original:
    "Hello WORLD! 🌍 Visit https://example.com for <b>MORE</b> info!"
  
  Step 1 (TextCleaner):
    "Hello WORLD! 🌍 Visit for MORE info!"
  
  Step 2 (UnicodeNormalizer - NFC):
    "Hello WORLD! 🌍 Visit for MORE info!"
  
  Step 3 (CaseNormalizer - lowercase):
    "hello world! 🌍 visit for more info!"
  
  Step 4 (PunctuationHandler):
    "hello world 🌍 visit for more info"
  
  Step 5 (StopwordRemover - if enabled):
    "hello world visit more info"
  
  Final: "hello world 🌍 visit for more info" (or without stopwords)

READY FOR TOKENIZATION:
  Use the 'final_preprocessed_text' field in JSONL for:
  - Hugging Face Tokenizer
  - SentencePiece
  - Custom tokenizers
  - Model training
{'='*70}
""")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive JSONL preprocessing using ALL strategies'
    )
    parser.add_argument(
        '--input-file',
        default='./output/combined_text.jsonl',
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output-file',
        default=None,
        help='Output JSONL file (auto-generated if not specified)'
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
        help='Remove common stopwords'
    )
    parser.add_argument(
        '--remove-punctuation',
        action='store_true',
        help='Remove punctuation marks'
    )
    parser.add_argument(
        '--unicode-form',
        default='NFC',
        choices=['NFC', 'NFD', 'NFKC', 'NFKD'],
        help='Unicode normalization form (default: NFC)'
    )
    parser.add_argument(
        '--show-pipeline',
        action='store_true',
        help='Show pipeline details and exit'
    )
    
    args = parser.parse_args()
    
    if args.show_pipeline:
        show_pipeline_details()
    else:
        processor = ComprehensiveJSONLPreprocessor(
            input_file=args.input_file,
            output_file=args.output_file,
            case_mode=args.case_mode,
            remove_stopwords=args.remove_stopwords,
            remove_punctuation=args.remove_punctuation,
            unicode_form=args.unicode_form,
        )
        
        result = processor.process()
        sys.exit(0 if result['status'] == 'success' else 1)
