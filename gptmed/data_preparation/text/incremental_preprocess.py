"""
Incremental Data Preprocessing Pipeline for PDFs

Process only NEW PDF files and append results to existing full_preprocessed.jsonl.
This avoids reprocessing all PDFs every time you add new files.

ARCHITECTURE:
1. pdfs/new/              # Place new PDFs here
2. output/
   ├── full_preprocessed.jsonl      # Master preprocessed data (appended to)
   ├── incremental_TIMESTAMP.jsonl  # New data from this run
   └── processing_manifest.json     # Tracks processed files

USAGE:
    # Add new PDFs to pdfs/new/
    python3 incremental_preprocess.py
    
    # With options:
    python3 incremental_preprocess.py --reset-manifest --workers 8
    
    # Reset and reprocess all files:
    python3 incremental_preprocess.py --reset-manifest

WHY THIS IS BETTER:
- 1st run: Process all PDFs in pdfs/new/ (depends on file size)
- 2nd run: Process only NEW files added to pdfs/new/ 
- No reprocessing of already-processed files
- Automatically appends to existing dataset
"""

import sys
import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime
import argparse
import importlib.util

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Lazy load modules to handle missing dependencies
PDFBatchProcessor = None
PDFRecord = None
ComprehensiveJSONLPreprocessor = None
FullPreprocessedRecord = None

def lazy_load_modules():
    """Lazily load pipeline modules when needed."""
    global PDFBatchProcessor, PDFRecord, ComprehensiveJSONLPreprocessor, FullPreprocessedRecord
    
    if PDFBatchProcessor is not None:
        return  # Already loaded
    
    try:
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
        
    except ImportError as e:
        logger.error(f"Failed to load pipeline modules: {e}")
        raise


# ============================================================================
# CONFIGURATION
# ============================================================================

class IncrementalConfig:
    """Configuration for incremental preprocessing."""
    
    # Directories
    NEW_PDF_DIR = Path('./pdfs/new')
    OUTPUT_DIR = Path('./output')
    
    # Files
    MANIFEST_FILE = OUTPUT_DIR / 'processing_manifest.json'
    FULL_PREPROCESSED_FILE = OUTPUT_DIR / 'full_preprocessed.jsonl'
    
    # Incremental output
    INCREMENTAL_PREFIX = 'incremental'  # incremental_20240115_120530.jsonl
    
    # Processing options
    CASE_MODE = 'lower'
    REMOVE_STOPWORDS = False
    REMOVE_PUNCTUATION = False
    WORKERS = 4


# ============================================================================
# SETUP LOGGING
# ============================================================================

def setup_logging():
    """Setup logging for incremental preprocessing."""
    log_dir = Path('./logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'incremental_preprocess.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# FILE TRACKING
# ============================================================================

class ProcessingManifest:
    """Tracks which PDF files have been processed."""
    
    def __init__(self, manifest_path: Path):
        """
        Args:
            manifest_path: Path to manifest JSON file
        """
        self.manifest_path = manifest_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        """Load manifest from file or create empty."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted manifest at {self.manifest_path}, creating new one")
                return self._create_empty()
        else:
            return self._create_empty()
    
    def _create_empty(self) -> Dict:
        """Create empty manifest structure."""
        return {
            'version': 1,
            'last_updated': None,
            'processed_files': {},  # filename -> hash
            'statistics': {
                'total_files_processed': 0,
                'total_records_processed': 0,
                'incremental_runs': 0,
            }
        }
    
    def save(self):
        """Save manifest to file."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        logger.info(f"✓ Manifest saved to {self.manifest_path}")
    
    def is_processed(self, file_path: Path) -> bool:
        """Check if file has been processed before."""
        file_hash = self._hash_file(file_path)
        filename = file_path.name  # Use just the filename
        
        if filename in self.data['processed_files']:
            stored_hash = self.data['processed_files'][filename]
            # If hash matches, file hasn't changed
            return stored_hash == file_hash
        
        return False
    
    def mark_processed(self, file_path: Path, record_count: int = 0):
        """Mark a file as processed."""
        file_hash = self._hash_file(file_path)
        filename = file_path.name
        
        self.data['processed_files'][filename] = file_hash
        self.data['last_updated'] = datetime.now().isoformat()
        self.data['statistics']['total_files_processed'] += 1
        self.data['statistics']['total_records_processed'] += record_count
        
        logger.info(f"  ✓ Marked as processed: {filename} ({record_count} records)")
    
    @staticmethod
    def _hash_file(file_path: Path, chunk_size: int = 8192) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()
    
    def get_unprocessed_files(self, directory: Path) -> List[Path]:
        """Get all unprocessed PDF files in directory."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        unprocessed = []
        for file_path in sorted(directory.glob('*.pdf')):
            if not self.is_processed(file_path):
                unprocessed.append(file_path)
        
        return unprocessed


# ============================================================================
# PDF PROCESSING
# ============================================================================

def process_new_pdfs(
    pdf_files: List[Path],
    output_dir: Path,
    workers: int,
    case_mode: str,
    remove_stopwords: bool,
    remove_punctuation: bool,
) -> tuple:
    """
    Process new PDF files through the pipeline.
    
    Args:
        pdf_files: List of new PDF file paths
        output_dir: Output directory
        workers: Number of parallel workers
        case_mode: Case normalization mode
        remove_stopwords: Whether to remove stopwords
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        (preprocessed_records, statistics) tuple
    """
    # Load pipeline modules when needed
    lazy_load_modules()
    
    logger.info(f"\nProcessing {len(pdf_files)} new PDF files...")
    
    # Step 1: Extract PDFs
    logger.info("  Step 1: Extracting text from PDFs...")
    
    # Create temporary directory for PDF processor output
    temp_extract_dir = output_dir / '_temp_extraction'
    temp_extract_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        processor = PDFBatchProcessor(
            input_dir=str(pdf_files[0].parent),  # Use parent directory of first PDF
            output_dir=str(temp_extract_dir),
            max_workers=workers,
        )
        
        # Process only the new files
        result = processor.process()
        pdf_records = result.get('records', [])
        
        logger.info(f"  ✓ Extracted {len(pdf_records)} records from PDFs")
        
        # Step 2: Preprocess text
        logger.info("  Step 2: Preprocessing extracted text...")
        
        # Create a dummy temp file just to initialize the preprocessor
        temp_input = temp_extract_dir / "_temp_input.jsonl"
        with open(temp_input, 'w') as f:
            f.write("{}\n")
        
        try:
            preprocess_processor = ComprehensiveJSONLPreprocessor(
                input_file=str(temp_input),
                output_file=None,
                case_mode=case_mode,
                remove_stopwords=remove_stopwords,
                remove_punctuation=remove_punctuation,
            )
        finally:
            temp_input.unlink()
        
        preprocessed_records = []
        for record in pdf_records:
            record_dict = {
                'filename': record.filename,
                'text': record.text,
                'word_count': record.word_count,
            }
            
            # Preprocess the record
            preprocessed = preprocess_processor.preprocess_record(record_dict)
            preprocessed_records.append(preprocessed)
        
        logger.info(f"  ✓ Preprocessed {len(preprocessed_records)} records")
        
        stats = {
            'pdf_files_processed': len(pdf_files),
            'records_extracted': len(pdf_records),
            'records_preprocessed': len(preprocessed_records),
        }
        
        return preprocessed_records, stats
        
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        import traceback
        traceback.print_exc()
        return [], {'error': str(e)}
    
    finally:
        # Cleanup temporary directory
        if temp_extract_dir.exists():
            import shutil
            shutil.rmtree(temp_extract_dir)


def save_incremental_data(records: List, output_dir: Path, timestamp: str) -> Path:
    """Save incrementally processed data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL with timestamp
    filename = f'{IncrementalConfig.INCREMENTAL_PREFIX}_{timestamp}.jsonl'
    output_file = output_dir / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            # Handle both FullPreprocessedRecord objects and dicts
            if hasattr(record, 'to_dict'):
                data = record.to_dict()
            elif isinstance(record, dict):
                data = record
            else:
                data = vars(record) if hasattr(record, '__dict__') else record
            
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logger.info(f"✓ Saved {len(records)} preprocessed records to {output_file}")
    return output_file


def append_to_master_dataset(incremental_file: Path, master_file: Path) -> int:
    """Append incremental data to master dataset file."""
    if not incremental_file.exists():
        logger.warning(f"Incremental file not found: {incremental_file}")
        return 0
    
    # Ensure master file exists
    master_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Append new data
    with open(incremental_file, 'r', encoding='utf-8') as inc_f:
        lines = inc_f.readlines()
    
    with open(master_file, 'a', encoding='utf-8') as master_f:
        master_f.writelines(lines)
    
    logger.info(f"✓ Appended {len(lines)} lines to master dataset: {master_file}")
    return len(lines)


def create_new_pdf_directory():
    """Create the pdfs/new directory structure if it doesn't exist."""
    config = IncrementalConfig()
    config.NEW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create README
    readme_content = """# New PDF Folder

Add your new PDF files here for incremental processing.

## Usage

1. Place your new PDFs in this directory
2. Run: `python3 incremental_preprocess.py`
3. Processing will:
   - Detect new/modified PDFs
   - Extract text from only new PDFs (skip already-processed)
   - Preprocess the text
   - Append results to output/full_preprocessed.jsonl
   - Update processing manifest

## Benefits

- **Fast**: Only process new PDFs (not all PDFs every time)
- **Safe**: Tracks all processed PDFs to avoid duplicates
- **Flexible**: Add PDFs incrementally without retraining from scratch
- **Transparent**: Manifest file shows what has been processed

## Manifest

The `output/processing_manifest.json` tracks:
- Which PDFs have been processed
- File hashes (detects modifications)
- Processing statistics
- Last update time

To reset and reprocess all PDFs:
```
python3 incremental_preprocess.py --reset-manifest
```
"""
    
    readme_file = config.NEW_PDF_DIR / 'README.md'
    if not readme_file.exists():
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        logger.info(f"✓ Created README.md in {config.NEW_PDF_DIR}")
    
    logger.info(f"✓ New PDF directory ready: {config.NEW_PDF_DIR}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_incremental_preprocessing(
    config: IncrementalConfig,
    reset_manifest: bool = False,
) -> Dict:
    """Run the incremental preprocessing pipeline."""
    
    print("\n" + "="*80)
    print("INCREMENTAL PDF PREPROCESSING PIPELINE")
    print("="*80)
    
    start_time = time.time()
    
    # Step 0: Initialize manifest
    if reset_manifest:
        logger.warning("Resetting processing manifest - will reprocess all files...")
        if config.MANIFEST_FILE.exists():
            config.MANIFEST_FILE.unlink()
    
    manifest = ProcessingManifest(config.MANIFEST_FILE)
    logger.info(f"Manifest: {config.MANIFEST_FILE}")
    logger.info(f"Previously processed files: {len(manifest.data['processed_files'])}")
    
    # Step 1: Find new PDFs
    logger.info("\nStep 1: Detecting new/modified PDFs...")
    new_pdf_files = manifest.get_unprocessed_files(config.NEW_PDF_DIR)
    
    logger.info(f"  New PDFs found: {len(new_pdf_files)}")
    
    if not new_pdf_files:
        print("\n✅ No new PDFs to process!")
        if manifest.data['last_updated']:
            print(f"   Last processed: {manifest.data['last_updated']}")
        print(f"   Total files processed so far: {manifest.data['statistics']['total_files_processed']}")
        print(f"   Total records processed so far: {manifest.data['statistics']['total_records_processed']}")
        return {
            'status': 'no_changes',
            'message': 'No new PDFs found',
            'files_processed': 0,
            'records_processed': 0,
        }
    
    # Step 2: Process new PDFs
    logger.info("\nStep 2: Processing new PDFs through pipeline...")
    preprocessed_records, process_stats = process_new_pdfs(
        pdf_files=new_pdf_files,
        output_dir=config.OUTPUT_DIR,
        workers=config.WORKERS,
        case_mode=config.CASE_MODE,
        remove_stopwords=config.REMOVE_STOPWORDS,
        remove_punctuation=config.REMOVE_PUNCTUATION,
    )
    
    if 'error' in process_stats:
        logger.error(f"Processing failed: {process_stats['error']}")
        return {
            'status': 'failure',
            'error': process_stats['error'],
        }
    
    if not preprocessed_records:
        logger.warning("No records were produced during processing")
        return {
            'status': 'failure',
            'error': 'No records produced',
        }
    
    # Step 3: Save incremental data
    logger.info("\nStep 3: Saving incremental data...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    incremental_file = save_incremental_data(preprocessed_records, config.OUTPUT_DIR, timestamp)
    
    # Step 4: Append to master dataset
    logger.info("\nStep 4: Appending to master dataset...")
    appended_count = append_to_master_dataset(incremental_file, config.FULL_PREPROCESSED_FILE)
    
    # Step 5: Update manifest
    logger.info("\nStep 5: Updating processing manifest...")
    for pdf_file in new_pdf_files:
        manifest.mark_processed(pdf_file, len(preprocessed_records) // len(new_pdf_files))
    
    manifest.data['statistics']['incremental_runs'] += 1
    manifest.save()
    
    # Calculate time
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("✅ INCREMENTAL PREPROCESSING COMPLETE")
    print("="*80)
    print(f"\nProcessing Results:")
    print(f"  New PDFs processed: {len(new_pdf_files)}")
    print(f"  Records extracted: {process_stats.get('records_extracted', 0)}")
    print(f"  Records preprocessed: {process_stats.get('records_preprocessed', 0)}")
    print(f"  Lines appended to master: {appended_count}")
    print(f"\nOutput Files:")
    print(f"  Incremental: {incremental_file}")
    print(f"  Master: {config.FULL_PREPROCESSED_FILE}")
    print(f"  Manifest: {config.MANIFEST_FILE}")
    print(f"\nStatistics:")
    print(f"  Total files processed (all-time): {manifest.data['statistics']['total_files_processed']}")
    print(f"  Total records processed (all-time): {manifest.data['statistics']['total_records_processed']}")
    print(f"  Total incremental runs: {manifest.data['statistics']['incremental_runs']}")
    print(f"\nTime: {total_time:.2f}s")
    print("="*80 + "\n")
    
    return {
        'status': 'success',
        'total_time': total_time,
        'files_processed': len(new_pdf_files),
        'records_processed': len(preprocessed_records),
        'appended_lines': appended_count,
        'incremental_file': str(incremental_file),
        'master_file': str(config.FULL_PREPROCESSED_FILE),
        'manifest_file': str(config.MANIFEST_FILE),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Incremental PDF preprocessing - process only new PDFs and append to dataset"
    )
    parser.add_argument(
        '--reset-manifest',
        action='store_true',
        help='Reset processing manifest (reprocess all PDFs)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers for PDF processing (default: 4)'
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
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only create pdfs/new directory structure (do not process)'
    )
    
    args = parser.parse_args()
    
    # Setup config
    config = IncrementalConfig()
    config.WORKERS = args.workers
    config.CASE_MODE = args.case_mode
    config.REMOVE_STOPWORDS = args.remove_stopwords
    config.REMOVE_PUNCTUATION = args.remove_punctuation
    
    # Create directory structure
    create_new_pdf_directory()
    
    if args.setup_only:
        print(f"\n✅ Directory structure is ready at {config.NEW_PDF_DIR}")
        print(f"   Add your PDF files and run: python3 incremental_preprocess.py")
        return 0
    
    # Run incremental preprocessing
    result = run_incremental_preprocessing(config, reset_manifest=args.reset_manifest)
    
    # Exit with appropriate code
    return 0 if result['status'] in ['success', 'no_changes'] else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
