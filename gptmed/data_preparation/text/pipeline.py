"""
Complete PDF → Tokens → Vocabulary Pipeline

Orchestrates the full preprocessing pipeline:
1. Extract text from PDFs (in-memory)
2. Preprocess text (in-memory)
3. Tokenize (saves merged_tokens.jsonl and token_stats.json)
4. Build Vocabulary (creates vocab.json, token_counts.json, and vocab_info.json)

This is the main entry point for generating training data with complete tokenization and vocabulary information.

Usage - Full pipeline (all PDFs):
    python3 pipeline.py \
        --input-dir ./pdfs \
        --output-dir ./output \
        --tokenizer-method huggingface \
        --tokenizer-model gpt2 \
        --workers 4

Usage - Incremental preprocessing (only new PDFs):
    python3 pipeline.py --incremental-preprocess
"""

import sys
import json
import logging
import time
import torch
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict
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
ParallelJSONLTokenizer = None
SimplifiedTokenizedRecord = None
VocabularyBuilder = None

def lazy_load_modules():
    """Lazily load pipeline modules when needed."""
    global PDFBatchProcessor, PDFRecord, ComprehensiveJSONLPreprocessor, FullPreprocessedRecord
    global ParallelJSONLTokenizer, SimplifiedTokenizedRecord, VocabularyBuilder
    
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
        
        # Load tokenize_jsonl
        spec3 = importlib.util.spec_from_file_location("tokenize_jsonl", Path(__file__).parent / "tokenize_jsonl.py")
        tokenize_module = importlib.util.module_from_spec(spec3)
        spec3.loader.exec_module(tokenize_module)
        ParallelJSONLTokenizer = tokenize_module.ParallelJSONLTokenizer
        SimplifiedTokenizedRecord = tokenize_module.SimplifiedTokenizedRecord
        
        # Load build_vocabulary
        spec4 = importlib.util.spec_from_file_location("build_vocabulary", Path(__file__).parent / "build_vocabulary.py")
        vocab_module = importlib.util.module_from_spec(spec4)
        spec4.loader.exec_module(vocab_module)
        VocabularyBuilder = vocab_module.VocabularyBuilder
        
    except ImportError as e:
        logger.error(f"Failed to load pipeline modules: {e}")
        raise

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

# Load build_vocabulary
spec4 = importlib.util.spec_from_file_location("build_vocabulary", Path(__file__).parent / "build_vocabulary.py")
vocab_module = importlib.util.module_from_spec(spec4)
spec4.loader.exec_module(vocab_module)
VocabularyBuilder = vocab_module.VocabularyBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# INCREMENTAL PREPROCESSING SUPPORT
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
        # Use relative path for consistency (handles nested folders)
        try:
            rel_path = str(file_path.relative_to(file_path.parent.parent))
        except ValueError:
            rel_path = file_path.name
        
        if rel_path in self.data['processed_files']:
            stored_hash = self.data['processed_files'][rel_path]
            return stored_hash == file_hash
        
        return False
    
    def mark_processed(self, file_path: Path, record_count: int = 0):
        """Mark a file as processed."""
        file_hash = self._hash_file(file_path)
        # Use relative path for consistency (handles nested folders)
        try:
            rel_path = str(file_path.relative_to(file_path.parent.parent))
        except ValueError:
            rel_path = file_path.name
        
        self.data['processed_files'][rel_path] = file_hash
        self.data['last_updated'] = datetime.now().isoformat()
        self.data['statistics']['total_files_processed'] += 1
        self.data['statistics']['total_records_processed'] += record_count
        
        logger.info(f"  ✓ Marked as processed: {rel_path} ({record_count} records)")
    
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
        """Get all unprocessed PDF files in directory (including nested folders)."""
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return []
        
        unprocessed = []
        # Use rglob to find PDFs in nested directories
        for file_path in sorted(directory.rglob('*.pdf')):
            if not self.is_processed(file_path):
                unprocessed.append(file_path)
        
        return unprocessed



class EndToEndPipeline:
    """Complete PDF to training data pipeline"""
    
    def __init__(
        self,
        input_dir: str = "./pdfs",
        output_dir: str = "./output",
        tokenizer_method: str = "huggingface",
        tokenizer_model: str = "gpt2",
        workers: int = 10,
        case_mode: str = "lower",
        remove_stopwords: bool = False,
        remove_punctuation: bool = False,
        device: str = "gpu",
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
            device: Processing device (gpu/cpu, default: gpu)
        """
        # Initialize logger FIRST (before device configuration)
        self.logger = logging.getLogger(self.__class__.__name__)
        
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
        
        # Configure device (GPU/CPU) - NOW logger is available
        self.device = None
        self._configure_device(device)
    
    def _configure_device(self, device: str):
        """
        Configure and validate GPU/CPU device
        
        Args:
            device: Device preference (gpu/cpu)
        """
        device = device.lower()
        
        if device == "gpu":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                self.logger.info(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
                self.logger.info(f"  Available GPUs: {gpu_count}")
                self.logger.info(f"  CUDA Version: {torch.version.cuda}")
                self.device = "cuda"
            else:
                self.logger.warning("GPU requested but not available. Falling back to CPU.")
                self.device = "cpu"
        elif device == "cpu":
            self.logger.info("Using CPU for processing")
            self.device = "cpu"
        else:
            self.logger.warning(f"Unknown device '{device}'. Using GPU if available, else CPU.")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def step1_extract_pdfs(self) -> List[PDFRecord]:
        """Step 1: Extract text from PDFs in-memory"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 1: PDF EXTRACTION")
        self.logger.info("="*70)
        self.logger.info(f"Processing Device: {self.device.upper()}")
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
    
    def step3_tokenize(self, preprocessed_records: List[FullPreprocessedRecord] = None) -> Dict[str, Any]:
        """Step 3: Tokenize preprocessed text
        
        Args:
            preprocessed_records: Records to tokenize (from step 2)
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 3: TOKENIZATION")
        self.logger.info("="*70)
        
        if preprocessed_records is None or len(preprocessed_records) == 0:
            raise ValueError("preprocessed_records required and cannot be empty")
        
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
        if temp_file.exists():
            temp_file.unlink()
        
        return result
    
    def step4_build_vocabulary(self, tokenization_result: Dict[str, Any]) -> Dict[str, Any]:
        """Step 4: Build vocabulary from tokenized data"""
        self.logger.info("\n" + "="*70)
        self.logger.info("STEP 4: VOCABULARY BUILDING")
        self.logger.info("="*70)
        
        try:
            # Get path to merged tokens
            tokens_dir = self.output_dir / "tokens"
            merged_tokens_file = tokens_dir / "merged_tokens.jsonl"
            
            if not merged_tokens_file.exists():
                self.logger.warning(f"Merged tokens file not found: {merged_tokens_file}")
                return {'status': 'failure', 'message': 'Merged tokens file not found'}
            
            # Build vocabulary
            builder = VocabularyBuilder(str(merged_tokens_file), str(tokens_dir))
            builder.build()
            builder.save()
            builder.print_summary()
            
            # Prepare vocabulary summary
            vocab_info = {
                'total_unique_tokens': len(builder.id_to_token),
                'total_token_instances': sum(builder.token_frequency.values()),
                'token_id_range': [
                    int(min(builder.token_frequency.keys())),
                    int(max(builder.token_frequency.keys()))
                ] if builder.token_frequency else [0, 0],
                'gpt2_vocab_enabled': bool(builder.gpt2_vocab),
                'top_tokens': [
                    {
                        'token_id': token_id,
                        'frequency': freq,
                        'label': builder.id_to_token[token_id]
                    }
                    for token_id, freq in builder.token_frequency.most_common(10)
                ]
            }
            
            return {
                'status': 'success',
                'vocabulary_info': vocab_info,
                'vocab_file': str(tokens_dir / 'vocab.json'),
                'token_counts_file': str(tokens_dir / 'token_counts.json'),
                'vocab_info_file': str(tokens_dir / 'vocab_info.json'),
            }
            
        except Exception as e:
            self.logger.error(f"Error building vocabulary: {str(e)}")
            return {'status': 'failure', 'message': str(e)}
    
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
            
            if tokenization_result.get('status') != 'success':
                self.logger.error("Tokenization failed. Exiting.")
                return {'status': 'failure', 'message': 'Tokenization failed'}
            
            # Step 4: Build Vocabulary
            vocabulary_result = self.step4_build_vocabulary(tokenization_result)
            
            total_time = time.time() - start_time
            
            # Final summary
            self.logger.info("\n" + "="*70)
            self.logger.info("PIPELINE COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"\nFinal Outputs:")
            self.logger.info(f"  1. {self.output_dir}/full_preprocessed.jsonl (cleaned text)")
            self.logger.info(f"  2. {self.output_dir}/tokens/merged_tokens.jsonl (training tokens)")
            self.logger.info(f"  3. {self.output_dir}/tokens/token_stats.json (token & text summary)")
            self.logger.info(f"  4. {self.output_dir}/tokens/vocab.json (vocabulary mapping)")
            self.logger.info(f"  5. {self.output_dir}/tokens/token_counts.json (token frequencies)")
            self.logger.info(f"  6. {self.output_dir}/tokens/vocab_info.json (vocabulary metadata)")
            
            self.logger.info(f"\nToken Summary Statistics:")
            if tokenization_result.get('status') == 'success':
                self.logger.info(f"  Total tokens: {tokenization_result.get('total_tokens', 0):,}")
                self.logger.info(f"  Total records: {tokenization_result.get('total_records', 0)}")
                self.logger.info(f"  Average tokens per file: {tokenization_result.get('average_tokens_per_record', 0):.1f}")
            
            self.logger.info(f"\nVocabulary Statistics:")
            if vocabulary_result.get('status') == 'success':
                vocab_info = vocabulary_result.get('vocabulary_info', {})
                self.logger.info(f"  Total unique tokens: {vocab_info.get('total_unique_tokens', 0):,}")
                self.logger.info(f"  Total token instances: {vocab_info.get('total_token_instances', 0):,}")
                token_range = vocab_info.get('token_id_range', [0, 0])
                self.logger.info(f"  Token ID range: {token_range[0]} - {token_range[1]}")
                self.logger.info(f"  GPT2 vocabulary enabled: {vocab_info.get('gpt2_vocab_enabled', False)}")
            
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
                'vocabulary': vocabulary_result,
                'output_dir': str(self.output_dir),
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'status': 'failure', 'error': str(e)}
    
    def incremental_run(self, reset_manifest: bool = False) -> Dict[str, Any]:
        """Execute incremental preprocessing (only new PDFs from pdfs/new/)"""
        start_time = time.time()
        
        self.logger.info("\n")
        self.logger.info("╔" + "="*68 + "╗")
        self.logger.info("║" + " "*15 + "INCREMENTAL PDF PREPROCESSING" + " "*25 + "║")
        self.logger.info("╚" + "="*68 + "╝")
        
        # Setup directories
        new_pdf_dir = Path(self.input_dir) / "new"
        master_file = Path(self.output_dir) / "full_preprocessed.jsonl"
        manifest_file = Path(self.output_dir) / "processing_manifest.json"
        
        # Initialize manifest
        if reset_manifest and manifest_file.exists():
            self.logger.warning("Resetting processing manifest - will reprocess all files...")
            manifest_file.unlink()
        
        manifest = ProcessingManifest(manifest_file)
        self.logger.info(f"Manifest: {manifest_file}")
        self.logger.info(f"Previously processed files: {len(manifest.data['processed_files'])}")
        
        # Find new PDFs
        self.logger.info("\nStep 1: Detecting new/modified PDFs...")
        new_pdf_files = manifest.get_unprocessed_files(new_pdf_dir)
        self.logger.info(f"  New PDFs found: {len(new_pdf_files)}")
        
        if not new_pdf_files:
            self.logger.info("\n✅ No new PDFs to process!")
            if manifest.data['last_updated']:
                self.logger.info(f"   Last processed: {manifest.data['last_updated']}")
            self.logger.info(f"   Total files processed so far: {manifest.data['statistics']['total_files_processed']}")
            self.logger.info(f"   Total records processed so far: {manifest.data['statistics']['total_records_processed']}")
            return {
                'status': 'no_changes',
                'message': 'No new PDFs found',
                'files_processed': 0,
                'records_processed': 0,
            }
        
        try:
            # Step 2: Extract new PDFs
            self.logger.info("\nStep 2: Extracting text from new PDFs...")
            temp_extract_dir = Path(self.output_dir) / '_temp_extraction'
            temp_extract_dir.mkdir(parents=True, exist_ok=True)
            
            processor = PDFBatchProcessor(
                input_dir=str(new_pdf_dir),
                output_dir=str(temp_extract_dir),
                max_workers=self.workers,
            )
            
            result = processor.process()
            pdf_records = result.get('records', [])
            self.logger.info(f"  ✓ Extracted {len(pdf_records)} records from {len(new_pdf_files)} PDFs")
            
            if not pdf_records:
                self.logger.warning("No records extracted from new PDFs")
                return {
                    'status': 'failure',
                    'error': 'No records extracted from PDFs',
                }
            
            # Step 3: Preprocess
            self.logger.info("\nStep 3: Preprocessing extracted text...")
            
            temp_input = temp_extract_dir / "_temp_input.jsonl"
            with open(temp_input, 'w') as f:
                f.write("{}\n")
            
            try:
                preprocess_processor = ComprehensiveJSONLPreprocessor(
                    input_file=str(temp_input),
                    output_file=None,
                    case_mode=self.case_mode,
                    remove_stopwords=self.remove_stopwords,
                    remove_punctuation=self.remove_punctuation,
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
                preprocessed = preprocess_processor.preprocess_record(record_dict)
                preprocessed_records.append(preprocessed)
            
            self.logger.info(f"  ✓ Preprocessed {len(preprocessed_records)} records")
            
            # Step 4: Save incremental data
            self.logger.info("\nStep 4: Saving incremental data...")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            incremental_file = Path(self.output_dir) / f"incremental_{timestamp}.jsonl"
            
            with open(incremental_file, 'w', encoding='utf-8') as f:
                for record in preprocessed_records:
                    json.dump(asdict(record), f, ensure_ascii=False)
                    f.write('\n')
            
            self.logger.info(f"  ✓ Saved incremental data: {incremental_file.name}")
            
            # Step 5: Append to master file
            self.logger.info("\nStep 5: Appending to master dataset...")
            with open(incremental_file, 'r', encoding='utf-8') as inc_f:
                lines = inc_f.readlines()
            
            with open(master_file, 'a', encoding='utf-8') as master_f:
                master_f.writelines(lines)
            
            self.logger.info(f"  ✓ Appended {len(lines)} records to {master_file.name}")
            
            # Step 6: Backup old tokens before tokenization overwrites them
            self.logger.info("\nStep 6: Backing up existing tokens...")
            old_tokens_file = None
            tokens_dir = Path(self.output_dir) / "tokens"
            merged_tokens_path = tokens_dir / "merged_tokens.jsonl"
            old_tokens_backup = tokens_dir / "_temp_old_merged_tokens.jsonl"
            
            if merged_tokens_path.exists():
                import shutil
                shutil.copy(merged_tokens_path, old_tokens_backup)
                old_tokens_file = old_tokens_backup
                self.logger.info(f"  ✓ Backed up existing tokens ({merged_tokens_path.stat().st_size / 1024 / 1024:.1f}MB)")
            else:
                self.logger.info("  ℹ No existing tokens to back up (first run)")
            
            # Step 6.5: Tokenize new preprocessed records
            self.logger.info("\nStep 6.5: Tokenizing new preprocessed records...")
            try:
                tokenization_result = self.step3_tokenize(preprocessed_records)
                self.logger.info(f"  ✓ Tokenization complete - {tokenization_result.get('total_tokens', 0):,} new tokens created")
            except Exception as e:
                self.logger.warning(f"  ⚠ Tokenization failed: {e}")
                tokenization_result = None
            
            # Step 6.7: Merge new tokens with old tokens
            self.logger.info("\nStep 6.7: Merging new tokens with existing tokens...")
            try:
                if old_tokens_file and old_tokens_file.exists() and merged_tokens_path.exists():
                    # Read new tokens (just generated)
                    with open(merged_tokens_path, 'r', encoding='utf-8') as f:
                        new_lines = f.readlines()
                    
                    # Read old tokens
                    with open(old_tokens_file, 'r', encoding='utf-8') as f:
                        old_lines = f.readlines()
                    
                    # Merge: old + new
                    with open(merged_tokens_path, 'w', encoding='utf-8') as f:
                        f.writelines(old_lines)
                        f.writelines(new_lines)
                    
                    old_lines_count = len(old_lines)
                    new_lines_count = len(new_lines)
                    total_lines = old_lines_count + new_lines_count
                    
                    self.logger.info(f"  ✓ Merged: {old_lines_count:,} old + {new_lines_count:,} new = {total_lines:,} total tokens")
                    
                    # Update tokenization result to reflect total
                    if tokenization_result:
                        tokenization_result['merged_old_new'] = True
                        tokenization_result['old_token_count'] = old_lines_count
                        tokenization_result['new_token_count'] = new_lines_count
                        tokenization_result['total_tokens'] = total_lines
                    
                    # Clean up temp old file
                    old_tokens_file.unlink()
                elif old_tokens_file is None:
                    self.logger.info("  ℹ First tokenization run - no merge needed")
                    
            except Exception as e:
                self.logger.warning(f"  ⚠ Token merging failed: {e}")
                # Don't fail, continue with vocabulary building
            
            # Step 7: Rebuild vocabulary from all tokens (including new ones)
            try:
                if tokenization_result:
                    vocabulary_result = self.step4_build_vocabulary(tokenization_result)
                    self.logger.info(f"  ✓ Vocabulary rebuilt - {vocabulary_result.get('vocabulary_info', {}).get('total_unique_tokens', 0):,} unique tokens")
                else:
                    vocabulary_result = None
            except Exception as e:
                self.logger.warning(f"  ⚠ Vocabulary building failed: {e}")
                vocabulary_result = None
            
            # Step 9: Update manifest
            self.logger.info("\nStep 9: Updating processing manifest...")
            for pdf_file in new_pdf_files:
                manifest.mark_processed(pdf_file, len(preprocessed_records) // len(new_pdf_files) if new_pdf_files else 0)
            
            manifest.data['statistics']['incremental_runs'] += 1
            manifest.save()
            
            # Cleanup
            import shutil
            if temp_extract_dir.exists():
                shutil.rmtree(temp_extract_dir)
            
            total_time = time.time() - start_time
            
            # Summary
            self.logger.info("\n" + "="*70)
            self.logger.info("✅ INCREMENTAL PREPROCESSING COMPLETE")
            self.logger.info("="*70)
            self.logger.info(f"\nProcessing Results:")
            self.logger.info(f"  New PDFs processed: {len(new_pdf_files)}")
            self.logger.info(f"  Records extracted: {len(pdf_records)}")
            self.logger.info(f"  Records preprocessed: {len(preprocessed_records)}")
            self.logger.info(f"  Records appended to master: {len(lines)}")
            self.logger.info(f"\nOutput Files:")
            self.logger.info(f"  Incremental: {incremental_file}")
            self.logger.info(f"  Master: {master_file}")
            self.logger.info(f"  Manifest: {manifest_file}")
            self.logger.info(f"\nStatistics:")
            self.logger.info(f"  Total files processed (all-time): {manifest.data['statistics']['total_files_processed']}")
            self.logger.info(f"  Total records processed (all-time): {manifest.data['statistics']['total_records_processed']}")
            self.logger.info(f"  Total incremental runs: {manifest.data['statistics']['incremental_runs']}")
            self.logger.info(f"\nTime: {total_time:.2f}s")
            self.logger.info("="*70)
            self.logger.info(f"\nProcessing Results:")
            self.logger.info(f"  New PDFs processed: {len(new_pdf_files)}")
            self.logger.info(f"  Records extracted: {len(pdf_records)}")
            self.logger.info(f"  Records preprocessed: {len(preprocessed_records)}")
            self.logger.info(f"  Records appended to master: {len(lines)}")
            
            # Log tokenization info if available
            if tokenization_result and tokenization_result.get('status') == 'success':
                self.logger.info(f"\nTokenization Results:")
                self.logger.info(f"  Total tokens: {tokenization_result.get('total_tokens', 0):,}")
                self.logger.info(f"  Token file: {tokenization_result.get('output_file', 'N/A')}")
            
            # Log vocabulary info if available
            if vocabulary_result and vocabulary_result.get('status') == 'success':
                vocab_info = vocabulary_result.get('vocabulary_info', {})
                self.logger.info(f"\nVocabulary Results:")
                self.logger.info(f"  Total unique tokens: {vocab_info.get('total_unique_tokens', 0):,}")
                self.logger.info(f"  Total token instances: {vocab_info.get('total_token_instances', 0):,}")
                self.logger.info(f"  Vocabulary file: {vocabulary_result.get('vocab_file', 'N/A')}")
            
            self.logger.info(f"\nOutput Files:")
            self.logger.info(f"  Incremental: {incremental_file}")
            self.logger.info(f"  Master: {master_file}")
            self.logger.info(f"  Manifest: {manifest_file}")
            self.logger.info(f"\nStatistics:")
            self.logger.info(f"  Total files processed (all-time): {manifest.data['statistics']['total_files_processed']}")
            self.logger.info(f"  Total records processed (all-time): {manifest.data['statistics']['total_records_processed']}")
            self.logger.info(f"  Total incremental runs: {manifest.data['statistics']['incremental_runs']}")
            self.logger.info(f"\nTime: {total_time:.2f}s")
            self.logger.info("="*70 + "\n")
            
            return {
                'status': 'success',
                'total_time': total_time,
                'files_processed': len(new_pdf_files),
                'records_processed': len(preprocessed_records),
                'appended_lines': len(lines),
                'incremental_file': str(incremental_file),
                'master_file': str(master_file),
                'manifest_file': str(manifest_file),
                'tokenization': tokenization_result,
                'vocabulary': vocabulary_result,
            }
        
        except Exception as e:
            self.logger.error(f"Incremental preprocessing failed: {str(e)}")
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
    parser.add_argument(
        '--device',
        default='gpu',
        choices=['gpu', 'cpu'],
        help='Processing device - gpu or cpu (default: gpu)'
    )
    parser.add_argument(
        '--incremental-preprocess',
        action='store_true',
        help='Run incremental preprocessing (new PDFs only from pdfs/new/)'
    )
    parser.add_argument(
        '--reset-manifest',
        action='store_true',
        help='Reset processing manifest (reprocess all new PDFs)'
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
        device=args.device,
    )
    
    # Run appropriate mode
    if args.incremental_preprocess:
        result = pipeline.incremental_run(reset_manifest=args.reset_manifest)
    else:
        result = pipeline.run()
    
    # Exit with appropriate code
    return 0 if result['status'] in ['success', 'no_changes'] else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
