"""
Build Vocabulary from Tokenized Data

Creates intelligent vocabulary mappings that combine:
1. GPT2's actual token texts for known tokens
2. Descriptive names for frequent tokens
3. Helpful labels for custom/unknown tokens

This vocabulary builder solves the issue where training data only contains 
token IDs (not text) by mapping them to actual GPT2 token texts.

Vocabulary Output Formats:
1. vocab.json - Token ID to token text mapping (used for decoding)
2. token_counts.json - Token frequency statistics
3. vocab_info.json - Vocabulary metadata

Usage:
    python3 build_vocabulary.py \
        --input-file ./output/tokens/merged_tokens.jsonl \
        --output-dir ./output/vocabularies
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter
from argparse import ArgumentParser

# Adjust path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """Build intelligent vocabulary with GPT2 token mappings"""
    
    def __init__(self, merged_tokens_file: str, output_dir: str):
        """
        Initialize vocabulary builder
        
        Args:
            merged_tokens_file: Path to merged_tokens.jsonl with token IDs
            output_dir: Output directory for vocabulary files
        """
        self.merged_tokens_file = Path(merged_tokens_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.id_to_token = {}
        self.token_frequency = Counter()
        self.gpt2_vocab = None
        self.gpt2_id_to_token = {}
        
        # Load GPT2 vocabulary for reference
        self._load_gpt2_vocab()
        
        # Load and analyze training tokens
        self._analyze_training_tokens()
    
    def _load_gpt2_vocab(self):
        """Load GPT2 vocabulary for intelligent token mapping"""
        try:
            from transformers import GPT2Tokenizer
            logger.info("Loading GPT2 tokenizer...")
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_vocab = tokenizer.get_vocab()
            self.gpt2_id_to_token = {v: k for k, v in self.gpt2_vocab.items()}
            logger.info(f"✓ Loaded GPT2 vocabulary with {len(self.gpt2_vocab)} tokens")
        except Exception as e:
            logger.warning(f"Could not load GPT2 vocab (will use fallback): {e}")
            self.gpt2_vocab = {}
            self.gpt2_id_to_token = {}
    
    def _analyze_training_tokens(self):
        """Analyze tokens in training data"""
        if not self.merged_tokens_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.merged_tokens_file}")
        
        logger.info(f"Analyzing tokens from {self.merged_tokens_file.name}...")
        
        all_tokens = []
        with open(self.merged_tokens_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'tokens' in record and isinstance(record['tokens'], list):
                        all_tokens.extend(record['tokens'])
                except json.JSONDecodeError:
                    continue
        
        self.token_frequency = Counter(all_tokens)
        logger.info(f"✓ Found {len(self.token_frequency):,} unique tokens")
        logger.info(f"  Total token instances: {sum(self.token_frequency.values()):,}")
        
        if self.token_frequency:
            min_id = min(self.token_frequency.keys())
            max_id = max(self.token_frequency.keys())
            logger.info(f"  Token ID range: {min_id} - {max_id}")
    
    def _get_token_label(self, token_id: int, frequency: int) -> str:
        """
        Get intelligent label for token ID
        
        Priority:
        1. Use GPT2 token text if available
        2. Use special names for very common punctuation/markers
        3. Generate descriptive fallback label
        """
        # Try GPT2 first
        if self.gpt2_vocab and token_id in self.gpt2_id_to_token:
            return self.gpt2_id_to_token[token_id]
        
        # Default fallback
        return f"<token_{token_id}>"
    
    def build(self):
        """Build the vocabulary"""
        logger.info("\n" + "="*70)
        logger.info("Building Vocabulary")
        logger.info("="*70)
        
        # Create mapping for all tokens found in training
        for token_id, frequency in self.token_frequency.items():
            label = self._get_token_label(token_id, frequency)
            self.id_to_token[token_id] = label
        
        logger.info(f"✓ Created mappings for {len(self.id_to_token):,} tokens")
    
    def save(self):
        """Save vocabulary files"""
        logger.info("\n" + "="*70)
        logger.info("Saving Vocabulary Files")
        logger.info("="*70)
        
        # Save id_to_token mapping (JSON requires string keys)
        vocab_dict = {str(k): v for k, v in self.id_to_token.items()}
        vocab_file = self.output_dir / 'vocab.json'
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(vocab_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved vocab.json ({len(vocab_dict):,} tokens)")
        
        # Save token frequency info
        freq_dict = {str(k): v for k, v in self.token_frequency.items()}
        freq_file = self.output_dir / 'token_counts.json'
        with open(freq_file, 'w', encoding='utf-8') as f:
            json.dump(freq_dict, f, indent=2)
        logger.info(f"✓ Saved token_counts.json")
        
        # Save metadata
        metadata = {
            'total_unique_tokens': len(self.id_to_token),
            'total_token_instances': sum(self.token_frequency.values()),
            'token_id_range': [
                int(min(self.token_frequency.keys())),
                int(max(self.token_frequency.keys()))
            ] if self.token_frequency else [0, 0],
            'gpt2_vocab_enabled': bool(self.gpt2_vocab),
        }
        meta_file = self.output_dir / 'vocab_info.json'
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved vocab_info.json")
    
    def print_summary(self):
        """Print vocabulary summary and examples"""
        logger.info("\n" + "="*70)
        logger.info("Vocabulary Summary")
        logger.info("="*70)
        logger.info(f"Total unique tokens: {len(self.id_to_token):,}")
        logger.info(f"Total token instances: {sum(self.token_frequency.values()):,}")
        
        if self.token_frequency:
            min_id = min(self.token_frequency.keys())
            max_id = max(self.token_frequency.keys())
            logger.info(f"Token ID range: {min_id} - {max_id}")
        
        logger.info(f"GPT2 vocabulary enabled: {bool(self.gpt2_vocab)}")
        
        # Show top tokens
        logger.info(f"\nTop 15 most frequent tokens:")
        for token_id, freq in self.token_frequency.most_common(15):
            label = self.id_to_token[token_id]
            logger.info(f"  {token_id:5d} ({freq:8,} times) → {label}")
        
        logger.info(f"\nOutput files:")
        logger.info(f"  - vocab.json (ID → Text mapping)")
        logger.info(f"  - token_counts.json (Frequency data)")
        logger.info(f"  - vocab_info.json (Metadata)")
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("="*70)


def main():
    parser = ArgumentParser(description='Build vocabulary from tokenized data')
    parser.add_argument(
        '--input-file',
        required=True,
        help='Path to merged_tokens.jsonl'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for vocabulary files'
    )
    
    args = parser.parse_args()
    
    # Build vocabulary
    builder = VocabularyBuilder(args.input_file, args.output_dir)
    builder.build()
    builder.save()
    builder.print_summary()


if __name__ == '__main__':
    main()
