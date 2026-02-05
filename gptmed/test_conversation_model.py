"""
Test script for Conversation Language Model

Tests the trained model with proper tokenization integration using actual token vocabulary.
"""

import sys
from pathlib import Path
import torch
import logging
import json
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from framework.conversation.inference.inference import ConversationInference, InferenceConfig
from framework.conversation.model.architecture import ConversationLanguageModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VocabularyTokenizer:
    """
    Tokenizer that loads vocabulary from vocab.json
    
    This tokenizer:
    1. Loads vocabulary mappings from vocab.json (created by build_vocabulary.py)
    2. Encodes new text using vocabulary + hash fallback
    3. Decodes output properly using the learned vocabulary
    4. Clamps token IDs to model's vocabulary size (10000 by default)
    """
    
    def __init__(self, vocab_file: str = None, data_file: str = None, vocab_size: int = 10000, model_vocab_size: int = 10000):
        """
        Initialize tokenizer
        
        Args:
            vocab_file: Path to vocab.json (created by build_vocabulary.py)
            data_file: Path to merged_tokens.jsonl (for backup/metadata)
            vocab_size: Size of loaded vocabulary
            model_vocab_size: Actual model vocab size (default: 10000) - token IDs are clamped to this range
        """
        self.vocab_size = vocab_size
        self.model_vocab_size = model_vocab_size  # Model's actual embedding vocab size
        self.id_to_word = {}
        self.word_to_id = {}
        self.token_frequency = Counter()
        
        # Try to load from vocab.json first
        if vocab_file and Path(vocab_file).exists():
            self._load_vocab_from_json(vocab_file)
            logger.info(f"✓ Loaded vocabulary from {vocab_file}")
        # Fall back to building from data file
        elif data_file and Path(data_file).exists():
            self._build_vocab_from_data(data_file)
            logger.info(f"✓ Built vocabulary from {data_file}")
        else:
            logger.warning(f"Could not find vocab.json or data file. Using default vocabulary.")
            self._build_default_vocab()
    
    def _load_vocab_from_json(self, vocab_file: str):
        """Load vocabulary from vocab.json"""
        logger.info(f"Loading vocabulary from {vocab_file}...")
        
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # vocab.json has string keys (token IDs as strings) mapping to token text
        for token_id_str, token_text in vocab_data.items():
            try:
                token_id = int(token_id_str)
                self.id_to_word[token_id] = token_text
                # Also create reverse mapping for encoding
                if token_text and not token_text.startswith('token_'):
                    self.word_to_id[token_text] = token_id
            except (ValueError, TypeError):
                continue
        
        logger.info(f"✓ Loaded {len(self.id_to_word)} tokens from vocabulary")
    
    def _build_vocab_from_data(self, data_file: str):
        """Build vocabulary from actual training tokens in merged_tokens.jsonl"""
        logger.info(f"Building vocabulary from {data_file}...")
        
        all_tokens = []
        
        # Read all tokens from JSONL file
        with open(data_file, 'r') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    if 'tokens' in record and isinstance(record['tokens'], list):
                        all_tokens.extend(record['tokens'])
                except json.JSONDecodeError:
                    continue
        
        # Count token frequencies
        self.token_frequency = Counter(all_tokens)
        
        # Create bidirectional mapping
        # Map tokens by frequency (most common = lower IDs)
        for token_id, count in self.token_frequency.most_common(self.vocab_size):
            if 0 <= token_id < self.vocab_size:
                self.id_to_word[token_id] = f'token_{token_id}'
                self.word_to_id[f'token_{token_id}'] = token_id
        
        logger.info(f"✓ Built vocabulary with {len(self.id_to_word)} tokens")
        logger.info(f"  Token frequency range: {min(self.token_frequency.values())} - {max(self.token_frequency.values())}")
    
    def _build_default_vocab(self):
        """Build simple default vocabulary"""
        # Medical terminology
        words = [
            'patient', 'treatment', 'diagnosis', 'disease', 'symptom',
            'medication', 'hospital', 'doctor', 'nurse', 'care',
            'cancer', 'heart', 'brain', 'surgery', 'therapy',
            'blood', 'organ', 'infection', 'virus', 'drug',
            'pain', 'fever', 'cough', 'breath', 'pressure',
            'health', 'condition', 'severe', 'mild',
            'what', 'is', 'the', 'how', 'why', 'when', 'where',
            'can', 'i', 'you', 'we', 'they', 'have', 'has', 'had',
            'do', 'does', 'did', 'be', 'been', 'should', 'would', 'could',
        ]
        
        for idx, word in enumerate(words):
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word
        
        for idx in range(len(words), self.vocab_size):
            self.id_to_word[idx] = f'unk_{idx}'
    
    def encode(self, text: str) -> list:
        """
        Encode text to token IDs
        
        Uses hash-based fallback for unknown words to ensure model compatibility.
        Clamps all token IDs to model's vocabulary size.
        """
        tokens = []
        words = text.lower().split()
        
        for word in words:
            # Clean word
            word = ''.join(c for c in word if c.isalnum())
            
            if word in self.word_to_id:
                token_id = self.word_to_id[word]
            else:
                # Use hash for unknown words
                token_id = hash(word) % self.vocab_size
            
            # Clamp to model's vocabulary size
            token_id = min(token_id, self.model_vocab_size - 1) if self.model_vocab_size else token_id
            tokens.append(token_id)
        
        return tokens if tokens else [1]
    
    def decode(self, token_ids: list) -> str:
        """
        Decode token IDs to text
        
        Uses vocabulary mapping to reconstruct meaningful text.
        Clamps token IDs to model's vocabulary size to avoid out-of-bounds errors.
        """
        words = []
        for token_id in token_ids:
            if isinstance(token_id, float):
                token_id = int(token_id)
            
            # Clamp token ID to model's vocabulary size
            clamped_id = min(token_id, self.model_vocab_size - 1) if self.model_vocab_size else token_id
            
            if clamped_id in self.id_to_word:
                word = self.id_to_word[clamped_id]
                # Skip placeholder tokens and special markers
                if word and not word.startswith('<') and word not in ['unk_', 'pad']:
                    words.append(word)
            else:
                # If token ID not in vocab, show it as unknown
                if clamped_id < self.vocab_size:
                    words.append(f'<unk_{clamped_id}>')
        
        # Join words and clean up spacing issues (e.g., from BPE tokens with Ġ prefix)
        result = ' '.join(words) if words else '[no output]'
        # Remove special BPE markers (Ġ represents space in BPE)
        result = result.replace('Ġ', ' ').strip()
        return result


def test_model():
    """Test the trained conversation model"""
    
    checkpoint_dir = Path(__file__).parent / 'framework' / 'conversation' / 'model' / 'checkpoints'
    
    # Build paths to data files
    data_file_candidates = [
        Path(__file__).parent / 'data' / 'conversation' / 'merged_tokens.jsonl',
        Path(__file__).parent / 'framework' / 'conversation' / 'data' / 'merged_tokens.jsonl',
        Path('data/conversation/merged_tokens.jsonl'),
    ]
    
    vocab_file_candidates = [
        Path(__file__).parent / 'data' / 'conversation' / 'vocabularies' / 'vocab.json',
        Path(__file__).parent / 'framework' / 'conversation' / 'data' / 'vocabularies' / 'vocab.json',
        Path('data/conversation/vocabularies/vocab.json'),
    ]
    
    data_file = None
    vocab_file = None
    
    for candidate in vocab_file_candidates:
        if candidate.exists():
            vocab_file = str(candidate)
            logger.info(f"Found vocabulary file: {vocab_file}")
            break
    
    for candidate in data_file_candidates:
        if candidate.exists():
            data_file = str(candidate)
            if not vocab_file:
                logger.info(f"Found data file: {data_file}")
            break
    
    if not vocab_file and not data_file:
        logger.warning("Could not find vocab.json or merged_tokens.jsonl")
        logger.warning("Expected vocab.json locations:")
        for candidate in vocab_file_candidates:
            logger.warning(f"  - {candidate}")
        logger.warning("Expected data file locations:")
        for candidate in data_file_candidates:
            logger.warning(f"  - {candidate}")
    
    logger.info("="*70)
    logger.info("CONVERSATION LANGUAGE MODEL TEST")
    logger.info("="*70)
    
    # Load model
    logger.info("\nLoading trained model...")
    config = InferenceConfig(
        checkpoint_dir=str(checkpoint_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_length=50,
        temperature=0.7,
        top_k=10,
    )
    
    try:
        inference = ConversationInference(config)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Setup tokenizer with actual vocabulary
    logger.info("\nSetting up tokenizer...")
    inference.tokenizer = VocabularyTokenizer(
        vocab_file=vocab_file,
        data_file=data_file
    )
    logger.info("✓ Tokenizer ready")
    
    # Test prompts
    test_prompts = [
        "What is treatment",
        "How to diagnose disease",
        "Patient symptoms",
        "Cancer therapy",
        "Heart surgery",
    ]
    
    logger.info("\n" + "="*70)
    logger.info("TESTING WITH SAMPLE PROMPTS")
    logger.info("="*70)
    
    for prompt in test_prompts:
        logger.info(f"\n{'─'*70}")
        logger.info(f"Input: {prompt}")
        logger.info(f"{'─'*70}")
        
        try:
            # Generate response
            response = inference.chat(
                prompt,
                max_tokens=30,
                temperature=0.7,
            )
            
            logger.info(f"Output: {response}")
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            continue
    
    # Interactive mode
    logger.info("\n" + "="*70)
    logger.info("INTERACTIVE MODE")
    logger.info("="*70)
    logger.info("Type 'quit' to exit\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            if not prompt:
                continue
            
            response = inference.chat(
                prompt,
                max_tokens=30,
                temperature=0.7,
            )
            
            print(f"BCI: {response}\n")
        
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue


if __name__ == '__main__':
    test_model()
