"""
Integration tests for data-preparation service
"""

import os
import json
import tempfile
from pathlib import Path
import unittest

from gptmed.data_preparation import (
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor,
    VideoPreprocessor,
    PreprocessingConfig,
    BaseDataPreprocessor,
)


class TestPreprocessingConfig(unittest.TestCase):
    """Test PreprocessingConfig"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def test_config_creation(self):
        """Test creating preprocessing config"""
        config = PreprocessingConfig(
            input_path="./data/raw",
            output_path="./data/processed",
            data_type="text"
        )
        
        self.assertEqual(config.input_path, "./data/raw")
        self.assertEqual(config.output_path, "./data/processed")
        self.assertEqual(config.data_type, "text")
        self.assertEqual(config.batch_size, 32)
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = PreprocessingConfig(
            input_path="./data/raw",
            output_path="./data/processed",
            data_type="text",
            batch_size=64
        )
        
        config_dict = config.to_dict()
        self.assertEqual(config_dict['batch_size'], 64)
        self.assertEqual(config_dict['data_type'], 'text')
    
    def test_config_save_load(self):
        """Test saving and loading config"""
        config_path = os.path.join(self.temp_dir, 'config.json')
        
        config = PreprocessingConfig(
            input_path="./data/raw",
            output_path="./data/processed",
            data_type="text",
            batch_size=64,
            custom_params={'test': 'value'}
        )
        
        config.save(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        loaded_config = PreprocessingConfig.load(config_path)
        self.assertEqual(loaded_config.batch_size, 64)
        self.assertEqual(loaded_config.custom_params['test'], 'value')


class TestTextPreprocessor(unittest.TestCase):
    """Test TextPreprocessor"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PreprocessingConfig(
            input_path="./data/raw",
            output_path="./data/processed",
            data_type="text"
        )
    
    def test_text_validation(self):
        """Test text validation"""
        preprocessor = TextPreprocessor(config=self.config)
        
        # Valid text
        self.assertTrue(preprocessor.validate("This is valid text"))
        
        # Invalid - not string
        self.assertFalse(preprocessor.validate(123))
        
        # Invalid - too short
        self.assertFalse(preprocessor.validate("ab"))
    
    def test_text_cleaning(self):
        """Test text cleaning"""
        preprocessor = TextPreprocessor(config=self.config)
        
        # Test HTML removal
        dirty_text = "This is <b>bold</b> text"
        cleaned = preprocessor.clean(dirty_text)
        self.assertNotIn("<b>", cleaned)
        
        # Test URL removal
        text_with_url = "Check this link http://example.com here"
        cleaned = preprocessor.clean(text_with_url)
        self.assertNotIn("http://", cleaned)
    
    def test_text_normalization(self):
        """Test text normalization"""
        preprocessor = TextPreprocessor(
            config=self.config,
            lowercase=True,
            remove_punctuation=False
        )
        
        text = "This is UPPERCASE text!"
        normalized = preprocessor.normalize(text)
        self.assertTrue(normalized.islower())
    
    def test_text_processing(self):
        """Test full text processing"""
        preprocessor = TextPreprocessor(config=self.config)
        
        raw_text = "This is <b>HTML</b> with URL http://example.com"
        processed = preprocessor.process(raw_text)
        
        self.assertIsNotNone(processed)
        self.assertNotIn("<b>", processed)
        self.assertNotIn("http://", processed)
    
    def test_text_tokenization(self):
        """Test text tokenization"""
        preprocessor = TextPreprocessor(config=self.config)
        
        tokens = preprocessor.tokenize("This is a test sentence")
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_text_stats(self):
        """Test text statistics"""
        preprocessor = TextPreprocessor(config=self.config)
        
        text = "This is a test sentence with multiple words"
        stats = preprocessor.get_text_stats(text)
        
        self.assertIn('word_count', stats)
        self.assertIn('sentence_count', stats)
        self.assertIn('vocabulary_diversity', stats)
        self.assertGreater(stats['word_count'], 0)
    
    def test_batch_processing(self):
        """Test batch processing"""
        preprocessor = TextPreprocessor(config=self.config)
        
        texts = [
            "First text sample",
            "Second text sample",
            "Third text sample",
        ]
        
        results = preprocessor.batch_process(texts)
        self.assertEqual(len(results), 3)
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        preprocessor = TextPreprocessor(config=self.config)
        
        texts = ["Valid text", "Short", "Another valid text"]
        preprocessor.batch_process(texts)
        
        stats = preprocessor.get_statistics()
        self.assertIn('input_count', stats)
        self.assertIn('output_count', stats)
        self.assertIn('errors', stats)
        self.assertEqual(stats['input_count'], 3)


class TestImagePreprocessor(unittest.TestCase):
    """Test ImagePreprocessor"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PreprocessingConfig(
            input_path="./data/raw/images",
            output_path="./data/processed/images",
            data_type="image"
        )
    
    def test_image_preprocessor_init(self):
        """Test image preprocessor initialization"""
        preprocessor = ImagePreprocessor(config=self.config)
        
        self.assertEqual(preprocessor.target_size, (224, 224))
        self.assertTrue(preprocessor.preserve_aspect_ratio)
        self.assertEqual(len(preprocessor.supported_formats), 5)
    
    def test_image_validation_nonexistent(self):
        """Test image validation with nonexistent file"""
        preprocessor = ImagePreprocessor(config=self.config)
        
        # File doesn't exist
        self.assertFalse(preprocessor.validate("/nonexistent/image.jpg"))


class TestAudioPreprocessor(unittest.TestCase):
    """Test AudioPreprocessor"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PreprocessingConfig(
            input_path="./data/raw/audio",
            output_path="./data/processed/audio",
            data_type="audio"
        )
    
    def test_audio_preprocessor_init(self):
        """Test audio preprocessor initialization"""
        preprocessor = AudioPreprocessor(config=self.config)
        
        self.assertEqual(preprocessor.target_sample_rate, 16000)
        self.assertTrue(preprocessor.mono)
        self.assertEqual(len(preprocessor.supported_formats), 5)


class TestVideoPreprocessor(unittest.TestCase):
    """Test VideoPreprocessor"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PreprocessingConfig(
            input_path="./data/raw/videos",
            output_path="./data/processed/videos",
            data_type="video"
        )
    
    def test_video_preprocessor_init(self):
        """Test video preprocessor initialization"""
        preprocessor = VideoPreprocessor(config=self.config)
        
        self.assertEqual(preprocessor.target_fps, 30)
        self.assertEqual(preprocessor.target_resolution, (640, 480))
        self.assertEqual(len(preprocessor.supported_formats), 6)


class TestBasePreprocessor(unittest.TestCase):
    """Test BaseDataPreprocessor"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = PreprocessingConfig(
            input_path="./data/raw",
            output_path=self.temp_dir,
            data_type="text"
        )
    
    def test_output_directory_creation(self):
        """Test that output directory is created"""
        preprocessor = TextPreprocessor(config=self.config)
        self.assertTrue(os.path.exists(self.temp_dir))
    
    def test_statistics_reset(self):
        """Test statistics reset"""
        preprocessor = TextPreprocessor(config=self.config)
        
        # Process some data
        preprocessor.process("Test text")
        stats1 = preprocessor.get_statistics()
        self.assertGreater(stats1['output_count'], 0)
        
        # Reset
        preprocessor.reset_statistics()
        stats2 = preprocessor.get_statistics()
        self.assertEqual(stats2['output_count'], 0)
    
    def test_save_statistics(self):
        """Test saving statistics"""
        preprocessor = TextPreprocessor(config=self.config)
        preprocessor.process("Test text")
        
        stats_path = os.path.join(self.temp_dir, 'stats.json')
        preprocessor.save_statistics(stats_path)
        
        self.assertTrue(os.path.exists(stats_path))
        
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        self.assertIn('input_count', stats)


if __name__ == '__main__':
    unittest.main()
