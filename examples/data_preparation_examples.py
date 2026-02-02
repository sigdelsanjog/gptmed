"""
Example usage of the data-preparation service

This script demonstrates how to use all preprocessors for different data types.
Run this to see the data preparation framework in action.
"""

import logging
from pathlib import Path

from gptmed.data_preparation import (
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor,
    VideoPreprocessor,
    PreprocessingConfig,
)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_text_preprocessing():
    """Example: Text preprocessing"""
    logger.info("=" * 60)
    logger.info("TEXT PREPROCESSING EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/text",
        output_path="./data/processed/text",
        data_type="text"
    )
    
    preprocessor = TextPreprocessor(
        config=config,
        lowercase=True,
        remove_stopwords=False,
        remove_punctuation=False,
        min_length=3
    )
    
    # Example raw texts
    raw_texts = [
        "This is a sample text with HTML <b>tags</b> and URLs http://example.com",
        "Email addresses like test@example.com should be removed",
        "Multiple   spaces   should   be   normalized",
        "Special characters like @#$% are cleaned",
        "UPPERCASE TEXT should be converted to lowercase",
    ]
    
    logger.info("\nProcessing sample texts...")
    for i, text in enumerate(raw_texts, 1):
        logger.info(f"\n--- Text {i} ---")
        logger.info(f"Raw: {text}")
        
        processed = preprocessor.process(text)
        logger.info(f"Cleaned: {processed}")
        
        stats = preprocessor.get_text_stats(text)
        logger.info(f"Stats: {stats}")
    
    logger.info(f"\nOverall statistics: {preprocessor.get_statistics()}")


def example_image_preprocessing():
    """Example: Image preprocessing (requires PIL)"""
    logger.info("\n" + "=" * 60)
    logger.info("IMAGE PREPROCESSING EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/images",
        output_path="./data/processed/images",
        data_type="image"
    )
    
    preprocessor = ImagePreprocessor(
        config=config,
        target_size=(224, 224),
        preserve_aspect_ratio=True,
        min_size=(32, 32),
        max_size=(4096, 4096)
    )
    
    logger.info("\nImage Preprocessor initialized with:")
    logger.info(f"  - Target size: {preprocessor.target_size}")
    logger.info(f"  - Preserve aspect ratio: {preprocessor.preserve_aspect_ratio}")
    logger.info(f"  - Supported formats: {preprocessor.supported_formats}")
    
    # Note: Requires actual image files to process
    logger.info("\nTo process images, run:")
    logger.info("  preprocessor.batch_process_directory('./data/raw/images')")


def example_audio_preprocessing():
    """Example: Audio preprocessing (requires librosa)"""
    logger.info("\n" + "=" * 60)
    logger.info("AUDIO PREPROCESSING EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/audio",
        output_path="./data/processed/audio",
        data_type="audio"
    )
    
    preprocessor = AudioPreprocessor(
        config=config,
        target_sample_rate=16000,
        mono=True,
        remove_silence=False,
        min_duration=0.5,
        max_duration=None
    )
    
    logger.info("\nAudio Preprocessor initialized with:")
    logger.info(f"  - Target sample rate: {preprocessor.target_sample_rate} Hz")
    logger.info(f"  - Mono conversion: {preprocessor.mono}")
    logger.info(f"  - Silence removal: {preprocessor.remove_silence}")
    logger.info(f"  - Min duration: {preprocessor.min_duration}s")
    logger.info(f"  - Supported formats: {preprocessor.supported_formats}")
    
    # Note: Requires actual audio files to process
    logger.info("\nTo process audio, run:")
    logger.info("  preprocessor.batch_process_directory('./data/raw/audio')")


def example_video_preprocessing():
    """Example: Video preprocessing (requires OpenCV)"""
    logger.info("\n" + "=" * 60)
    logger.info("VIDEO PREPROCESSING EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/videos",
        output_path="./data/processed/videos",
        data_type="video"
    )
    
    preprocessor = VideoPreprocessor(
        config=config,
        target_fps=30,
        target_resolution=(640, 480),
        preserve_aspect_ratio=True,
        min_duration=1.0,
        extract_frames=False
    )
    
    logger.info("\nVideo Preprocessor initialized with:")
    logger.info(f"  - Target FPS: {preprocessor.target_fps}")
    logger.info(f"  - Target resolution: {preprocessor.target_resolution}")
    logger.info(f"  - Preserve aspect ratio: {preprocessor.preserve_aspect_ratio}")
    logger.info(f"  - Min duration: {preprocessor.min_duration}s")
    logger.info(f"  - Supported formats: {preprocessor.supported_formats}")
    
    # Note: Requires actual video files to process
    logger.info("\nTo process videos, run:")
    logger.info("  preprocessor.batch_process_directory('./data/raw/videos')")


def example_configuration_management():
    """Example: Configuration management"""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION MANAGEMENT EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw",
        output_path="./data/processed",
        data_type="text",
        batch_size=64,
        num_workers=4,
        verbose=True,
        validation_split=0.1,
        test_split=0.1,
        custom_params={
            'language': 'english',
            'domain': 'medical',
        }
    )
    
    logger.info("\nConfiguration created:")
    for key, value in config.to_dict().items():
        logger.info(f"  {key}: {value}")
    
    # Save configuration
    config_path = "./configs/example_config.json"
    logger.info(f"\nSaving configuration to {config_path}")
    config.save(config_path)
    
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    loaded_config = PreprocessingConfig.load(config_path)
    logger.info(f"Loaded config data_type: {loaded_config.data_type}")


def example_batch_processing_pipeline():
    """Example: Multi-data-type batch processing pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("BATCH PROCESSING PIPELINE EXAMPLE")
    logger.info("=" * 60)
    
    # Create configs for all data types
    configs = {
        'text': PreprocessingConfig(
            input_path="./data/raw/text",
            output_path="./data/processed/text",
            data_type="text"
        ),
        'image': PreprocessingConfig(
            input_path="./data/raw/images",
            output_path="./data/processed/images",
            data_type="image"
        ),
        'audio': PreprocessingConfig(
            input_path="./data/raw/audio",
            output_path="./data/processed/audio",
            data_type="audio"
        ),
        'video': PreprocessingConfig(
            input_path="./data/raw/videos",
            output_path="./data/processed/videos",
            data_type="video"
        ),
    }
    
    # Initialize all preprocessors
    preprocessors = {
        'text': TextPreprocessor(config=configs['text'], lowercase=True),
        'image': ImagePreprocessor(config=configs['image']),
        'audio': AudioPreprocessor(config=configs['audio']),
        'video': VideoPreprocessor(config=configs['video']),
    }
    
    logger.info("\nInitialized preprocessors for all data types:")
    for data_type, preprocessor in preprocessors.items():
        logger.info(f"  - {data_type}: {preprocessor.__class__.__name__}")
    
    logger.info("\nTo process all data types, run:")
    logger.info("  for data_type, preprocessor in preprocessors.items():")
    logger.info("    preprocessor.batch_process_directory(...)")


def main():
    """Run all examples"""
    logger.info("\n" + "=" * 60)
    logger.info("DATA PREPARATION SERVICE - USAGE EXAMPLES")
    logger.info("=" * 60)
    
    # Run examples
    example_text_preprocessing()
    example_image_preprocessing()
    example_audio_preprocessing()
    example_video_preprocessing()
    example_configuration_management()
    example_batch_processing_pipeline()
    
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLES COMPLETE")
    logger.info("=" * 60)
    logger.info("\nFor more information, see: DATA_PREPARATION_GUIDE.md")
    logger.info("For CLI usage, run: data-preparation --help")


if __name__ == '__main__':
    main()
