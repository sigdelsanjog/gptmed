"""
CLI interface for data-preparation service

Provides command-line tools for preprocessing and cleaning various data types.

Usage:
    data-preparation text --input data/raw/text --output data/processed/text
    data-preparation image --input data/raw/images --output data/processed/images
    data-preparation audio --input data/raw/audio --output data/processed/audio
    data-preparation video --input data/raw/videos --output data/processed/videos
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Optional, Any

from .base import PreprocessingConfig
from .text import TextPreprocessor
from .image import ImagePreprocessor
from .audio import AudioPreprocessor
from .video import VideoPreprocessor


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparationCLI:
    """CLI handler for data preparation tasks"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            prog='data-preparation',
            description='Data preprocessing and cleaning toolkit for text, image, audio, and video',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Text preprocessing
  data-preparation text \\
    --input ./data/raw/text \\
    --output ./data/processed/text \\
    --lowercase \\
    --remove-stopwords

  # Image preprocessing
  data-preparation image \\
    --input ./data/raw/images \\
    --output ./data/processed/images \\
    --target-size 224 224 \\
    --batch-size 32

  # Audio preprocessing
  data-preparation audio \\
    --input ./data/raw/audio \\
    --output ./data/processed/audio \\
    --target-sample-rate 16000 \\
    --mono

  # Video preprocessing
  data-preparation video \\
    --input ./data/raw/videos \\
    --output ./data/processed/videos \\
    --extract-frames \\
    --frame-sample-rate 30
            """
        )
        
        # Global arguments
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--version',
            action='version',
            version='data-preparation 0.1.0'
        )
        
        # Subcommands
        subparsers = parser.add_subparsers(dest='command', help='Data type to process')
        
        # Text preprocessing
        text_parser = subparsers.add_parser('text', help='Text data preprocessing')
        self._add_text_arguments(text_parser)
        
        # Image preprocessing
        image_parser = subparsers.add_parser('image', help='Image data preprocessing')
        self._add_image_arguments(image_parser)
        
        # Audio preprocessing
        audio_parser = subparsers.add_parser('audio', help='Audio data preprocessing')
        self._add_audio_arguments(audio_parser)
        
        # Video preprocessing
        video_parser = subparsers.add_parser('video', help='Video data preprocessing')
        self._add_video_arguments(video_parser)
        
        return parser
    
    def _add_text_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add text preprocessing arguments"""
        parser.add_argument('--input', required=True, help='Input text file or directory')
        parser.add_argument('--output', required=True, help='Output directory')
        parser.add_argument('--lowercase', action='store_true', help='Convert to lowercase')
        parser.add_argument('--remove-stopwords', action='store_true', help='Remove stopwords')
        parser.add_argument('--remove-punctuation', action='store_true', help='Remove punctuation')
        parser.add_argument('--min-length', type=int, default=3, help='Minimum text length')
        parser.add_argument('--max-length', type=int, help='Maximum text length')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch processing size')
        parser.add_argument('--save-stats', action='store_true', help='Save processing statistics')
        parser.set_defaults(func=self.process_text)
    
    def _add_image_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add image preprocessing arguments"""
        parser.add_argument('--input', required=True, help='Input image directory')
        parser.add_argument('--output', required=True, help='Output directory')
        parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224],
                          help='Target image size (height width)')
        parser.add_argument('--preserve-aspect', action='store_true', default=True,
                          help='Preserve aspect ratio')
        parser.add_argument('--output-format', default='jpg', help='Output image format')
        parser.add_argument('--quality', type=int, default=95, help='JPEG quality (0-100)')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch processing size')
        parser.set_defaults(func=self.process_image)
    
    def _add_audio_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add audio preprocessing arguments"""
        parser.add_argument('--input', required=True, help='Input audio directory')
        parser.add_argument('--output', required=True, help='Output directory')
        parser.add_argument('--target-sample-rate', type=int, default=16000,
                          help='Target sample rate (Hz)')
        parser.add_argument('--mono', action='store_true', help='Convert to mono')
        parser.add_argument('--remove-silence', action='store_true', help='Remove silence')
        parser.add_argument('--min-duration', type=float, default=0.5,
                          help='Minimum audio duration (seconds)')
        parser.add_argument('--output-format', default='wav', help='Output audio format')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch processing size')
        parser.set_defaults(func=self.process_audio)
    
    def _add_video_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add video preprocessing arguments"""
        parser.add_argument('--input', required=True, help='Input video directory')
        parser.add_argument('--output', required=True, help='Output directory')
        parser.add_argument('--target-fps', type=int, default=30, help='Target frames per second')
        parser.add_argument('--target-resolution', type=int, nargs=2, default=[640, 480],
                          help='Target resolution (width height)')
        parser.add_argument('--extract-frames', action='store_true',
                          help='Extract frames from videos')
        parser.add_argument('--frame-sample-rate', type=int, default=30,
                          help='Extract every Nth frame')
        parser.add_argument('--min-duration', type=float, default=1.0,
                          help='Minimum video duration (seconds)')
        parser.add_argument('--batch-size', type=int, default=32, help='Batch processing size')
        parser.set_defaults(func=self.process_video)
    
    def process_text(self, args: argparse.Namespace) -> int:
        """Process text data"""
        try:
            logger.info("Starting text preprocessing...")
            
            config = PreprocessingConfig(
                input_path=args.input,
                output_path=args.output,
                data_type='text',
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            
            preprocessor = TextPreprocessor(
                config=config,
                remove_stopwords=args.remove_stopwords,
                remove_punctuation=args.remove_punctuation,
                lowercase=args.lowercase,
                min_length=args.min_length,
                max_length=args.max_length,
            )
            
            input_path = Path(args.input)
            
            # Process single file or directory
            if input_path.is_file():
                with open(input_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                results = preprocessor.batch_process([text])
                logger.info(f"Processed text. Output: {args.output}")
                
            elif input_path.is_dir():
                results = preprocessor.batch_process_files(args.input, args.output)
                logger.info(f"Processed directory: {args.input}")
            else:
                logger.error(f"Input path not found: {args.input}")
                return 1
            
            # Save statistics if requested
            if args.save_stats:
                stats_file = Path(args.output) / 'processing_stats.json'
                preprocessor.save_statistics(str(stats_file))
                logger.info(f"Statistics saved to {stats_file}")
            
            logger.info("Text preprocessing complete!")
            return 0
            
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            return 1
    
    def process_image(self, args: argparse.Namespace) -> int:
        """Process image data"""
        try:
            logger.info("Starting image preprocessing...")
            
            config = PreprocessingConfig(
                input_path=args.input,
                output_path=args.output,
                data_type='image',
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            
            preprocessor = ImagePreprocessor(
                config=config,
                target_size=tuple(args.target_size),
                preserve_aspect_ratio=args.preserve_aspect,
            )
            
            results = preprocessor.batch_process_directory(
                args.input,
                args.output,
                output_format=args.output_format,
                quality=args.quality,
            )
            
            logger.info(f"Processed images from {args.input}")
            logger.info(f"Results: {results['stats']}")
            logger.info("Image preprocessing complete!")
            return 0
            
        except Exception as e:
            logger.error(f"Error in image preprocessing: {str(e)}")
            return 1
    
    def process_audio(self, args: argparse.Namespace) -> int:
        """Process audio data"""
        try:
            logger.info("Starting audio preprocessing...")
            
            config = PreprocessingConfig(
                input_path=args.input,
                output_path=args.output,
                data_type='audio',
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            
            preprocessor = AudioPreprocessor(
                config=config,
                target_sample_rate=args.target_sample_rate,
                mono=args.mono,
                remove_silence=args.remove_silence,
                min_duration=args.min_duration,
            )
            
            results = preprocessor.batch_process_directory(
                args.input,
                args.output,
                output_format=args.output_format,
            )
            
            logger.info(f"Processed audio from {args.input}")
            logger.info(f"Results: {results['stats']}")
            logger.info("Audio preprocessing complete!")
            return 0
            
        except Exception as e:
            logger.error(f"Error in audio preprocessing: {str(e)}")
            return 1
    
    def process_video(self, args: argparse.Namespace) -> int:
        """Process video data"""
        try:
            logger.info("Starting video preprocessing...")
            
            config = PreprocessingConfig(
                input_path=args.input,
                output_path=args.output,
                data_type='video',
                batch_size=args.batch_size,
                verbose=args.verbose,
            )
            
            preprocessor = VideoPreprocessor(
                config=config,
                target_fps=args.target_fps,
                target_resolution=tuple(args.target_resolution),
                min_duration=args.min_duration,
            )
            
            results = preprocessor.batch_process_directory(
                args.input,
                args.output,
                extract_frames=args.extract_frames,
                frame_sample_rate=args.frame_sample_rate,
            )
            
            logger.info(f"Processed videos from {args.input}")
            logger.info(f"Results: {results['stats']}")
            logger.info("Video preprocessing complete!")
            return 0
            
        except Exception as e:
            logger.error(f"Error in video preprocessing: {str(e)}")
            return 1
    
    def run(self, args: Optional[list] = None) -> int:
        """Run CLI"""
        parsed_args = self.parser.parse_args(args)
        
        if not hasattr(parsed_args, 'func'):
            self.parser.print_help()
            return 1
        
        return parsed_args.func(parsed_args)


def main():
    """Main entry point"""
    cli = DataPreparationCLI()
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
