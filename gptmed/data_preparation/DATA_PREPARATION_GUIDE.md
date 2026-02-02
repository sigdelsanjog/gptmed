# Data Preparation Service

A comprehensive data preprocessing and cleaning framework integrated into the gptmed package. Provides modular support for preparing **text, image, audio, and video** data as a preprocessing baseline for the aggregated machine learning framework.

## Features

### Text Preprocessing

- **Text Cleaning**: Remove HTML, URLs, emails, special characters
- **Normalization**: Case conversion, punctuation handling, Unicode normalization
- **Tokenization**: Word-level tokenization with stopword removal
- **Validation**: Length checks and content validation
- **Statistics**: Text metrics (word count, unique words, diversity metrics)
- **Batch Processing**: Process single files or entire directories

### Image Preprocessing

- **Format Validation**: Support for JPG, PNG, BMP, WebP, etc.
- **Resizing**: With aspect ratio preservation or explicit resizing
- **Normalization**: Automatic format conversion to RGB
- **Quality Checks**: Min/max size constraints
- **Metadata Extraction**: Resolution, format, size information
- **Batch Directory Processing**: Process multiple images efficiently

### Audio Preprocessing

- **Format Support**: WAV, MP3, FLAC, OGG, M4A
- **Resampling**: Convert to target sample rate (default: 16kHz)
- **Mono Conversion**: Stereo to mono conversion
- **Silence Removal**: Optional silence detection and removal
- **Amplitude Normalization**: Peak and loudness normalization
- **Duration Validation**: Min/max duration constraints
- **Comprehensive Metadata**: Sample rate, duration, channels, bitrate

### Video Preprocessing

- **Format Support**: MP4, AVI, MOV, MKV, FLV, WMV
- **Frame Extraction**: Extract frames at specified intervals
- **Resolution Management**: Resize with aspect ratio preservation
- **Quality Validation**: Min/max resolution checks
- **Duration Validation**: Min/max duration constraints
- **Metadata Extraction**: FPS, resolution, duration, bitrate

## Installation

The data-preparation service is included in the gptmed package. Install with optional dependencies:

```bash
# Basic installation
pip install gptmed

# With data preparation support
pip install gptmed[data-preparation]

# Full installation with all extras
pip install gptmed[data-preparation,training,visualization,xai]
```

### Optional Dependencies

For enhanced functionality, install these separately:

```bash
# For image processing
pip install pillow>=9.0.0

# For audio processing
pip install librosa>=0.10.0 soundfile>=0.12.0

# For video processing
pip install opencv-python>=4.5.0

# For advanced audio (optional)
pip install librosa[all]
```

## CLI Usage

### Text Processing

```bash
# Basic text cleaning
data-preparation text \
  --input ./data/raw/text \
  --output ./data/processed/text

# Advanced text cleaning
data-preparation text \
  --input ./data/raw/text \
  --output ./data/processed/text \
  --lowercase \
  --remove-stopwords \
  --remove-punctuation \
  --min-length 5 \
  --max-length 1000 \
  --save-stats

# Process single file
data-preparation text \
  --input ./document.txt \
  --output ./processed/ \
  --lowercase
```

### Image Processing

```bash
# Basic image resizing
data-preparation image \
  --input ./data/raw/images \
  --output ./data/processed/images

# Advanced image processing
data-preparation image \
  --input ./data/raw/images \
  --output ./data/processed/images \
  --target-size 224 224 \
  --preserve-aspect \
  --output-format jpg \
  --quality 95
```

### Audio Processing

```bash
# Basic audio resampling
data-preparation audio \
  --input ./data/raw/audio \
  --output ./data/processed/audio

# Advanced audio processing
data-preparation audio \
  --input ./data/raw/audio \
  --output ./data/processed/audio \
  --target-sample-rate 16000 \
  --mono \
  --remove-silence \
  --min-duration 0.5 \
  --output-format wav
```

### Video Processing

```bash
# Basic video validation
data-preparation video \
  --input ./data/raw/videos \
  --output ./data/processed/videos

# Extract frames
data-preparation video \
  --input ./data/raw/videos \
  --output ./data/processed/videos \
  --extract-frames \
  --frame-sample-rate 30 \
  --target-fps 30 \
  --target-resolution 640 480
```

## Python API Usage

### Text Preprocessing

```python
from gptmed.data_preparation import TextPreprocessor, PreprocessingConfig

# Create config
config = PreprocessingConfig(
    input_path="./data/raw/text",
    output_path="./data/processed/text",
    data_type="text"
)

# Initialize preprocessor
preprocessor = TextPreprocessor(
    config=config,
    lowercase=True,
    remove_stopwords=False,
    remove_punctuation=False,
    min_length=3
)

# Process single text
cleaned_text = preprocessor.process("Raw text with HTML <b>tags</b> and URLs http://example.com")

# Get statistics
stats = preprocessor.get_text_stats(cleaned_text)
print(f"Word count: {stats['word_count']}")
print(f"Vocabulary diversity: {stats['vocabulary_diversity']:.2%}")

# Process entire directory
results = preprocessor.batch_process_files(
    "./data/raw/text",
    "./data/processed/text",
    pattern="*.txt"
)
```

### Image Preprocessing

```python
from gptmed.data_preparation import ImagePreprocessor, PreprocessingConfig
from PIL import Image

config = PreprocessingConfig(
    input_path="./data/raw/images",
    output_path="./data/processed/images",
    data_type="image"
)

preprocessor = ImagePreprocessor(
    config=config,
    target_size=(224, 224),
    preserve_aspect_ratio=True
)

# Process single image
if preprocessor.validate("path/to/image.jpg"):
    img = Image.open("path/to/image.jpg")
    processed = preprocessor.process(img)
    processed.save("output.jpg")

# Process directory
results = preprocessor.batch_process_directory(
    "./data/raw/images",
    "./data/processed/images",
    output_format="jpg",
    quality=95
)
print(results['stats'])
```

### Audio Preprocessing

```python
from gptmed.data_preparation import AudioPreprocessor, PreprocessingConfig

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
    min_duration=0.5
)

# Process directory
results = preprocessor.batch_process_directory(
    "./data/raw/audio",
    "./data/processed/audio",
    output_format="wav"
)

# Get statistics
stats = preprocessor.get_audio_stats("path/to/audio.wav")
print(f"Duration: {stats['duration_seconds']:.2f}s")
print(f"Sample rate: {stats['sample_rate']}Hz")
```

### Video Preprocessing

```python
from gptmed.data_preparation import VideoPreprocessor, PreprocessingConfig

config = PreprocessingConfig(
    input_path="./data/raw/videos",
    output_path="./data/processed/videos",
    data_type="video"
)

preprocessor = VideoPreprocessor(
    config=config,
    target_fps=30,
    target_resolution=(640, 480),
    min_duration=1.0
)

# Process directory with frame extraction
results = preprocessor.batch_process_directory(
    "./data/raw/videos",
    "./data/processed/videos",
    extract_frames=True,
    frame_sample_rate=30
)

# Get video statistics
stats = preprocessor.get_video_stats("path/to/video.mp4")
print(f"Resolution: {stats['resolution']}")
print(f"Duration: {stats['duration_seconds']:.2f}s")
```

## Advanced Usage

### Batch Processing Pipeline

```python
from gptmed.data_preparation import (
    TextPreprocessor, ImagePreprocessor,
    AudioPreprocessor, VideoPreprocessor,
    PreprocessingConfig
)

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
    # ... more configs
}

# Initialize all preprocessors
preprocessors = {
    'text': TextPreprocessor(config=configs['text'], lowercase=True),
    'image': ImagePreprocessor(config=configs['image']),
    'audio': AudioPreprocessor(config=configs['audio']),
    'video': VideoPreprocessor(config=configs['video']),
}

# Process all data types
results = {}
for data_type, preprocessor in preprocessors.items():
    results[data_type] = preprocessor.batch_process_directory(
        configs[data_type].input_path,
        configs[data_type].output_path
    )

# Save all statistics
for data_type, result in results.items():
    stats_path = f"./stats/{data_type}_stats.json"
    preprocessors[data_type].save_statistics(stats_path)
```

### Custom Preprocessing Configuration

```python
from gptmed.data_preparation import TextPreprocessor, PreprocessingConfig

# Create custom config
config = PreprocessingConfig(
    input_path="./data/raw",
    output_path="./data/processed",
    data_type="text",
    save_format="json",
    batch_size=64,
    num_workers=4,
    verbose=True,
    validation_split=0.1,
    test_split=0.1,
    random_seed=42,
    custom_params={
        'language': 'english',
        'domain': 'medical',
    }
)

# Save config for reproducibility
config.save("./configs/preprocessing_config.json")

# Load config later
loaded_config = PreprocessingConfig.load("./configs/preprocessing_config.json")

preprocessor = TextPreprocessor(config=loaded_config)
```

### Statistics and Monitoring

```python
preprocessor = TextPreprocessor(config=config)

# Process data
results = preprocessor.batch_process_files("./data/raw", "./data/processed")

# Get statistics
stats = preprocessor.get_statistics()
print(f"Input count: {stats['input_count']}")
print(f"Output count: {stats['output_count']}")
print(f"Errors: {stats['errors']}")
print(f"Skipped: {stats['skipped']}")

# Save statistics
preprocessor.save_statistics("./stats/processing_stats.json")

# Reset for next batch
preprocessor.reset_statistics()
```

## Architecture

### Directory Structure

```
gptmed/
├── data_preparation/
│   ├── __init__.py                 # Main module exports
│   ├── base.py                     # Base classes and configs
│   ├── cli.py                      # CLI interface
│   ├── text/
│   │   └── __init__.py            # Text preprocessor
│   ├── image/
│   │   └── __init__.py            # Image preprocessor
│   ├── audio/
│   │   └── __init__.py            # Audio preprocessor
│   └── video/
│       └── __init__.py            # Video preprocessor
```

### Base Architecture

All preprocessors inherit from `BaseDataPreprocessor` which provides:

- **Validation**: Check if data is valid for processing
- **Cleaning**: Remove artifacts and normalize format
- **Normalization**: Standardize data properties
- **Batch Processing**: Process multiple items efficiently
- **Statistics Tracking**: Monitor processing metrics
- **Configuration Management**: Save/load configs

## Configuration

### PreprocessingConfig

```python
@dataclass
class PreprocessingConfig:
    input_path: str                 # Input data path
    output_path: str                # Output data path
    data_type: str                  # 'text', 'image', 'audio', 'video'
    save_format: str = "json"       # Output format
    batch_size: int = 32            # Batch size
    num_workers: int = 4            # Number of workers
    verbose: bool = True            # Verbose output
    validation_split: float = 0.1   # Validation split
    test_split: float = 0.1         # Test split
    random_seed: int = 42           # Random seed
    custom_params: Dict[str, Any] = {}  # Custom parameters
```

## Error Handling

All preprocessors handle errors gracefully:

```python
preprocessor = TextPreprocessor(config=config)

# Process with error handling
result = preprocessor.process(potentially_invalid_data)

if result is None:
    print(f"Processing failed. Errors: {preprocessor.get_statistics()['errors']}")
else:
    print(f"Processing successful: {result}")
```

## Best Practices

1. **Validate Configuration**: Always validate your config before processing
2. **Save Statistics**: Keep track of processing metrics for reproducibility
3. **Batch Processing**: Use batch processing for better performance
4. **Memory Management**: Process large files in batches
5. **Error Logging**: Enable verbose mode for debugging
6. **Test First**: Test on small data samples before processing large datasets

## Performance Tips

- Use batch processing for large datasets
- Enable multi-worker processing (num_workers > 1)
- Adjust batch_size based on available memory
- Use preserve_aspect_ratio=True for faster image processing
- Extract only necessary frames from videos

## Troubleshooting

### Missing Dependencies

If you get import errors, install optional dependencies:

```bash
# For specific data type
pip install gptmed[data-preparation]

# Or install individually
pip install pillow librosa soundfile opencv-python
```

### Memory Issues

Reduce batch size:

```python
config = PreprocessingConfig(
    batch_size=8,  # Smaller batch size
    num_workers=1  # Fewer workers
)
```

### Encoding Issues

Specify encoding explicitly:

```python
with open(file_path, 'r', encoding='utf-8-sig') as f:
    text = f.read()
```

## Contributing

To extend the data-preparation service:

1. Inherit from `BaseDataPreprocessor`
2. Implement required methods: `validate()`, `clean()`, `normalize()`
3. Add CLI integration in `cli.py`
4. Update documentation

## License

MIT License - See LICENSE file for details

## Support

For issues and feature requests, visit: https://github.com/sigdelsanjog/gptmed/issues
