# Data-Preparation Service Module

A comprehensive data preprocessing and cleaning framework for text, image, audio, and video data.

## Quick Start

### Installation

```bash
# Install with data-preparation support
pip install gptmed[data-preparation]
```

### CLI Usage

```bash
# Text preprocessing
data-preparation text --input ./raw --output ./processed --lowercase

# Image preprocessing
data-preparation image --input ./raw/images --output ./processed/images

# Audio preprocessing
data-preparation audio --input ./raw/audio --output ./processed/audio --target-sample-rate 16000

# Video preprocessing
data-preparation video --input ./raw/videos --output ./processed/videos --extract-frames
```

### Python API

```python
from gptmed.data_preparation import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned = preprocessor.process("Raw text with HTML <b>tags</b>")
print(cleaned)
```

## Module Structure

```
data_preparation/
├── __init__.py              # Main module exports
├── base.py                  # Base classes and configuration
├── cli.py                   # CLI interface
├── text/                    # Text preprocessing
│   └── __init__.py
├── image/                   # Image preprocessing
│   └── __init__.py
├── audio/                   # Audio preprocessing
│   └── __init__.py
└── video/                   # Video preprocessing
    └── __init__.py
```

## Features

### Text Processing

- HTML/URL/email removal
- Unicode and whitespace normalization
- Case conversion
- Punctuation handling
- Stopword removal
- Tokenization
- Text statistics

### Image Processing

- Format validation and conversion
- Resizing with aspect ratio preservation
- Quality checks
- Metadata extraction
- Batch processing

### Audio Processing

- Resampling to target sample rate
- Mono conversion
- Amplitude normalization
- Silence removal
- Duration validation
- Comprehensive metadata

### Video Processing

- Frame extraction
- Resolution management
- Duration validation
- Metadata extraction
- Batch processing

## Documentation

- **DATA_PREPARATION_GUIDE.md** - Comprehensive usage guide
- **DATA_PREPARATION_QUICK_REFERENCE.md** - Quick reference cheatsheet
- **examples/data_preparation_examples.py** - Usage examples

## Architecture

### BaseDataPreprocessor

All preprocessors inherit from this abstract base class:

```python
class BaseDataPreprocessor(ABC):
    def validate(self, data) -> bool
    def clean(self, data) -> Any
    def normalize(self, data) -> Any
    def process(self, data) -> Any
    def batch_process(self, data_list) -> List[Any]
    def get_statistics(self) -> Dict
    def save_statistics(self, path) -> None
```

### PreprocessingConfig

Unified configuration for all preprocessors:

```python
config = PreprocessingConfig(
    input_path="./data/raw",
    output_path="./data/processed",
    data_type="text",
    batch_size=32,
    num_workers=4,
    verbose=True
)
```

## Supported Formats

| Data Type | Formats                            |
| --------- | ---------------------------------- |
| Text      | .txt, .md, .json, .csv             |
| Image     | .jpg, .jpeg, .png, .bmp, .webp     |
| Audio     | .wav, .mp3, .flac, .ogg, .m4a      |
| Video     | .mp4, .avi, .mov, .mkv, .flv, .wmv |

## Examples

### Basic Text Processing

```python
from gptmed.data_preparation import TextPreprocessor, PreprocessingConfig

config = PreprocessingConfig(
    input_path="./data/raw/text",
    output_path="./data/processed/text",
    data_type="text"
)

preprocessor = TextPreprocessor(
    config=config,
    lowercase=True,
    remove_stopwords=False
)

# Process single text
text = "Check http://example.com and email@test.com here"
cleaned = preprocessor.process(text)
print(cleaned)  # Output: "Check and here"

# Process directory
results = preprocessor.batch_process_files("./data/raw/text")
print(preprocessor.get_statistics())
```

### Image Processing

```python
from gptmed.data_preparation import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(224, 224))

# Process directory
results = preprocessor.batch_process_directory(
    "./data/raw/images",
    "./data/processed/images"
)
```

### Audio Processing

```python
from gptmed.data_preparation import AudioPreprocessor

preprocessor = AudioPreprocessor(
    target_sample_rate=16000,
    mono=True
)

# Process directory
preprocessor.batch_process_directory(
    "./data/raw/audio",
    "./data/processed/audio"
)
```

### Video Processing

```python
from gptmed.data_preparation import VideoPreprocessor

preprocessor = VideoPreprocessor(extract_frames=True)

# Process directory with frame extraction
preprocessor.batch_process_directory(
    "./data/raw/videos",
    "./data/processed/videos"
)
```

## Configuration Management

```python
from gptmed.data_preparation import PreprocessingConfig

# Create and save config
config = PreprocessingConfig(
    input_path="./data/raw",
    output_path="./data/processed",
    data_type="text",
    batch_size=64,
    custom_params={'language': 'english'}
)
config.save("./config.json")

# Load config later
loaded_config = PreprocessingConfig.load("./config.json")
```

## Statistics and Monitoring

```python
preprocessor = TextPreprocessor(config)

# Process data
results = preprocessor.batch_process(data_list)

# Get statistics
stats = preprocessor.get_statistics()
print(f"Processed: {stats['output_count']}")
print(f"Errors: {stats['errors']}")

# Save statistics
preprocessor.save_statistics("./stats.json")
```

## Error Handling

All preprocessors handle errors gracefully:

```python
preprocessor = TextPreprocessor(config)

result = preprocessor.process(potentially_invalid_data)

if result is None:
    # Processing failed
    stats = preprocessor.get_statistics()
    print(f"Errors: {stats['errors']}")
else:
    # Processing succeeded
    print(f"Result: {result}")
```

## Performance Tips

1. Use batch processing for large datasets
2. Adjust `batch_size` based on available memory
3. Set `num_workers > 1` for parallel processing
4. Use `preserve_aspect_ratio=True` for faster image processing
5. Extract only necessary frames from videos
6. Enable verbose mode for debugging

## Requirements

### Core

- `numpy>=1.24.0`
- `pyyaml>=6.0`

### Optional (for data-preparation)

- `pillow>=9.0.0` - Image processing
- `librosa>=0.10.0` - Audio processing
- `soundfile>=0.12.0` - Audio I/O
- `opencv-python>=4.5.0` - Video processing

## Testing

Run tests:

```bash
pytest tests/test_data_preparation.py
```

Test coverage:

- Configuration management
- Text preprocessing
- Input validation
- Error handling
- Statistics tracking
- Batch processing

## Extending the Framework

To add a new data type:

1. Create module in `data_preparation/{type}/`
2. Inherit from `BaseDataPreprocessor`
3. Implement `validate()`, `clean()`, `normalize()`
4. Add CLI support in `cli.py`
5. Add tests in `tests/test_data_preparation.py`

Example:

```python
from .base import BaseDataPreprocessor, PreprocessingConfig

class CustomPreprocessor(BaseDataPreprocessor):
    def validate(self, data):
        # Validation logic
        return True

    def clean(self, data):
        # Cleaning logic
        return cleaned_data

    def normalize(self, data):
        # Normalization logic
        return normalized_data
```

## Troubleshooting

### Module Not Found

```bash
pip install gptmed[data-preparation]
```

### Missing Dependencies

```bash
# Install all optional dependencies
pip install pillow librosa soundfile opencv-python
```

### Memory Issues

```python
config = PreprocessingConfig(batch_size=8, num_workers=1)
```

### Encoding Issues

```python
with open(file_path, 'r', encoding='utf-8-sig') as f:
    text = f.read()
```

## License

MIT License - See LICENSE file

## Support

For issues and feature requests:
https://github.com/sigdelsanjog/gptmed/issues

## See Also

- [DATA_PREPARATION_GUIDE.md](DATA_PREPARATION_GUIDE.md) - Comprehensive guide
- [DATA_PREPARATION_QUICK_REFERENCE.md](DATA_PREPARATION_QUICK_REFERENCE.md) - Quick reference
- [examples/data_preparation_examples.py](examples/data_preparation_examples.py) - Usage examples
