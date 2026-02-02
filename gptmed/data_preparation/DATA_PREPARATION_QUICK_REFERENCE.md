# Data-Preparation Quick Reference

## Installation

```bash
# Install with data-preparation support
pip install gptmed[data-preparation]
```

## CLI Quick Start

### Text Processing

```bash
# Basic cleaning
data-preparation text --input ./raw --output ./processed

# Advanced options
data-preparation text \
  --input ./raw \
  --output ./processed \
  --lowercase \
  --remove-stopwords \
  --remove-punctuation
```

### Image Processing

```bash
# Basic resizing
data-preparation image --input ./raw --output ./processed

# With options
data-preparation image \
  --input ./raw \
  --output ./processed \
  --target-size 224 224 \
  --quality 95
```

### Audio Processing

```bash
# Basic resampling
data-preparation audio --input ./raw --output ./processed

# With options
data-preparation audio \
  --input ./raw \
  --output ./processed \
  --target-sample-rate 16000 \
  --mono
```

### Video Processing

```bash
# Basic validation
data-preparation video --input ./raw --output ./processed

# With frame extraction
data-preparation video \
  --input ./raw \
  --output ./processed \
  --extract-frames \
  --frame-sample-rate 30
```

## Python API Quick Start

### Text

```python
from gptmed.data_preparation import TextPreprocessor

preprocessor = TextPreprocessor()
cleaned = preprocessor.process("Raw text with <html> and http://url.com")
```

### Image

```python
from gptmed.data_preparation import ImagePreprocessor
from PIL import Image

preprocessor = ImagePreprocessor(target_size=(224, 224))
img = Image.open("photo.jpg")
processed = preprocessor.process(img)
```

### Audio

```python
from gptmed.data_preparation import AudioPreprocessor

preprocessor = AudioPreprocessor(target_sample_rate=16000)
preprocessor.batch_process_directory("./audio_files/")
```

### Video

```python
from gptmed.data_preparation import VideoPreprocessor

preprocessor = VideoPreprocessor(extract_frames=True)
preprocessor.batch_process_directory("./videos/")
```

## Common Tasks

### Process All Data Types

```python
from gptmed.data_preparation import (
    TextPreprocessor,
    ImagePreprocessor,
    AudioPreprocessor,
    VideoPreprocessor,
    PreprocessingConfig
)

configs = {
    'text': PreprocessingConfig('./raw/text', './proc/text', 'text'),
    'image': PreprocessingConfig('./raw/images', './proc/images', 'image'),
    'audio': PreprocessingConfig('./raw/audio', './proc/audio', 'audio'),
    'video': PreprocessingConfig('./raw/videos', './proc/videos', 'video'),
}

preprocessors = {
    'text': TextPreprocessor(config=configs['text']),
    'image': ImagePreprocessor(config=configs['image']),
    'audio': AudioPreprocessor(config=configs['audio']),
    'video': VideoPreprocessor(config=configs['video']),
}

for data_type, preprocessor in preprocessors.items():
    preprocessor.batch_process_directory(configs[data_type].input_path)
```

### Save Configuration

```python
from gptmed.data_preparation import PreprocessingConfig

config = PreprocessingConfig(
    input_path='./raw',
    output_path='./processed',
    data_type='text',
    batch_size=64
)
config.save('./config.json')
loaded = PreprocessingConfig.load('./config.json')
```

### Get Statistics

```python
preprocessor = TextPreprocessor(config)
results = preprocessor.batch_process(data_list)
stats = preprocessor.get_statistics()
print(f"Processed: {stats['output_count']}, Errors: {stats['errors']}")
preprocessor.save_statistics('./stats.json')
```

## Preprocessor Options

### TextPreprocessor

- `lowercase`: Convert to lowercase
- `remove_stopwords`: Remove common words
- `remove_punctuation`: Remove punctuation
- `min_length`: Minimum text length
- `max_length`: Maximum text length

### ImagePreprocessor

- `target_size`: Output size (height, width)
- `preserve_aspect_ratio`: Keep original proportions
- `normalize`: Normalize pixel values
- `min_size`: Minimum image size
- `max_size`: Maximum image size

### AudioPreprocessor

- `target_sample_rate`: Output sample rate (Hz)
- `mono`: Convert to mono
- `normalize_amplitude`: Normalize volume
- `remove_silence`: Remove silent parts
- `min_duration`: Minimum duration (seconds)
- `max_duration`: Maximum duration (seconds)

### VideoPreprocessor

- `target_fps`: Output frames per second
- `target_resolution`: Output size (width, height)
- `preserve_aspect_ratio`: Keep original proportions
- `extract_frames`: Extract individual frames
- `min_duration`: Minimum duration (seconds)
- `max_duration`: Maximum duration (seconds)

## File Formats

| Data Type | Supported Formats                  |
| --------- | ---------------------------------- |
| Text      | .txt, .md, .json, .csv             |
| Image     | .jpg, .jpeg, .png, .bmp, .webp     |
| Audio     | .wav, .mp3, .flac, .ogg, .m4a      |
| Video     | .mp4, .avi, .mov, .mkv, .flv, .wmv |

## Help & Documentation

```bash
# Get help for data-preparation
data-preparation --help

# Get help for specific command
data-preparation text --help
data-preparation image --help
data-preparation audio --help
data-preparation video --help
```

## Troubleshooting

| Issue                 | Solution                               |
| --------------------- | -------------------------------------- |
| Module not found      | `pip install gptmed[data-preparation]` |
| PIL not available     | `pip install pillow`                   |
| librosa not available | `pip install librosa soundfile`        |
| OpenCV not available  | `pip install opencv-python`            |
| Memory error          | Reduce batch_size, use num_workers=1   |

## Performance Tips

1. Use batch processing for large datasets
2. Adjust `batch_size` based on available memory
3. Enable `num_workers` for parallel processing
4. For images, use `preserve_aspect_ratio=True`
5. For video, use smaller `frame_sample_rate`

## Examples

See `examples/data_preparation_examples.py` for complete examples.

Run with:

```bash
python -m gptmed.examples.data_preparation_examples
```

## Testing

Run tests:

```bash
pytest tests/test_data_preparation.py
```

## License

MIT - See LICENSE file
