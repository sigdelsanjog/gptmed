# GptMed Quick Start Guide 🚀

The **simplest way** to train your own GPT model in 3 steps!

## Installation

```bash
pip install gptmed
```

## Training in 3 Simple Steps

### Step 1: Create Configuration File

```python
import gptmed

# Create a config file template
gptmed.create_config('my_config.yaml')
```

This creates a `my_config.yaml` file with all the settings you need.

### Step 2: Edit Configuration

Open `my_config.yaml` and update your settings:

```yaml
model:
  size: small # tiny, small, or medium

data:
  train_data: ./data/tokenized/train.npy
  val_data: ./data/tokenized/val.npy

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 0.0003
```

### Step 3: Train Your Model

```python
import gptmed

# Train using the config file
results = gptmed.train_from_config('my_config.yaml')

# Your model is ready!
print(f"Best model: {results['best_checkpoint']}")
```

## Using Your Trained Model

```python
import gptmed

# Generate answers
answer = gptmed.generate(
    checkpoint='model/checkpoints/best_model.pt',
    tokenizer='tokenizer/my_tokenizer.model',
    prompt='Your question here?',
    max_length=100,
    temperature=0.7
)

print(answer)
```

## Complete Example (main.py)

Create a file called `main.py`:

```python
import gptmed

# 1. Create config (first time only)
gptmed.create_config('training_config.yaml')

# Edit training_config.yaml with your settings, then:

# 2. Train the model
results = gptmed.train_from_config('training_config.yaml')

# 3. Test it
answer = gptmed.generate(
    checkpoint=results['best_checkpoint'],
    tokenizer='tokenizer/my_tokenizer.model',
    prompt='What is machine learning?'
)

print(f"Answer: {answer}")
```

Run it:

```bash
python main.py
```

That's it! 🎉

## Configuration File Options

The configuration file supports all training parameters:

```yaml
# Model
model:
  size: small # tiny (~2M), small (~10M), medium (~50M)

# Data paths
data:
  train_data: ./data/tokenized/train.npy
  val_data: ./data/tokenized/val.npy

# Training hyperparameters
training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 100

# Checkpointing
checkpointing:
  checkpoint_dir: ./model/checkpoints
  save_every: 1
  keep_last_n: 3

# Device
device:
  device: cuda # or cpu
  seed: 42
```

## Next Steps

- See [USER_MANUAL.md](USER_MANUAL.md) for complete training pipeline
- Check [examples/](examples/) folder for more examples
- Read [README.md](README.md) for detailed documentation

## Need Help?

- 📖 [User Manual](USER_MANUAL.md)
- 💬 [GitHub Discussions](https://github.com/sigdelsanjog/gptmed/discussions)
- 🐛 [Issues](https://github.com/sigdelsanjog/gptmed/issues)
