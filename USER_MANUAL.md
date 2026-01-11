# GptMed User Manual 📚

A complete step-by-step guide to train your own custom GPT model on any Q&A dataset.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Step-by-Step Training Pipeline](#step-by-step-training-pipeline)
4. [Inference (Using Your Trained Model)](#inference-using-your-trained-model)
5. [Command-Line Reference](#command-line-reference)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

---

## Quick Start

**Got a Q&A dataset? Here's the fastest way to train a model:**

```bash
# 1. Install
pip install gptmed

# 2. Prepare your Q&A data in text format (see Step 3.1 below)

# 3. Train tokenizer
python -m gptmed.tokenizer.train_tokenizer \
    --input data/my_qa_data.txt \
    --output-prefix tokenizer/my_tokenizer \
    --vocab-size 8000

# 4. Tokenize your data
python -m gptmed.tokenizer.tokenize_data \
    --input data/my_qa_data.txt \
    --tokenizer tokenizer/my_tokenizer.model \
    --output-dir data/tokenized

# 5. Train the model
gptmed-train \
    --model-size small \
    --num-epochs 10 \
    --batch-size 16 \
    --train-data data/tokenized/train.npy \
    --val-data data/tokenized/val.npy

# 6. Generate answers!
gptmed-generate \
    --prompt "Your question here?" \
    --max-length 100
```

---

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install gptmed
```

### Option 2: Install from Source (For Development)

```bash
git clone https://github.com/sigdelsanjog/gptmed.git
cd gptmed
pip install -e .
```

### Option 3: Install with Optional Dependencies

```bash
# For training with TensorBoard/W&B
pip install gptmed[training]

# For development
pip install gptmed[dev]

# Everything
pip install gptmed[dev,training]
```

### Verify Installation

```bash
gptmed-train --help
gptmed-generate --help
```

---

## Step-by-Step Training Pipeline

### Step 1: Prepare Your Q&A Dataset

#### 1.1 Data Format Requirements

Your Q&A data should be in a simple text format with questions and answers clearly marked:

**Format 1: Simple Q&A Format (Recommended)**

```text
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

Q: What is a neural network?
A: A neural network is a series of algorithms that attempt to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.

Q: What is deep learning?
A: Deep learning is a subset of machine learning that uses multi-layered neural networks to analyze data with a structure similar to the human brain.
```

**Key Requirements:**

- Each Q&A pair separated by double newline (`\n\n`)
- Questions start with `Q: `
- Answers start with `A: `
- One newline between question and answer

**Format 2: JSON Format**

```json
[
  {
    "question": "What is machine learning?",
    "answer": "Machine learning is a subset of artificial intelligence..."
  },
  {
    "question": "What is a neural network?",
    "answer": "A neural network is a series of algorithms..."
  }
]
```

_Note: If using JSON, you'll need to convert it to text format (see conversion script below)._

#### 1.2 Convert JSON to Text Format

If your data is in JSON format, use this Python script:

```python
import json

# Read JSON
with open('qa_data.json', 'r') as f:
    data = json.load(f)

# Convert to text format
with open('qa_data.txt', 'w') as f:
    for item in data:
        f.write(f"Q: {item['question']}\n")
        f.write(f"A: {item['answer']}\n\n")

print(f"Converted {len(data)} Q&A pairs to text format")
```

#### 1.3 Data Quality Tips

✅ **Good practices:**

- At least 1,000 Q&A pairs for decent results
- 5,000+ pairs for good quality
- 20,000+ pairs for high quality
- Clean, grammatically correct text
- Consistent formatting

❌ **Avoid:**

- Very short answers (< 10 words)
- Duplicates
- Mixed languages
- Extremely long answers (> 500 words)

---

### Step 2: Train Your Custom Tokenizer

The tokenizer converts text into numbers (tokens) that the model can understand.

#### 2.1 Basic Tokenizer Training

```bash
python -m gptmed.tokenizer.train_tokenizer \
    --input data/my_qa_data.txt \
    --output-prefix tokenizer/my_tokenizer \
    --vocab-size 8000
```

**What this does:**

- Reads your Q&A text file
- Learns the vocabulary from your data
- Creates two files:
  - `my_tokenizer.model` (tokenizer model)
  - `my_tokenizer.vocab` (vocabulary file)

#### 2.2 Tokenizer Parameters Explained

| Parameter              | Default | Description                | When to Change                                                              |
| ---------------------- | ------- | -------------------------- | --------------------------------------------------------------------------- |
| `--vocab-size`         | 8000    | Number of unique tokens    | Increase for larger datasets (16K-32K), decrease for small datasets (4K-6K) |
| `--model-type`         | bpe     | Tokenization algorithm     | Keep as 'bpe' (better for domain-specific text)                             |
| `--character-coverage` | 0.9995  | Unicode character coverage | Keep default (handles special symbols)                                      |

#### 2.3 Example: Training Tokenizer

```bash
# Create directories
mkdir -p tokenizer data

# Train tokenizer on your data
python -m gptmed.tokenizer.train_tokenizer \
    --input data/tech_support_qa.txt \
    --output-prefix tokenizer/tech_tokenizer \
    --vocab-size 8000 \
    --model-type bpe
```

**Output:**

```
Training SentencePiece Tokenizer
============================================================
Input: data/tech_support_qa.txt
Output prefix: tokenizer/tech_tokenizer
Vocab size: 8000
Model type: bpe

Training tokenizer...
✓ Tokenizer trained successfully!
  Model: tokenizer/tech_tokenizer.model
  Vocab: tokenizer/tech_tokenizer.vocab
```

---

### Step 3: Tokenize Your Dataset

Convert your text into numerical tokens for training.

#### 3.1 Basic Tokenization

```bash
python -m gptmed.tokenizer.tokenize_data \
    --input data/my_qa_data.txt \
    --tokenizer tokenizer/my_tokenizer.model \
    --output-dir data/tokenized \
    --max-length 512 \
    --train-split 0.9
```

**What this does:**

- Converts text → token IDs
- Splits into train/validation sets (90/10 by default)
- Creates:
  - `data/tokenized/train.npy` (training data)
  - `data/tokenized/val.npy` (validation data)

#### 3.2 Tokenization Parameters

| Parameter       | Default    | Description                                       |
| --------------- | ---------- | ------------------------------------------------- |
| `--max-length`  | 512        | Maximum sequence length in tokens                 |
| `--train-split` | 0.9        | Training data ratio (0.9 = 90% train, 10% val)    |
| `--stride`      | max-length | Sliding window stride (for overlapping sequences) |

#### 3.3 Example: Tokenizing Data

```bash
python -m gptmed.tokenizer.tokenize_data \
    --input data/tech_support_qa.txt \
    --tokenizer tokenizer/tech_tokenizer.model \
    --output-dir data/tokenized \
    --max-length 512 \
    --train-split 0.9
```

**Output:**

```
Tokenizing Dataset
============================================================
Input: data/tech_support_qa.txt
Tokenizer: tokenizer/tech_tokenizer.model
Max length: 512
Train split: 0.9

Processing...
✓ Tokenization complete!
  Train sequences: 8,542
  Val sequences: 949
  Files created:
    - data/tokenized/train.npy
    - data/tokenized/val.npy
```

---

### Step 4: Train Your Model

Now train your GPT model on the tokenized data!

#### 4.1 Basic Training

```bash
gptmed-train \
    --model-size small \
    --num-epochs 10 \
    --batch-size 16 \
    --train-data data/tokenized/train.npy \
    --val-data data/tokenized/val.npy \
    --checkpoint-dir model/checkpoints
```

#### 4.2 Model Sizes

| Size       | Parameters | RAM Required | Training Time\* | Quality         |
| ---------- | ---------- | ------------ | --------------- | --------------- |
| **tiny**   | ~2M        | 4GB          | 1-2 hours       | Testing only    |
| **small**  | ~10M       | 6GB          | 4-8 hours       | **Recommended** |
| **medium** | ~50M       | 12GB         | 12-24 hours     | High quality    |

\*Approximate times on GTX 1080 for 10K Q&A pairs

#### 4.3 Training Parameters

| Parameter         | Default | Description     | Recommendation                                               |
| ----------------- | ------- | --------------- | ------------------------------------------------------------ |
| `--model-size`    | small   | Model size      | Use 'small' for most cases                                   |
| `--num-epochs`    | 10      | Training epochs | 10-20 for small datasets, 5-10 for large                     |
| `--batch-size`    | 16      | Batch size      | Reduce if OOM (8, 4), increase if GPU underutilized (32, 64) |
| `--learning-rate` | 3e-4    | Learning rate   | Keep default unless you know what you're doing               |
| `--device`        | cuda    | Device to use   | 'cuda' or 'cpu'                                              |

#### 4.4 Example: Training Session

```bash
# Create checkpoint directory
mkdir -p model/checkpoints

# Start training
gptmed-train \
    --model-size small \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 3e-4 \
    --train-data data/tokenized/train.npy \
    --val-data data/tokenized/val.npy \
    --checkpoint-dir model/checkpoints \
    --device cuda
```

**Training Output:**

```
============================================================
GPT Training Script
============================================================

Setting random seed: 42
Loading configurations...
Model config: small
  d_model: 768
  n_layers: 12
  n_heads: 12

Creating model...
  Total parameters: 10,234,567

Loading data...
  Train sequences: 8,542
  Val sequences: 949

Starting training...

Epoch 1/10
  Train Loss: 3.245 | Val Loss: 3.102 | Time: 24m 32s
  ✓ Saved checkpoint: model/checkpoints/checkpoint_epoch_1.pt

Epoch 2/10
  Train Loss: 2.891 | Val Loss: 2.754 | Time: 24m 18s
  ✓ Saved checkpoint: model/checkpoints/checkpoint_epoch_2.pt

...

Training complete! 🎉
Best checkpoint: model/checkpoints/best_model.pt
```

#### 4.5 Monitoring Training

Watch for these indicators:

✅ **Good training:**

- Train loss decreases steadily
- Val loss decreases (may plateau)
- Val loss < Train loss (no overfitting)

⚠️ **Problems:**

- Train loss not decreasing → Learning rate too high/low
- Val loss increasing → Overfitting (stop training, use early stopping)
- Loss = NaN → Gradient explosion (reduce learning rate)

---

## Inference (Using Your Trained Model)

### Step 5: Generate Answers

#### 5.1 Using Command Line

```bash
gptmed-generate \
    --checkpoint model/checkpoints/best_model.pt \
    --tokenizer tokenizer/my_tokenizer.model \
    --prompt "How do I reset my password?" \
    --max-length 100 \
    --temperature 0.7
```

#### 5.2 Using Python API

```python
import torch
from gptmed.inference.generator import TextGenerator
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import get_small_config

# Load model
config = get_small_config()
model = GPTTransformer(config)

# Load trained checkpoint
checkpoint = torch.load('model/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Create generator
generator = TextGenerator(
    model=model,
    tokenizer_path='tokenizer/my_tokenizer.model'
)

# Generate answer
question = "How do I reset my password?"
answer = generator.generate(
    prompt=question,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)

print(f"Q: {question}")
print(f"A: {answer}")
```

#### 5.3 Generation Parameters

| Parameter       | Default | Effect                                 | Recommendation                            |
| --------------- | ------- | -------------------------------------- | ----------------------------------------- |
| `--temperature` | 0.7     | Randomness (0=deterministic, 1=random) | 0.5-0.7 for factual, 0.8-1.0 for creative |
| `--top-k`       | 50      | Sample from top K tokens               | 40-50 for balanced                        |
| `--top-p`       | 0.9     | Nucleus sampling threshold             | 0.85-0.95 for quality                     |
| `--max-length`  | 100     | Maximum tokens to generate             | Adjust based on expected answer length    |

#### 5.4 Batch Generation

```python
questions = [
    "How do I reset my password?",
    "What are your operating hours?",
    "How do I contact support?"
]

for q in questions:
    answer = generator.generate(q, max_length=80, temperature=0.6)
    print(f"Q: {q}")
    print(f"A: {answer}\n")
```

---

## Command-Line Reference

### Training Commands

```bash
# Quick test run (small batches, few epochs)
gptmed-train --quick-test

# Resume from checkpoint
gptmed-train --resume

# Resume from specific checkpoint
gptmed-train --resume-from model/checkpoints/checkpoint_epoch_5.pt

# Full training with all options
gptmed-train \
    --model-size small \
    --num-epochs 20 \
    --batch-size 32 \
    --learning-rate 3e-4 \
    --train-data data/tokenized/train.npy \
    --val-data data/tokenized/val.npy \
    --checkpoint-dir model/checkpoints \
    --device cuda \
    --seed 42
```

### Tokenizer Commands

```bash
# Train tokenizer
python -m gptmed.tokenizer.train_tokenizer \
    --input data/my_data.txt \
    --output-prefix tokenizer/my_tok \
    --vocab-size 8000 \
    --model-type bpe

# Tokenize data
python -m gptmed.tokenizer.tokenize_data \
    --input data/my_data.txt \
    --tokenizer tokenizer/my_tok.model \
    --output-dir data/tokenized \
    --max-length 512 \
    --train-split 0.9
```

### Generation Commands

```bash
# Generate with defaults
gptmed-generate --prompt "Your question?"

# Generate with custom parameters
gptmed-generate \
    --checkpoint model/checkpoints/best_model.pt \
    --tokenizer tokenizer/my_tok.model \
    --prompt "Your question?" \
    --max-length 150 \
    --temperature 0.7 \
    --top-k 50 \
    --top-p 0.9
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Error

**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**

```bash
# Reduce batch size
gptmed-train --batch-size 8  # or 4

# Use smaller model
gptmed-train --model-size tiny

# Use CPU (slower but no memory limit)
gptmed-train --device cpu
```

#### 2. Loss Not Decreasing

**Symptoms:** Loss stays constant or increases

**Solutions:**

- Check data quality (are Q&A pairs properly formatted?)
- Reduce learning rate: `--learning-rate 1e-4`
- Increase training epochs: `--num-epochs 20`
- Verify tokenization worked correctly

#### 3. Model Generating Garbage

**Symptoms:** Output is random tokens or repeating text

**Solutions:**

- Train longer (model not converged)
- Lower temperature: `--temperature 0.5`
- Use top-k sampling: `--top-k 40`
- Check if checkpoint loaded correctly

#### 4. File Not Found Errors

**Error:** `FileNotFoundError: data/tokenized/train.npy`

**Solution:**

```bash
# Check file paths
ls data/tokenized/

# Make sure you ran tokenization step
python -m gptmed.tokenizer.tokenize_data \
    --input data/my_data.txt \
    --tokenizer tokenizer/my_tok.model \
    --output-dir data/tokenized
```

#### 5. Slow Training

**Symptoms:** Training taking too long

**Solutions:**

- Check GPU usage: `nvidia-smi`
- Increase batch size if GPU not fully utilized: `--batch-size 32`
- Use quick test config for debugging: `--quick-test`
- Reduce model size temporarily: `--model-size tiny`

#### 6. Import Errors

**Error:** `ModuleNotFoundError: No module named 'gptmed'`

**Solution:**

```bash
# Reinstall package
pip install --upgrade gptmed

# Or install from source
cd gptmed
pip install -e .
```

---

## Advanced Configuration

### Custom Training Configuration

Create a Python script for advanced control:

```python
from gptmed.training.train import main
from gptmed.configs.train_config import TrainingConfig
from gptmed.model.configs.model_config import get_small_config

# Custom training config
train_config = TrainingConfig(
    batch_size=32,
    learning_rate=3e-4,
    num_epochs=15,
    warmup_steps=500,
    grad_clip=1.0,
    weight_decay=0.01,
    save_every=1,  # Save checkpoint every epoch
    eval_every=100,  # Evaluate every 100 steps
    train_data_path='data/tokenized/train.npy',
    val_data_path='data/tokenized/val.npy',
    checkpoint_dir='model/checkpoints',
    device='cuda',
    seed=42
)

# Train
main()
```

### Custom Model Configuration

```python
from gptmed.model.configs.model_config import ModelConfig

# Create custom model config
custom_config = ModelConfig(
    vocab_size=8000,      # Match your tokenizer
    d_model=512,          # Hidden dimension
    n_layers=8,           # Number of transformer layers
    n_heads=8,            # Attention heads
    d_ff=2048,            # Feed-forward dimension
    max_seq_len=512,      # Max sequence length
    dropout=0.1,          # Dropout rate
    use_rope=True         # Use RoPE positional encoding
)
```

### Using with Different Data Formats

#### CSV Format

```python
import pandas as pd

# Read CSV
df = pd.read_csv('qa_data.csv')

# Convert to text format
with open('qa_data.txt', 'w') as f:
    for _, row in df.iterrows():
        f.write(f"Q: {row['question']}\n")
        f.write(f"A: {row['answer']}\n\n")
```

#### Database Format

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('qa_database.db')
cursor = conn.cursor()

# Query Q&A pairs
cursor.execute("SELECT question, answer FROM qa_table")

# Convert to text format
with open('qa_data.txt', 'w') as f:
    for question, answer in cursor.fetchall():
        f.write(f"Q: {question}\n")
        f.write(f"A: {answer}\n\n")

conn.close()
```

---

## Best Practices

### 1. Data Preparation

- ✅ Clean your data before training
- ✅ Remove duplicates
- ✅ Validate Q&A format consistency
- ✅ Split into train/validation properly

### 2. Training

- ✅ Start with small model for testing
- ✅ Monitor validation loss for overfitting
- ✅ Save checkpoints regularly
- ✅ Use GPU if available
- ✅ Set random seed for reproducibility

### 3. Inference

- ✅ Lower temperature for factual answers
- ✅ Higher temperature for creative responses
- ✅ Use top-k/top-p for better quality
- ✅ Limit max_length to avoid rambling

### 4. Iteration

- ✅ Evaluate generated answers
- ✅ Collect failure cases
- ✅ Add more training data if needed
- ✅ Fine-tune hyperparameters

---

## Next Steps

1. **Try it out:** Follow the Quick Start guide above
2. **Experiment:** Try different model sizes and parameters
3. **Evaluate:** Test your model on held-out questions
4. **Improve:** Add more data and retrain
5. **Deploy:** Use the inference API in your application

## Getting Help

- 📫 **Issues:** [GitHub Issues](https://github.com/sigdelsanjog/gptmed/issues)
- 💬 **Discussions:** [GitHub Discussions](https://github.com/sigdelsanjog/gptmed/discussions)
- 📧 **Email:** sanjog.sigdel@ku.edu.np

---

**Happy Training! 🚀**
