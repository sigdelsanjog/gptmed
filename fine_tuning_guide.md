# Fine-Tuning Strategy for GptMed Conversation Architecture

## Executive Summary

Your biomedical data (1.3M tokens, 10K vocabulary) is ideal for fine-tuning existing models rather than training from scratch. This guide provides specific recommendations.

---

## Part 1: Data Optimization

### Current Data State

```
Location: /data_preparation/text/output/
├── full_preprocessed.jsonl       (preprocessed Q&A pairs)
├── incremental_20260213.jsonl    (latest batch)
├── tokens/
│   ├── merged_tokens.jsonl       (1.3M tokens total)
│   ├── token_counts.json         (frequency statistics)
│   ├── vocab.json                (10K vocabulary)
│   └── vocab_info.json
└── processing_manifest.json      (92 biomedical papers processed)
```

### Data Enhancement Recommendations

#### 1. **Augment Dataset Size**

```python
# Current: ~1.3M tokens
# Recommended: 5-10M tokens for strong fine-tuning
# Actions:
# - Add more biomedical papers (MedArxiv, Semantic Scholar)
# - Include PubMed abstracts related to your papers
# - Create synthetic Q&A pairs using existing content
```

#### 2. **Data Distribution Analysis**

```python
# Check token distribution
from collections import Counter
import json

with open('/data_preparation/text/output/tokens/token_counts.json') as f:
    counts = json.load(f)

# Tokens: Count distribution analysis
print(f"Total unique tokens: {len(counts)}")
print(f"Top 20 tokens: {sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20]}")

# Action: Remove highly frequent stop words, keep domain-specific terms
```

#### 3. **Training/Validation Split**

```
Recommended split for your 1.3M tokens:
- Training:   80% (1.04M tokens)  → Fine-tune model
- Validation: 10% (130K tokens)   → Early stopping
- Test:       10% (130K tokens)   → Final evaluation
```

---

## Part 2: Model Selection & Recommendations

### Tier 1: Best for Biomedical Domain

```yaml
Model Name: PubMedBERT
- Base model: BERT pre-trained on PubMed
- Uses: Medical entity recognition, semantic similarity
- Size: 340M parameters
- Fine-tuning cost: LOW (BERT encoder is fast)
- Performance gain: HIGH for medical Q&A

Model Name: Biomedical-RoBERTa
- Better contextual understanding than BERT
- Pre-trained on 13M PubMed abstracts
- Performance: Strong on medical NER and extraction
```

### Tier 2: Conversation-Ready Models

```yaml
Model Name: DistilGPT-2 (Recommended START)
- Size: 82M parameters (vs GPT-2 124M)
- Pre-trained on general English
- Fast fine-tuning (3-4 hours on single GPU)
- Good for: Conversational Q&A

Model Name: GPT-2 Medium
- Size: 355M parameters
- Better quality than DistilGPT-2
- Fine-tuning time: 6-8 hours
- Better output quality but slower
```

### Tier 3: Domain-Specific

```yaml
Model Name: SciBERT
- Pre-trained on 1.2M scientific papers
- Good for: Medical document understanding
- Performance: Best semantic understanding

Model Name: Huggingface "mBERT" (Multilingual BERT)
- If multilingual support needed
```

---

## Part 3: Step-by-Step Fine-Tuning Implementation

### Step 1: Setup Environment

```bash
# Install transformers library
pip install transformers torch datasets accelerate tensorboard

# For your specific needs:
pip install sentencepiece  # For tokenization
pip install scikit-learn   # For evaluation metrics
```

### Step 2: Prepare Data in HuggingFace Format

```python
# Convert your JSONL to format expected by transformers

import json
from pathlib import Path

def prepare_data_for_finetuning(input_jsonl, output_dir):
    """
    Convert preprocessed data to fine-tuning format
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Read your preprocessed data
    texts = []
    with open(input_jsonl) as f:
        for line in f:
            data = json.loads(line)
            # Combine question + answer for causal LM
            text = f"Question: {data['text']}\n"
            texts.append(text)

    # Save training data
    train_size = int(len(texts) * 0.8)

    with open(f"{output_dir}/train.txt", "w") as f:
        f.write("\n".join(texts[:train_size]))

    with open(f"{output_dir}/val.txt", "w") as f:
        f.write("\n".join(texts[train_size:]))

    return len(texts[:train_size]), len(texts[train_size:])
```

### Step 3: Fine-Tune DistilGPT-2 (Recommended Start)

```python
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare datasets
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path/to/train.txt",
    block_size=128  # Increase to 256 if GPU memory allows
)

eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="path/to/val.txt",
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # For causal LM (GPT-style)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/finetuned_distilgpt2_biomedical",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_steps=500,
    save_total_limit=3,
    eval_steps=1000,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    seed=42,
    fp16=True,  # Mixed precision for faster training
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Fine-tune!
trainer.train()

# Save final model
model.save_pretrained("./models/biomedical_gpt2")
tokenizer.save_pretrained("./models/biomedical_gpt2")
```

### Step 4: Fine-Tune PubMedBERT (Better for Medical Tasks)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# PubMedBERT is more specialized
model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Same training procedure as above
# But use MLM=True for BERT-style masked language modeling
```

---

## Part 4: Conversation Architecture Integration

### Current Issue

Your conversation model may suffer from:

1. **Insufficient training data** (1.3M tokens is small)
2. **Generic base model** not adapted to biomedical domain
3. **No domain-specific vocabulary** optimization

### Solution: Hybrid Approach

```
Step 1: Fine-tune pre-trained model on your biomedical data
              ↓
Step 2: Use fine-tuned model as backbone for conversation layer
              ↓
Step 3: Train conversation wrapper (dialog management) on your data
              ↓
Step 4: Integrate with your gptmed conversation architecture
```

### Implementation

```python
# In your conversation architecture

from transformers import GPT2Tokenizer, GPT2LMHeadModel

class BiomedicalgptConversation:
    def __init__(self, model_path="./models/biomedical_gpt2"):
        """
        Use fine-tuned model instead of training from scratch
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = "cuda"  # or "cpu"
        self.model.to(self.device)

    def generate_response(self, context, max_length=100):
        """
        Generate biomedically-aware responses
        """
        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            top_p=0.9,          # Nucleus sampling for diversity
            top_k=50,           # Keep top 50 tokens
            temperature=0.7,    # Lower for more focused responses
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)
```

---

## Part 5: Performance Metrics & Evaluation

### Benchmark Your Fine-Tuned Model

```python
from datasets import load_metric
from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(model, test_dataset):
    """
    Measure if fine-tuning improved performance
    """
    perplexity_metric = load_metric("perplexity")

    # Test on biomedical Q&A
    results = {
        'perplexity': perplexity_metric.compute(predictions=predictions),
        'bleu_score': bleu_score(predictions, references),
        'rouge_score': rouge_score(predictions, references),
    }

    return results
```

### Expected Improvements After Fine-Tuning

```
Generic GPT-2:
- Perplexity: ~25-30
- BLEU: ~15-20
- Understanding of medical terms: POOR

Fine-tuned GPT-2 (on your data):
- Perplexity: ~8-12
- BLEU: ~25-35
- Medical term understanding: EXCELLENT
- Coherence on biomedical topics: HIGH
```

---

## Part 6: Quick Start Commands

### Option A: Fine-tune DistilGPT-2 (Fastest, Recommended)

```bash
python -m transformers.examples.language-modeling.run_language_modeling \
    --output_dir models/biomedical_distilgpt2 \
    --model_type gpt2 \
    --model_name_or_path distilgpt2 \
    --do_train \
    --do_eval \
    --train_data_file data_preparation/text/output/full_preprocessed.jsonl \
    --eval_data_file data_preparation/text/output/incremental_20260213.jsonl \
    --num_train_epochs 3 \
    --block_size 128 \
    --per_device_train_batch_size 16 \
    --learning_rate 5e-5 \
    --evaluate_during_training \
    --save_steps 500
```

### Option B: Use Hugging Face Trainer (Recommended)

See Step 3 above for detailed implementation

---

## Part 7: Troubleshooting & Best Practices

### Issue: Overfitting on small dataset

**Solution**:

- Use smaller model (DistilGPT-2 instead of GPT-2 Large)
- Add data augmentation
- Increase dropout: `hidden_dropout_prob: 0.3`
- Use weight decay: `weight_decay: 0.01`

### Issue: Loss not decreasing

**Solution**:

- Reduce learning rate: `5e-5` → `2e-5`
- Check data format (ensure proper tokenization)
- Increase warmup steps: `warmup_steps: 1000`

### Issue: Out of Memory (OOM)

**Solution**:

- Reduce batch_size: `16` → `8` → `4`
- Reduce block_size: `256` → `128`
- Enable gradient accumulation
- Use 16-bit precision: `fp16=True`

---

## Summary Table

| Model        | Training Speed | Quality    | Domain Fit        | Recommended For                  |
| ------------ | -------------- | ---------- | ----------------- | -------------------------------- |
| DistilGPT-2  | ⚡⚡⚡ (2-3h)  | ⭐⭐⭐     | General           | Quick baseline, MVP              |
| GPT-2 Medium | ⚡⚡ (6-8h)    | ⭐⭐⭐⭐   | General           | Production biomedical Q&A        |
| PubMedBERT   | ⚡⚡ (4-6h)    | ⭐⭐⭐⭐⭐ | Biomedical        | Best for medical NER, extraction |
| BioBERT      | ⚡⚡ (4-6h)    | ⭐⭐⭐⭐   | Biomedical        | Entity recognition               |
| SciBERT      | ⚡⚡ (5-7h)    | ⭐⭐⭐⭐   | Scientific papers | Semantic understanding           |

**Recommendation: Start with DistilGPT-2, then scale to GPT-2 Medium if quality not sufficient**

---

## Next Steps

1. **Prepare data** → Convert to HF format
2. **Choose model** → Start with DistilGPT-2
3. **Fine-tune** → Run training script
4. **Evaluate** → Measure perplexity, BLEU scores
5. **Integrate** → Replace base model in conversation architecture
6. **Deploy** → Test with real conversations

---

## Resources

- HuggingFace Model Hub: https://huggingface.co/models
- Paper: "Domain-Specific Language Model Pretraining for Biomedical" (ArXiv)
- BioBERT: https://github.com/naver/biobert
- PubMedBERT: https://microsoft.com/en-us/research/
