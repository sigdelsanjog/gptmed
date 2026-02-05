# PDF Preprocessing and Token Generation Guide

This guide provides step-by-step instructions to preprocess PDF files and generate tokens for training. The pipeline processes PDFs through three main stages: extraction, preprocessing, and tokenization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Directory Structure](#directory-structure)
3. [Step-by-Step Instructions](#step-by-step-instructions)
4. [Configuration Options](#configuration-options)
5. [Output Files](#output-files)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Packages

Ensure you have the following dependencies installed in your Python environment:

```bash
pip install pdfplumber transformers torch sentencepiece tqdm
```

### Environment Setup

Make sure you're in the gptmed project directory:

```bash
cd /path/to/gptmed
```

---

## Directory Structure

### Input Directory (pdfs/)

Create a `pdfs/` folder in the text directory and place your PDF files there. The pipeline supports both flat and nested directory structures:

**Flat Structure (Simple):**

```
data_preparation/text/
├── pdfs/                    # Your PDF input files
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── output/                  # Generated output (auto-created)
└── pipeline.py
```

**Nested Structure (Recommended for organization):**

```
data_preparation/text/
├── pdfs/                    # Your PDF input files (supports nested subdirectories)
│   ├── medical/
│   │   ├── cardiology/
│   │   │   ├── paper1.pdf
│   │   │   └── paper2.pdf
│   │   └── neurology/
│   │       ├── paper3.pdf
│   │       └── paper4.pdf
│   ├── research/
│   │   ├── study1.pdf
│   │   └── study2.pdf
│   └── document1.pdf
├── output/                  # Generated output (auto-created)
└── pipeline.py
```

The pipeline automatically discovers and processes all PDFs in any nested subdirectory structure.

### Output Directory (output/)

The pipeline automatically creates this directory with the following structure:

```
output/
├── full_preprocessed.jsonl  # Cleaned and preprocessed text
├── tokens/
│   ├── merged_tokens.jsonl  # Final training tokens (main output)
│   ├── token_stats.json     # Tokenization statistics
│   └── ...
└── extraction_report.json   # PDF extraction summary
```

---

## Step-by-Step Instructions

### Step 1: Prepare Your PDF Files

1. **Collect PDF Files**
   - Gather all PDF documents you want to process
   - Ensure PDFs are readable and not corrupted
   - Suggested file naming: use descriptive names without special characters

2. **Create Input Directory**

   ```bash
   mkdir -p data_preparation/text/pdfs
   ```

3. **Place PDFs in Input Directory**

   You can place PDFs directly or organize them in nested subdirectories:

   **Flat Structure:**

   ```bash
   cp your_documents/*.pdf data_preparation/text/pdfs/
   ```

   **Nested Structure (Recommended):**

   ```bash
   mkdir -p data_preparation/text/pdfs/cardiology
   mkdir -p data_preparation/text/pdfs/neurology
   cp cardiology_papers/*.pdf data_preparation/text/pdfs/cardiology/
   cp neurology_papers/*.pdf data_preparation/text/pdfs/neurology/
   ```

4. **Verify PDF Placement**

   ```bash
   # List all PDFs recursively
   find data_preparation/text/pdfs/ -name "*.pdf"
   ```

5. **Verify PDF Placement**
   ```bash
   ls -la data_preparation/text/pdfs/
   ```

### Step 2: Run the Pipeline

The pipeline is controlled by the `pipeline.py` script. You can run it with default settings or customize various parameters.

#### Option A: Run with Default Settings

```bash
cd data_preparation/text/
python3 pipeline.py
```

**Default Configuration:**

- Input: `./pdfs/`
- Output: `./output/`
- Tokenizer: HuggingFace GPT-2
- Workers: 4 parallel processes
- Case Mode: lowercase

#### Option B: Run with Custom Parameters

```bash
python3 pipeline.py \
    --input-dir ./pdfs \
    --output-dir ./output \
    --tokenizer-method huggingface \
    --tokenizer-model gpt2 \
    --workers 4 \
    --case-mode lower \
    --remove-stopwords \
    --remove-punctuation
```

### Step 3: Monitor Pipeline Execution

The pipeline provides detailed real-time progress monitoring with visual indicators. You'll see three main stages:

#### Real-Time Progress Monitoring

During PDF extraction, you'll see a live progress bar showing:

- **Progress Bar**: Visual representation of completion percentage
- **Processed/Total**: Count of files processed vs. total files
- **Time Elapsed**: How long processing has been running
- **ETA**: Estimated time remaining
- **Files Remaining**: Number of files still to process
- **Per-File Metrics**: For each file - processing time and file size

Example output:

```
================================================================================
PDF BATCH PROCESSING MONITOR STARTED
================================================================================
Total files to process: 5
================================================================================

[██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 3/5 (60.0%)
  [00:00:45 elapsed | ETA: 00:00:20] [↓ 2 remaining] ✓ document1.pdf          256.3KB    1.23s
[██████████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 4/5 (80.0%)
  [00:00:52 elapsed | ETA: 00:00:10] [↓ 1 remaining] ✓ document2.pdf          512.8KB    2.15s
[██████████████████████████████████████████████] 5/5 (100.0%)
  [00:00:58 elapsed | ETA: 00:00:00] [↓ 0 remaining] ✓ document3.pdf          128.5KB    0.87s
```

#### Detailed File Metrics Summary

After all files are processed, you'll see a detailed breakdown:

```
================================================================================
DETAILED FILE PROCESSING METRICS
================================================================================
Status Filename                                    Size      Time
--------------------------------------------------------------------------------
✓ document1.pdf                              256.3KB    1.23s
✓ document2.pdf                              512.8KB    2.15s
✓ document3.pdf                              128.5KB    0.87s
✓ document4.pdf                              342.1KB    1.67s
✓ document5.pdf                              189.4KB    1.12s
================================================================================

================================================================================
PROCESSING SUMMARY STATISTICS
================================================================================
Total Files Processed: 5
  ✓ Successful: 5
  ✗ Failed: 0

Total Data Size: 1429.1 KB (1.39 MB)
Total Processing Time: 7.04s

Average Time Per File: 1.41s

Fastest File:
  File: document3.pdf
  Size: 128.5 KB
  Time: 0.87s

Slowest File:
  File: document2.pdf
  Size: 512.8 KB
  Time: 2.15s
================================================================================
```

#### Stage 1: PDF Extraction

# The extraction phase shows real-time progress with per-file details.

# STEP 3: TOKENIZATION

Tokenizing with huggingface (gpt2)
✓ Tokenization complete
✓ Saved: merged_tokens.jsonl

````

### Step 4: Verify Output

After successful execution, check the output directory:

```bash
ls -lh output/
````

Expected files:

- ✅ `full_preprocessed.jsonl` - Preprocessed text data
- ✅ `tokens/merged_tokens.jsonl` - Final training tokens
- ✅ `tokens/token_stats.json` - Statistics about tokenization
- ✅ `extraction_report.json` - Summary of PDF extraction

### Step 5: Review Generated Tokens

Inspect the token output to verify quality:

```bash
# View first few token records
head -n 3 output/tokens/merged_tokens.jsonl | python3 -m json.tool
```

Expected format:

```json
{
  "filename": "document1.pdf",
  "tokens": [220, 1043, 1043, 2058, ...],
  "token_count": 1500,
  "original_word_count": 850
}
```

---

## Progress Monitoring and Performance Tracking

The pipeline includes integrated progress monitoring through the **PDFBatchProcessMonitor** module. This provides real-time visibility into the processing pipeline with detailed metrics.

### Monitor Features

- **Real-Time Progress Bar**: Visual progress indicator with percentage completion
- **Time Tracking**: Elapsed time, estimated time remaining (ETA)
- **Per-File Metrics**: File size and processing time for each PDF
- **Performance Statistics**: Average processing time, fastest/slowest files
- **Detailed Summary**: Complete breakdown of all processed files

### Understanding Monitor Output

The monitor displays information in real-time as files are processed:

```
[████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 2/10 (20.0%)
  [00:00:15 elapsed | ETA: 00:01:15] [↓ 8 remaining] ✓ doc1.pdf        256.3KB    1.23s
```

**Breakdown:**

- `████████░░░░...`: Visual progress bar (filled/unfilled)
- `2/10 (20.0%)`: 2 files processed out of 10 total
- `00:00:15 elapsed`: Time spent so far
- `ETA: 00:01:15`: Estimated time to completion
- `↓ 8 remaining`: Number of files still to process
- `doc1.pdf`: Filename
- `256.3KB`: File size in kilobytes
- `1.23s`: Time taken to process this file

### Monitor Module Location

The monitoring functionality is implemented in:

```
gptmed/data_preparation/PDFBatchProcessMonitor.py
```

This module can be used independently for other batch processing tasks.

---

## Configuration Options

### Command-Line Arguments

| Argument               | Default       | Description                                                                         |
| ---------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `--input-dir`          | `./pdfs`      | Directory containing PDF files                                                      |
| `--output-dir`         | `./output`    | Directory for output files                                                          |
| `--tokenizer-method`   | `huggingface` | Tokenization method (`huggingface`, `custom`, `sentencepiece`)                      |
| `--tokenizer-model`    | `gpt2`        | Tokenizer model name (e.g., `gpt2`, `bert-base-uncased`, `distilbert-base-uncased`) |
| `--workers`            | `4`           | Number of parallel workers for processing                                           |
| `--case-mode`          | `lower`       | Case normalization mode (`lower`, `upper`, `title`, `sentence`)                     |
| `--remove-stopwords`   | `False`       | Remove common stopwords (flag)                                                      |
| `--remove-punctuation` | `False`       | Remove punctuation marks (flag)                                                     |

### Example Configurations

**Lightweight Processing (Minimal Preprocessing):**

```bash
python3 pipeline.py \
    --input-dir ./pdfs \
    --output-dir ./output \
    --tokenizer-model gpt2 \
    --workers 2
```

**Medical Text Optimization:**

```bash
python3 pipeline.py \
    --input-dir ./pdfs \
    --output-dir ./output \
    --tokenizer-method huggingface \
    --tokenizer-model distilbert-base-uncased \
    --workers 4 \
    --case-mode lower
```

**Aggressive Cleaning:**

```bash
python3 pipeline.py \
    --input-dir ./pdfs \
    --output-dir ./output \
    --tokenizer-model gpt2 \
    --workers 4 \
    --remove-stopwords \
    --remove-punctuation \
    --case-mode lower
```

---

## Output Files

### 1. full_preprocessed.jsonl

Contains cleaned text after all preprocessing steps.

```json
{
  "filename": "document1.pdf",
  "text": "medical text with preprocessing applied...",
  "word_count": 850,
  "char_count": 5234,
  "has_special_chars": false,
  "unicode_normalized": true
}
```

### 2. merged_tokens.jsonl (Primary Output)

Contains tokenized representations ready for model training.

```json
{
  "filename": "document1.pdf",
  "tokens": [220, 1043, 2058, 1539, ...],
  "token_count": 1500,
  "original_word_count": 850
}
```

### 3. token_stats.json

Summary statistics of the tokenization process.

```json
{
  "total_files": 2,
  "total_tokens": 3500,
  "average_tokens_per_file": 1750,
  "total_words": 2000,
  "processing_time_seconds": 12.45
}
```

### 4. extraction_report.json

Summary of PDF extraction results.

```json
{
  "total_pdfs": 2,
  "successfully_extracted": 2,
  "failed_extractions": 0,
  "files": [
    {
      "filename": "document1.pdf",
      "status": "success",
      "words_extracted": 850,
      "file_size_kb": 245
    }
  ]
}
```

---

## Preprocessing Steps Explained

The preprocessing pipeline applies the following transformations in order:

### 1. **Unicode Normalization**

- Converts special Unicode characters to standard forms
- Ensures consistency across different text encodings

### 2. **Case Normalization** (Configurable)

- **lower**: Converts all text to lowercase
- **upper**: Converts all text to uppercase
- **title**: Title case formatting
- **sentence**: Sentence case formatting

### 3. **Text Cleaning**

- Removes extra whitespace and line breaks
- Handles special characters
- Normalizes spacing around punctuation

### 4. **Stopword Removal** (Optional)

- Removes common English words (the, a, an, etc.)
- Reduces vocabulary size
- Improves signal-to-noise ratio

### 5. **Punctuation Handling** (Optional)

- Removes or normalizes punctuation marks
- Configurable based on requirements

---

## Troubleshooting

### Issue: "No PDF files found in input directory"

**Cause:** PDFs not placed in the correct location or nested directories not being discovered

**Solution:**

```bash
# Verify file location (searches recursively)
find data_preparation/text/pdfs/ -name "*.pdf"

# Check flat directory
ls -la data_preparation/text/pdfs/

# Check nested directories
find data_preparation/text/pdfs/ -type d

# Ensure files have correct permissions
chmod 644 data_preparation/text/pdfs/**/*.pdf
chmod 644 data_preparation/text/pdfs/*.pdf
```

**Note:** The pipeline automatically discovers PDFs in nested subdirectories. You can organize PDFs in any nested folder structure, and they will all be found and processed.

### Issue: "MemoryError during tokenization"

**Cause:** Processing too many PDFs in parallel

**Solution:**

```bash
# Reduce number of workers
python3 pipeline.py --workers 2
```

### Issue: "PDF extraction failed for file X"

**Cause:** PDF file is corrupted or uses unsupported encoding

**Solution:**

1. Verify PDF integrity:

   ```bash
   pdfinfo data_preparation/text/pdfs/problematic.pdf
   ```

2. Try re-saving the PDF with a different PDF tool

3. Skip the problematic file and continue with others

### Issue: "Output directory permission denied"

**Cause:** Insufficient write permissions

**Solution:**

```bash
chmod 755 data_preparation/text/output/
chmod 755 data_preparation/text/
```

### Issue: "Module not found errors"

**Cause:** Missing dependencies

**Solution:**

```bash
# Reinstall requirements
pip install -r requirements.txt

# Verify installation
python3 -c "import transformers; print(transformers.__version__)"
```

---

## Performance Tips

### For Large PDF Collections

1. **Use Multiple Workers:**

   ```bash
   python3 pipeline.py --workers 8  # For 8+ core systems
   ```

2. **Process in Batches:**
   - Split large PDF collections into smaller batches
   - Run pipeline separately on each batch
   - Concatenate output files:
     ```bash
     cat output_batch1/merged_tokens.jsonl output_batch2/merged_tokens.jsonl > final_tokens.jsonl
     ```

3. **Monitor System Resources:**
   ```bash
   # In another terminal, monitor CPU and memory
   watch -n 1 'top -b -n 1 | head -n 15'
   ```

### For Quality Optimization

1. **Experiment with Parameters:**
   - Test different tokenizer models (gpt2, distilbert, etc.)
   - Try different case modes
   - Evaluate impact of stopword removal

2. **Validate Output:**
   ```bash
   # Check token distribution
   python3 -c "
   import json
   with open('output/tokens/merged_tokens.jsonl') as f:
       tokens = [json.loads(line)['token_count'] for line in f]
   print(f'Min tokens: {min(tokens)}, Max: {max(tokens)}, Avg: {sum(tokens)/len(tokens):.0f}')
   "
   ```

---

## Next Steps

After generating tokens:

1. **Use tokens for training:**
   - Load `merged_tokens.jsonl` in your training script
   - Use token arrays as input to your model

2. **Evaluate quality:**
   - Check token distribution
   - Verify against ground truth for sample documents
   - Adjust preprocessing parameters if needed

3. **Archive results:**
   ```bash
   tar -czf tokens_backup_$(date +%Y%m%d).tar.gz output/tokens/
   ```

---

## Support and Further Information

For detailed information about specific preprocessing steps, see:

- [Base Strategy Documentation](base_strategy.py)
- [Text Cleaner Details](text_cleaner.py)
- [Tokenizer Configuration](tokenizer.py)
- [Main Data Preparation Guide](../DATA_PREPARATION_GUIDE.md)

For issues or questions, check the pipeline logs and verify all prerequisites are met.
