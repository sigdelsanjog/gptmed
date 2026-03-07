# New PDF Folder

Add your new PDF files here for incremental processing.

## Usage

1. Place your new PDFs in this directory
2. Run: `python3 incremental_preprocess.py`
3. Processing will:
   - Detect new/modified PDFs
   - Extract text from only new PDFs (skip already-processed)
   - Preprocess the text
   - Append results to output/full_preprocessed.jsonl
   - Update processing manifest

## Benefits

- **Fast**: Only process new PDFs (not all PDFs every time)
- **Safe**: Tracks all processed PDFs to avoid duplicates
- **Flexible**: Add PDFs incrementally without retraining from scratch
- **Transparent**: Manifest file shows what has been processed

## Manifest

The `output/processing_manifest.json` tracks:
- Which PDFs have been processed
- File hashes (detects modifications)
- Processing statistics
- Last update time

To reset and reprocess all PDFs:
```
python3 incremental_preprocess.py --reset-manifest
```
