"""
PDF to Text Conversion and CSV Export Examples

Demonstrates PDF processing and CSV export capabilities of the TextPreprocessor
"""

import logging
from pathlib import Path

from gptmed.data_preparation import TextPreprocessor, PreprocessingConfig


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_pdf_to_text_extraction():
    """Example: Extract text from a single PDF file"""
    logger.info("=" * 60)
    logger.info("PDF TO TEXT EXTRACTION EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/pdfs",
        output_path="./data/processed/pdfs",
        data_type="text"
    )
    
    preprocessor = TextPreprocessor(config=config)
    
    pdf_file = "./data/raw/sample.pdf"
    
    logger.info(f"\nExtracting text from: {pdf_file}")
    
    text = preprocessor.extract_text_from_pdf(pdf_file)
    
    if text:
        logger.info(f"✓ Successfully extracted {len(text)} characters")
        logger.info(f"First 200 characters: {text[:200]}...")
        
        # Get text statistics
        stats = preprocessor.get_text_stats(text)
        logger.info(f"\nText Statistics:")
        logger.info(f"  - Word count: {stats['word_count']}")
        logger.info(f"  - Sentence count: {stats['sentence_count']}")
        logger.info(f"  - Unique words: {stats['unique_words']}")
        logger.info(f"  - Vocabulary diversity: {stats['vocabulary_diversity']:.2%}")
    else:
        logger.warning("Failed to extract text from PDF")


def example_batch_pdf_processing():
    """Example: Process multiple PDF files from a directory"""
    logger.info("\n" + "=" * 60)
    logger.info("BATCH PDF PROCESSING EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/pdfs",
        output_path="./data/processed/pdfs",
        data_type="text"
    )
    
    preprocessor = TextPreprocessor(
        config=config,
        lowercase=True,
        remove_stopwords=False
    )
    
    input_dir = "./data/raw/pdfs"
    output_dir = "./data/processed/pdfs_text"
    
    logger.info(f"\nProcessing PDFs from: {input_dir}")
    logger.info(f"Saving output to: {output_dir}")
    
    results = preprocessor.batch_process_pdfs(input_dir, output_dir)
    
    logger.info(f"\nProcessing Results:")
    logger.info(f"  - Total processed: {results['stats']['output_count']}")
    logger.info(f"  - Errors: {results['stats']['errors']}")
    logger.info(f"  - Skipped: {results['stats']['skipped']}")
    
    for result in results['results'][:3]:  # Show first 3
        logger.info(f"\n  File: {result['file']}")
        logger.info(f"  Status: {result['status']}")
        if result['status'] == 'success' and 'stats' in result:
            logger.info(f"  Word count: {result['stats']['word_count']}")


def example_pdf_to_csv_basic():
    """Example: Export PDF texts to CSV (basic format)"""
    logger.info("\n" + "=" * 60)
    logger.info("PDF TO CSV EXPORT (BASIC) EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/pdfs",
        output_path="./data/processed",
        data_type="text"
    )
    
    preprocessor = TextPreprocessor(config=config, lowercase=True)
    
    input_dir = "./data/raw/pdfs"
    output_csv = "./data/processed/pdf_texts.csv"
    
    logger.info(f"\nExporting PDFs to CSV...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"CSV Columns: [pdf_filename, text_content]")
    
    results = preprocessor.export_to_csv(input_dir, output_csv, process_text=True)
    
    logger.info(f"\nExport Results:")
    logger.info(f"  - Rows exported: {results['rows_exported']}")
    logger.info(f"  - Errors: {results['errors']}")
    logger.info(f"  - Total PDFs: {results['total_pdfs']}")
    logger.info(f"  - Success: {results['success']}")
    
    if results['success']:
        logger.info(f"\n✓ CSV saved to: {results['output_file']}")


def example_pdf_to_csv_detailed():
    """Example: Export PDF texts to CSV (with statistics)"""
    logger.info("\n" + "=" * 60)
    logger.info("PDF TO CSV EXPORT (DETAILED WITH STATS) EXAMPLE")
    logger.info("=" * 60)
    
    config = PreprocessingConfig(
        input_path="./data/raw/pdfs",
        output_path="./data/processed",
        data_type="text"
    )
    
    preprocessor = TextPreprocessor(config=config, lowercase=False)
    
    input_dir = "./data/raw/pdfs"
    output_csv = "./data/processed/pdf_texts_detailed.csv"
    
    logger.info(f"\nExporting PDFs to CSV with statistics...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output CSV: {output_csv}")
    logger.info(f"CSV Columns: [pdf_filename, text_content, word_count, char_count, sentence_count]")
    
    results = preprocessor.export_to_csv_detailed(
        input_dir,
        output_csv,
        process_text=True,
        include_stats=True
    )
    
    logger.info(f"\nExport Results:")
    logger.info(f"  - Rows exported: {results['rows_exported']}")
    logger.info(f"  - Errors: {results['errors']}")
    logger.info(f"  - Total PDFs: {results['total_pdfs']}")
    logger.info(f"  - Success: {results['success']}")
    
    if results['success']:
        logger.info(f"\n✓ Detailed CSV saved to: {results['output_file']}")
        logger.info("\nCSV contains:")
        logger.info("  1. pdf_filename - Name of the PDF file")
        logger.info("  2. text_content - Extracted and processed text")
        logger.info("  3. word_count - Total words in document")
        logger.info("  4. char_count - Total characters in text")
        logger.info("  5. sentence_count - Number of sentences")


def example_workflow():
    """Example: Complete PDF processing workflow"""
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE PDF PROCESSING WORKFLOW EXAMPLE")
    logger.info("=" * 60)
    
    # Create preprocessor
    preprocessor = TextPreprocessor(
        config=PreprocessingConfig(
            input_path="./data/raw/pdfs",
            output_path="./data/processed",
            data_type="text"
        ),
        lowercase=True,
        remove_stopwords=False,
        remove_punctuation=False
    )
    
    pdf_dir = "./data/raw/pdfs"
    
    logger.info("\nWorkflow steps:")
    logger.info("1. Extract and process PDFs")
    logger.info("2. Save individual text files")
    logger.info("3. Export to CSV with statistics")
    
    # Step 1: Batch process PDFs
    logger.info("\n[Step 1] Processing PDFs...")
    text_results = preprocessor.batch_process_pdfs(pdf_dir)
    logger.info(f"✓ Processed {text_results['stats']['output_count']} PDFs")
    
    # Step 2: Export to detailed CSV
    logger.info("\n[Step 2] Exporting to CSV...")
    csv_results = preprocessor.export_to_csv_detailed(
        pdf_dir,
        "./data/processed/pdfs_export.csv",
        process_text=True,
        include_stats=True
    )
    logger.info(f"✓ Exported {csv_results['rows_exported']} records to CSV")
    
    # Step 3: Summary
    logger.info("\n[Step 3] Summary:")
    logger.info(f"  - Total PDFs processed: {text_results['stats']['output_count']}")
    logger.info(f"  - Records in CSV: {csv_results['rows_exported']}")
    logger.info(f"  - Output CSV: {csv_results.get('output_file', 'N/A')}")


def main():
    """Run all examples"""
    logger.info("\n" + "=" * 80)
    logger.info("PDF TO TEXT CONVERSION & CSV EXPORT - USAGE EXAMPLES")
    logger.info("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("NOTE: These examples require actual PDF files in ./data/raw/pdfs/")
    logger.info("Ensure PyPDF2 is installed: pip install PyPDF2")
    logger.info("=" * 80)
    
    # Run examples
    try:
        example_pdf_to_text_extraction()
    except FileNotFoundError:
        logger.info("PDF files not found - skipping extraction example")
    
    try:
        example_batch_pdf_processing()
    except FileNotFoundError:
        logger.info("PDF directory not found - skipping batch processing example")
    
    try:
        example_pdf_to_csv_basic()
    except FileNotFoundError:
        logger.info("PDF directory not found - skipping CSV export example")
    
    try:
        example_pdf_to_csv_detailed()
    except FileNotFoundError:
        logger.info("PDF directory not found - skipping detailed CSV export example")
    
    # Show workflow example (informational)
    logger.info("\n" + "=" * 60)
    logger.info("WORKFLOW EXAMPLE (Information)")
    logger.info("=" * 60)
    logger.info("\nTo use the complete workflow:")
    logger.info("\n1. Place PDF files in ./data/raw/pdfs/")
    logger.info("\n2. Use in Python:")
    logger.info("""
    from gptmed.data_preparation import TextPreprocessor
    
    preprocessor = TextPreprocessor()
    
    # Extract text from PDFs
    results = preprocessor.batch_process_pdfs('./data/raw/pdfs')
    
    # Export to CSV with statistics
    csv_results = preprocessor.export_to_csv_detailed(
        './data/raw/pdfs',
        './output.csv',
        process_text=True,
        include_stats=True
    )
    
    print(f"Exported {csv_results['rows_exported']} PDFs to CSV")
    """)
    
    logger.info("\n3. Or use CLI:")
    logger.info("   data-preparation text --help")
    logger.info("   # PDF support available through Python API")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLES COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
