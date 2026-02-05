"""
Tokenization Report Generator

Analyzes merged_tokens.jsonl and provides:
1. Token count per document
2. Total token statistics
3. Training data format analysis
4. Recommendations for training

Usage:
    from config.TokenReportGenerator import TokenReportGenerator
    
    generator = TokenReportGenerator(
        merged_tokens_file="output/tokens/merged_tokens.jsonl",
        output_dir="output/reports"
    )
    generator.generate_comprehensive_report()
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class DocumentTokenStats:
    """Statistics for a single document"""
    filename: str
    token_count: int
    status: str
    
    def __repr__(self) -> str:
        return (f"  {self.filename:<45} "
                f"{self.token_count:>8} tokens | "
                f"{self.status}")


@dataclass
class TokenizationReportData:
    """Complete tokenization report data"""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_tokens: int
    average_tokens_per_doc: float
    median_tokens_per_doc: float
    min_tokens: int
    max_tokens: int
    documents: List[DocumentTokenStats] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_documents': self.total_documents,
            'successful_documents': self.successful_documents,
            'failed_documents': self.failed_documents,
            'total_tokens': self.total_tokens,
            'average_tokens_per_doc': round(self.average_tokens_per_doc, 2),
            'median_tokens_per_doc': round(self.median_tokens_per_doc, 2),
            'min_tokens': self.min_tokens,
            'max_tokens': self.max_tokens,
            'documents': [
                {
                    'filename': d.filename,
                    'token_count': d.token_count,
                    'status': d.status
                }
                for d in self.documents
            ]
        }


class TokenReportGenerator:
    """Generate comprehensive tokenization reports"""
    
    def __init__(
        self,
        merged_tokens_file: str,
        output_dir: str = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize report generator
        
        Args:
            merged_tokens_file: Path to merged_tokens.jsonl
            output_dir: Output directory for reports
            logger: Optional logger instance
        """
        self.merged_tokens_file = Path(merged_tokens_file)
        
        if output_dir is None:
            output_dir = self.merged_tokens_file.parent / 'reports'
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(self.__class__.__name__)
        
        self.report_data = None
    
    def _read_merged_tokens(self) -> List[Dict[str, Any]]:
        """
        Read merged tokens from JSONL file
        
        Returns:
            List of token records
        """
        records = []
        
        if not self.merged_tokens_file.exists():
            self.logger.error(f"File not found: {self.merged_tokens_file}")
            return records
        
        try:
            with open(self.merged_tokens_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Error parsing line {line_num}: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error reading file: {str(e)}")
        
        return records
    
    def analyze_tokens(self) -> TokenizationReportData:
        """
        Analyze token distribution and statistics
        
        Returns:
            TokenizationReportData object
        """
        records = self._read_merged_tokens()
        
        if not records:
            self.logger.warning("No token records found")
            return TokenizationReportData(
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_tokens=0,
                average_tokens_per_doc=0,
                median_tokens_per_doc=0,
                min_tokens=0,
                max_tokens=0
            )
        
        # Extract data
        doc_stats = []
        token_counts = []
        total_tokens = 0
        successful = 0
        failed = 0
        
        for record in records:
            filename = record.get('filename', 'unknown')
            token_count = record.get('token_count', 0)
            status = record.get('status', 'unknown')
            
            doc_stat = DocumentTokenStats(
                filename=filename,
                token_count=token_count,
                status=status
            )
            doc_stats.append(doc_stat)
            
            if status == 'success':
                successful += 1
                token_counts.append(token_count)
                total_tokens += token_count
            else:
                failed += 1
        
        # Calculate statistics
        avg_tokens = total_tokens / len(token_counts) if token_counts else 0
        median_tokens = statistics.median(token_counts) if token_counts else 0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        
        # Sort by token count (descending)
        doc_stats.sort(key=lambda x: x.token_count, reverse=True)
        
        self.report_data = TokenizationReportData(
            total_documents=len(records),
            successful_documents=successful,
            failed_documents=failed,
            total_tokens=total_tokens,
            average_tokens_per_doc=avg_tokens,
            median_tokens_per_doc=median_tokens,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            documents=doc_stats
        )
        
        return self.report_data
    
    def log_token_report(self):
        """Log detailed token report to logger"""
        if not self.report_data:
            self.logger.warning("No report data available. Run analyze_tokens() first.")
            return
        
        data = self.report_data
        
        # Header
        self.logger.info("\n" + "="*90)
        self.logger.info("TOKENIZATION REPORT")
        self.logger.info("="*90)
        self.logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Source File: {self.merged_tokens_file.name}")
        self.logger.info("="*90)
        
        # Summary Statistics
        self.logger.info("\n" + "-"*90)
        self.logger.info("SUMMARY STATISTICS")
        self.logger.info("-"*90)
        self.logger.info(f"Total Documents: {data.total_documents}")
        self.logger.info(f"  ✓ Successfully Tokenized: {data.successful_documents}")
        self.logger.info(f"  ✗ Failed: {data.failed_documents}")
        self.logger.info(f"\nTOTAL TOKENS: {data.total_tokens:,}")
        self.logger.info(f"Average Tokens per Document: {data.average_tokens_per_doc:.2f}")
        self.logger.info(f"Median Tokens per Document: {data.median_tokens_per_doc:.0f}")
        self.logger.info(f"Min Tokens per Document: {data.min_tokens}")
        self.logger.info(f"Max Tokens per Document: {data.max_tokens}")
        self.logger.info(f"Token Range (Max - Min): {data.max_tokens - data.min_tokens}")
        
        # Per-Document Breakdown
        self.logger.info("\n" + "-"*90)
        self.logger.info("PER-DOCUMENT TOKEN BREAKDOWN")
        self.logger.info("-"*90)
        self.logger.info(f"{'Document':<45} {'Tokens':>8} | Status")
        self.logger.info("-"*90)
        
        for doc in data.documents:
            self.logger.info(str(doc))
        
        self.logger.info("="*90 + "\n")
    
    def save_report_json(self) -> bool:
        """
        Save report as JSON file
        
        Returns:
            True if successful, False otherwise
        """
        if not self.report_data:
            self.logger.error("No report data available")
            return False
        
        try:
            output_file = self.output_dir / 'tokenization_report.json'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.report_data.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✓ Saved JSON report: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving JSON report: {str(e)}")
            return False
    
    def save_report_csv(self) -> bool:
        """
        Save per-document report as CSV
        
        Returns:
            True if successful, False otherwise
        """
        if not self.report_data:
            self.logger.error("No report data available")
            return False
        
        try:
            output_file = self.output_dir / 'tokenization_per_document.csv'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("Document Name,Token Count,Status\n")
                
                # Data
                for doc in self.report_data.documents:
                    f.write(f'"{doc.filename}",{doc.token_count},"{doc.status}"\n')
            
            self.logger.info(f"✓ Saved CSV report: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving CSV report: {str(e)}")
            return False
    
    def get_training_format_recommendation(self) -> Dict[str, Any]:
        """
        Provide recommendations for training data format
        
        Returns:
            Dictionary with recommendations
        """
        if not self.report_data:
            return {}
        
        data = self.report_data
        recommendation = {
            'current_format': 'JSONL with filename, tokens, token_count, tokenizer_method, status',
            'total_tokens': data.total_tokens,
            'total_documents': data.successful_documents,
            'average_tokens_per_doc': round(data.average_tokens_per_doc, 2),
            'token_distribution': {
                'min': data.min_tokens,
                'max': data.max_tokens,
                'median': data.median_tokens_per_doc,
                'average': data.average_tokens_per_doc
            }
        }
        
        # Recommendations
        recommendations = []
        
        if data.average_tokens_per_doc < 100:
            recommendations.append(
                "WARNING: Average tokens per document is low (<100). "
                "Consider combining documents for better training."
            )
        
        if data.max_tokens > 4096:
            recommendations.append(
                "INFO: Some documents exceed 4096 tokens. "
                "Consider chunking documents for models with smaller context windows."
            )
        
        if data.failed_documents > 0:
            recommendations.append(
                f"WARNING: {data.failed_documents} documents failed tokenization. "
                "Review error status for details."
            )
        
        if data.total_tokens < 10000:
            recommendations.append(
                "WARNING: Total token count is low (<10,000). "
                "Recommend collecting more training data for better model performance."
            )
        elif data.total_tokens < 100000:
            recommendations.append(
                "INFO: Total token count is moderate (10K-100K). "
                "Suitable for fine-tuning small models or initial testing."
            )
        else:
            recommendations.append(
                "INFO: Total token count is substantial (>100K). "
                "Good for training larger models or full pre-training."
            )
        
        recommendation['recommendations'] = recommendations
        
        return recommendation
    
    def log_training_recommendations(self):
        """Log training format recommendations"""
        recommendation = self.get_training_format_recommendation()
        
        self.logger.info("\n" + "="*90)
        self.logger.info("TRAINING DATA FORMAT & RECOMMENDATIONS")
        self.logger.info("="*90)
        
        self.logger.info(f"\nCurrent Format:")
        self.logger.info(f"  File Format: {recommendation['current_format']}")
        self.logger.info(f"\nData Summary:")
        self.logger.info(f"  Total Documents: {recommendation['total_documents']}")
        self.logger.info(f"  Total Tokens: {recommendation['total_tokens']:,}")
        self.logger.info(f"  Average Tokens/Doc: {recommendation['average_tokens_per_doc']:.2f}")
        
        self.logger.info(f"\nToken Distribution:")
        dist = recommendation['token_distribution']
        self.logger.info(f"  Min: {dist['min']}")
        self.logger.info(f"  Max: {dist['max']}")
        self.logger.info(f"  Median: {dist['median']:.0f}")
        self.logger.info(f"  Average: {dist['average']:.2f}")
        
        self.logger.info(f"\nRecommendations:")
        for i, rec in enumerate(recommendation['recommendations'], 1):
            self.logger.info(f"  {i}. {rec}")
        
        self.logger.info("="*90 + "\n")
    
    def generate_comprehensive_report(self) -> TokenizationReportData:
        """
        Generate comprehensive tokenization report
        
        Returns:
            TokenizationReportData with all analysis
        """
        self.logger.info("Starting tokenization analysis...")
        
        # Analyze tokens
        self.analyze_tokens()
        
        # Log reports
        self.log_token_report()
        self.log_training_recommendations()
        
        # Save reports
        self.save_report_json()
        self.save_report_csv()
        
        self.logger.info("✓ Report generation complete")
        
        return self.report_data


# Standalone usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate tokenization reports'
    )
    parser.add_argument(
        '--merged-tokens-file',
        required=True,
        help='Path to merged_tokens.jsonl file'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    generator = TokenReportGenerator(
        merged_tokens_file=args.merged_tokens_file,
        output_dir=args.output_dir
    )
    
    generator.generate_comprehensive_report()
