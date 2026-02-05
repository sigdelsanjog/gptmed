"""
PDF Batch Processing Monitor

Tracks and displays progress during PDF batch processing with:
- Real-time progress bar
- Per-file metrics (size, processing time)
- Average processing time calculation
- Comprehensive summary report

Usage:
    from PDFBatchProcessMonitor import PDFBatchProcessMonitor
    
    monitor = PDFBatchProcessMonitor(total_files=10)
    monitor.start_logging()
    
    # For each file processed
    monitor.record_file_processed(
        filename="doc.pdf",
        file_size_kb=256,
        processing_time=2.5,
        success=True
    )
    
    # Get final report
    report = monitor.get_final_summary()
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FileMetrics:
    """Metrics for a single PDF file"""
    filename: str
    file_size_kb: float
    processing_time: float
    success: bool
    status_message: str = ""
    
    def __str__(self) -> str:
        size_str = f"{self.file_size_kb:.1f}KB"
        time_str = f"{self.processing_time:.2f}s"
        status = "✓" if self.success else "✗"
        return f"{status} {self.filename:<40} {size_str:>8} {time_str:>8}"


class PDFBatchProcessMonitor:
    """Monitor and track PDF batch processing progress"""
    
    def __init__(self, total_files: int = 0, logger: Optional[logging.Logger] = None):
        """
        Initialize the monitor
        
        Args:
            total_files: Total number of files to process
            logger: Optional logger instance
        """
        self.total_files = total_files
        self.processed_count = 0
        self.successful_count = 0
        self.failed_count = 0
        self.file_metrics: List[FileMetrics] = []
        self.start_time = None
        self.end_time = None
        
        # Use provided logger or create one
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
    
    def start_logging(self):
        """Start the monitoring process"""
        self.start_time = time.time()
        self.logger.info("\n" + "="*80)
        self.logger.info("PDF BATCH PROCESSING MONITOR STARTED")
        self.logger.info("="*80)
        self.logger.info(f"Total files to process: {self.total_files}")
        self.logger.info("="*80 + "\n")
    
    def _get_progress_bar(self, width: int = 50) -> str:
        """
        Generate a visual progress bar
        
        Args:
            width: Width of the progress bar
            
        Returns:
            Formatted progress bar string
        """
        if self.total_files == 0:
            return "[" + "█" * width + "] 0/0"
        
        filled = int(width * self.processed_count / self.total_files)
        empty = width - filled
        percentage = (self.processed_count / self.total_files) * 100
        
        bar = "[" + "█" * filled + "░" * empty + "]"
        progress = f"{bar} {self.processed_count}/{self.total_files} ({percentage:.1f}%)"
        return progress
    
    def _get_elapsed_time(self) -> str:
        """Get formatted elapsed time"""
        if self.start_time is None:
            return "00:00:00"
        
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _estimate_remaining_time(self) -> str:
        """Estimate remaining processing time"""
        if self.processed_count == 0 or self.total_files == 0:
            return "--:--:--"
        
        if self.start_time is None:
            return "--:--:--"
        
        elapsed = time.time() - self.start_time
        avg_time_per_file = elapsed / self.processed_count
        remaining_files = self.total_files - self.processed_count
        estimated_remaining = avg_time_per_file * remaining_files
        
        hours = int(estimated_remaining // 3600)
        minutes = int((estimated_remaining % 3600) // 60)
        seconds = int(estimated_remaining % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def record_file_processed(
        self,
        filename: str,
        file_size_kb: float,
        processing_time: float,
        success: bool = True,
        status_message: str = ""
    ):
        """
        Record a processed file
        
        Args:
            filename: Name of the PDF file
            file_size_kb: Size of file in KB
            processing_time: Time taken to process in seconds
            success: Whether processing was successful
            status_message: Optional status/error message
        """
        self.processed_count += 1
        if success:
            self.successful_count += 1
        else:
            self.failed_count += 1
        
        # Record metrics
        metrics = FileMetrics(
            filename=filename,
            file_size_kb=file_size_kb,
            processing_time=processing_time,
            success=success,
            status_message=status_message
        )
        self.file_metrics.append(metrics)
        
        # Log with progress bar
        remaining = self.total_files - self.processed_count
        elapsed = self._get_elapsed_time()
        eta = self._estimate_remaining_time()
        
        progress_bar = self._get_progress_bar()
        
        self.logger.info(f"{progress_bar}")
        self.logger.info(
            f"  [{elapsed} elapsed | ETA: {eta}] "
            f"[↓ {remaining} remaining] {metrics}"
        )
    
    def get_average_processing_time(self) -> float:
        """
        Calculate average processing time per file
        
        Returns:
            Average processing time in seconds
        """
        if not self.file_metrics:
            return 0.0
        
        total_time = sum(m.processing_time for m in self.file_metrics)
        return total_time / len(self.file_metrics)
    
    def get_total_size(self) -> float:
        """
        Get total size of all processed files
        
        Returns:
            Total size in KB
        """
        return sum(m.file_size_kb for m in self.file_metrics)
    
    def get_file_summary(self) -> List[str]:
        """
        Get detailed summary for each file
        
        Returns:
            List of formatted file summaries
        """
        summaries = []
        summaries.append("\n" + "="*80)
        summaries.append("DETAILED FILE PROCESSING METRICS")
        summaries.append("="*80)
        summaries.append(f"{'Status':<5} {'Filename':<40} {'Size':>8} {'Time':>8}")
        summaries.append("-"*80)
        
        for metrics in self.file_metrics:
            summaries.append(str(metrics))
        
        summaries.append("="*80)
        
        return summaries
    
    def get_final_summary(self) -> Dict[str, Any]:
        """
        Generate final processing summary
        
        Returns:
            Dictionary containing comprehensive summary
        """
        self.end_time = time.time()
        total_elapsed = self.end_time - self.start_time if self.start_time else 0
        
        successful = [m for m in self.file_metrics if m.success]
        failed = [m for m in self.file_metrics if not m.success]
        
        avg_time = self.get_average_processing_time()
        total_size = self.get_total_size()
        
        # Calculate additional stats
        fastest_file = min(self.file_metrics, key=lambda m: m.processing_time) if self.file_metrics else None
        slowest_file = max(self.file_metrics, key=lambda m: m.processing_time) if self.file_metrics else None
        
        summary = {
            'total_files': len(self.file_metrics),
            'successful': len(successful),
            'failed': len(failed),
            'total_size_kb': total_size,
            'total_processing_time': total_elapsed,
            'average_time_per_file': avg_time,
            'fastest_file': {
                'filename': fastest_file.filename if fastest_file else 'N/A',
                'time': fastest_file.processing_time if fastest_file else 0,
                'size_kb': fastest_file.file_size_kb if fastest_file else 0
            },
            'slowest_file': {
                'filename': slowest_file.filename if slowest_file else 'N/A',
                'time': slowest_file.processing_time if slowest_file else 0,
                'size_kb': slowest_file.file_size_kb if slowest_file else 0
            },
            'file_metrics': self.file_metrics
        }
        
        return summary
    
    def log_final_summary(self):
        """Log the final comprehensive summary to logger"""
        summary = self.get_final_summary()
        
        # Log detailed file metrics
        self.logger.info("\n" + "="*80)
        self.logger.info("DETAILED FILE PROCESSING METRICS")
        self.logger.info("="*80)
        self.logger.info(f"{'Status':<5} {'Filename':<40} {'Size':>8} {'Time':>8}")
        self.logger.info("-"*80)
        
        for metrics in self.file_metrics:
            self.logger.info(str(metrics))
        
        # Log summary statistics
        self.logger.info("\n" + "="*80)
        self.logger.info("PROCESSING SUMMARY STATISTICS")
        self.logger.info("="*80)
        self.logger.info(f"Total Files Processed: {summary['total_files']}")
        self.logger.info(f"  ✓ Successful: {summary['successful']}")
        self.logger.info(f"  ✗ Failed: {summary['failed']}")
        self.logger.info(f"\nTotal Data Size: {summary['total_size_kb']:.2f} KB ({summary['total_size_kb']/1024:.2f} MB)")
        self.logger.info(f"Total Processing Time: {summary['total_processing_time']:.2f}s")
        self.logger.info(f"\nAverage Time Per File: {summary['average_time_per_file']:.2f}s")
        
        if summary['fastest_file']['filename'] != 'N/A':
            self.logger.info(f"\nFastest File:")
            self.logger.info(f"  File: {summary['fastest_file']['filename']}")
            self.logger.info(f"  Size: {summary['fastest_file']['size_kb']:.1f} KB")
            self.logger.info(f"  Time: {summary['fastest_file']['time']:.2f}s")
        
        if summary['slowest_file']['filename'] != 'N/A':
            self.logger.info(f"\nSlowest File:")
            self.logger.info(f"  File: {summary['slowest_file']['filename']}")
            self.logger.info(f"  Size: {summary['slowest_file']['size_kb']:.1f} KB")
            self.logger.info(f"  Time: {summary['slowest_file']['time']:.2f}s")
        
        self.logger.info("="*80 + "\n")
