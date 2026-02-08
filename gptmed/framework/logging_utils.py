"""
Logging utility for gptmed framework training

Handles log file creation and management for all training processes.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_framework_logs_dir() -> Path:
    """
    Get the framework logs directory, creating it if it doesn't exist.
    
    Returns:
        Path to the framework logs directory
    """
    # Get the framework directory (parent of this file)
    framework_dir = Path(__file__).parent
    logs_dir = framework_dir / "logs"
    
    # Create logs directory if it doesn't exist
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    return logs_dir


def setup_training_logger(
    name: str,
    model_type: str = "conversation",
    log_file_prefix: str = "training"
) -> logging.Logger:
    """
    Set up a logger for training with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        model_type: Type of model being trained (e.g., "conversation", "qna")
        log_file_prefix: Prefix for log file name
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level - captures everything)
    logs_dir = get_framework_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"{log_file_prefix}_{model_type}_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Add a separate summary file for each model type
    summary_file = logs_dir / f"{model_type}_training_summary.log"
    summary_handler = logging.FileHandler(summary_file, mode='a')
    summary_handler.setLevel(logging.INFO)
    summary_handler.setFormatter(detailed_formatter)
    logger.addHandler(summary_handler)
    
    logger.info(f"Logging started - Log file: {log_file}")
    
    return logger


def get_logs_dir_path() -> Path:
    """
    Returns the path to the logs directory.
    Ensures it exists before returning.
    
    Returns:
        Path to logs directory
    """
    return get_framework_logs_dir()
