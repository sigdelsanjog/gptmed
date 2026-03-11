"""
Logging system for agentic framework.
Provides consistent logging across all components.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class AgentLogger:
    """Centralized logger for the agentic framework."""
    
    _logger = None
    _log_dir = None
    
    @classmethod
    def setup(cls, log_dir: str = None, level: str = "INFO"):
        """
        Setup logging configuration.
        
        Args:
            log_dir: Directory to store logs (default: ./logs/)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if log_dir is None:
            log_dir = os.path.join(os.path.dirname(__file__), "../../logs")
        
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        cls._logger = logging.getLogger("AgenticFramework")
        cls._logger.setLevel(getattr(logging, level))
        
        # Create formatters and handlers
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, level))
        console_handler.setFormatter(formatter)
        cls._logger.addHandler(console_handler)
        
        # File handler
        log_file = cls._log_dir / f"agentic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_handler.setFormatter(formatter)
        cls._logger.addHandler(file_handler)
    
    @classmethod
    def get(cls) -> logging.Logger:
        """Get the agentic logger."""
        if cls._logger is None:
            cls.setup()
        return cls._logger
    
    @classmethod
    def debug(cls, message: str, *args, **kwargs):
        """Log debug message."""
        cls.get().debug(message, *args, **kwargs)
    
    @classmethod
    def info(cls, message: str, *args, **kwargs):
        """Log info message."""
        cls.get().info(message, *args, **kwargs)
    
    @classmethod
    def warning(cls, message: str, *args, **kwargs):
        """Log warning message."""
        cls.get().warning(message, *args, **kwargs)
    
    @classmethod
    def error(cls, message: str, *args, **kwargs):
        """Log error message."""
        cls.get().error(message, *args, **kwargs)
    
    @classmethod
    def critical(cls, message: str, *args, **kwargs):
        """Log critical message."""
        cls.get().critical(message, *args, **kwargs)
