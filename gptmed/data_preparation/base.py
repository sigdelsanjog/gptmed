"""
Base classes and configurations for data preprocessing
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    
    input_path: str
    output_path: str
    data_type: str  # 'text', 'image', 'audio', 'video'
    save_format: str = "json"  # json, csv, parquet
    batch_size: int = 32
    num_workers: int = 4
    verbose: bool = True
    clean_cache: bool = False
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save(self, path: str) -> None:
        """Save config to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'PreprocessingConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class BaseDataPreprocessor(ABC):
    """Abstract base class for all data preprocessors"""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: PreprocessingConfig instance
        """
        self.config = config
        self.data_type = config.data_type
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup logging
        if config.verbose:
            self.logger.setLevel(logging.INFO)
        
        # Create output directory
        Path(self.config.output_path).mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'errors': 0,
            'skipped': 0,
            'processing_time': 0.0,
        }
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate input data
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    @abstractmethod
    def clean(self, data: Any) -> Any:
        """
        Clean the input data
        
        Args:
            data: Input data to clean
            
        Returns:
            Cleaned data
        """
        pass
    
    @abstractmethod
    def normalize(self, data: Any) -> Any:
        """
        Normalize the data
        
        Args:
            data: Data to normalize
            
        Returns:
            Normalized data
        """
        pass
    
    def process(self, data: Any) -> Any:
        """
        Full preprocessing pipeline
        
        Args:
            data: Input data
            
        Returns:
            Processed data
        """
        try:
            # Validate
            if not self.validate(data):
                self.logger.warning("Validation failed for input data")
                self.stats['skipped'] += 1
                return None
            
            # Clean
            cleaned = self.clean(data)
            
            # Normalize
            normalized = self.normalize(cleaned)
            
            self.stats['output_count'] += 1
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            self.stats['errors'] += 1
            return None
    
    def batch_process(self, data_list: List[Any]) -> List[Any]:
        """
        Process a batch of data items
        
        Args:
            data_list: List of data items
            
        Returns:
            List of processed data
        """
        self.stats['input_count'] = len(data_list)
        results = []
        
        for i, data in enumerate(data_list):
            if self.config.verbose and (i + 1) % max(1, len(data_list) // 10) == 0:
                self.logger.info(f"Processing: {i + 1}/{len(data_list)}")
            
            result = self.process(data)
            if result is not None:
                results.append(result)
        
        self.logger.info(f"Batch processing complete. Stats: {self.stats}")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset statistics"""
        self.stats = {
            'input_count': 0,
            'output_count': 0,
            'errors': 0,
            'skipped': 0,
            'processing_time': 0.0,
        }
    
    def save_statistics(self, path: str) -> None:
        """Save statistics to file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        self.logger.info(f"Statistics saved to {path}")
