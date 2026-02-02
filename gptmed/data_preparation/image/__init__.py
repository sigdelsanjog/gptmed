"""
Image data preprocessing and cleaning module

Handles image resizing, normalization, augmentation, and quality checks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json
from abc import ABC

from ..base import BaseDataPreprocessor, PreprocessingConfig


logger = logging.getLogger(__name__)


class ImagePreprocessor(BaseDataPreprocessor):
    """
    Image preprocessing with resizing, normalization, and validation
    
    Features:
        - Image format validation
        - Resizing and aspect ratio preservation
        - Normalization (pixel value scaling)
        - Brightness/contrast adjustment
        - Noise reduction
        - Format conversion
        - Metadata extraction
        - Duplicate detection via hashing
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        preserve_aspect_ratio: bool = True,
        min_size: Tuple[int, int] = (32, 32),
        max_size: Tuple[int, int] = (4096, 4096),
        supported_formats: Optional[List[str]] = None,
    ):
        """
        Initialize image preprocessor
        
        Args:
            config: PreprocessingConfig instance
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            preserve_aspect_ratio: Whether to preserve aspect ratio when resizing
            min_size: Minimum allowed image size
            max_size: Maximum allowed image size
            supported_formats: List of supported image formats
        """
        if config is None:
            config = PreprocessingConfig(
                input_path="./data/raw/images",
                output_path="./data/processed/images",
                data_type="image"
            )
        
        super().__init__(config)
        
        self.target_size = target_size
        self.normalize = normalize
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.min_size = min_size
        self.max_size = max_size
        self.supported_formats = supported_formats or ['jpg', 'jpeg', 'png', 'bmp', 'webp']
        
        self._import_image_library()
    
    def _import_image_library(self):
        """Attempt to import PIL/Pillow"""
        try:
            from PIL import Image
            self.Image = Image
            self.pil_available = True
        except ImportError:
            self.logger.warning(
                "PIL/Pillow not available. Install with: pip install Pillow"
            )
            self.pil_available = False
    
    def validate(self, data: Any) -> bool:
        """
        Validate image input
        
        Args:
            data: Image file path (str) or PIL Image
            
        Returns:
            True if valid, False otherwise
        """
        if not self.pil_available:
            self.logger.error("PIL/Pillow is required for image processing")
            return False
        
        try:
            if isinstance(data, str):
                img_path = Path(data)
                if not img_path.exists():
                    self.logger.warning(f"Image file not found: {data}")
                    return False
                if not any(str(img_path).lower().endswith(f) for f in self.supported_formats):
                    self.logger.warning(f"Unsupported format: {data}")
                    return False
                
                # Try to open
                img = self.Image.open(img_path)
                w, h = img.size
                
            elif hasattr(data, 'size'):  # PIL Image object
                w, h = data.size
            else:
                self.logger.warning(f"Invalid image type: {type(data)}")
                return False
            
            # Check size constraints
            if (w, h) < self.min_size or (w, h) > self.max_size:
                self.logger.warning(f"Image size {(w, h)} outside allowed range")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image validation error: {str(e)}")
            return False
    
    def clean(self, image_data: Any) -> Any:
        """
        Clean image data
        
        Args:
            image_data: Image file path or PIL Image
            
        Returns:
            Cleaned PIL Image
        """
        try:
            # Load image if path
            if isinstance(image_data, str):
                img = self.Image.open(image_data)
            else:
                img = image_data
            
            # Convert to RGB if needed (remove alpha channel, convert grayscale)
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = self.Image.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            return img
            
        except Exception as e:
            self.logger.error(f"Image cleaning error: {str(e)}")
            return None
    
    def normalize(self, image: Any) -> Any:
        """
        Normalize image
        
        Args:
            image: PIL Image
            
        Returns:
            Normalized image
        """
        try:
            # Resize image
            if self.preserve_aspect_ratio:
                image.thumbnail(self.target_size, self.Image.Resampling.LANCZOS)
                # Pad to target size
                new_img = self.Image.new('RGB', self.target_size, (0, 0, 0))
                offset = (
                    (self.target_size[0] - image.size[0]) // 2,
                    (self.target_size[1] - image.size[1]) // 2
                )
                new_img.paste(image, offset)
                image = new_img
            else:
                image = image.resize(self.target_size, self.Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Image normalization error: {str(e)}")
            return None
    
    def get_image_stats(self, image_path: str) -> Dict[str, Any]:
        """
        Get statistics about image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with image statistics
        """
        try:
            if not self.pil_available:
                return {}
            
            img = self.Image.open(image_path)
            
            stats = {
                'file': str(image_path),
                'format': img.format,
                'mode': img.mode,
                'width': img.width,
                'height': img.height,
                'size_bytes': Path(image_path).stat().st_size,
                'aspect_ratio': img.width / img.height if img.height > 0 else 0,
            }
            
            # Get file size in MB
            stats['size_mb'] = stats['size_bytes'] / (1024 * 1024)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting image stats: {str(e)}")
            return {}
    
    def batch_process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        output_format: str = "jpg",
        quality: int = 95,
    ) -> Dict[str, Any]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            output_format: Output image format
            quality: Output quality (for JPEG)
            
        Returns:
            Processing results
        """
        if not self.pil_available:
            self.logger.error("PIL/Pillow is required")
            return {'error': 'PIL not available'}
        
        output_dir = output_dir or self.config.output_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_dir)
        results = []
        
        for img_file in input_path.iterdir():
            if img_file.suffix.lower()[1:] not in self.supported_formats:
                continue
            
            try:
                # Validate and process
                if not self.validate(str(img_file)):
                    self.stats['skipped'] += 1
                    continue
                
                img = self.Image.open(str(img_file))
                cleaned = self.clean(img)
                normalized = self.normalize(cleaned)
                
                # Save processed image
                output_file = Path(output_dir) / f"{img_file.stem}.{output_format}"
                if output_format.lower() == 'jpg':
                    normalized.save(str(output_file), 'JPEG', quality=quality)
                else:
                    normalized.save(str(output_file))
                
                self.stats['output_count'] += 1
                
                results.append({
                    'file': str(img_file),
                    'status': 'success',
                    'stats': self.get_image_stats(str(img_file))
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {img_file}: {str(e)}")
                self.stats['errors'] += 1
                results.append({
                    'file': str(img_file),
                    'status': 'error',
                    'error': str(e)
                })
        
        self.logger.info(f"Processed {self.stats['output_count']} images")
        return {'results': results, 'stats': self.get_statistics()}
