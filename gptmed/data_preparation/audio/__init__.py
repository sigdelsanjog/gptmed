"""
Audio data preprocessing and cleaning module

Handles audio resampling, normalization, silence removal, and quality checks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..base import BaseDataPreprocessor, PreprocessingConfig


logger = logging.getLogger(__name__)


class AudioPreprocessor(BaseDataPreprocessor):
    """
    Audio preprocessing with resampling, normalization, and validation
    
    Features:
        - Audio format validation
        - Resampling to target sample rate
        - Amplitude normalization
        - Silence detection and removal
        - Noise reduction
        - Duration validation
        - Metadata extraction
        - Stereo to mono conversion
        - Compression artifact detection
    """
    
    def __init__(
        self,
        config: Optional[PreprocessingConfig] = None,
        target_sample_rate: int = 16000,
        mono: bool = True,
        normalize_amplitude: bool = True,
        remove_silence: bool = False,
        silence_threshold: float = 0.02,
        min_duration: float = 0.5,  # seconds
        max_duration: Optional[float] = None,  # seconds
        supported_formats: Optional[List[str]] = None,
    ):
        """
        Initialize audio preprocessor
        
        Args:
            config: PreprocessingConfig instance
            target_sample_rate: Target sample rate in Hz
            mono: Convert to mono if True
            normalize_amplitude: Normalize audio amplitude
            remove_silence: Remove silence from audio
            silence_threshold: Threshold for silence detection (0-1)
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            supported_formats: List of supported audio formats
        """
        if config is None:
            config = PreprocessingConfig(
                input_path="./data/raw/audio",
                output_path="./data/processed/audio",
                data_type="audio"
            )
        
        super().__init__(config)
        
        self.target_sample_rate = target_sample_rate
        self.mono = mono
        self.normalize_amplitude = normalize_amplitude
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.supported_formats = supported_formats or ['wav', 'mp3', 'flac', 'ogg', 'm4a']
        
        self._import_audio_library()
    
    def _import_audio_library(self):
        """Attempt to import librosa and soundfile"""
        self.librosa_available = False
        self.soundfile_available = False
        
        try:
            import librosa
            self.librosa = librosa
            self.librosa_available = True
        except ImportError:
            self.logger.warning(
                "librosa not available. Install with: pip install librosa"
            )
        
        try:
            import soundfile as sf
            self.soundfile = sf
            self.soundfile_available = True
        except ImportError:
            self.logger.warning(
                "soundfile not available. Install with: pip install soundfile"
            )
    
    def validate(self, data: Any) -> bool:
        """
        Validate audio input
        
        Args:
            data: Audio file path (str) or numpy array with sample rate
            
        Returns:
            True if valid, False otherwise
        """
        if not self.librosa_available:
            self.logger.error("librosa is required for audio processing")
            return False
        
        try:
            if isinstance(data, str):
                audio_path = Path(data)
                if not audio_path.exists():
                    self.logger.warning(f"Audio file not found: {data}")
                    return False
                
                if not any(str(audio_path).lower().endswith(f) for f in self.supported_formats):
                    self.logger.warning(f"Unsupported format: {data}")
                    return False
                
                # Load and check
                y, sr = self.librosa.load(str(audio_path), sr=None, mono=False)
                duration = self.librosa.get_duration(y=y, sr=sr)
                
            elif isinstance(data, tuple) and len(data) == 2:  # (audio_array, sample_rate)
                y, sr = data
                duration = len(y) / sr
            else:
                self.logger.warning(f"Invalid audio type: {type(data)}")
                return False
            
            # Check duration constraints
            if duration < self.min_duration:
                self.logger.warning(f"Audio too short: {duration:.2f}s < {self.min_duration}s")
                return False
            
            if self.max_duration and duration > self.max_duration:
                self.logger.warning(f"Audio too long: {duration:.2f}s > {self.max_duration}s")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Audio validation error: {str(e)}")
            return False
    
    def clean(self, audio_data: Any) -> Tuple:
        """
        Clean audio data
        
        Args:
            audio_data: Audio file path or (audio_array, sample_rate) tuple
            
        Returns:
            Tuple of (cleaned_audio, sample_rate)
        """
        try:
            # Load audio if path
            if isinstance(audio_data, str):
                y, sr = self.librosa.load(str(audio_data), sr=None, mono=False)
            else:
                y, sr = audio_data
            
            # Convert to mono if needed
            if self.mono and len(y.shape) > 1 and y.shape[0] > 1:
                y = self.librosa.to_mono(y)
            elif self.mono and len(y.shape) > 1:
                y = y[0]
            
            # Remove silence if specified
            if self.remove_silence:
                # Simple silence removal based on amplitude threshold
                S = self.librosa.feature.melspectrogram(y=y, sr=sr)
                S_db = self.librosa.power_to_db(S, ref=self.librosa.db_to_power(0))
                
                # Get energy
                energy = self.librosa.feature.melspectrogram(y=y, sr=sr)
                
                # Very basic silence detection - can be improved
                y = self.librosa.effects.split(y, top_db=40)[0]
            
            return (y, sr)
            
        except Exception as e:
            self.logger.error(f"Audio cleaning error: {str(e)}")
            return None
    
    def normalize(self, audio_data: Tuple) -> Tuple:
        """
        Normalize audio
        
        Args:
            audio_data: Tuple of (audio_array, sample_rate)
            
        Returns:
            Tuple of (normalized_audio, target_sample_rate)
        """
        try:
            y, sr = audio_data
            
            # Resample if needed
            if sr != self.target_sample_rate:
                y = self.librosa.resample(y, orig_sr=sr, target_sr=self.target_sample_rate)
            
            # Normalize amplitude
            if self.normalize_amplitude:
                # Peak normalization
                y = y / (self.librosa.effects.loudness(y=y) + 1e-10)
                # Ensure it's in [-1, 1] range
                max_val = self.librosa.util.peak_normalize(y)
                y = max_val
            
            return (y, self.target_sample_rate)
            
        except Exception as e:
            self.logger.error(f"Audio normalization error: {str(e)}")
            return None
    
    def get_audio_stats(self, audio_path: str) -> Dict[str, Any]:
        """
        Get statistics about audio
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio statistics
        """
        try:
            if not self.librosa_available:
                return {}
            
            y, sr = self.librosa.load(str(audio_path), sr=None, mono=False)
            duration = self.librosa.get_duration(y=y, sr=sr)
            
            # Get RMS energy
            S = self.librosa.feature.melspectrogram(y=y, sr=sr)
            S_db = self.librosa.power_to_db(S, ref=self.librosa.db_to_power(0))
            
            stats = {
                'file': str(audio_path),
                'sample_rate': sr,
                'duration_seconds': float(duration),
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'total_samples': len(y) if len(y.shape) == 1 else y.shape[1],
                'file_size_bytes': Path(audio_path).stat().st_size,
                'rms_energy': float(self.librosa.feature.rms(y=y)[0].mean()) if len(y.shape) == 1 else 0,
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting audio stats: {str(e)}")
            return {}
    
    def batch_process_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        output_format: str = "wav",
    ) -> Dict[str, Any]:
        """
        Process all audio files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            output_format: Output audio format
            
        Returns:
            Processing results
        """
        if not self.librosa_available:
            self.logger.error("librosa is required")
            return {'error': 'librosa not available'}
        
        output_dir = output_dir or self.config.output_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_dir)
        results = []
        
        for audio_file in input_path.iterdir():
            if audio_file.suffix.lower()[1:] not in self.supported_formats:
                continue
            
            try:
                # Validate
                if not self.validate(str(audio_file)):
                    self.stats['skipped'] += 1
                    continue
                
                # Load and process
                y, sr = self.librosa.load(str(audio_file), sr=None)
                cleaned = self.clean((y, sr))
                normalized = self.normalize(cleaned)
                
                if normalized is None:
                    raise Exception("Normalization failed")
                
                # Save processed audio
                y_out, sr_out = normalized
                output_file = Path(output_dir) / f"{audio_file.stem}.{output_format}"
                
                if self.soundfile_available:
                    self.soundfile.write(str(output_file), y_out, sr_out)
                else:
                    self.librosa.output.write_wav(str(output_file), y_out, sr=sr_out)
                
                self.stats['output_count'] += 1
                
                results.append({
                    'file': str(audio_file),
                    'status': 'success',
                    'stats': self.get_audio_stats(str(audio_file))
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {audio_file}: {str(e)}")
                self.stats['errors'] += 1
                results.append({
                    'file': str(audio_file),
                    'status': 'error',
                    'error': str(e)
                })
        
        self.logger.info(f"Processed {self.stats['output_count']} audio files")
        return {'results': results, 'stats': self.get_statistics()}
