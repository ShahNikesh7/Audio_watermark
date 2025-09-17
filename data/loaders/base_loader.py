"""
Base data loader for audio datasets.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseAudioLoader(ABC):
    """Base class for audio data loaders."""
    
    def __init__(self, 
                 data_dir: str,
                 sample_rate: int = 44100,
                 chunk_length: int = 44100,
                 cache_size: int = 1000):
        """
        Initialize the base audio loader.
        
        Args:
            data_dir: Directory containing audio files
            sample_rate: Target sample rate for audio
            chunk_length: Length of audio chunks in samples
            cache_size: Maximum number of items to cache
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.chunk_length = chunk_length
        self.cache_size = cache_size
        
        self.audio_files = []
        self.cache = {}
        
        self._discover_files()
    
    def _discover_files(self):
        """Discover audio files in the data directory."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if any(file.lower().endswith(fmt) for fmt in supported_formats):
                    file_path = os.path.join(root, file)
                    self.audio_files.append(file_path)
        
        logger.info(f"Discovered {len(self.audio_files)} audio files")
    
    @abstractmethod
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio data as numpy array
        """
        pass
    
    @abstractmethod
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio data.
        
        Args:
            audio: Raw audio data
            
        Returns:
            Preprocessed audio data
        """
        pass
    
    def get_audio_chunks(self, audio: np.ndarray, overlap: float = 0.0) -> List[np.ndarray]:
        """
        Split audio into chunks.
        
        Args:
            audio: Audio data
            overlap: Overlap between chunks (0.0 to 1.0)
            
        Returns:
            List of audio chunks
        """
        if len(audio) <= self.chunk_length:
            return [audio]
        
        step_size = int(self.chunk_length * (1 - overlap))
        chunks = []
        
        for start in range(0, len(audio) - self.chunk_length + 1, step_size):
            chunk = audio[start:start + self.chunk_length]
            chunks.append(chunk)
        
        return chunks
    
    def get_item(self, index: int) -> Dict[str, Any]:
        """
        Get item by index.
        
        Args:
            index: Item index
            
        Returns:
            Dictionary containing audio data and metadata
        """
        if index >= len(self.audio_files):
            raise IndexError(f"Index {index} out of range")
        
        file_path = self.audio_files[index]
        
        # Check cache first
        if file_path in self.cache:
            return self.cache[file_path]
        
        # Load and preprocess audio
        try:
            audio = self.load_audio(file_path)
            audio = self.preprocess_audio(audio)
            
            item = {
                'audio': audio,
                'file_path': file_path,
                'sample_rate': self.sample_rate,
                'duration': len(audio) / self.sample_rate,
                'filename': os.path.basename(file_path)
            }
            
            # Cache if not full
            if len(self.cache) < self.cache_size:
                self.cache[file_path] = item
            
            return item
            
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def get_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get batch of items.
        
        Args:
            indices: List of item indices
            
        Returns:
            List of items
        """
        return [self.get_item(index) for index in indices]
    
    def get_random_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Get random batch of items.
        
        Args:
            batch_size: Number of items in batch
            
        Returns:
            List of random items
        """
        indices = np.random.choice(len(self.audio_files), batch_size, replace=False)
        return self.get_batch(indices.tolist())
    
    def __len__(self) -> int:
        """Get number of audio files."""
        return len(self.audio_files)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item by index (for compatibility with PyTorch Dataset)."""
        return self.get_item(index)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            'num_files': len(self.audio_files),
            'sample_rate': self.sample_rate,
            'chunk_length': self.chunk_length,
            'cache_size': len(self.cache),
            'max_cache_size': self.cache_size
        }
