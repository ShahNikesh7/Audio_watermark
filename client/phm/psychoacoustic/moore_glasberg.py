"""
Psychoacoustic Analysis Module - Moore-Glasberg Model Implementation
Advanced implementation for audio watermarking with real-time optimization.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Tuple, Dict, Optional, Union
import numba
from numba import jit, prange
import logging

logger = logging.getLogger(__name__)


@jit(nopython=True, cache=True)
def _moore_glasberg_masking_threshold_jit(power_spectrum: np.ndarray, 
                                        frequencies: np.ndarray,
                                        bark_frequencies: np.ndarray) -> np.ndarray:
    """
    JIT-compiled Moore-Glasberg masking threshold calculation for performance.
    
    Args:
        power_spectrum: Power spectrum of the audio signal
        frequencies: Frequency bins
        bark_frequencies: Bark scale frequency mapping
        
    Returns:
        Masking threshold for each frequency bin
    """
    n_freqs = len(frequencies)
    masking_threshold = np.zeros(n_freqs)
    
    # Constants for Moore-Glasberg model
    spreading_function_slope_low = 27.0  # dB/Bark below masker
    spreading_function_slope_high = -27.0  # dB/Bark above masker
    
    for i in prange(n_freqs):
        if power_spectrum[i] <= 0:
            continue
            
        masker_bark = bark_frequencies[i]
        masker_level = 10 * np.log10(power_spectrum[i] + 1e-10)
        
        for j in range(n_freqs):
            maskee_bark = bark_frequencies[j]
            bark_distance = maskee_bark - masker_bark
            
            # Calculate spreading function
            if bark_distance >= 0:
                # Above masker frequency
                spreading = spreading_function_slope_high * bark_distance
            else:
                # Below masker frequency
                spreading = spreading_function_slope_low * bark_distance
            
            # Apply spreading and absolute threshold
            masked_threshold = masker_level + spreading
            
            # Convert back to linear scale and accumulate
            masking_threshold[j] += 10 ** (masked_threshold / 10.0)
    
    return masking_threshold


@jit(nopython=True, cache=True)
def _calculate_bark_scale_jit(frequencies: np.ndarray) -> np.ndarray:
    """
    JIT-compiled Bark scale conversion for performance optimization.
    
    Args:
        frequencies: Frequency array in Hz
        
    Returns:
        Bark scale values
    """
    # Traunmüller's formula for Bark scale
    bark_values = np.zeros(len(frequencies))
    for i in prange(len(frequencies)):
        f = frequencies[i]
        if f > 0:
            bark_values[i] = 26.81 * f / (1960 + f) - 0.53
        else:
            bark_values[i] = 0.0
    return bark_values


@jit(nopython=True, cache=True)
def _calculate_critical_bands_jit(bark_frequencies: np.ndarray,
                                 n_bands: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled critical band calculation.
    
    Args:
        bark_frequencies: Bark scale frequency values
        n_bands: Number of critical bands
        
    Returns:
        Tuple of (band_indices, band_centers)
    """
    bark_max = np.max(bark_frequencies)
    band_edges = np.linspace(0, bark_max, n_bands + 1)
    band_indices = np.zeros(len(bark_frequencies), dtype=np.int32)
    band_centers = np.zeros(n_bands)
    
    for i in prange(len(bark_frequencies)):
        for j in range(n_bands):
            if band_edges[j] <= bark_frequencies[i] < band_edges[j + 1]:
                band_indices[i] = j
                break
    
    for j in prange(n_bands):
        band_centers[j] = (band_edges[j] + band_edges[j + 1]) / 2.0
    
    return band_indices, band_centers


class MooreGlasbergAnalyzer:
    """
    Moore-Glasberg psychoacoustic model implementation optimized for real-time performance.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 1000,
                 hop_length: int = 512,
                 n_critical_bands: int = 24):
        """
        Initialize the Moore-Glasberg analyzer.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size for frequency analysis
            hop_length: Hop length for STFT
            n_critical_bands: Number of critical bands
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_critical_bands = n_critical_bands
        
        # Pre-compute frequency mappings for efficiency
        self.frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        self.bark_frequencies = self._compute_bark_frequencies()
        self.band_indices, self.band_centers = self._compute_critical_bands()
        
        # Absolute threshold of hearing (quiet threshold)
        self.absolute_threshold = self._compute_absolute_threshold()
        
        logger.info(f"Initialized Moore-Glasberg analyzer with {n_critical_bands} critical bands")
    
    def _compute_bark_frequencies(self) -> np.ndarray:
        """Compute Bark scale frequencies for all frequency bins."""
        return _calculate_bark_scale_jit(self.frequencies)
    
    def _compute_critical_bands(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute critical band mapping."""
        return _calculate_critical_bands_jit(self.bark_frequencies, self.n_critical_bands)
    
    def _compute_absolute_threshold(self) -> np.ndarray:
        """
        Compute absolute threshold of hearing for each frequency bin.
        Based on ISO 226 standard.
        """
        # Absolute threshold approximation (dB SPL)
        threshold_db = np.zeros(len(self.frequencies))
        
        for i, freq in enumerate(self.frequencies):
            if freq < 20:
                threshold_db[i] = 80  # Very high threshold for very low frequencies
            elif freq < 1000:
                # Low frequency region
                threshold_db[i] = 3.64 * (freq / 1000.0) ** -0.8 - 6.5 * np.exp(-0.6 * (freq / 1000.0 - 3.3) ** 2) + 1e-3 * (freq / 1000.0) ** 4
            else:
                # High frequency region
                threshold_db[i] = 3.64 * (freq / 1000.0) ** -0.8 - 6.5 * np.exp(-0.6 * (freq / 1000.0 - 3.3) ** 2) + 1e-3 * (freq / 1000.0) ** 4
        
        # Convert to linear scale (power)
        return 10 ** (threshold_db / 10.0)
    
    def analyze_masking_threshold(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze audio and compute masking thresholds using Moore-Glasberg model.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary containing masking analysis results
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        power_spectrum = np.abs(stft) ** 2
        
        # Initialize results
        n_frames = power_spectrum.shape[1]
        masking_thresholds = np.zeros((len(self.frequencies), n_frames))
        band_thresholds = np.zeros((self.n_critical_bands, n_frames))
        
        # Process each time frame
        for frame_idx in range(n_frames):
            frame_power = power_spectrum[:, frame_idx]
            
            # Calculate masking threshold for this frame
            frame_masking = _moore_glasberg_masking_threshold_jit(
                frame_power, self.frequencies, self.bark_frequencies
            )
            
            # Combine with absolute threshold
            combined_threshold = np.maximum(frame_masking, self.absolute_threshold)
            masking_thresholds[:, frame_idx] = combined_threshold
            
            # Calculate per-band thresholds
            for band_idx in range(self.n_critical_bands):
                band_mask = self.band_indices == band_idx
                if np.any(band_mask):
                    band_thresholds[band_idx, frame_idx] = np.mean(combined_threshold[band_mask])
        
        return {
            'masking_thresholds': masking_thresholds,
            'band_thresholds': band_thresholds,
            'frequencies': self.frequencies,
            'band_centers': self.band_centers,
            'band_indices': self.band_indices,
            'power_spectrum': power_spectrum
        }
    
    def calculate_band_masking_thresholds(self, 
                                        power_spectrum: np.ndarray) -> np.ndarray:
        """
        Calculate masking thresholds for critical bands.
        
        Args:
            power_spectrum: Power spectrum of audio (freq_bins x time_frames)
            
        Returns:
            Band masking thresholds (n_bands x time_frames)
        """
        n_frames = power_spectrum.shape[1]
        band_thresholds = np.zeros((self.n_critical_bands, n_frames))
        
        for frame_idx in range(n_frames):
            frame_power = power_spectrum[:, frame_idx]
            
            # Calculate full masking threshold
            masking_threshold = _moore_glasberg_masking_threshold_jit(
                frame_power, self.frequencies, self.bark_frequencies
            )
            
            # Combine with absolute threshold
            combined_threshold = np.maximum(masking_threshold, self.absolute_threshold)
            
            # Average within each critical band
            for band_idx in range(self.n_critical_bands):
                band_mask = self.band_indices == band_idx
                if np.any(band_mask):
                    band_thresholds[band_idx, frame_idx] = np.mean(combined_threshold[band_mask])
        
        return band_thresholds

    def masking_threshold(self, audio: np.ndarray) -> np.ndarray:
        """
        Simplified interface to compute masking threshold.
        Returns the global masking threshold array for compatibility with tests.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Global masking threshold array
        """
        # Validate input
        if audio.ndim > 2:
            raise ValueError("Audio must be 1D or 2D")
        
        # Convert to mono if stereo
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        
        # Use existing analyze_masking_threshold method
        analysis = self.analyze_masking_threshold(audio)
        
        # Return the masking thresholds averaged over time frames
        masking_thresholds = analysis['masking_thresholds']
        
        # Average over time dimension if multiple frames
        if masking_thresholds.ndim == 2:
            return np.mean(masking_thresholds, axis=1)
        else:
            return masking_thresholds


class PerceptualAnalyzer(nn.Module):
    """
    Neural network wrapper for psychoacoustic analysis integration with watermarking.
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 1000,
                 hop_length: int = 512,
                 n_critical_bands: int = 24):
        super().__init__()
        
        self.moore_glasberg = MooreGlasbergAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_critical_bands=n_critical_bands
        )
        
        # Learnable parameters for adaptive masking
        self.masking_adaptation = nn.Parameter(torch.ones(n_critical_bands))
        self.threshold_offset = nn.Parameter(torch.zeros(n_critical_bands))
        
        # Temporal smoothing
        self.temporal_smoother = nn.Conv1d(
            in_channels=n_critical_bands,
            out_channels=n_critical_bands,
            kernel_size=5,
            padding=2,
            groups=n_critical_bands
        )
        
    def forward(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for perceptual analysis.
        
        Args:
            audio: Input audio tensor (batch_size, samples)
            
        Returns:
            Dictionary with masking analysis results
        """
        batch_size = audio.shape[0]
        device = audio.device
        
        # Process each sample in batch
        batch_results = []
        
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            
            # Analyze with Moore-Glasberg model
            analysis = self.moore_glasberg.analyze_masking_threshold(audio_np)
            
            # Convert to tensors
            band_thresholds = torch.from_numpy(analysis['band_thresholds']).float().to(device)
            
            batch_results.append(band_thresholds)
        
        # Stack batch results
        band_thresholds = torch.stack(batch_results)  # (batch, bands, time)
        
        # Apply learnable adaptations
        adapted_thresholds = band_thresholds * self.masking_adaptation.unsqueeze(0).unsqueeze(-1)
        adapted_thresholds = adapted_thresholds + self.threshold_offset.unsqueeze(0).unsqueeze(-1)
        
        # Temporal smoothing
        smoothed_thresholds = self.temporal_smoother(adapted_thresholds)
        
        return {
            'band_thresholds': smoothed_thresholds,
            'raw_thresholds': band_thresholds,
            'masking_adaptation': self.masking_adaptation,
            'threshold_offset': self.threshold_offset
        }


def integrate_masking_with_watermark(watermark_spectrum: np.ndarray,
                                   masking_thresholds: np.ndarray,
                                   band_indices: np.ndarray,
                                   safety_factor: float = 0.5) -> np.ndarray:
    """
    Integrate masking thresholds with watermark embedding process.
    
    Args:
        watermark_spectrum: Watermark signal in frequency domain
        masking_thresholds: Masking thresholds per critical band
        band_indices: Mapping of frequency bins to critical bands
        safety_factor: Safety factor to ensure imperceptibility (0-1)
        
    Returns:
        Scaled watermark spectrum that respects masking thresholds
    """
    scaled_watermark = watermark_spectrum.copy()
    n_bands = len(masking_thresholds)
    
    for band_idx in range(n_bands):
        # Find frequency bins in this band
        band_mask = band_indices == band_idx
        
        if np.any(band_mask):
            # Current watermark power in this band
            band_watermark = scaled_watermark[band_mask]
            current_power = np.mean(np.abs(band_watermark) ** 2)
            
            # Target power based on masking threshold
            target_power = masking_thresholds[band_idx] * safety_factor
            
            if current_power > target_power and target_power > 0:
                # Scale down watermark to respect masking threshold
                scale_factor = np.sqrt(target_power / current_power)
                scaled_watermark[band_mask] *= scale_factor
    
    return scaled_watermark
