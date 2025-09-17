# File: client/phm/psychoacoustic/band_thresholds.py
"""
Micro Task 1.1.1.2: Calculate per-critical-band masking thresholds

This module derives per-band thresholds from the global masking curve produced by
the Moore-Glasberg psychoacoustic model. It includes robust input validation,
handles edge cases (e.g., short or multi-channel signals), and caches intermediate
computations for reuse.
"""
import numpy as np
import librosa
from typing import Tuple, Dict, Union

from client.phm.psychoacoustic.moore_glasberg import MooreGlasbergAnalyzer


class BandThresholdCalculator:
    """
    Computes per-band masking thresholds using Bark-scale critical bands.
    """
    def __init__(
        self,
        sample_rate: int,
    n_fft: int = 1000,
        hop_length: int = 512,
        n_bands: int = 24
    ):
        # Validate inputs
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_fft <= 0 or hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive integers")
        if n_bands < 1:
            raise ValueError("n_bands must be at least 1")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands

        # Precompute FFT bin frequencies
        self.frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        # Precompute band mapping
        self.band_indices, self.band_centers = self._compute_band_mapping()

        # Instantiate psychoacoustic analyzer
        self._analyzer = MooreGlasbergAnalyzer(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def _compute_band_mapping(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a Bark filterbank and map each FFT bin to a critical band index.

        Returns:
            band_indices: array of length freq_bins mapping each bin to [0..n_bands-1]
            band_centers: center frequency (Hz) of each band
        """
        # Convert frequencies to Bark scale
        bark_freqs = self._hz_to_bark(self.frequencies)
        
        # Create band edges in Bark scale
        bark_max = self._hz_to_bark(self.sample_rate / 2)
        bark_edges = np.linspace(0, bark_max, self.n_bands + 1)
        
        # Assign each frequency bin to a band
        band_indices = np.digitize(bark_freqs, bark_edges) - 1
        band_indices = np.clip(band_indices, 0, self.n_bands - 1)

        # Compute band centers as mean frequency of bins in that band
        band_centers = np.array([
            np.mean(self.frequencies[band_indices == b]) if np.any(band_indices == b) else 0.0
            for b in range(self.n_bands)
        ])

        return band_indices, band_centers
    
    def _hz_to_bark(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Convert frequencies in Hz to Bark scale using Traunmüller's formula.
        
        Args:
            frequencies: Frequencies in Hz
            
        Returns:
            Bark scale values
        """
        return 26.81 * frequencies / (1960 + frequencies) - 0.53

    def compute(
        self,
        audio: Union[np.ndarray, Tuple[np.ndarray, int]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute global and per-band masking thresholds.

        Args:
            audio: 1D mono array or 2D array (channels x samples) of audio samples,
                   or a tuple (audio_array, sample_rate).

        Returns:
            dict with:
              'frequencies': (freq_bins,)
              'band_indices': (freq_bins,)
              'band_centers': (n_bands,)
              'global_masking': (freq_bins,)
              'band_thresholds': (n_bands,)
              'power_spectrum': (freq_bins,)
        """
        # Unpack if tuple provided
        if isinstance(audio, tuple):
            audio_array, sr = audio
            if sr != self.sample_rate:
                raise ValueError(f"Expected sample_rate {self.sample_rate}, got {sr}")
        else:
            audio_array = audio

        # Handle multi-channel: average channels
        if audio_array.ndim == 2:
            audio_mono = np.mean(audio_array, axis=0)
        elif audio_array.ndim == 1:
            audio_mono = audio_array
        else:
            raise ValueError("audio must be 1D or 2D numpy array of samples")

        # Zero-pad if shorter than n_fft
        if audio_mono.size < self.n_fft:
            pad_amount = self.n_fft - audio_mono.size
            audio_mono = np.pad(audio_mono, (0, pad_amount), mode='constant')

        # 1. Compute STFT and power spectrum (freq_bins, frames)
        stft = librosa.stft(
            y=audio_mono,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=True
        )
        power_spec = np.mean(np.abs(stft)**2, axis=1)

        # 2. Compute global per-bin masking threshold
        global_mask = self._analyzer.masking_threshold(audio_mono)
        if global_mask.shape != power_spec.shape:
            raise RuntimeError(
                f"Mask length {global_mask.shape} != freq_bins {power_spec.shape}"
            )

        # 3. Compute per-band thresholds
        band_thresholds = np.zeros(self.n_bands)
        for b in range(self.n_bands):
            mask = (self.band_indices == b)
            if np.any(mask):
                band_thresholds[b] = np.mean(global_mask[mask])
            else:
                band_thresholds[b] = 0.0

        return {
            'frequencies': self.frequencies,
            'band_indices': self.band_indices,
            'band_centers': self.band_centers,
            'global_masking': global_mask,
            'band_thresholds': band_thresholds,
            'power_spectrum': power_spec
        }


def calculate_per_band_threshold(
    audio: Union[np.ndarray, Tuple[np.ndarray, int]],
    sample_rate: int = 44100,
    n_fft: int = 1000,
    hop_length: int = 512,
    n_bands: int = 24
) -> Dict[str, np.ndarray]:
    """
    Functional API wrapper: instantiate BandThresholdCalculator and compute thresholds.

    Args:
        audio: 1D or 2D numpy array (or tuple (array, sr)).
    """
    calculator = BandThresholdCalculator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands
    )
    return calculator.compute(audio)
