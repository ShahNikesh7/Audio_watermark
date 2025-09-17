# File: client/phm/adaptive/adaptive_strength_module.py
"""
Task 1.1.2.1: Adaptive Watermarking in the Temporal Domain

This module dynamically adjusts watermark strength based on perceptual masking
thresholds. It integrates tightly with the psychoacoustic module from Task 1.1.1.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from client.phm.psychoacoustic.band_thresholds import BandThresholdCalculator
from client.phm.psychoacoustic.integration import generate_dummy_watermark  # Replace with actual gen fn


class TemporalAdaptiveStrength(nn.Module):
    """
    Learns a per-band, per-frame gain mask using masking thresholds and temporal context.
    """
    def __init__(self,
                 n_bands: int = 24,
                 hidden_channels: int = 64,
                 kernel_size: int = 5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv1d(n_bands, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, n_bands, kernel_size=3, padding=1),
            nn.Sigmoid()  # Limit gains ∈ [0, 1]
        )

    def forward(self, band_thresholds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            band_thresholds: (batch_size, n_bands, time) - masking thresholds

        Returns:
            gain_mask: (batch_size, n_bands, time) - adaptive gains per band
        """
        return self.network(band_thresholds)


def apply_temporal_adaptive_gain(
    watermark: torch.Tensor,
    gains: torch.Tensor,
    band_indices: np.ndarray
) -> torch.Tensor:
    """
    Apply per-band adaptive gains to a watermark spectrum.

    Args:
        watermark: Tensor (batch, freq_bins, time)
        gains: Tensor (batch, n_bands, time)
        band_indices: Numpy array (freq_bins,) mapping bins → bands

    Returns:
        scaled_watermark: Tensor (batch, freq_bins, time)
    """
    B, F, T = watermark.shape
    band_tensor = torch.from_numpy(band_indices).long().to(watermark.device)  # (F,)

    # Build a gain matrix per frequency bin
    gain_map = torch.zeros(B, F, T, device=watermark.device)

    for b in range(gains.shape[1]):
        bin_mask = (band_tensor == b).unsqueeze(0).unsqueeze(-1)  # (1, F, 1)
        gain_slice = gains[:, b:b+1, :]  # (B, 1, T)
        gain_map += bin_mask * gain_slice

    return watermark * gain_map


def integrate_temporal_adaptive_module(
    audio: np.ndarray,
    sample_rate: int,
    adaptive_model: TemporalAdaptiveStrength,
    n_fft: int = 1000,
    hop_length: int = 512,
    n_bands: int = 24
) -> torch.Tensor:
    """
    Integrated pipeline: (1) compute masking thresholds (2) generate watermark (3) scale with adaptive gains.

    Args:
        audio: numpy (samples,) input audio
        sample_rate: sample rate of audio
        adaptive_model: instance of TemporalAdaptiveStrength

    Returns:
        scaled watermark tensor: (1, freq_bins, time)
    """
    # Step 1: Psychoacoustic analysis
    btc = BandThresholdCalculator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands
    )
    results = btc.compute(audio)

    band_thresholds = results['band_thresholds']
    band_indices = results['band_indices']

    # Step 2: Watermark generation (placeholder)
    raw_watermark = generate_dummy_watermark(results['power_spectrum'].shape[0], audio.shape[0] // hop_length)

    # Step 3: Adaptively scale watermark
    adaptive_model.eval()
    with torch.no_grad():
        band_thresh_tensor = torch.from_numpy(band_thresholds.reshape(n_bands, -1)).unsqueeze(0).float()
        watermark_tensor = torch.from_numpy(raw_watermark).unsqueeze(0).float()

        gains = adaptive_model(band_thresh_tensor)
        scaled = apply_temporal_adaptive_gain(watermark_tensor, gains, band_indices)

    return scaled
