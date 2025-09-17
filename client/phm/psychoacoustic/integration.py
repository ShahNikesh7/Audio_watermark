# File: client/phm/psychoacoustic/integration.py
"""
Micro Task 1.1.1.3 & 1.1.2.2: Integrate masking thresholds with watermark embedding
and optionally apply adaptive strength control (temporal domain).

This module handles perceptual scaling of watermark energy, including optional
adaptive gain control using TemporalAdaptiveStrength model.
Implements frequency-domain embedding using STFT and extraction using
correlation-based detection.
"""
import numpy as np
import librosa
import scipy.signal
from typing import Dict, Any, Optional, Tuple


def integrate_masking_with_watermark(
    watermark_spectrum: np.ndarray,
    masking_data: Dict[str, np.ndarray],
    safety_factor: float = 0.5
) -> np.ndarray:
    if safety_factor <= 0:
        raise ValueError("Safety factor must be positive")

    band_indices = masking_data['band_indices']
    band_thresholds = masking_data['band_thresholds']
    n_bands = band_thresholds.shape[0]

    if np.iscomplexobj(watermark_spectrum):
        power = np.abs(watermark_spectrum)**2
    else:
        power = watermark_spectrum**2

    scales = np.ones_like(power)

    for b in range(n_bands):
        mask = (band_indices == b)
        if not np.any(mask):
            continue
        current_power = np.mean(power[mask])
        allowed = band_thresholds[b] * safety_factor
        if allowed <= 0:
            scales[mask] = 0.0
        elif current_power > 0:
            factor = np.sqrt(allowed / current_power)
            scales[mask] = min(factor, 1.0)

    return watermark_spectrum * scales


def generate_dummy_watermark(freq_bins: int, time_frames: int) -> np.ndarray:
    np.random.seed(42)
    pattern = np.random.choice([-1.0, 1.0], size=(freq_bins, time_frames)) * 1e-4
    return pattern


def embed_watermark(audio: np.ndarray,
                    watermark_spec: np.ndarray,
                    n_fft: int = 1000,
                    hop_length: int = 512) -> np.ndarray:
    """
    Embed watermark by adding scaled watermark to STFT magnitude spectrum.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)

    # Align shapes
    min_time = min(watermark_spec.shape[1], mag.shape[1])
    mag[:, :min_time] += watermark_spec[:, :min_time]

    watermarked = mag[:, :min_time] * np.exp(1j * phase[:, :min_time])
    return librosa.istft(watermarked, hop_length=hop_length, length=len(audio))


def extract_watermark(audio: np.ndarray,
                      original_watermark: np.ndarray,
                      n_fft: int = 1000,
                      hop_length: int = 512) -> Tuple[np.ndarray, float]:
    """
    Extract watermark from signal using correlation with original watermark.
    Returns estimated spectrum and similarity score.
    """
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)

    freq_bins, time_frames = original_watermark.shape
    extracted = mag[:freq_bins, :time_frames]

    # Normalize
    orig_norm = original_watermark / (np.linalg.norm(original_watermark) + 1e-10)
    extracted_norm = extracted / (np.linalg.norm(extracted) + 1e-10)

    # Correlation
    similarity = np.sum(orig_norm * extracted_norm)
    return extracted, similarity


def full_integration_pipeline(
    audio: np.ndarray,
    raw_watermark_spec: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 1000,
    hop_length: int = 512,
    n_bands: int = 24,
    safety_factor: float = 0.5,
    adaptive_model: Optional[Any] = None
) -> Dict[str, Any]:
    from client.phm.psychoacoustic.band_thresholds import BandThresholdCalculator
    from client.phm.adaptive.adaptive_strength_module import apply_temporal_adaptive_gain

    btc = BandThresholdCalculator(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_bands=n_bands
    )
    results = btc.compute(audio)

    band_indices = results['band_indices']
    band_thresholds = results['band_thresholds']

    if adaptive_model is not None:
        import torch
        adaptive_model.eval()
        with torch.no_grad():
            B = 1
            band_thresh_tensor = torch.from_numpy(band_thresholds.reshape(n_bands, -1)).unsqueeze(0).float()
            watermark_tensor = torch.from_numpy(raw_watermark_spec).unsqueeze(0).float()
            gains = adaptive_model(band_thresh_tensor)
            scaled_tensor = apply_temporal_adaptive_gain(watermark_tensor, gains, band_indices)
            scaled_watermark = scaled_tensor.squeeze(0).cpu().numpy()
    else:
        masking_data = {
            'band_indices': band_indices,
            'band_thresholds': band_thresholds
        }
        scaled_watermark = integrate_masking_with_watermark(
            raw_watermark_spec,
            masking_data,
            safety_factor
        )

    # Embed into audio
    watermarked_audio = embed_watermark(audio, scaled_watermark, n_fft=n_fft, hop_length=hop_length)

    results['scaled_watermark'] = scaled_watermark
    results['watermarked_audio'] = watermarked_audio
    return results
