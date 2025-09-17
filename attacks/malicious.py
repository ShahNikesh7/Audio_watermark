"""
Malicious Attacks for Watermark Security Testing
==============================================

This module implements various malicious attacks specifically designed to
evade, remove, or corrupt audio watermarks. These attacks are used for
security testing and robustness evaluation.

Attack types include:
- Watermark inversion and removal attempts
- Cut-and-paste attacks
- Collage attacks (mixing multiple watermarked sources)
- Averaging attacks
- Desynchronization attacks
- Frequency masking attacks
- Replacement attacks
"""

import numpy as np
from typing import Optional, Union, List, Dict, Any, Tuple
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


def watermark_inversion(audio: np.ndarray, sample_rate: int = 44100, 
                       severity: float = 0.5, watermark_estimate: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Attempt to invert/remove watermark by estimation and subtraction.
    
    Args:
        audio: Input audio signal (potentially watermarked)
        sample_rate: Sample rate
        severity: Inversion severity (0.0 to 1.0)
        watermark_estimate: Estimated watermark signal
    
    Returns:
        Audio with attempted watermark removal
    """
    if watermark_estimate is None:
        # Estimate watermark using high-pass filtering
        # Assumption: watermark is high-frequency content
        nyquist = sample_rate / 2
        cutoff = 8000 / nyquist  # 8kHz cutoff
        
        b, a = signal.butter(4, cutoff, btype='high')
        watermark_estimate = signal.filtfilt(b, a, audio)
        
        # Scale estimate based on severity
        watermark_estimate *= severity
    
    # Attempt watermark removal
    cleaned_audio = audio - watermark_estimate
    
    # Apply adaptive filtering to smooth artifacts
    if severity > 0.5:
        # Additional smoothing for aggressive removal
        window_size = int(0.01 * sample_rate)  # 10ms window
        kernel = np.ones(window_size) / window_size
        cleaned_audio = np.convolve(cleaned_audio, kernel, mode='same')
    
    return cleaned_audio


def cut_and_paste(audio: np.ndarray, sample_rate: int = 44100, 
                 severity: float = 0.5, segment_length: Optional[float] = None,
                 num_segments: Optional[int] = None) -> np.ndarray:
    """
    Apply cut-and-paste attack by rearranging audio segments.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        segment_length: Length of segments in seconds
        num_segments: Number of segments to rearrange
    
    Returns:
        Rearranged audio
    """
    if segment_length is None:
        # Map severity to segment length (0.5s to 0.05s)
        segment_length = 0.5 - severity * 0.45
    
    segment_samples = int(segment_length * sample_rate)
    
    if num_segments is None:
        # Calculate number of segments
        num_segments = len(audio) // segment_samples
    
    # Create segments
    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = min(start + segment_samples, len(audio))
        segments.append(audio[start:end])
    
    # Shuffle segments based on severity
    np.random.seed(42)  # For reproducibility
    shuffle_indices = np.arange(len(segments))
    
    # Shuffle a percentage of segments based on severity
    num_shuffled = int(severity * len(segments))
    shuffled_indices = np.random.choice(len(segments), num_shuffled, replace=False)
    
    # Rearrange shuffled segments
    shuffled_segments = [segments[i] for i in shuffled_indices]
    np.random.shuffle(shuffled_segments)
    
    # Reconstruct audio
    result = np.zeros_like(audio)
    current_pos = 0
    
    shuffle_idx = 0
    for i, segment in enumerate(segments):
        if i in shuffled_indices:
            # Use shuffled segment
            segment = shuffled_segments[shuffle_idx]
            shuffle_idx += 1
        
        # Place segment in result
        end_pos = min(current_pos + len(segment), len(result))
        result[current_pos:end_pos] = segment[:end_pos-current_pos]
        current_pos = end_pos
    
    return result


def collage_attack(audio_list: List[np.ndarray], sample_rate: int = 44100, 
                  severity: float = 0.5, mix_method: str = "average") -> np.ndarray:
    """
    Apply collage attack by mixing multiple watermarked audio sources.
    
    Args:
        audio_list: List of audio signals to mix
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        mix_method: Method for mixing ("average", "weighted", "random")
    
    Returns:
        Mixed audio result
    """
    if len(audio_list) == 0:
        raise ValueError("At least one audio signal required")
    
    if len(audio_list) == 1:
        return audio_list[0]
    
    # Find minimum length
    min_length = min(len(audio) for audio in audio_list)
    
    # Truncate all audio to same length
    truncated_audio = [audio[:min_length] for audio in audio_list]
    
    if mix_method == "average":
        # Simple averaging
        result = np.mean(truncated_audio, axis=0)
        
    elif mix_method == "weighted":
        # Weighted mixing based on severity
        weights = np.random.uniform(0.1, 1.0, len(truncated_audio))
        weights = weights / np.sum(weights)
        
        # Apply severity to weight distribution
        if severity > 0.5:
            # More uneven weighting
            weights = weights ** (1 + severity)
            weights = weights / np.sum(weights)
        
        result = np.sum([w * audio for w, audio in zip(weights, truncated_audio)], axis=0)
        
    elif mix_method == "random":
        # Random sample-by-sample mixing
        np.random.seed(42)
        result = np.zeros(min_length)
        
        for i in range(min_length):
            # Choose random source for each sample
            if np.random.random() < severity:
                # Random selection
                source_idx = np.random.randint(len(truncated_audio))
                result[i] = truncated_audio[source_idx][i]
            else:
                # Use first source
                result[i] = truncated_audio[0][i]
    
    else:
        # Default to average
        result = np.mean(truncated_audio, axis=0)
    
    return result


def averaging_attack(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, window_size: Optional[int] = None) -> np.ndarray:
    """
    Apply averaging attack to smooth out watermark.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        window_size: Averaging window size in samples
    
    Returns:
        Averaged audio
    """
    if window_size is None:
        # Map severity to window size (1ms to 10ms)
        window_size = int((1 + severity * 9) * sample_rate / 1000)
    
    # Apply moving average
    kernel = np.ones(window_size) / window_size
    averaged_audio = np.convolve(audio, kernel, mode='same')
    
    # Mix with original based on severity
    result = (1 - severity) * audio + severity * averaged_audio
    
    return result


def desynchronization(audio: np.ndarray, sample_rate: int = 44100, 
                     severity: float = 0.5, desync_type: str = "jitter") -> np.ndarray:
    """
    Apply desynchronization attack.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        desync_type: Type of desynchronization ("jitter", "drift", "random")
    
    Returns:
        Desynchronized audio
    """
    if desync_type == "jitter":
        # Random time jitter
        max_jitter = int(severity * 0.001 * sample_rate)  # Max 1ms jitter
        
        result = np.zeros_like(audio)
        np.random.seed(42)
        
        for i in range(len(audio)):
            jitter = np.random.randint(-max_jitter, max_jitter + 1)
            source_idx = i + jitter
            
            if 0 <= source_idx < len(audio):
                result[i] = audio[source_idx]
            else:
                result[i] = 0  # Silence for out-of-bounds
        
        return result
    
    elif desync_type == "drift":
        # Time drift (gradual timing change)
        max_drift = severity * 0.01  # Max 1% drift
        
        # Create time warping function
        time_indices = np.arange(len(audio))
        drift_factor = 1 + max_drift * np.sin(2 * np.pi * time_indices / len(audio))
        
        # Apply time warping
        warped_indices = np.cumsum(drift_factor)
        warped_indices = warped_indices * len(audio) / warped_indices[-1]
        
        # Interpolate to get warped audio
        result = np.interp(time_indices, warped_indices, audio)
        
        return result
    
    elif desync_type == "random":
        # Random sample drops/insertions
        drop_rate = severity * 0.1  # Max 10% drop rate
        
        np.random.seed(42)
        result = []
        
        for i in range(len(audio)):
            if np.random.random() > drop_rate:
                result.append(audio[i])
            elif np.random.random() < 0.5:
                # Double sample (insertion)
                result.append(audio[i])
                result.append(audio[i])
        
        # Convert to numpy array and ensure same length
        result = np.array(result)
        if len(result) > len(audio):
            result = result[:len(audio)]
        elif len(result) < len(audio):
            result = np.pad(result, (0, len(audio) - len(result)), 'constant')
        
        return result
    
    else:
        return audio


def temporal_desync(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, shift_samples: Optional[int] = None) -> np.ndarray:
    """
    Apply temporal desynchronization by shifting audio in time.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        shift_samples: Number of samples to shift
    
    Returns:
        Time-shifted audio
    """
    if shift_samples is None:
        # Map severity to shift amount (0 to 1000 samples)
        shift_samples = int(severity * 1000)
    
    # Apply circular shift
    shifted_audio = np.roll(audio, shift_samples)
    
    # Optional: Add fade in/out to reduce artifacts
    if severity > 0.5:
        fade_samples = min(100, shift_samples // 2)
        
        # Fade out beginning
        fade_out = np.linspace(1, 0, fade_samples)
        shifted_audio[:fade_samples] *= fade_out
        
        # Fade in end
        fade_in = np.linspace(0, 1, fade_samples)
        shifted_audio[-fade_samples:] *= fade_in
    
    return shifted_audio


def frequency_masking(audio: np.ndarray, sample_rate: int = 44100, 
                     severity: float = 0.5, mask_bands: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Apply frequency masking attack by adding noise in specific frequency bands.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        mask_bands: List of (low_freq, high_freq) tuples to mask
    
    Returns:
        Frequency-masked audio
    """
    if mask_bands is None:
        # Default masking bands (common watermark frequencies)
        mask_bands = [(2000, 4000), (6000, 8000), (10000, 12000)]
    
    result = audio.copy()
    
    for low_freq, high_freq in mask_bands:
        # Generate band-limited noise
        noise = np.random.normal(0, 1, len(audio))
        
        # Filter noise to band
        nyquist = sample_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        # Ensure valid frequency range
        low_norm = max(0.01, min(0.98, low_norm))
        high_norm = max(0.02, min(0.99, high_norm))
        
        if low_norm < high_norm:
            b, a = signal.butter(4, [low_norm, high_norm], btype='band')
            band_noise = signal.filtfilt(b, a, noise)
            
            # Scale noise based on severity
            noise_level = severity * 0.1 * np.std(audio)
            band_noise = band_noise * noise_level / np.std(band_noise)
            
            # Add to result
            result += band_noise
    
    return result


def replacement_attack(audio: np.ndarray, sample_rate: int = 44100, 
                      severity: float = 0.5, replacement_audio: Optional[np.ndarray] = None,
                      segment_length: Optional[float] = None) -> np.ndarray:
    """
    Apply replacement attack by replacing segments with other audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        replacement_audio: Audio to use for replacement
        segment_length: Length of replacement segments in seconds
    
    Returns:
        Audio with replaced segments
    """
    if replacement_audio is None:
        # Generate replacement audio (noise or silence)
        replacement_audio = np.random.normal(0, 0.1, len(audio))
    
    if segment_length is None:
        # Map severity to segment length (0.1s to 1.0s)
        segment_length = 0.1 + severity * 0.9
    
    segment_samples = int(segment_length * sample_rate)
    num_segments = len(audio) // segment_samples
    
    # Determine number of segments to replace
    num_replacements = int(severity * num_segments)
    
    # Choose random segments to replace
    np.random.seed(42)
    replacement_indices = np.random.choice(num_segments, num_replacements, replace=False)
    
    result = audio.copy()
    
    for idx in replacement_indices:
        start = idx * segment_samples
        end = min(start + segment_samples, len(audio))
        
        # Replace segment
        if end - start <= len(replacement_audio):
            result[start:end] = replacement_audio[:end-start]
        else:
            # Repeat replacement audio if needed
            replacement_segment = np.tile(replacement_audio, 
                                       (end - start) // len(replacement_audio) + 1)
            result[start:end] = replacement_segment[:end-start]
    
    return result


def watermark_estimation_attack(audio: np.ndarray, sample_rate: int = 44100, 
                               severity: float = 0.5, estimation_method: str = "spectral") -> np.ndarray:
    """
    Attack by estimating and removing watermark using various methods.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
        estimation_method: Method for watermark estimation
    
    Returns:
        Audio with estimated watermark removed
    """
    if estimation_method == "spectral":
        # Spectral estimation - assume watermark is high-frequency
        fft_audio = fft(audio)
        freqs = fftfreq(len(audio), 1/sample_rate)
        
        # Create mask for high frequencies
        high_freq_mask = np.abs(freqs) > 4000  # Above 4kHz
        
        # Estimate watermark spectrum
        watermark_spectrum = fft_audio.copy()
        watermark_spectrum[~high_freq_mask] = 0
        
        # Scale by severity
        watermark_spectrum *= severity
        
        # Remove estimated watermark
        cleaned_spectrum = fft_audio - watermark_spectrum
        cleaned_audio = np.real(ifft(cleaned_spectrum))
        
        return cleaned_audio
    
    elif estimation_method == "temporal":
        # Temporal estimation - assume watermark is high-frequency temporal pattern
        # Apply high-pass filter to estimate watermark
        nyquist = sample_rate / 2
        cutoff = 6000 / nyquist  # 6kHz cutoff
        
        b, a = signal.butter(6, cutoff, btype='high')
        estimated_watermark = signal.filtfilt(b, a, audio)
        
        # Scale by severity
        estimated_watermark *= severity
        
        # Remove estimated watermark
        cleaned_audio = audio - estimated_watermark
        
        return cleaned_audio
    
    elif estimation_method == "adaptive":
        # Adaptive estimation using local statistics
        window_size = int(0.1 * sample_rate)  # 100ms windows
        
        result = audio.copy()
        
        for i in range(0, len(audio), window_size):
            end = min(i + window_size, len(audio))
            segment = audio[i:end]
            
            # Estimate local watermark as deviation from mean
            local_mean = np.mean(segment)
            local_std = np.std(segment)
            
            # Estimate watermark as high-deviation samples
            watermark_estimate = segment - local_mean
            watermark_estimate = np.where(np.abs(watermark_estimate) > local_std, 
                                        watermark_estimate, 0)
            
            # Remove estimated watermark
            result[i:end] = segment - severity * watermark_estimate
        
        return result
    
    else:
        # Default: simple high-pass filtering
        nyquist = sample_rate / 2
        cutoff = 5000 / nyquist  # 5kHz cutoff
        
        b, a = signal.butter(4, cutoff, btype='high')
        estimated_watermark = signal.filtfilt(b, a, audio)
        
        return audio - severity * estimated_watermark


def synchronized_averaging(audio_list: List[np.ndarray], sample_rate: int = 44100, 
                          severity: float = 0.5) -> np.ndarray:
    """
    Apply synchronized averaging attack across multiple copies.
    
    Args:
        audio_list: List of audio signals (same content, different watermarks)
        sample_rate: Sample rate
        severity: Attack severity (0.0 to 1.0)
    
    Returns:
        Averaged audio with reduced watermark
    """
    if len(audio_list) == 0:
        raise ValueError("At least one audio signal required")
    
    if len(audio_list) == 1:
        return audio_list[0]
    
    # Find minimum length
    min_length = min(len(audio) for audio in audio_list)
    
    # Truncate all audio to same length
    truncated_audio = [audio[:min_length] for audio in audio_list]
    
    # Compute average
    averaged = np.mean(truncated_audio, axis=0)
    
    # Mix with original based on severity
    original = truncated_audio[0]  # Use first signal as reference
    result = (1 - severity) * original + severity * averaged
    
    return result


# Export all functions
__all__ = [
    "watermark_inversion", "cut_and_paste", "collage_attack", "averaging_attack",
    "desynchronization", "temporal_desync", "frequency_masking", "replacement_attack",
    "watermark_estimation_attack", "synchronized_averaging"
]
