"""
Audio Filtering Attacks for Watermark Robustness Testing
======================================================

This module implements various audio filtering attacks using scipy and other libraries
for testing watermark robustness against frequency domain manipulations.

Filtering types include:
- Frequency domain filters (lowpass, highpass, bandpass, notch)
- Dynamic range processing (compression, limiting, expansion)
- Distortion effects (clipping, saturation, harmonic distortion)
- Phase manipulation (phase shift, all-pass filters)
- Time-domain effects (comb filtering, flanging, phasing)
"""

import numpy as np
from typing import Optional, Union, Tuple
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
import warnings
warnings.filterwarnings('ignore')


def lowpass_filter(audio: np.ndarray, sample_rate: int = 44100, 
                  severity: float = 0.5, cutoff: Optional[float] = None) -> np.ndarray:
    """
    Apply lowpass filter to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        cutoff: Cutoff frequency in Hz (None for automatic)
    
    Returns:
        Lowpass filtered audio
    """
    if cutoff is None:
        # Map severity to cutoff frequency (20kHz to 4kHz)
        cutoff = 20000 - severity * 16000
    
    # Design Butterworth lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is within valid range
    normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))
    
    # Design filter
    b, a = signal.butter(4, normalized_cutoff, btype='low')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def highpass_filter(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, cutoff: Optional[float] = None) -> np.ndarray:
    """
    Apply highpass filter to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        cutoff: Cutoff frequency in Hz (None for automatic)
    
    Returns:
        Highpass filtered audio
    """
    if cutoff is None:
        # Map severity to cutoff frequency (20Hz to 1kHz)
        cutoff = 20 + severity * 980
    
    # Design Butterworth highpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff / nyquist
    
    # Ensure cutoff is within valid range
    normalized_cutoff = max(0.01, min(0.99, normalized_cutoff))
    
    # Design filter
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def bandpass_filter(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, low_freq: Optional[float] = None, 
                   high_freq: Optional[float] = None) -> np.ndarray:
    """
    Apply bandpass filter to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        low_freq: Low cutoff frequency in Hz
        high_freq: High cutoff frequency in Hz
    
    Returns:
        Bandpass filtered audio
    """
    if low_freq is None:
        # Map severity to frequency range
        low_freq = 100 + severity * 400  # 100Hz to 500Hz
    
    if high_freq is None:
        high_freq = 8000 - severity * 4000  # 8kHz to 4kHz
    
    # Ensure proper frequency ordering
    if low_freq >= high_freq:
        low_freq = high_freq * 0.5
    
    # Design Butterworth bandpass filter
    nyquist = sample_rate / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    # Ensure frequencies are within valid range
    low_normalized = max(0.01, min(0.98, low_normalized))
    high_normalized = max(0.02, min(0.99, high_normalized))
    
    # Design filter
    b, a = signal.butter(4, [low_normalized, high_normalized], btype='band')
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def notch_filter(audio: np.ndarray, sample_rate: int = 44100, 
                severity: float = 0.5, notch_freq: Optional[float] = None,
                q_factor: Optional[float] = None) -> np.ndarray:
    """
    Apply notch filter to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        notch_freq: Notch frequency in Hz
        q_factor: Q factor of the notch
    
    Returns:
        Notch filtered audio
    """
    if notch_freq is None:
        # Common notch frequencies (50Hz, 60Hz, 1kHz, 5kHz)
        notch_frequencies = [50, 60, 1000, 5000]
        notch_freq = notch_frequencies[int(severity * (len(notch_frequencies) - 1))]
    
    if q_factor is None:
        # Map severity to Q factor (higher severity = sharper notch)
        q_factor = 1 + severity * 9  # Q from 1 to 10
    
    # Design notch filter
    nyquist = sample_rate / 2
    normalized_freq = notch_freq / nyquist
    
    # Ensure frequency is within valid range
    normalized_freq = max(0.01, min(0.99, normalized_freq))
    
    # Design IIR notch filter
    b, a = signal.iirnotch(normalized_freq, q_factor)
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def equalization(audio: np.ndarray, sample_rate: int = 44100, 
                severity: float = 0.5, eq_type: str = "graphic") -> np.ndarray:
    """
    Apply equalization to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: EQ severity (0.0 to 1.0)
        eq_type: Type of equalization ("graphic", "parametric", "shelf")
    
    Returns:
        Equalized audio
    """
    if eq_type == "graphic":
        # Graphic EQ with standard frequency bands
        frequencies = [60, 170, 310, 600, 1000, 3000, 6000, 12000, 14000, 16000]
        
        # Generate random gain adjustments based on severity
        np.random.seed(42)  # For reproducibility
        gains = np.random.uniform(-severity * 12, severity * 12, len(frequencies))
        
        # Apply each frequency band
        result = audio.copy()
        for freq, gain in zip(frequencies, gains):
            if gain != 0:
                # Design peaking filter for this frequency
                nyquist = sample_rate / 2
                normalized_freq = freq / nyquist
                
                if 0.01 < normalized_freq < 0.99:
                    # Use peaking filter approximation with bandpass
                    Q = 1.0
                    bw = normalized_freq / Q
                    low_freq = max(0.01, normalized_freq - bw/2)
                    high_freq = min(0.99, normalized_freq + bw/2)
                    
                    # Create bandpass filter and apply gain
                    b, a = signal.butter(2, [low_freq, high_freq], btype='band')
                    filtered = signal.filtfilt(b, a, result)
                    
                    # Mix with original based on gain
                    gain_linear = 10**(gain/20)
                    result = result + filtered * (gain_linear - 1)
        
        return result
    
    elif eq_type == "parametric":
        # Parametric EQ with configurable frequency, Q, and gain
        center_freq = 1000 + severity * 2000  # 1kHz to 3kHz
        q_factor = 0.5 + severity * 2.5  # Q from 0.5 to 3.0
        gain = (severity - 0.5) * 12  # ±6dB
        
        nyquist = sample_rate / 2
        normalized_freq = center_freq / nyquist
        
        if 0.01 < normalized_freq < 0.99:
            # Use peaking filter approximation with bandpass
            Q = q_factor
            bw = normalized_freq / Q
            low_freq = max(0.01, normalized_freq - bw/2)
            high_freq = min(0.99, normalized_freq + bw/2)
            
            # Create bandpass filter and apply gain
            b, a = signal.butter(2, [low_freq, high_freq], btype='band')
            filtered = signal.filtfilt(b, a, audio)
            
            # Mix with original based on gain
            gain_linear = 10**(gain/20)
            return audio + filtered * (gain_linear - 1)
        else:
            return audio
    
    elif eq_type == "shelf":
        # Shelf EQ (high and low shelf)
        shelf_freq = 1000 + severity * 4000  # 1kHz to 5kHz
        shelf_gain = (severity - 0.5) * 12  # ±6dB
        
        nyquist = sample_rate / 2
        normalized_freq = shelf_freq / nyquist
        
        if 0.01 < normalized_freq < 0.99:
            # High shelf filter
            b, a = signal.iirfilter(2, normalized_freq, btype='high', 
                                  ftype='butter', output='ba')
            return signal.filtfilt(b, a, audio) * (10**(shelf_gain/20))
        else:
            return audio
    
    else:
        return audio


def dynamic_range_compression(audio: np.ndarray, sample_rate: int = 44100, 
                            severity: float = 0.5, threshold: Optional[float] = None,
                            ratio: Optional[float] = None) -> np.ndarray:
    """
    Apply dynamic range compression.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        threshold: Compression threshold in dB
        ratio: Compression ratio
    
    Returns:
        Compressed audio
    """
    if threshold is None:
        # Map severity to threshold (-40dB to -10dB)
        threshold = -40 + severity * 30
    
    if ratio is None:
        # Map severity to ratio (1:1 to 20:1)
        ratio = 1 + severity * 19
    
    # Convert to dB
    audio_db = 20 * np.log10(np.maximum(np.abs(audio), 1e-10))
    
    # Apply compression
    compressed_db = audio_db.copy()
    above_threshold = audio_db > threshold
    
    # Compress signals above threshold
    compressed_db[above_threshold] = threshold + (audio_db[above_threshold] - threshold) / ratio
    
    # Convert back to linear scale
    compressed_audio = np.sign(audio) * (10 ** (compressed_db / 20))
    
    # Apply makeup gain to restore average level
    makeup_gain = np.mean(np.abs(audio)) / np.mean(np.abs(compressed_audio))
    compressed_audio *= makeup_gain
    
    return compressed_audio


def limiter(audio: np.ndarray, sample_rate: int = 44100, 
           severity: float = 0.5, threshold: Optional[float] = None) -> np.ndarray:
    """
    Apply audio limiter.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Limiter severity (0.0 to 1.0)
        threshold: Limiter threshold
    
    Returns:
        Limited audio
    """
    if threshold is None:
        # Map severity to threshold (0.9 to 0.1)
        threshold = 0.9 - severity * 0.8
    
    # Apply hard limiting
    limited_audio = np.clip(audio, -threshold, threshold)
    
    # Apply soft limiting for more natural sound
    soft_threshold = threshold * 0.8
    
    # Soft limit above soft threshold
    above_soft = np.abs(audio) > soft_threshold
    limited_audio[above_soft] = np.sign(audio[above_soft]) * (
        soft_threshold + (threshold - soft_threshold) * 
        np.tanh((np.abs(audio[above_soft]) - soft_threshold) / (threshold - soft_threshold))
    )
    
    return limited_audio


def clipping(audio: np.ndarray, sample_rate: int = 44100, 
            severity: float = 0.5, clip_level: Optional[float] = None) -> np.ndarray:
    """
    Apply clipping distortion.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Clipping severity (0.0 to 1.0)
        clip_level: Clipping level
    
    Returns:
        Clipped audio
    """
    if clip_level is None:
        # Map severity to clipping level (0.8 to 0.1)
        clip_level = 0.8 - severity * 0.7
    
    # Apply hard clipping
    clipped_audio = np.clip(audio, -clip_level, clip_level)
    
    return clipped_audio


def phase_shift(audio: np.ndarray, sample_rate: int = 44100, 
               severity: float = 0.5, phase_deg: Optional[float] = None) -> np.ndarray:
    """
    Apply phase shift to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Phase shift severity (0.0 to 1.0)
        phase_deg: Phase shift in degrees
    
    Returns:
        Phase-shifted audio
    """
    if phase_deg is None:
        # Map severity to phase shift (0° to 180°)
        phase_deg = severity * 180
    
    # Convert to radians
    phase_rad = np.radians(phase_deg)
    
    # Apply phase shift in frequency domain
    fft_audio = fft(audio)
    
    # Create phase shift vector
    phase_shift_vector = np.exp(1j * phase_rad)
    
    # Apply phase shift
    fft_shifted = fft_audio * phase_shift_vector
    
    # Convert back to time domain
    shifted_audio = np.real(ifft(fft_shifted))
    
    return shifted_audio


def all_pass_filter(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, center_freq: Optional[float] = None) -> np.ndarray:
    """
    Apply all-pass filter (phase-only filtering).
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        center_freq: Center frequency for all-pass filter
    
    Returns:
        All-pass filtered audio
    """
    if center_freq is None:
        # Map severity to center frequency (100Hz to 5kHz)
        center_freq = 100 + severity * 4900
    
    # Design all-pass filter
    nyquist = sample_rate / 2
    normalized_freq = center_freq / nyquist
    
    # Ensure frequency is within valid range
    normalized_freq = max(0.01, min(0.99, normalized_freq))
    
    # All-pass filter coefficients
    r = 0.9  # Pole radius
    theta = 2 * np.pi * normalized_freq
    
    # Filter coefficients
    a = [1, -2*r*np.cos(theta), r**2]
    b = [r**2, -2*r*np.cos(theta), 1]
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def comb_filter(audio: np.ndarray, sample_rate: int = 44100, 
               severity: float = 0.5, delay_ms: Optional[float] = None,
               feedback: Optional[float] = None) -> np.ndarray:
    """
    Apply comb filter effect.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        delay_ms: Delay in milliseconds
        feedback: Feedback amount
    
    Returns:
        Comb-filtered audio
    """
    if delay_ms is None:
        # Map severity to delay (1ms to 20ms)
        delay_ms = 1 + severity * 19
    
    if feedback is None:
        # Map severity to feedback (0.0 to 0.7)
        feedback = severity * 0.7
    
    # Calculate delay in samples
    delay_samples = int(delay_ms * sample_rate / 1000)
    
    # Initialize output array
    output = np.zeros(len(audio))
    
    # Apply comb filter
    for i in range(len(audio)):
        output[i] = audio[i]
        
        # Add delayed and feedback signal
        if i >= delay_samples:
            output[i] += feedback * output[i - delay_samples]
    
    return output


def harmonic_distortion(audio: np.ndarray, sample_rate: int = 44100, 
                       severity: float = 0.5, distortion_type: str = "tube") -> np.ndarray:
    """
    Apply harmonic distortion.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Distortion severity (0.0 to 1.0)
        distortion_type: Type of distortion ("tube", "transistor", "digital")
    
    Returns:
        Distorted audio
    """
    if distortion_type == "tube":
        # Tube-style soft saturation
        drive = 1 + severity * 9  # Drive from 1 to 10
        distorted = np.tanh(audio * drive) / drive
        
    elif distortion_type == "transistor":
        # Transistor-style asymmetric distortion
        drive = 1 + severity * 4
        distorted = audio * drive
        
        # Asymmetric clipping
        pos_clip = 0.7 - severity * 0.3
        neg_clip = -0.8 + severity * 0.3
        distorted = np.clip(distorted, neg_clip, pos_clip)
        
    elif distortion_type == "digital":
        # Digital bit-crushing effect
        bits = 16 - int(severity * 8)  # 16-bit to 8-bit
        levels = 2 ** bits
        
        # Quantize audio
        distorted = np.round(audio * levels) / levels
        
    else:
        # Default: simple overdrive
        drive = 1 + severity * 3
        distorted = np.tanh(audio * drive)
    
    return distorted


def spectral_subtraction(audio: np.ndarray, sample_rate: int = 44100, 
                        severity: float = 0.5, noise_floor: Optional[float] = None) -> np.ndarray:
    """
    Apply spectral subtraction (noise reduction technique that can affect watermarks).
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Subtraction severity (0.0 to 1.0)
        noise_floor: Noise floor estimate
    
    Returns:
        Spectrally processed audio
    """
    if noise_floor is None:
        # Estimate noise floor from quiet sections
        noise_floor = np.percentile(np.abs(audio), 10 + severity * 20)
    
    # Apply FFT
    fft_audio = fft(audio)
    magnitude = np.abs(fft_audio)
    phase = np.angle(fft_audio)
    
    # Spectral subtraction
    alpha = severity * 2.0  # Over-subtraction factor
    
    # Subtract noise spectrum
    subtracted_magnitude = magnitude - alpha * noise_floor
    
    # Prevent negative values
    subtracted_magnitude = np.maximum(subtracted_magnitude, 0.1 * magnitude)
    
    # Reconstruct signal
    reconstructed_fft = subtracted_magnitude * np.exp(1j * phase)
    reconstructed_audio = np.real(ifft(reconstructed_fft))
    
    return reconstructed_audio


def sample_suppression(audio: np.ndarray, sample_rate: int = 44100, 
                      severity: float = 0.5, threshold: Optional[float] = None) -> np.ndarray:
    """
    Apply sample suppression attack by removing samples below threshold.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Suppression severity (0.0 to 1.0)
        threshold: Suppression threshold
    
    Returns:
        Sample-suppressed audio
    """
    if threshold is None:
        # Map severity to threshold (0.01 to 0.3)
        threshold = 0.01 + severity * 0.29
    
    # Suppress samples below threshold
    suppressed_audio = audio.copy()
    below_threshold = np.abs(audio) < threshold
    suppressed_audio[below_threshold] = 0
    
    return suppressed_audio


def median_filter(audio: np.ndarray, sample_rate: int = 44100, 
                 severity: float = 0.5, kernel_size: Optional[int] = None) -> np.ndarray:
    """
    Apply median filtering to audio signal.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Filter severity (0.0 to 1.0)
        kernel_size: Median filter kernel size
    
    Returns:
        Median-filtered audio
    """
    from scipy import ndimage
    
    if kernel_size is None:
        # Map severity to kernel size (3 to 21 samples)
        kernel_size = int(3 + severity * 18)
        # Ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1
    
    # Apply median filter
    filtered_audio = ndimage.median_filter(audio, size=kernel_size)
    
    return filtered_audio


def resampling(audio: np.ndarray, sample_rate: int = 44100, 
              severity: float = 0.5, target_rate: Optional[int] = None) -> np.ndarray:
    """
    Apply resampling attack by changing sample rate and resampling back.
    
    Args:
        audio: Input audio signal
        sample_rate: Original sample rate
        severity: Resampling severity (0.0 to 1.0)
        target_rate: Target sample rate for resampling
    
    Returns:
        Resampled audio
    """
    import librosa
    
    if target_rate is None:
        # Map severity to target rate (8kHz to 22kHz)
        target_rate = int(8000 + severity * 14000)
    
    # Resample to target rate and back
    resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_rate)
    restored = librosa.resample(resampled, orig_sr=target_rate, target_sr=sample_rate)
    
    # Ensure output length matches input
    if len(restored) != len(audio):
        if len(restored) > len(audio):
            restored = restored[:len(audio)]
        else:
            padding = len(audio) - len(restored)
            restored = np.pad(restored, (0, padding), mode='edge')
    
    return restored


def amplitude_scaling(audio: np.ndarray, sample_rate: int = 44100, 
                     severity: float = 0.5, scale_factor: Optional[float] = None) -> np.ndarray:
    """
    Apply amplitude scaling attack.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Scaling severity (0.0 to 1.0)
        scale_factor: Amplitude scale factor
    
    Returns:
        Amplitude-scaled audio
    """
    if scale_factor is None:
        # Map severity to scale factor (0.1 to 2.0)
        if severity < 0.5:
            scale_factor = 0.1 + severity * 1.8  # 0.1 to 1.0
        else:
            scale_factor = 1.0 + (severity - 0.5) * 2.0  # 1.0 to 2.0
    
    # Apply amplitude scaling
    scaled_audio = audio * scale_factor
    
    # Clip to prevent overflow
    scaled_audio = np.clip(scaled_audio, -1.0, 1.0)
    
    return scaled_audio


def quantization(audio: np.ndarray, sample_rate: int = 44100, 
                severity: float = 0.5, bits: Optional[int] = None) -> np.ndarray:
    """
    Apply quantization attack by reducing bit depth.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Quantization severity (0.0 to 1.0)
        bits: Target bit depth
    
    Returns:
        Quantized audio
    """
    if bits is None:
        # Map severity to bit depth (4 to 16 bits)
        bits = int(16 - severity * 12)
    
    # Quantize to target bit depth
    levels = 2 ** bits
    quantized_audio = np.round(audio * levels) / levels
    
    return quantized_audio


def echo_addition(audio: np.ndarray, sample_rate: int = 44100, 
                 severity: float = 0.5, delay_ms: Optional[float] = None,
                 decay: Optional[float] = None) -> np.ndarray:
    """
    Apply echo addition attack.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Echo severity (0.0 to 1.0)
        delay_ms: Echo delay in milliseconds
        decay: Echo decay factor
    
    Returns:
        Audio with echo
    """
    if delay_ms is None:
        # Map severity to delay (50ms to 500ms)
        delay_ms = 50 + severity * 450
    
    if decay is None:
        # Map severity to decay (0.1 to 0.7)
        decay = 0.1 + severity * 0.6
    
    # Calculate delay in samples
    delay_samples = int(delay_ms * sample_rate / 1000)
    
    # Create echo
    echo_audio = audio.copy()
    
    # Add delayed and decayed signal
    if delay_samples < len(audio):
        echo_audio[delay_samples:] += decay * audio[:-delay_samples]
    
    return echo_audio


# Export all functions
__all__ = [
    "lowpass_filter", "highpass_filter", "bandpass_filter", "notch_filter",
    "equalization", "dynamic_range_compression", "limiter", "clipping",
    "phase_shift", "all_pass_filter", "comb_filter", "harmonic_distortion",
    "spectral_subtraction", "sample_suppression", "median_filter",
    "resampling", "amplitude_scaling", "quantization", "echo_addition"
]
