"""
Audio Augmentations for Watermark Robustness Testing
==================================================

This module implements various audio augmentations using audiomentations library
for testing watermark robustness against common audio processing operations.

Augmentations include:
- Additive noise (Gaussian, uniform, colored)
- Reverb and room simulation
- Time stretching and pitch shifting
- Background noise addition
- Impulse response convolution
"""

import numpy as np
import librosa
from typing import Optional, Union, Dict, Any
from audiomentations import (
    AddGaussianNoise, AddBackgroundNoise, TimeStretch, PitchShift,
    RoomSimulator, ApplyImpulseResponse, Normalize, Compose
)
import warnings
warnings.filterwarnings('ignore')


def additive_noise(audio: np.ndarray, sample_rate: int = 44100, 
                  severity: float = 0.5, noise_type: str = "gaussian") -> np.ndarray:
    """
    Add noise to audio signal.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate of audio
        severity: Noise severity (0.0 to 1.0)
        noise_type: Type of noise ("gaussian", "uniform", "pink")
    
    Returns:
        Noisy audio signal
    """
    if noise_type == "gaussian":
        # SNR ranges from 40dB (mild) to 10dB (severe)
        snr_db = 40 - severity * 30
        noise_power = np.mean(audio**2) / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        return audio + noise
    
    elif noise_type == "uniform":
        noise_amplitude = severity * 0.1
        noise = np.random.uniform(-noise_amplitude, noise_amplitude, audio.shape)
        return audio + noise
    
    elif noise_type == "pink":
        # Generate pink noise using inverse FFT
        N = len(audio)
        freqs = np.fft.fftfreq(N, 1/sample_rate)
        # Pink noise has 1/f spectrum
        spectrum = np.zeros(N, dtype=complex)
        spectrum[1:N//2] = np.random.normal(0, 1, N//2-1) / np.sqrt(np.abs(freqs[1:N//2]))
        spectrum[N//2+1:] = np.conj(spectrum[1:N//2][::-1])
        pink_noise = np.fft.ifft(spectrum).real
        
        # Scale by severity
        pink_noise = pink_noise * severity * 0.1
        return audio + pink_noise
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def gaussian_noise(audio: np.ndarray, sample_rate: int = 44100, 
                  severity: float = 0.5) -> np.ndarray:
    """Add Gaussian noise using audiomentations."""
    # Convert severity to SNR range
    min_snr = 40 - severity * 30  # 40dB to 10dB
    max_snr = min_snr + 5
    
    try:
        # Try different parameter names for different audiomentations versions
        transform = AddGaussianNoise(
            min_amplitude=0.001,
            max_amplitude=severity * 0.1,
            p=1.0
        )
    except TypeError:
        # Fallback: manual Gaussian noise
        noise_power = severity * 0.05
        noise = np.random.normal(0, noise_power, audio.shape)
        return audio + noise
    
    return transform(samples=audio, sample_rate=sample_rate)


def reverb(audio: np.ndarray, sample_rate: int = 44100, 
          severity: float = 0.5, room_size: Optional[float] = None) -> np.ndarray:
    """
    Apply reverb effect to audio using RoomSimulator.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Reverb intensity (0.0 to 1.0)
        room_size: Room size parameter (None for automatic)
    
    Returns:
        Reverberated audio
    """
    if room_size is None:
        room_size = 0.2 + severity * 0.6  # 0.2 to 0.8
    
    # Use RoomSimulator as reverb alternative
    try:
        transform = RoomSimulator(
            room_size=room_size,
            absorption_value=0.2 + severity * 0.6,
            damping=0.2 + severity * 0.4,
            p=1.0
        )
        return transform(samples=audio, sample_rate=sample_rate)
    except Exception as e:
        # Fallback: simple echo-based reverb
        delay_samples = int(0.05 * sample_rate)  # 50ms delay
        decay = 0.3 + severity * 0.4
        
        reverb_audio = audio.copy()
        for i in range(3):  # 3 echoes
            delay = delay_samples * (i + 1)
            if delay < len(audio):
                reverb_audio[delay:] += audio[:-delay] * (decay ** (i + 1))
        
        return reverb_audio


def time_stretch(audio: np.ndarray, sample_rate: int = 44100, 
                severity: float = 0.5, stretch_factor: Optional[float] = None) -> np.ndarray:
    """
    Apply time stretching without pitch change.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Stretch severity (0.0 to 1.0)
        stretch_factor: Stretch factor (None for automatic)
    
    Returns:
        Time-stretched audio
    """
    if stretch_factor is None:
        # Severity 0.5 = no change, 0.0 = 0.8x speed, 1.0 = 1.2x speed
        stretch_factor = 0.8 + severity * 0.4
    
    try:
        transform = TimeStretch(
            min_rate=stretch_factor,
            max_rate=stretch_factor,
            leave_length_unchanged=False,
            p=1.0
        )
        return transform(samples=audio, sample_rate=sample_rate)
    except:
        # Fallback: use librosa
        return librosa.effects.time_stretch(audio, rate=stretch_factor)


def pitch_shift(audio: np.ndarray, sample_rate: int = 44100, 
               severity: float = 0.5, semitones: Optional[float] = None) -> np.ndarray:
    """
    Apply pitch shifting without time change.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Pitch shift severity (0.0 to 1.0)
        semitones: Semitones to shift (None for automatic)
    
    Returns:
        Pitch-shifted audio
    """
    if semitones is None:
        # Severity 0.5 = no change, range ±2 semitones
        semitones = (severity - 0.5) * 4
    
    try:
        transform = PitchShift(
            min_semitones=semitones,
            max_semitones=semitones,
            p=1.0
        )
        return transform(samples=audio, sample_rate=sample_rate)
    except:
        # Fallback: use librosa
        return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=semitones)


def background_noise(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, noise_type: str = "white") -> np.ndarray:
    """
    Add background noise to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Noise severity (0.0 to 1.0)
        noise_type: Type of background noise
    
    Returns:
        Audio with background noise
    """
    # Generate background noise
    if noise_type == "white":
        noise = np.random.normal(0, 1, audio.shape)
    elif noise_type == "pink":
        # Generate pink noise
        N = len(audio)
        freqs = np.fft.fftfreq(N, 1/sample_rate)
        spectrum = np.zeros(N, dtype=complex)
        spectrum[1:N//2] = np.random.normal(0, 1, N//2-1) / np.sqrt(np.abs(freqs[1:N//2]))
        spectrum[N//2+1:] = np.conj(spectrum[1:N//2][::-1])
        noise = np.fft.ifft(spectrum).real
    elif noise_type == "brown":
        # Generate brown noise
        N = len(audio)
        freqs = np.fft.fftfreq(N, 1/sample_rate)
        spectrum = np.zeros(N, dtype=complex)
        spectrum[1:N//2] = np.random.normal(0, 1, N//2-1) / np.abs(freqs[1:N//2])
        spectrum[N//2+1:] = np.conj(spectrum[1:N//2][::-1])
        noise = np.fft.ifft(spectrum).real
    else:
        noise = np.random.normal(0, 1, audio.shape)
    
    # Normalize and scale noise
    noise = noise / np.std(noise)
    
    # SNR based on severity
    snr_db = 30 - severity * 20  # 30dB to 10dB
    noise_power = np.mean(audio**2) / (10**(snr_db/10))
    noise = noise * np.sqrt(noise_power)
    
    return audio + noise


def impulse_response(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, ir_type: str = "room") -> np.ndarray:
    """
    Apply impulse response convolution using ApplyImpulseResponse.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Effect severity (0.0 to 1.0)
        ir_type: Type of impulse response
    
    Returns:
        Audio convolved with impulse response
    """
    # Generate synthetic impulse response
    if ir_type == "room":
        # Room impulse response with exponential decay
        length = int(0.5 * sample_rate)  # 0.5 second IR
        t = np.arange(length) / sample_rate
        
        # Exponential decay with some early reflections
        decay_rate = 2 + severity * 6  # 2 to 8 decay rate
        ir = np.exp(-decay_rate * t)
        
        # Add some early reflections
        for i in range(3):
            delay = int(0.01 * (i + 1) * sample_rate)
            if delay < length:
                ir[delay] += 0.3 * np.exp(-decay_rate * t[delay])
        
        # Add some random variations
        ir += np.random.normal(0, 0.01, length)
        
    elif ir_type == "hall":
        # Hall impulse response with longer decay
        length = int(1.0 * sample_rate)  # 1 second IR
        t = np.arange(length) / sample_rate
        
        decay_rate = 1 + severity * 3  # 1 to 4 decay rate
        ir = np.exp(-decay_rate * t)
        
        # Add multiple reflections
        for i in range(5):
            delay = int(0.02 * (i + 1) * sample_rate)
            if delay < length:
                ir[delay] += 0.2 * np.exp(-decay_rate * t[delay])
    
    else:
        # Default room response
        length = int(0.3 * sample_rate)
        t = np.arange(length) / sample_rate
        ir = np.exp(-3 * t)
    
    # Normalize impulse response
    ir = ir / np.max(np.abs(ir))
    
    # Apply wet/dry mix based on severity
    wet_level = severity * 0.5
    dry_level = 1.0 - wet_level * 0.5
    
    # Convolve with impulse response
    convolved = np.convolve(audio, ir, mode='same')
    
    # Mix wet and dry signals
    result = dry_level * audio + wet_level * convolved
    
    return result


def room_simulation(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, room_type: str = "medium") -> np.ndarray:
    """
    Simulate room acoustics using RoomSimulator.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Room effect severity (0.0 to 1.0)
        room_type: Type of room simulation
    
    Returns:
        Audio with room simulation
    """
    # Room parameters based on type
    if room_type == "small":
        room_size = 0.2 + severity * 0.3  # 0.2 to 0.5
        absorption = 0.3 + severity * 0.4  # 0.3 to 0.7
    elif room_type == "medium":
        room_size = 0.4 + severity * 0.3  # 0.4 to 0.7
        absorption = 0.2 + severity * 0.5  # 0.2 to 0.7
    elif room_type == "large":
        room_size = 0.6 + severity * 0.3  # 0.6 to 0.9
        absorption = 0.1 + severity * 0.4  # 0.1 to 0.5
    else:
        room_size = 0.4 + severity * 0.3
        absorption = 0.2 + severity * 0.5
    
    # Try to use RoomSimulator
    try:
        transform = RoomSimulator(
            room_size=room_size,
            absorption_value=absorption,
            damping=0.3 + severity * 0.4,
            p=1.0
        )
        return transform(samples=audio, sample_rate=sample_rate)
    except Exception as e:
        # Fallback: manual room simulation
        result = audio.copy()
        
        # Calculate reflection parameters
        if room_type == "small":
            delay_time = 0.02  # 20ms
            decay_time = 0.3   # 300ms
            reflections = 3
        elif room_type == "medium":
            delay_time = 0.03  # 30ms
            decay_time = 0.6   # 600ms
            reflections = 5
        elif room_type == "large":
            delay_time = 0.05  # 50ms
            decay_time = 1.2   # 1.2s
            reflections = 7
        else:
            delay_time = 0.03
            decay_time = 0.6
            reflections = 5
        
        # Apply room simulation
        for i in range(reflections):
            delay_samples = int(delay_time * (i + 1) * sample_rate)
            if delay_samples < len(audio):
                # Calculate reflection amplitude with decay
                reflection_amp = 0.3 * np.exp(-i / decay_time) * severity
                
                # Add reflection
                result[delay_samples:] += audio[:-delay_samples] * reflection_amp
        
        # Apply slight filtering to simulate absorption
        if severity > 0.3:
            # Simple lowpass filter for high frequencies absorption
            from scipy import signal
            b, a = signal.butter(2, 8000 / (sample_rate / 2), btype='lowpass')
            result = signal.filtfilt(b, a, result)
        
        return result


# Compose transforms for batch processing
def create_augmentation_pipeline(severity: float = 0.5, 
                               transforms: Optional[list] = None) -> Compose:
    """
    Create a pipeline of audio augmentations.
    
    Args:
        severity: Overall severity of augmentations
        transforms: List of transform names to include
    
    Returns:
        Composed augmentation pipeline
    """
    if transforms is None:
        transforms = ["gaussian_noise", "room_simulation", "time_stretch"]
    
    pipeline = []
    
    for transform_name in transforms:
        if transform_name == "gaussian_noise":
            pipeline.append(AddGaussianNoise(
                min_snr_in_db=40-severity*30,
                max_snr_in_db=45-severity*30,
                p=0.7
            ))
        elif transform_name == "room_simulation":
            pipeline.append(RoomSimulator(
                room_size=0.2+severity*0.6,
                absorption_value=0.2+severity*0.6,
                damping=0.3+severity*0.4,
                p=0.5
            ))
        elif transform_name == "time_stretch":
            stretch_range = 0.1 * severity
            pipeline.append(TimeStretch(
                min_rate=1.0-stretch_range,
                max_rate=1.0+stretch_range,
                p=0.3
            ))
    
    # Always normalize at the end
    pipeline.append(Normalize(p=1.0))
    
    return Compose(pipeline)


# Export all functions
__all__ = [
    "additive_noise", "gaussian_noise", "reverb", "time_stretch", "pitch_shift",
    "background_noise", "impulse_response", "room_simulation", 
    "create_augmentation_pipeline"
]
