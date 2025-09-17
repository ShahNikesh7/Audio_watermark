#!/usr/bin/env python3
"""
Generate test audio files for watermark evaluation.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# Configuration
TEST_AUDIO_DIR = Path("data/test_audio")
SAMPLE_RATE = 22050
DURATION = 3.0  # seconds

def generate_test_audio():
    """Generate various test audio files."""
    
    # 1. Sine wave tone
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))
    sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz A note
    sf.write(TEST_AUDIO_DIR / "sine_440hz.wav", sine_wave, SAMPLE_RATE)
    
    # 2. Chirp signal (frequency sweep)
    f0, f1 = 100, 2000  # Start and end frequencies
    chirp = 0.3 * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / DURATION / 2) * t)
    sf.write(TEST_AUDIO_DIR / "chirp_100_2000hz.wav", chirp, SAMPLE_RATE)
    
    # 3. White noise
    white_noise = 0.1 * np.random.randn(int(SAMPLE_RATE * DURATION))
    sf.write(TEST_AUDIO_DIR / "white_noise.wav", white_noise, SAMPLE_RATE)
    
    # 4. Pink noise (1/f spectrum)
    pink_noise = 0.1 * generate_pink_noise(int(SAMPLE_RATE * DURATION))
    sf.write(TEST_AUDIO_DIR / "pink_noise.wav", pink_noise, SAMPLE_RATE)
    
    # 5. Complex harmonic signal
    harmonics = np.zeros_like(t)
    for harmonic in range(1, 6):  # First 5 harmonics
        harmonics += (0.2 / harmonic) * np.sin(2 * np.pi * 220 * harmonic * t)
    sf.write(TEST_AUDIO_DIR / "harmonics_220hz.wav", harmonics, SAMPLE_RATE)
    
    # 6. Impulse train
    impulse_train = np.zeros_like(t)
    impulse_interval = int(SAMPLE_RATE * 0.1)  # 10 Hz impulses
    impulse_train[::impulse_interval] = 0.5
    sf.write(TEST_AUDIO_DIR / "impulse_train.wav", impulse_train, SAMPLE_RATE)
    
    print(f"Generated 6 test audio files in {TEST_AUDIO_DIR}")


def generate_pink_noise(n_samples):
    """Generate pink noise (1/f spectrum)."""
    # Generate white noise
    white = np.random.randn(n_samples)
    
    # Apply pink noise filter in frequency domain
    fft_white = np.fft.fft(white)
    freqs = np.fft.fftfreq(n_samples)
    
    # Apply 1/f filtering (avoid division by zero)
    mask = np.abs(freqs) > 0
    fft_pink = fft_white.copy()
    fft_pink[mask] = fft_white[mask] / np.sqrt(np.abs(freqs[mask]))
    
    # Convert back to time domain
    pink = np.real(np.fft.ifft(fft_pink))
    
    # Normalize
    pink = pink / np.max(np.abs(pink))
    
    return pink


if __name__ == "__main__":
    generate_test_audio()
