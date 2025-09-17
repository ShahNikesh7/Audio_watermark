"""
SoundSafe.ai Audio Watermark Attack Suite
==========================================

This module provides comprehensive attack simulations for testing watermark robustness.
Includes various types of attacks from benign audio processing to malicious evasion attempts.

Available attack categories:
- Augmentations: Noise, reverb, time stretching (audiomentations)
- Compression: MP3/AAC/OGG lossy compression (pydub)
- Filtering: EQ, clipping, phasing, distortion (scipy)
- Malicious: Watermark evasion, inversion, cut-paste attacks

Usage:
    from attacks import simulate_attack
    
    # Run a specific attack
    attacked_audio = simulate_attack(audio, attack_type="mp3_compression", severity=0.5)
    
    # Run comprehensive attack suite
    from attacks.simulate import run_attack_suite
    results = run_attack_suite(audio, watermark_data)
"""

from .simulate import simulate_attack, run_attack_suite
from .augmentations import *
from .compression import *
from .filtering import *
from .malicious import *

__version__ = "1.0.0"
__author__ = "SoundSafe.ai Team"

# Available attack types
ATTACK_TYPES = [
    # Augmentations
    "additive_noise", "gaussian_noise", "reverb", "time_stretch", "pitch_shift",
    "background_noise", "impulse_response", "room_simulation",
    
    # Compression
    "mp3_compression", "aac_compression", "ogg_compression", "opus_compression",
    "variable_bitrate", "lossy_format_conversion",
    
    # Filtering
    "lowpass_filter", "highpass_filter", "bandpass_filter", "notch_filter",
    "equalization", "dynamic_range_compression", "limiter", "clipping",
    "phase_shift", "all_pass_filter", "comb_filter", "harmonic_distortion",
    "sample_suppression", "median_filter", "resampling", "amplitude_scaling", "quantization", "echo_addition",
    
    # Malicious
    "watermark_inversion", "cut_and_paste", "collage_attack", "averaging_attack",
    "desynchronization", "temporal_desync", "frequency_masking", "replacement_attack"
]

# Severity levels
SEVERITY_LEVELS = {
    "mild": 0.2,
    "moderate": 0.5,
    "severe": 0.8,
    "extreme": 1.0
}
