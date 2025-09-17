"""
🚀 Master Attack Runner - simulate.py
====================================

Orchestrates all attack types and provides unified interface for testing watermark robustness.
Supports batch processing, severity scaling, and comprehensive attack suites.
"""

import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path
import json
import time

# Import attack modules
from .augmentations import (
    additive_noise, gaussian_noise, reverb, 
    time_stretch, pitch_shift, background_noise, impulse_response, room_simulation
)
from .compression import (
    mp3_compression, aac_compression, ogg_compression,
    opus_compression, variable_bitrate, lossy_format_conversion
)
from .filtering import (
    lowpass_filter, highpass_filter, bandpass_filter, notch_filter,
    equalization, dynamic_range_compression, limiter, clipping, phase_shift,
    all_pass_filter, comb_filter, harmonic_distortion,
    sample_suppression, median_filter as median_filter_attack, resampling as resampling_attack,
    amplitude_scaling, quantization as quantization_attack, echo_addition
)
from .malicious import (
    watermark_inversion, cut_and_paste, collage_attack,
    averaging_attack, desynchronization, temporal_desync, frequency_masking, replacement_attack
)

logger = logging.getLogger(__name__)

# Attack type mappings
ATTACK_FUNCTIONS = {
    # Augmentations
    "additive_noise": additive_noise,
    "gaussian_noise": gaussian_noise,
    "reverb": reverb,
    "time_stretch": time_stretch,
    "pitch_shift": pitch_shift,
    "background_noise": background_noise,
    "impulse_response": impulse_response,
    "room_simulation": room_simulation,
    
    # Compression
    "mp3_compression": mp3_compression,
    "aac_compression": aac_compression,
    "ogg_compression": ogg_compression,
    "opus_compression": opus_compression,
    "variable_bitrate": variable_bitrate,
    "lossy_format_conversion": lossy_format_conversion,
    
    # Filtering
    "lowpass_filter": lowpass_filter,
    "highpass_filter": highpass_filter,
    "bandpass_filter": bandpass_filter,
    "notch_filter": notch_filter,
    "equalization": equalization,
    "dynamic_range_compression": dynamic_range_compression,
    "limiter": limiter,
    "clipping": clipping,
    "phase_shift": phase_shift,
    "all_pass_filter": all_pass_filter,
    "comb_filter": comb_filter,
    "harmonic_distortion": harmonic_distortion,
    "sample_suppression": sample_suppression,
    "median_filter": median_filter_attack,
    "resampling": resampling_attack,
    "amplitude_scaling": amplitude_scaling,
    "quantization": quantization_attack,
    "echo_addition": echo_addition,
    
    # Malicious
    "watermark_inversion": watermark_inversion,
    "cut_and_paste": cut_and_paste,
    "collage_attack": collage_attack,
    "averaging_attack": averaging_attack,
    "desynchronization": desynchronization,
    "temporal_desync": temporal_desync,
    "frequency_masking": frequency_masking,
    "replacement_attack": replacement_attack,
}


def simulate_attack(audio: np.ndarray, 
                   attack_type: str, 
                   severity: float = 0.5,
                   sample_rate: int = 44100,
                   **kwargs) -> np.ndarray:
    """
    Apply a single attack to audio signal.
    
    Args:
        audio: Input audio signal
        attack_type: Type of attack to apply
        severity: Attack severity (0.0 = mild, 1.0 = extreme)
        sample_rate: Audio sample rate
        **kwargs: Additional parameters for specific attacks
        
    Returns:
        Attacked audio signal
        
    Raises:
        ValueError: If attack_type is not supported
    """
    if attack_type not in ATTACK_FUNCTIONS:
        available = ", ".join(ATTACK_FUNCTIONS.keys())
        raise ValueError(f"Unsupported attack type '{attack_type}'. Available: {available}")
    
    logger.info(f"Applying {attack_type} attack with severity {severity}")
    
    try:
        attack_func = ATTACK_FUNCTIONS[attack_type]
        
        # Special handling for collage_attack which requires multiple audio inputs
        if attack_type == "collage_attack":
            # Create multiple copies with slight variations for collage attack
            audio_list = [audio]
            for i in range(2):  # Create 2 additional variants
                variant = audio + np.random.normal(0, 0.01, audio.shape)
                audio_list.append(variant)
            attacked_audio = attack_func(audio_list, severity=severity, sample_rate=sample_rate, **kwargs)
        else:
            attacked_audio = attack_func(audio, severity=severity, sample_rate=sample_rate, **kwargs)
        
        # Ensure output is same length as input
        if len(attacked_audio) != len(audio):
            logger.warning(f"Attack {attack_type} changed audio length: {len(audio)} -> {len(attacked_audio)}")
        
        return attacked_audio
        
    except Exception as e:
        logger.error(f"Attack {attack_type} failed: {e}")
        return audio  # Return original audio if attack fails


def run_attack_suite(audio: np.ndarray,
                    watermark_data: Optional[str] = None,
                    sample_rate: int = 44100,
                    attack_types: Optional[List[str]] = None,
                    severity_levels: Optional[List[float]] = None,
                    include_baseline: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive attack suite for watermark robustness testing.
    
    Args:
        audio: Input audio signal
        watermark_data: Original watermark data for comparison
        sample_rate: Audio sample rate
        attack_types: List of attacks to test (None = all attacks)
        severity_levels: List of severity levels to test
        include_baseline: Whether to include unattacked baseline
        
    Returns:
        Dictionary containing attack results and statistics
    """
    if attack_types is None:
        attack_types = list(ATTACK_FUNCTIONS.keys())
    
    if severity_levels is None:
        severity_levels = [0.2, 0.5, 0.8]
    
    logger.info(f"Running attack suite: {len(attack_types)} attacks × {len(severity_levels)} severities")
    
    results = {
        "metadata": {
            "audio_length": len(audio),
            "sample_rate": sample_rate,
            "attack_types": attack_types,
            "severity_levels": severity_levels,
            "timestamp": time.time()
        },
        "attacks": {},
        "statistics": {}
    }
    
    # Baseline (no attack)
    if include_baseline:
        results["attacks"]["baseline"] = {
            "severity": 0.0,
            "audio": audio.copy(),
            "success": True,
            "error": None
        }
    
    # Run attacks
    total_attacks = len(attack_types) * len(severity_levels)
    completed = 0
    failed = 0
    
    for attack_type in attack_types:
        results["attacks"][attack_type] = {}
        
        for severity in severity_levels:
            attack_key = f"severity_{severity}"
            
            try:
                start_time = time.time()
                attacked_audio = simulate_attack(audio, attack_type, severity, sample_rate)
                duration = time.time() - start_time
                
                results["attacks"][attack_type][attack_key] = {
                    "severity": severity,
                    "audio": attacked_audio,
                    "success": True,
                    "duration": duration,
                    "error": None
                }
                completed += 1
                
            except Exception as e:
                logger.error(f"Attack {attack_type} (severity {severity}) failed: {e}")
                results["attacks"][attack_type][attack_key] = {
                    "severity": severity,
                    "audio": audio,  # Return original if failed
                    "success": False,
                    "duration": 0.0,
                    "error": str(e)
                }
                failed += 1
    
    # Calculate statistics
    results["statistics"] = {
        "total_attacks": total_attacks,
        "completed": completed,
        "failed": failed,
        "success_rate": completed / total_attacks if total_attacks > 0 else 0.0,
        "attack_categories": {
            "augmentations": len([a for a in attack_types if a in ["additive_noise", "gaussian_noise", "reverb", "time_stretch", "pitch_shift", "background_noise"]]),
            "compression": len([a for a in attack_types if a in ["mp3_compression", "aac_compression", "ogg_compression", "opus_compression", "variable_bitrate", "format_conversion"]]),
            "filtering": len([a for a in attack_types if a in ["lowpass_filter", "highpass_filter", "bandpass_filter", "equalization", "dynamic_compression", "clipping", "phase_shift"]]),
            "malicious": len([a for a in attack_types if a in ["watermark_inversion", "cut_and_paste", "collage_attack", "averaging_attack", "desynchronization", "frequency_masking"]])
        }
    }
    
    logger.info(f"Attack suite completed: {completed}/{total_attacks} successful ({failed} failed)")
    return results


def batch_attack_test(audio_files: List[Union[str, Path]],
                     attack_configs: List[Dict[str, Any]],
                     output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Run batch attack testing on multiple audio files.
    
    Args:
        audio_files: List of audio file paths
        attack_configs: List of attack configurations
        output_dir: Directory to save results (optional)
        
    Returns:
        Batch testing results
    """
    import librosa
    
    logger.info(f"Starting batch attack test: {len(audio_files)} files × {len(attack_configs)} configs")
    
    batch_results = {
        "metadata": {
            "num_files": len(audio_files),
            "num_configs": len(attack_configs),
            "timestamp": time.time()
        },
        "files": {},
        "summary": {}
    }
    
    for file_path in audio_files:
        file_name = Path(file_path).name
        logger.info(f"Processing file: {file_name}")
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            
            batch_results["files"][file_name] = {
                "path": str(file_path),
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "attacks": {}
            }
            
            # Run attack configurations
            for i, config in enumerate(attack_configs):
                config_name = f"config_{i}"
                attack_type = config.get("attack_type", "gaussian_noise")
                severity = config.get("severity", 0.5)
                
                try:
                    attacked_audio = simulate_attack(audio, attack_type, severity, sr)
                    
                    batch_results["files"][file_name]["attacks"][config_name] = {
                        "config": config,
                        "success": True,
                        "audio_length": len(attacked_audio),
                        "error": None
                    }
                    
                except Exception as e:
                    batch_results["files"][file_name]["attacks"][config_name] = {
                        "config": config,
                        "success": False,
                        "audio_length": 0,
                        "error": str(e)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")
            batch_results["files"][file_name] = {
                "path": str(file_path),
                "error": str(e),
                "attacks": {}
            }
    
    # Calculate summary statistics
    total_tests = len(audio_files) * len(attack_configs)
    successful_tests = sum(
        len([a for a in file_data.get("attacks", {}).values() if a.get("success", False)])
        for file_data in batch_results["files"].values()
    )
    
    batch_results["summary"] = {
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "success_rate": successful_tests / total_tests if total_tests > 0 else 0.0
    }
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir) / f"batch_attack_results_{int(time.time())}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = _prepare_for_json(batch_results)
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Batch results saved to: {output_path}")
    
    return batch_results


def _prepare_for_json(data: Any) -> Any:
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: _prepare_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_prepare_for_json(item) for item in data]
    else:
        return data


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create test audio signal
    duration = 3.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    print("🚀 Attack Simulation Demo")
    print("=" * 40)
    
    # Test individual attack
    print("Testing individual attack...")
    attacked = simulate_attack(test_audio, "gaussian_noise", severity=0.3, sample_rate=sample_rate)
    print(f"Original length: {len(test_audio)}, Attacked length: {len(attacked)}")
    
    # Test attack suite
    print("\nTesting attack suite...")
    suite_results = run_attack_suite(
        test_audio, 
        sample_rate=sample_rate,
        attack_types=["gaussian_noise", "mp3_compression", "lowpass_filter"],
        severity_levels=[0.2, 0.5]
    )
    
    print(f"Suite completed: {suite_results['statistics']['success_rate']:.1%} success rate")
    print(f"Categories tested: {suite_results['statistics']['attack_categories']}")
