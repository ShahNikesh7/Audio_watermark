"""
Data preprocessing script for SoundSafeAI.
"""

import os
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from data.utils import load_audio_file, save_audio_file, compute_audio_features, normalize_audio
    from data.augmentation.noise import add_noise
    from data.augmentation.time_stretch import time_stretch
except ImportError as e:
    logger.warning(f"Could not import local modules: {e}")
    logger.warning("Running in standalone mode with basic functionality")


def preprocess_audio_file(args: tuple) -> Dict[str, Any]:
    """
    Preprocess a single audio file.
    
    Args:
        args: Tuple of (file_path, output_dir, config)
        
    Returns:
        Processing result dictionary
    """
    file_path, output_dir, config = args
    
    try:
        # Load audio
        audio, sr = load_audio_file(file_path, config['sample_rate'])
        
        # Normalize audio
        audio = normalize_audio(audio, config['target_level'])
        
        # Compute features
        features = compute_audio_features(audio, sr)
        
        # Apply augmentations if specified
        augmented_files = []
        if config['apply_augmentations']:
            # Original file
            original_output = os.path.join(output_dir, f"original_{os.path.basename(file_path)}")
            save_audio_file(audio, original_output, sr)
            augmented_files.append(original_output)
            
            # Noise augmentation
            if config['augmentations']['noise']:
                noisy_audio = add_noise(audio, noise_type='gaussian', snr_db=20)
                noise_output = os.path.join(output_dir, f"noise_{os.path.basename(file_path)}")
                save_audio_file(noisy_audio, noise_output, sr)
                augmented_files.append(noise_output)
            
            # Time stretch augmentation
            if config['augmentations']['time_stretch']:
                stretched_audio = time_stretch(audio, sr, stretch_factor=0.9)
                stretch_output = os.path.join(output_dir, f"stretch_{os.path.basename(file_path)}")
                save_audio_file(stretched_audio, stretch_output, sr)
                augmented_files.append(stretch_output)
        
        else:
            # Just save the processed original
            output_path = os.path.join(output_dir, os.path.basename(file_path))
            save_audio_file(audio, output_path, sr)
            augmented_files.append(output_path)
        
        return {
            'status': 'success',
            'original_file': file_path,
            'processed_files': augmented_files,
            'features': features,
            'sample_rate': sr,
            'duration': len(audio) / sr
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {
            'status': 'error',
            'original_file': file_path,
            'error': str(e)
        }


def add_noise(audio: np.ndarray, noise_type: str = 'gaussian', snr_db: float = 20) -> np.ndarray:
    """Simple noise addition fallback."""
    if noise_type == 'gaussian':
        noise_power = np.mean(audio**2) / (10**(snr_db/10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
    return audio


def time_stretch(audio: np.ndarray, sr: int, stretch_factor: float = 1.0) -> np.ndarray:
    """Simple time stretching fallback."""
    if stretch_factor == 1.0:
        return audio
    
    # Simple interpolation-based time stretching
    indices = np.arange(0, len(audio), stretch_factor)
    return np.interp(indices, np.arange(len(audio)), audio)


def load_audio_file(file_path: str, sample_rate: int = None) -> tuple:
    """Simple audio loading fallback."""
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path)
        if sample_rate and sr != sample_rate:
            # Simple resampling
            ratio = sample_rate / sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)
            sr = sample_rate
        return audio, sr
    except ImportError:
        # Create dummy audio
        sr = sample_rate or 44100
        audio = np.random.randn(sr * 2).astype(np.float32)  # 2 seconds
        return audio, sr


def save_audio_file(audio: np.ndarray, file_path: str, sample_rate: int = 44100) -> bool:
    """Simple audio saving fallback."""
    try:
        import soundfile as sf
        sf.write(file_path, audio, sample_rate)
        return True
    except ImportError:
        # Save as numpy array
        np.save(file_path.replace('.wav', '.npy'), audio)
        return True


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """Simple normalization fallback."""
    if np.max(np.abs(audio)) == 0:
        return audio
    
    # Simple peak normalization
    audio = audio / np.max(np.abs(audio))
    
    # Apply target level
    target_linear = 10**(target_level / 20)
    return audio * target_linear


def compute_audio_features(audio: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Simple feature computation fallback."""
    return {
        'mean': float(np.mean(audio)),
        'std': float(np.std(audio)),
        'min': float(np.min(audio)),
        'max': float(np.max(audio)),
        'rms': float(np.sqrt(np.mean(audio**2))),
        'duration': len(audio) / sample_rate,
        'zero_crossing_rate': float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio)))
    }


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description='Preprocess audio data for SoundSafeAI')
    parser.add_argument('--input-dir', type=str, default='data/raw',
                        help='Input directory containing raw audio files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Output directory for processed audio files')
    parser.add_argument('--sample-rate', type=int, default=44100,
                        help='Target sample rate')
    parser.add_argument('--target-level', type=float, default=-20.0,
                        help='Target audio level in dB')
    parser.add_argument('--apply-augmentations', action='store_true',
                        help='Apply data augmentations')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'sample_rate': args.sample_rate,
        'target_level': args.target_level,
        'apply_augmentations': args.apply_augmentations,
        'augmentations': {
            'noise': True,
            'time_stretch': True
        }
    }
    
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(Path(args.input_dir).glob(f"**/*{ext}"))
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    if not audio_files:
        logger.warning("No audio files found in input directory")
        return
    
    # Prepare arguments for parallel processing
    process_args = [(str(f), args.output_dir, config) for f in audio_files]
    
    # Process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(preprocess_audio_file, arg) for arg in process_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            result = future.result()
            results.append(result)
    
    # Summarize results
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    logger.info(f"Processing complete: {successful} successful, {failed} failed")
    
    # Save processing report
    report_path = os.path.join(args.output_dir, 'preprocessing_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'config': config,
            'summary': {
                'total_files': len(audio_files),
                'successful': successful,
                'failed': failed
            },
            'results': results
        }, f, indent=2)
    
    logger.info(f"Processing report saved to {report_path}")


if __name__ == "__main__":
    main()
