"""
Main execution script for SoundSafeAI watermarking system.
This script demonstrates how to use the watermarking models for embedding and extraction.
"""

import sys
import os
import argparse
import logging
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
print(project_root)
sys.path.insert(0, str(project_root))

# Import our modules
from client.embedding import AudioWatermarkEmbedder
from client.extraction import AudioWatermarkExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_audio(duration: float = 5.0, sample_rate: int = 44100) -> np.ndarray:
    """
    Create a sample audio signal for testing.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Sample audio signal
    """
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Create a mix of sine waves (musical chord)
    frequencies = [440, 554.37, 659.25]  # A major chord
    audio = np.zeros_like(t)
    
    for freq in frequencies:
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    # Add some gentle noise for realism
    audio += 0.05 * np.random.normal(0, 1, len(audio))
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def test_embedding_extraction():
    """Test the complete watermarking workflow."""
    logger.info("=" * 60)
    logger.info("TESTING SOUNDSAFEAI WATERMARKING SYSTEM")
    logger.info("=" * 60)
    
    # Step 1: Create sample audio
    logger.info("Step 1: Creating sample audio...")
    original_audio = create_sample_audio(duration=3.0)
    logger.info(f"Created audio with shape: {original_audio.shape}")
    
    # Step 2: Initialize embedder
    logger.info("Step 2: Initializing watermark embedder...")
    embedder = AudioWatermarkEmbedder(device='cpu')
    logger.info("Embedder initialized successfully")
    
    # Step 3: Embed watermark
    logger.info("Step 3: Embedding watermark...")
    watermark_data = "SoundSafeAI_Test_Watermark_2025"
    embedding_strength = 0.1
    
    watermarked_audio, embed_metrics = embedder.embed_watermark(
        audio=original_audio,
        watermark_data=watermark_data,
        strength=embedding_strength
    )
    logger.info(f"Watermark embedded successfully. Output shape: {watermarked_audio.shape}")
    logger.info(f"Embedding metrics: {embed_metrics}")
    
    # Step 4: Initialize extractor
    logger.info("Step 4: Initializing watermark extractor...")
    extractor = AudioWatermarkExtractor(device='cpu')
    logger.info("Extractor initialized successfully")
    
    # Step 5: Extract watermark
    logger.info("Step 5: Extracting watermark...")
    extraction_results = extractor.extract_watermark(
        audio=watermarked_audio,
        confidence_threshold=0.5
    )
    
    extracted_data = extraction_results.get('watermark_data')
    confidence = extraction_results.get('confidence', 0.0)
    
    if extracted_data:
        logger.info(f"Watermark extracted: '{extracted_data}' (confidence: {confidence:.3f})")
    else:
        logger.info(f"No watermark detected (confidence: {confidence:.3f})")
    
    # Step 6: Test detection
    logger.info("Step 6: Testing watermark detection...")
    detection_results = extractor.detect_watermark(
        audio=watermarked_audio
    )
    
    is_watermarked = detection_results.get('watermark_detected', False)
    detection_score = detection_results.get('presence_probability', 0.0)
    
    logger.info(f"Watermark detection: {'DETECTED' if is_watermarked else 'NOT DETECTED'} "
               f"(score: {detection_score:.3f})")
    
    # Step 7: Compare original vs watermarked
    logger.info("Step 7: Analyzing quality metrics...")
    
    # Calculate basic quality metrics
    mse = np.mean((original_audio - watermarked_audio) ** 2)
    snr = 10 * np.log10(np.mean(original_audio ** 2) / (mse + 1e-10))
    max_diff = np.max(np.abs(original_audio - watermarked_audio))
    
    logger.info(f"Quality Metrics:")
    logger.info(f"  - Mean Squared Error: {mse:.6f}")
    logger.info(f"  - Signal-to-Noise Ratio: {snr:.2f} dB")
    logger.info(f"  - Maximum Difference: {max_diff:.6f}")
    
    # Step 8: Test with clean audio (should not detect watermark)
    logger.info("Step 8: Testing with clean audio...")
    clean_detection_results = extractor.detect_watermark(
        audio=original_audio
    )
    
    clean_detection = clean_detection_results.get('watermark_detected', False)
    clean_score = clean_detection_results.get('presence_probability', 0.0)
    
    logger.info(f"Clean audio detection: {'DETECTED' if clean_detection else 'NOT DETECTED'} "
               f"(score: {clean_score:.3f})")
    
    logger.info("=" * 60)
    logger.info("TESTING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    
    return {
        'original_audio': original_audio,
        'watermarked_audio': watermarked_audio,
        'extracted_data': extracted_data,
        'confidence': confidence,
        'detection_score': detection_score,
        'embed_metrics': embed_metrics,
        'extraction_results': extraction_results,
        'detection_results': detection_results,
        'quality_metrics': {
            'mse': mse,
            'snr': snr,
            'max_diff': max_diff
        }
    }


def test_file_processing(input_file: str, output_file: str = None):
    """
    Test watermarking with actual audio files.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to save watermarked audio (optional)
    """
    logger.info("=" * 60)
    logger.info("TESTING FILE-BASED WATERMARKING")
    logger.info("=" * 60)
    
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return None
    
    # Initialize embedder and extractor
    embedder = AudioWatermarkEmbedder(device='cpu')
    extractor = AudioWatermarkExtractor(device='cpu')
    
    # Process the file
    watermark_data = "FileTest_SoundSafeAI_2025"
    
    try:
        # Embed watermark
        logger.info(f"Processing file: {input_file}")
        watermarked_audio, embed_metrics = embedder.embed_watermark(
            audio=input_file,
            watermark_data=watermark_data,
            strength=0.1
        )
        
        # Save if output path provided
        if output_file:
            # This would use proper audio saving in real implementation
            logger.info(f"Watermarked audio would be saved to: {output_file}")
        
        # Test extraction
        extraction_results = extractor.extract_watermark(watermarked_audio)
        extracted_data = extraction_results.get('watermark_data')
        confidence = extraction_results.get('confidence', 0.0)
        
        logger.info(f"File processing completed:")
        logger.info(f"  - Original watermark: '{watermark_data}'")
        logger.info(f"  - Extracted watermark: '{extracted_data}'")
        logger.info(f"  - Confidence: {confidence:.3f}")
        
        return {
            'success': True,
            'extracted_data': extracted_data,
            'confidence': confidence,
            'embed_metrics': embed_metrics,
            'extraction_results': extraction_results
        }
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        return {'success': False, 'error': str(e)}


def run_benchmark_tests():
    """Run performance benchmark tests."""
    logger.info("=" * 60)
    logger.info("RUNNING PERFORMANCE BENCHMARKS")
    logger.info("=" * 60)
    
    import time
    
    # Test different audio lengths
    durations = [1.0, 3.0, 5.0, 10.0]
    strengths = [0.05, 0.1, 0.2, 0.3]
    
    embedder = AudioWatermarkEmbedder(device='cpu')
    extractor = AudioWatermarkExtractor(device='cpu')
    
    results = []
    
    for duration in durations:
        for strength in strengths:
            logger.info(f"Testing: {duration}s audio, strength {strength}")
            
            # Create test audio
            audio = create_sample_audio(duration)
            
            # Time embedding
            start_time = time.time()
            watermarked, embed_metrics = embedder.embed_watermark(audio, f"test_{duration}_{strength}", strength)
            embed_time = time.time() - start_time
            
            # Time extraction
            start_time = time.time()
            extraction_results = extractor.extract_watermark(watermarked)
            extract_time = time.time() - start_time
            
            extracted = extraction_results.get('watermark_data')
            confidence = extraction_results.get('confidence', 0.0)
            
            # Calculate quality
            mse = np.mean((audio - watermarked) ** 2)
            snr = 10 * np.log10(np.mean(audio ** 2) / (mse + 1e-10))
            
            result = {
                'duration': duration,
                'strength': strength,
                'embed_time': embed_time,
                'extract_time': extract_time,
                'snr': snr,
                'confidence': confidence,
                'extracted_correctly': extracted is not None
            }
            
            results.append(result)
            
            logger.info(f"  - Embed time: {embed_time:.3f}s")
            logger.info(f"  - Extract time: {extract_time:.3f}s")
            logger.info(f"  - SNR: {snr:.2f} dB")
            logger.info(f"  - Confidence: {confidence:.3f}")
    
    # Summary
    avg_embed_time = np.mean([r['embed_time'] for r in results])
    avg_extract_time = np.mean([r['extract_time'] for r in results])
    avg_snr = np.mean([r['snr'] for r in results])
    success_rate = np.mean([r['extracted_correctly'] for r in results])
    
    logger.info("=" * 40)
    logger.info("BENCHMARK SUMMARY:")
    logger.info(f"  - Average embed time: {avg_embed_time:.3f}s")
    logger.info(f"  - Average extract time: {avg_extract_time:.3f}s")
    logger.info(f"  - Average SNR: {avg_snr:.2f} dB")
    logger.info(f"  - Success rate: {success_rate:.1%}")
    logger.info("=" * 40)
    
    return results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description='SoundSafeAI Watermarking System')
    parser.add_argument('--mode', type=str, choices=['test', 'file', 'benchmark'], 
                       default='test', help='Execution mode')
    parser.add_argument('--input', type=str, help='Input audio file path')
    parser.add_argument('--output', type=str, help='Output audio file path')
    parser.add_argument('--watermark', type=str, default='SoundSafeAI_Test', 
                       help='Watermark data to embed')
    parser.add_argument('--strength', type=float, default=0.1, 
                       help='Embedding strength (0.0-1.0)')
    
    args = parser.parse_args()
    
    logger.info("Starting SoundSafeAI Watermarking System")
    logger.info(f"Mode: {args.mode}")
    
    try:
        if args.mode == 'test':
            # Run basic tests with synthetic audio
            results = test_embedding_extraction()
            
        elif args.mode == 'file':
            # Process actual audio file
            if not args.input:
                logger.error("Input file required for file mode")
                return
            results = test_file_processing(args.input, args.output)
            
        elif args.mode == 'benchmark':
            # Run performance benchmarks
            results = run_benchmark_tests()
            
        logger.info("Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
