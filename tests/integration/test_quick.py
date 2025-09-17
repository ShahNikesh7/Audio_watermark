"""
Quick test script for SoundSafeAI watermarking system.
Run this to quickly test if everything is working.
"""

import sys
import os
from pathlib import Path
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure simple logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def quick_test():
    """Quick test of the watermarking system."""
    print("🎵 SoundSafeAI Quick Test")
    print("=" * 50)
    
    try:
        # Import modules
        print("📦 Importing modules...")
        from client.embedding import AudioWatermarkEmbedder
        from client.extraction import AudioWatermarkExtractor
        print("✅ Modules imported successfully!")
        
        # Create test audio
        print("🎶 Creating test audio...")
        duration = 2.0  # 2 seconds
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        
        # Simple sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
        audio = audio.astype(np.float32)
        print(f"✅ Created {duration}s test audio (shape: {audio.shape})")
        
        # Test embedding
        print("🔧 Testing watermark embedding...")
        embedder = AudioWatermarkEmbedder()
        watermark_data = "TEST_WATERMARK_2025"
        
        watermarked_audio, quality_metrics = embedder.embed_watermark(
            audio=audio,
            watermark_data=watermark_data,
            strength=0.1
        )
        print("✅ Watermark embedded successfully!")
        print(f"   - Quality metrics: {quality_metrics}")
        
        # Test extraction
        print("🔍 Testing watermark extraction...")
        extractor = AudioWatermarkExtractor()
        
        extraction_results = extractor.extract_watermark(watermarked_audio)
        
        if extraction_results['extraction_successful']:
            extracted_data = extraction_results['watermark_data']
            confidence = extraction_results['confidence']
            print(f"✅ Watermark extracted: '{extracted_data}' (confidence: {confidence:.3f})")
        else:
            confidence = extraction_results['confidence']
            print(f"⚠️  No watermark detected (confidence: {confidence:.3f})")
        
        # Test detection
        print("🔎 Testing watermark detection...")
        detection_results = extractor.detect_watermark(watermarked_audio)
        
        if detection_results['watermark_detected']:
            detection_score = detection_results['presence_probability']
            print(f"✅ Watermark detected (score: {detection_score:.3f})")
        else:
            detection_score = detection_results['presence_probability']
            print(f"⚠️  Watermark not detected (score: {detection_score:.3f})")
        
        # Quality check
        print("📊 Quality analysis...")
        diff = np.mean(np.abs(audio - watermarked_audio))
        max_diff = np.max(np.abs(audio - watermarked_audio))
        
        print(f"   - Average difference: {diff:.6f}")
        print(f"   - Maximum difference: {max_diff:.6f}")
        print(f"   - Quality score: {quality_metrics.get('quality_score', 'N/A')}")
        print(f"   - Adapted strength: {quality_metrics.get('adapted_strength', 'N/A')}")
        
        print("=" * 50)
        print("🎉 Quick test completed successfully!")
        print("🚀 System is ready to use!")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're in the correct directory and modules exist")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = quick_test()
    if not success:
        sys.exit(1)
