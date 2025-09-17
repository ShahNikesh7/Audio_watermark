import pytest
import numpy as np
import torch
from client.embedding import AudioWatermarkEmbedder
from client.extraction import AudioWatermarkExtractor


class TestAudioWatermarkEmbedder:
    """Test cases for Audio Watermark Embedder"""

    def test_init(self):
        """Test embedder initialization"""
        embedder = AudioWatermarkEmbedder()
        assert embedder is not None
        assert hasattr(embedder, 'phm_model')

    def test_embed_basic(self):
        """Test basic embedding functionality"""
        embedder = AudioWatermarkEmbedder()
        
        # Create test audio data
        sample_rate = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Test embedding
        result = embedder.embed(audio_data, sample_rate)
        assert result is not None
        assert isinstance(result, dict)
        assert 'watermarked_audio' in result
        assert 'metadata' in result

    def test_embed_invalid_input(self):
        """Test embedding with invalid input"""
        embedder = AudioWatermarkEmbedder()
        
        # Test with None input
        with pytest.raises((ValueError, TypeError)):
            embedder.embed(None, 16000)
        
        # Test with empty array
        with pytest.raises((ValueError, IndexError)):
            embedder.embed(np.array([]), 16000)

    def test_embed_different_sample_rates(self):
        """Test embedding with different sample rates"""
        embedder = AudioWatermarkEmbedder()
        
        for sr in [8000, 16000, 22050, 44100, 48000]:
            audio_data = np.random.randn(sr).astype(np.float32)
            result = embedder.embed(audio_data, sr)
            assert result is not None


class TestAudioWatermarkExtractor:
    """Test cases for Audio Watermark Extractor"""

    def test_init(self):
        """Test extractor initialization"""
        extractor = AudioWatermarkExtractor()
        assert extractor is not None
        assert hasattr(extractor, 'phm_model')

    def test_extract_basic(self):
        """Test basic extraction functionality"""
        extractor = AudioWatermarkExtractor()
        
        # Create test audio data
        sample_rate = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Test extraction
        result = extractor.extract(audio_data, sample_rate)
        assert result is not None
        assert isinstance(result, dict)
        assert 'watermark_detected' in result
        assert 'confidence' in result

    def test_extract_invalid_input(self):
        """Test extraction with invalid input"""
        extractor = AudioWatermarkExtractor()
        
        # Test with None input
        with pytest.raises((ValueError, TypeError)):
            extractor.extract(None, 16000)
        
        # Test with empty array
        with pytest.raises((ValueError, IndexError)):
            extractor.extract(np.array([]), 16000)

    def test_extract_different_sample_rates(self):
        """Test extraction with different sample rates"""
        extractor = AudioWatermarkExtractor()
        
        for sr in [8000, 16000, 22050, 44100, 48000]:
            audio_data = np.random.randn(sr).astype(np.float32)
            result = extractor.extract(audio_data, sr)
            assert result is not None


class TestWatermarkIntegration:
    """Integration tests for watermark embedding and extraction"""

    def test_embed_extract_roundtrip(self):
        """Test embedding and extraction roundtrip"""
        embedder = AudioWatermarkEmbedder()
        extractor = AudioWatermarkExtractor()
        
        # Create test audio data
        sample_rate = 16000
        duration = 2.0
        audio_data = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        # Embed watermark
        embed_result = embedder.embed(audio_data, sample_rate)
        watermarked_audio = embed_result['watermarked_audio']
        
        # Extract watermark
        extract_result = extractor.extract(watermarked_audio, sample_rate)
        
        # Validate results
        assert extract_result is not None
        assert isinstance(extract_result['watermark_detected'], bool)
        assert isinstance(extract_result['confidence'], (int, float))
        assert 0 <= extract_result['confidence'] <= 1


if __name__ == '__main__':
    pytest.main([__file__])
