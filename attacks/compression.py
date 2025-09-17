"""
Audio Compression Attacks for Watermark Robustness Testing
========================================================

This module implements various audio compression attacks using pydub and other libraries
for testing watermark robustness against lossy compression formats.

Compression types include:
- MP3 compression (various bitrates)
- AAC compression 
- OGG Vorbis compression
- Opus compression
- Variable bitrate encoding
- Format conversion chains
"""

import numpy as np
import tempfile
import os
from typing import Optional, Union
from pydub import AudioSegment
from pydub.utils import mediainfo
import warnings
warnings.filterwarnings('ignore')


def _array_to_audiosegment(audio: np.ndarray, sample_rate: int = 44100) -> AudioSegment:
    """Convert numpy array to AudioSegment."""
    # Convert to 16-bit integer
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_int16.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=1
    )
    
    return audio_segment


def _audiosegment_to_array(audio_segment: AudioSegment) -> np.ndarray:
    """Convert AudioSegment to numpy array."""
    # Convert to numpy array
    audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1, 1]
    audio_array = audio_array / 32767.0
    
    return audio_array


def mp3_compression(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, bitrate: Optional[int] = None) -> np.ndarray:
    """
    Apply MP3 compression to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        bitrate: MP3 bitrate in kbps (None for automatic)
    
    Returns:
        MP3 compressed audio
    """
    if bitrate is None:
        # Map severity to bitrate (320kbps to 32kbps)
        bitrate = int(320 - severity * 288)
    
    # Convert to AudioSegment
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Convert to MP3 with specified bitrate
                audio_segment.export(tmp_mp3.name, format='mp3', bitrate=f"{bitrate}k")
                
                # Load MP3 back
                compressed_segment = AudioSegment.from_mp3(tmp_mp3.name)
                
                # Convert back to numpy array
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length (padding/truncation)
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_mp3.name)
                except:
                    pass


def aac_compression(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, bitrate: Optional[int] = None) -> np.ndarray:
    """
    Apply AAC compression to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        bitrate: AAC bitrate in kbps (None for automatic)
    
    Returns:
        AAC compressed audio
    """
    if bitrate is None:
        # Map severity to bitrate (256kbps to 32kbps)
        bitrate = int(256 - severity * 224)
    
    # Convert to AudioSegment
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as tmp_aac:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Convert to AAC with specified bitrate
                audio_segment.export(tmp_aac.name, format='mp4', bitrate=f"{bitrate}k")
                
                # Load AAC back
                compressed_segment = AudioSegment.from_file(tmp_aac.name)
                
                # Convert back to numpy array
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            except Exception as e:
                # Fallback to MP3 if AAC fails
                print(f"AAC compression failed, falling back to MP3: {e}")
                return mp3_compression(audio, sample_rate, severity, bitrate)
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_aac.name)
                except:
                    pass


def ogg_compression(audio: np.ndarray, sample_rate: int = 44100, 
                   severity: float = 0.5, quality: Optional[int] = None) -> np.ndarray:
    """
    Apply OGG Vorbis compression to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        quality: OGG quality level 0-10 (None for automatic)
    
    Returns:
        OGG compressed audio
    """
    if quality is None:
        # Map severity to quality (10 to 0)
        quality = int(10 - severity * 10)
    
    # Convert to AudioSegment
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as tmp_ogg:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Convert to OGG with specified quality
                audio_segment.export(tmp_ogg.name, format='ogg', 
                                   parameters=['-q', str(quality)])
                
                # Load OGG back
                compressed_segment = AudioSegment.from_ogg(tmp_ogg.name)
                
                # Convert back to numpy array
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            except Exception as e:
                # Fallback to MP3 if OGG fails
                print(f"OGG compression failed, falling back to MP3: {e}")
                return mp3_compression(audio, sample_rate, severity, quality * 32)
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_ogg.name)
                except:
                    pass


def opus_compression(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, bitrate: Optional[int] = None) -> np.ndarray:
    """
    Apply Opus compression to audio.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        bitrate: Opus bitrate in kbps (None for automatic)
    
    Returns:
        Opus compressed audio
    """
    if bitrate is None:
        # Map severity to bitrate (320kbps to 32kbps)
        bitrate = int(320 - severity * 288)
    
    # Convert to AudioSegment
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.opus', delete=False) as tmp_opus:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Convert to Opus with specified bitrate
                audio_segment.export(tmp_opus.name, format='opus', 
                                   bitrate=f"{bitrate}k")
                
                # Load Opus back
                compressed_segment = AudioSegment.from_file(tmp_opus.name)
                
                # Convert back to numpy array
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            except Exception as e:
                # Fallback to MP3 if Opus fails
                print(f"Opus compression failed, falling back to MP3: {e}")
                return mp3_compression(audio, sample_rate, severity, bitrate)
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_opus.name)
                except:
                    pass


def variable_bitrate(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, vbr_quality: Optional[int] = None) -> np.ndarray:
    """
    Apply variable bitrate MP3 compression.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        vbr_quality: VBR quality level 0-9 (None for automatic)
    
    Returns:
        VBR compressed audio
    """
    if vbr_quality is None:
        # Map severity to VBR quality (0 = best, 9 = worst)
        vbr_quality = int(severity * 9)
    
    # Convert to AudioSegment
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Convert to VBR MP3
                audio_segment.export(tmp_mp3.name, format='mp3', 
                                   parameters=['-q', str(vbr_quality)])
                
                # Load MP3 back
                compressed_segment = AudioSegment.from_mp3(tmp_mp3.name)
                
                # Convert back to numpy array
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_mp3.name)
                except:
                    pass


def lossy_format_conversion(audio: np.ndarray, sample_rate: int = 44100, 
                          severity: float = 0.5, format_chain: Optional[list] = None) -> np.ndarray:
    """
    Apply multiple lossy format conversions in sequence.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        format_chain: List of formats to convert through
    
    Returns:
        Multi-format compressed audio
    """
    if format_chain is None:
        # Default chain: WAV -> MP3 -> OGG -> MP3
        format_chain = ['mp3', 'ogg', 'mp3']
    
    current_audio = audio.copy()
    
    # Apply each compression in sequence
    for format_type in format_chain:
        if format_type == 'mp3':
            current_audio = mp3_compression(current_audio, sample_rate, severity)
        elif format_type == 'aac':
            current_audio = aac_compression(current_audio, sample_rate, severity)
        elif format_type == 'ogg':
            current_audio = ogg_compression(current_audio, sample_rate, severity)
        elif format_type == 'opus':
            current_audio = opus_compression(current_audio, sample_rate, severity)
        else:
            print(f"Unknown format: {format_type}, skipping")
    
    return current_audio


def adaptive_bitrate(audio: np.ndarray, sample_rate: int = 44100, 
                    severity: float = 0.5, target_quality: str = "medium") -> np.ndarray:
    """
    Apply adaptive bitrate compression based on audio content.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        target_quality: Target quality level ("low", "medium", "high")
    
    Returns:
        Adaptively compressed audio
    """
    # Analyze audio content to determine appropriate bitrate
    rms_energy = np.sqrt(np.mean(audio**2))
    spectral_centroid = np.mean(np.fft.fft(audio))
    
    # Base bitrate on quality target
    if target_quality == "low":
        base_bitrate = 64
    elif target_quality == "medium":
        base_bitrate = 128
    elif target_quality == "high":
        base_bitrate = 192
    else:
        base_bitrate = 128
    
    # Adjust bitrate based on content analysis
    if rms_energy > 0.1:  # High energy content
        bitrate_multiplier = 1.5
    elif rms_energy < 0.01:  # Low energy content
        bitrate_multiplier = 0.7
    else:
        bitrate_multiplier = 1.0
    
    # Apply severity
    final_bitrate = int(base_bitrate * bitrate_multiplier * (1 - severity * 0.5))
    final_bitrate = max(32, min(320, final_bitrate))  # Clamp to valid range
    
    return mp3_compression(audio, sample_rate, severity, final_bitrate)


def psychoacoustic_compression(audio: np.ndarray, sample_rate: int = 44100, 
                             severity: float = 0.5, use_joint_stereo: bool = False) -> np.ndarray:
    """
    Apply psychoacoustic model-based compression.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        use_joint_stereo: Whether to use joint stereo encoding
    
    Returns:
        Psychoacoustically compressed audio
    """
    # Map severity to quality settings
    if severity < 0.3:
        bitrate = 192
        quality = "high"
    elif severity < 0.7:
        bitrate = 128
        quality = "medium"
    else:
        bitrate = 64
        quality = "low"
    
    # Apply MP3 compression with psychoacoustic model
    audio_segment = _array_to_audiosegment(audio, sample_rate)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_wav:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_mp3:
            try:
                # Export to WAV first
                audio_segment.export(tmp_wav.name, format='wav')
                
                # Apply psychoacoustic compression
                parameters = ['-b', str(bitrate)]
                if use_joint_stereo:
                    parameters.extend(['-m', 'j'])
                
                audio_segment.export(tmp_mp3.name, format='mp3', 
                                   parameters=parameters)
                
                # Load compressed audio back
                compressed_segment = AudioSegment.from_mp3(tmp_mp3.name)
                result = _audiosegment_to_array(compressed_segment)
                
                # Ensure same length
                if len(result) > len(audio):
                    result = result[:len(audio)]
                elif len(result) < len(audio):
                    result = np.pad(result, (0, len(audio) - len(result)), 'constant')
                
                return result
                
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_wav.name)
                    os.unlink(tmp_mp3.name)
                except:
                    pass


def compression_pipeline(audio: np.ndarray, sample_rate: int = 44100, 
                        severity: float = 0.5, methods: Optional[list] = None) -> np.ndarray:
    """
    Apply multiple compression methods in sequence.
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        severity: Compression severity (0.0 to 1.0)
        methods: List of compression methods to apply
    
    Returns:
        Multi-compressed audio
    """
    if methods is None:
        methods = ["mp3_compression", "ogg_compression"]
    
    current_audio = audio.copy()
    
    for method in methods:
        if method == "mp3_compression":
            current_audio = mp3_compression(current_audio, sample_rate, severity)
        elif method == "aac_compression":
            current_audio = aac_compression(current_audio, sample_rate, severity)
        elif method == "ogg_compression":
            current_audio = ogg_compression(current_audio, sample_rate, severity)
        elif method == "opus_compression":
            current_audio = opus_compression(current_audio, sample_rate, severity)
        elif method == "variable_bitrate":
            current_audio = variable_bitrate(current_audio, sample_rate, severity)
        else:
            print(f"Unknown compression method: {method}")
    
    return current_audio


# Export all functions
__all__ = [
    "mp3_compression", "aac_compression", "ogg_compression", "opus_compression",
    "variable_bitrate", "lossy_format_conversion", "adaptive_bitrate",
    "psychoacoustic_compression", "compression_pipeline"
]
