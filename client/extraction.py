"""
Audio Watermark Extraction Module
Advanced extraction system using Parallel Hybrid Model (PHM) for robust watermark detection.
Includes Moore-Glasberg psychoacoustic analysis for enhanced detection.
"""

import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Union, Tuple, Dict, Optional
import logging
import os

from .phm.perceptual_cnn.mobilenetv3 import MobileNetV3
from .phm.technical_rnn.gru_module import GRUModule
from .phm.technical_rnn.conformer_lite import ConformerLite
from .phm.fusion.fusion_layer import FusionLayer
from .phm.psychoacoustic.moore_glasberg import (
    MooreGlasbergAnalyzer,
    PerceptualAnalyzer
)

logger = logging.getLogger(__name__)


class WatermarkDetector(nn.Module):
    """Neural network for watermark detection and extraction."""
    
    def __init__(self, audio_channels=1, watermark_dim=128):
        super(WatermarkDetector, self).__init__()
        self.audio_channels = audio_channels
        self.watermark_dim = watermark_dim
        
        # Detection encoder
        self.detector = nn.Sequential(
            nn.Conv1d(audio_channels, 64, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(64, 128, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(256, 512, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Watermark presence classifier
        self.presence_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Watermark extractor
        self.watermark_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, watermark_dim),
            nn.Tanh()
        )
    
    def forward(self, audio):
        # Extract features
        features = self.detector(audio)
        pooled_features = self.global_pool(features).squeeze(-1)
        
        # Detect presence
        presence_prob = self.presence_classifier(pooled_features)
        
        # Extract watermark
        watermark_vector = self.watermark_extractor(pooled_features)
        
        return presence_prob, watermark_vector


class AudioWatermarkExtractor:
    """
    Advanced audio watermark extractor using Parallel Hybrid Model (PHM).
    Provides robust watermark detection and extraction with confidence assessment.
    Includes Moore-Glasberg psychoacoustic analysis for enhanced detection.
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 device: str = 'cpu', 
                 use_phm: bool = True,
                 sample_rate: int = 16000,
                 enable_psychoacoustic: bool = True):
        """
        Initialize the watermark extractor.
        
        Args:
            model_path: Path to pre-trained models
            device: Device to run inference on ('cpu' or 'cuda')
            use_phm: Whether to use PHM for quality assessment
            sample_rate: Audio sample rate for processing
            enable_psychoacoustic: Enable Moore-Glasberg psychoacoustic analysis
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_phm = use_phm
        self.sample_rate = sample_rate
        self.enable_psychoacoustic = enable_psychoacoustic
        
        # Initialize psychoacoustic analyzer
        if enable_psychoacoustic:
            self._init_psychoacoustic_analyzer()
        
        # Initialize watermark detector
        self.watermark_detector = WatermarkDetector().to(self.device)
        
        # Initialize PHM components if enabled
        if self.use_phm:
            self._init_phm_models()
        
        # Load pre-trained weights if provided
        if self.model_path:
            self._load_models()
    
    def _init_psychoacoustic_analyzer(self):
        """Initialize psychoacoustic analysis components (Moore-Glasberg + neural analyzer)."""
        try:
            self.n_fft = 2048
            self.hop_length = 256
            self.n_critical_bands = 24
            
            self.moore_glasberg_analyzer = MooreGlasbergAnalyzer(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_critical_bands=self.n_critical_bands
            )
            
            self.perceptual_analyzer = PerceptualAnalyzer(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_critical_bands=self.n_critical_bands
            ).to(self.device)
            
            logger.info("Psychoacoustic analyzer for extraction initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize psychoacoustic analyzer: {e}")
            self.enable_psychoacoustic = False
    
    def _init_phm_models(self):
        """Initialize Parallel Hybrid Model components."""
        logger.info("Initializing PHM components for extraction...")
        
        # Perceptual CNN (MobileNetV3)
        self.perceptual_cnn = MobileNetV3(
            num_classes=128,  # Feature dimension
            input_channels=1,
            variant='small'
        ).to(self.device)
        
        # Technical RNN (GRU)
        self.technical_gru = GRUModule(
            input_size=32,  # Technical features dimension
            hidden_size=64,
            num_layers=2
        ).to(self.device)
        
        # Technical Conformer
        self.technical_conformer = ConformerLite(
            input_dim=32,
            num_heads=4,
            ff_dim=128,
            num_layers=2
        ).to(self.device)
        
        # Fusion layer
        self.fusion_layer = FusionLayer(
            perceptual_dim=128,
            technical_dim=1,  # Output from technical models
            fusion_dim=128,
            num_heads=4
        ).to(self.device)
        
        logger.info("PHM components for extraction initialized successfully")
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            logger.info(f"Loading extraction models from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'watermark_detector' in checkpoint:
                self.watermark_detector.load_state_dict(checkpoint['watermark_detector'])
            
            if self.use_phm:
                if 'perceptual_cnn' in checkpoint:
                    self.perceptual_cnn.load_state_dict(checkpoint['perceptual_cnn'])
                if 'technical_gru' in checkpoint:
                    self.technical_gru.load_state_dict(checkpoint['technical_gru'])
                if 'technical_conformer' in checkpoint:
                    self.technical_conformer.load_state_dict(checkpoint['technical_conformer'])
                if 'fusion_layer' in checkpoint:
                    self.fusion_layer.load_state_dict(checkpoint['fusion_layer'])
            
            logger.info("Extraction models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}. Using default initialization.")
    
    def _extract_perceptual_features(self, audio: np.ndarray, sr: int = 22050) -> torch.Tensor:
        """Extract perceptual features using mel-spectrogram."""
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, hop_length=256, win_length=2048
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_range = mel_spec_db.max() - mel_spec_db.min()
        if mel_range > 0:
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / mel_range
        else:
            # Handle constant values (e.g., silence)
            mel_spec_norm = np.zeros_like(mel_spec_db)
        
        # Convert to tensor and add batch/channel dimensions
        mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
        
        # Resize to fixed size for CNN
        mel_tensor = torch.nn.functional.interpolate(
            mel_tensor, size=(128, 128), mode='bilinear', align_corners=False
        )
        
        return mel_tensor.to(self.device)
    
    def _extract_technical_features(self, audio: np.ndarray, sr: int = 22050) -> torch.Tensor:
        """Extract technical features for quality assessment."""
        features = []
        # RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=256)[0]
        features.extend([float(np.mean(rms)), float(np.std(rms))])

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([float(np.mean(spectral_centroids)), float(np.std(spectral_centroids))])

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.extend([float(np.mean(spectral_rolloff)), float(np.std(spectral_rolloff))])

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features.extend([float(np.mean(spectral_bandwidth)), float(np.std(spectral_bandwidth))])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
        features.extend([float(np.mean(zcr)), float(np.std(zcr))])

        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend([float(np.mean(mfccs)), float(np.std(mfccs))])

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([float(np.mean(chroma)), float(np.std(chroma))])

        # Tonnetz
        try:
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            features.extend([float(np.mean(tonnetz)), float(np.std(tonnetz))])
        except Exception:
            # Some audio may be too short / edge cases
            features.extend([0.0, 0.0])

        # Pad or truncate to 32 features
        while len(features) < 32:
            features.append(0.0)
        features = features[:32]

        # Create simple temporal sequence by tiling (sequence length proportional to duration)
        seq_len = max(1, min(100, len(audio) // 512))
        feature_seq = np.tile(np.array(features, dtype=np.float32), (seq_len, 1))

        return torch.from_numpy(feature_seq).unsqueeze(0).to(self.device)
    
    def _assess_extraction_quality(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, float]:
        """Assess audio quality for extraction confidence."""
        if not self.use_phm:
            return {'quality_score': 0.8, 'extraction_confidence': 0.8}
        
        with torch.no_grad():
            # Extract features
            perceptual_features = self._extract_perceptual_features(audio, sr)
            technical_features = self._extract_technical_features(audio, sr)
            
            # Get perceptual score
            perceptual_output = self.perceptual_cnn(perceptual_features)
            perceptual_score = torch.sigmoid(perceptual_output).mean().item()
            
            # Get technical scores
            technical_gru_output = self.technical_gru(technical_features)
            technical_gru_score = torch.sigmoid(technical_gru_output).mean().item()
            
            technical_conformer_output = self.technical_conformer(technical_features)
            technical_conformer_score = torch.sigmoid(technical_conformer_output).mean().item()
            
            # Average technical scores
            technical_score = (technical_gru_score + technical_conformer_score) / 2
            
            # Fuse scores
            fusion_output = self.fusion_layer(
                perceptual_output.mean(dim=0, keepdim=True),
                torch.tensor([[technical_score]], device=self.device)
            )
            quality_score = torch.sigmoid(fusion_output).item()
            
            # Handle NaN values
            if np.isnan(perceptual_score):
                perceptual_score = 0.5
            if np.isnan(technical_score):
                technical_score = 0.5
            if np.isnan(quality_score):
                quality_score = 0.5
            
            # Extraction confidence based on quality
            extraction_confidence = min(1.0, quality_score * 1.2)  # Boost confidence slightly
            
            return {
                'quality_score': quality_score,
                'extraction_confidence': extraction_confidence,
                'perceptual_score': perceptual_score,
                'technical_score': technical_score
            }
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio from file."""
        logger.info(f"Loading audio from {file_path}")
        try:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            # Return synthetic audio for demo
            sr = 22050
            duration = 3  # 3 seconds
            audio = np.random.randn(sr * duration) * 0.1
            return audio, sr
    
    def _decode_watermark_vector(self, watermark_vector: torch.Tensor) -> Optional[str]:
        """Decode watermark vector back to original data."""
        try:
            # Convert to numpy and denormalize
            vector_np = watermark_vector.cpu().numpy().flatten()
            vector_denorm = ((vector_np + 1.0) * 127.5).astype(np.uint8)
            
            # Convert to bytes and decode
            watermark_bytes = vector_denorm.tobytes()
            
            # Try to decode as UTF-8 string
            try:
                decoded_str = watermark_bytes.decode('utf-8', errors='ignore')
                # Remove null characters and non-printable characters
                decoded_str = ''.join(char for char in decoded_str if ord(char) >= 32 and ord(char) <= 126)
                return decoded_str if decoded_str else None
            except:
                # Return hex representation if UTF-8 fails
                return watermark_bytes.hex()[:64]  # Limit length
                
        except Exception as e:
            logger.warning(f"Failed to decode watermark vector: {e}")
            return None
    
    def detect_watermark(self, audio: Union[np.ndarray, str]) -> Dict[str, any]:
        """
        Detect presence of watermark in audio.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            
        Returns:
            Dictionary containing detection results
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = 22050  # Default sample rate
        
        logger.info("Detecting watermark presence...")
        
        # Assess extraction quality
        quality_metrics = self._assess_extraction_quality(audio_data, sr)
        
        # Prepare audio for detection
        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Detect watermark
        with torch.no_grad():
            presence_prob, watermark_vector = self.watermark_detector(audio_tensor)
        
        presence_score = presence_prob.item()
        
        # Determine if watermark is present
        presence_threshold = 0.5
        watermark_detected = presence_score > presence_threshold
        
        results = {
            'watermark_detected': watermark_detected,
            'presence_probability': presence_score,
            'presence_threshold': presence_threshold,
            'watermark_vector': watermark_vector.cpu().numpy(),
            'quality_metrics': quality_metrics
        }
        
        logger.info(f"Watermark detection completed. Detected: {watermark_detected}, "
                   f"Probability: {presence_score:.3f}")
        
        return results
    
    def extract_watermark(self, 
                         audio: Union[np.ndarray, str],
                         confidence_threshold: float = 0.7) -> Dict[str, any]:
        """
        Extract watermark data from audio signal.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            confidence_threshold: Minimum confidence required for extraction
            
        Returns:
            Dictionary containing extraction results
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = 22050  # Default sample rate
        
        logger.info("Extracting watermark...")
        
        # First detect watermark
        detection_results = self.detect_watermark(audio_data)
        
        results = {
            'extraction_successful': False,
            'watermark_data': None,
            'confidence': 0.0,
            'detection_results': detection_results
        }
        
        if not detection_results['watermark_detected']:
            logger.info("No watermark detected")
            return results
        
        # Calculate extraction confidence
        presence_prob = detection_results['presence_probability']
        quality_confidence = detection_results['quality_metrics']['extraction_confidence']
        overall_confidence = (presence_prob + quality_confidence) / 2
        
        results['confidence'] = overall_confidence
        
        if overall_confidence < confidence_threshold:
            logger.info(f"Extraction confidence {overall_confidence:.3f} below threshold {confidence_threshold}")
            return results
        
        # Extract watermark data
        watermark_vector = torch.FloatTensor(detection_results['watermark_vector'])
        decoded_data = self._decode_watermark_vector(watermark_vector)
        
        if decoded_data:
            results['extraction_successful'] = True
            results['watermark_data'] = decoded_data
            logger.info(f"Watermark extracted successfully: {decoded_data[:50]}...")
        else:
            logger.warning("Failed to decode watermark data")
        
        return results
    
    def extract(self, 
                audio: Union[np.ndarray, str],
                sample_rate: int = None,
                confidence_threshold: float = 0.7,
                use_psychoacoustic: bool = True) -> Dict[str, any]:
        """
        Extract watermark from audio signal with psychoacoustic analysis.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            sample_rate: Sample rate of the audio
            confidence_threshold: Minimum confidence required for extraction
            use_psychoacoustic: Whether to use Moore-Glasberg psychoacoustic analysis
            
        Returns:
            Dictionary containing extraction results and confidence metrics
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = sample_rate or self.sample_rate
        
        logger.info(f"Extracting watermark with psychoacoustic analysis enabled: {use_psychoacoustic}")
        
        if use_psychoacoustic and self.enable_psychoacoustic:
            # Use psychoacoustic-aware extraction
            results = self._extract_with_psychoacoustic_analysis(
                audio_data, sr, confidence_threshold
            )
        else:
            # Use standard extraction
            results = self.extract_watermark(audio_data, confidence_threshold)
        
        # Add metadata
        results['metadata'] = {
            'sample_rate': sr,
            'psychoacoustic_enabled': use_psychoacoustic and self.enable_psychoacoustic,
            'confidence_threshold': confidence_threshold,
            'audio_length': len(audio_data)
        }
        
        return results
    
    def extract_psychoacoustic(self, 
                             audio: np.ndarray, 
                             sample_rate: int = None,
                             confidence_threshold: float = 0.7) -> Dict[str, any]:
        """
        Extract watermark from audio using psychoacoustic analysis.
        
        Args:
            audio: Input audio signal
            sample_rate: Sample rate of the audio
            confidence_threshold: Minimum confidence required for extraction
            
        Returns:
            Dictionary containing extraction results
        """
        sr = sample_rate or self.sample_rate
        
        # Use psychoacoustic-aware extraction
        results = self._extract_with_psychoacoustic_analysis(
            audio, sr, confidence_threshold
        )
        
        return results

    def _extract_with_psychoacoustic_analysis(self,
                                            audio: np.ndarray,
                                            sample_rate: int,
                                            confidence_threshold: float) -> Dict[str, any]:
        """
        Extract watermark using psychoacoustic analysis for enhanced detection.
        
        Args:
            audio: Input audio signal
            sample_rate: Audio sample rate
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Enhanced extraction results with psychoacoustic confidence
        """
        logger.info("Performing psychoacoustic analysis for enhanced extraction...")
        
        # Analyze masking thresholds
        masking_analysis = self.moore_glasberg_analyzer.analyze_masking_threshold(audio)
        
        # Compute STFT of audio
        stft_audio = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Analyze potential watermark regions using psychoacoustic masking
        watermark_regions = self._identify_watermark_regions(stft_audio, masking_analysis)
        
        # Standard detection on identified regions
        detection_results = self.detect_watermark(audio)
        
        # Enhance confidence using psychoacoustic analysis
        psychoacoustic_confidence = self._calculate_psychoacoustic_confidence(
            stft_audio, masking_analysis, watermark_regions
        )
        
        # Combine standard and psychoacoustic confidence
        presence_prob = detection_results['presence_probability']
        quality_confidence = detection_results['quality_metrics']['extraction_confidence']
        
        # Weighted combination of confidences
        overall_confidence = (
            0.4 * presence_prob + 
            0.3 * quality_confidence + 
            0.3 * psychoacoustic_confidence
        )
        
        # Determine if watermark is detected
        watermark_detected = overall_confidence >= confidence_threshold
        
        results = {
            'watermark_detected': watermark_detected,
            'confidence': overall_confidence,
            'presence_probability': presence_prob,
            'quality_confidence': quality_confidence,
            'psychoacoustic_confidence': psychoacoustic_confidence,
            'watermark_regions': watermark_regions,
            'masking_analysis': {
                'n_critical_bands': len(masking_analysis['band_centers']),
                'avg_masking_threshold': float(np.mean(masking_analysis['masking_thresholds'])),
                'band_energy_distribution': np.mean(masking_analysis['band_thresholds'], axis=1).tolist()
            },
            'extraction_successful': False,
            'watermark_data': None
        }
        
        # If watermark detected with sufficient confidence, attempt extraction
        if watermark_detected:
            logger.info(f"Watermark detected with confidence {overall_confidence:.3f}")
            
            # Extract watermark data
            if 'watermark_vector' in detection_results:
                watermark_vector = torch.FloatTensor(detection_results['watermark_vector'])
                decoded_data = self._decode_watermark_vector(watermark_vector)
                
                if decoded_data:
                    results['extraction_successful'] = True
                    results['watermark_data'] = decoded_data
                    logger.info("Watermark data extracted successfully")
                else:
                    logger.warning("Failed to decode watermark data")
            else:
                logger.warning("No watermark vector available for decoding")
        else:
            logger.info(f"Watermark confidence {overall_confidence:.3f} below threshold {confidence_threshold}")
        
        return results
    
    def _identify_watermark_regions(self,
                                   stft_audio: np.ndarray,
                                   masking_analysis: Dict) -> Dict[str, np.ndarray]:
        """
        Identify potential watermark regions using psychoacoustic analysis.
        
        Args:
            stft_audio: STFT of the audio signal
            masking_analysis: Results from Moore-Glasberg analysis
            
        Returns:
            Dictionary with watermark region analysis
        """
        masking_thresholds = masking_analysis['masking_thresholds']
        power_spectrum = masking_analysis['power_spectrum']
        
        # Calculate signal-to-masking ratio
        signal_power = np.abs(stft_audio) ** 2
        smr = np.divide(signal_power, masking_thresholds, 
                       out=np.zeros_like(signal_power), where=masking_thresholds!=0)
        
        # Identify regions where signal is close to masking threshold (potential watermark)
        watermark_mask = (smr > 0.1) & (smr < 2.0)  # Signal just above masking threshold
        
        # Temporal and frequency analysis of potential watermark regions
        time_presence = np.mean(watermark_mask, axis=0)  # Average across frequencies
        freq_presence = np.mean(watermark_mask, axis=1)  # Average across time
        
        return {
            'watermark_mask': watermark_mask,
            'time_presence': time_presence,
            'freq_presence': freq_presence,
            'total_coverage': float(np.mean(watermark_mask)),
            'smr': smr
        }
    
    def _calculate_psychoacoustic_confidence(self,
                                           stft_audio: np.ndarray,
                                           masking_analysis: Dict,
                                           watermark_regions: Dict) -> float:
        """
        Calculate confidence based on psychoacoustic analysis.
        
        Args:
            stft_audio: STFT of the audio signal
            masking_analysis: Results from Moore-Glasberg analysis
            watermark_regions: Identified watermark regions
            
        Returns:
            Psychoacoustic confidence score (0-1)
        """
        # Analyze consistency of watermark-like patterns
        watermark_mask = watermark_regions['watermark_mask']
        smr = watermark_regions['smr']
        
        # Check for consistent patterns in watermark regions
        consistency_score = self._measure_pattern_consistency(watermark_mask)
        
        # Check if SMR distribution matches watermark characteristics
        smr_score = self._analyze_smr_distribution(smr, watermark_mask)
        
        # Check temporal continuity of watermark regions
        temporal_score = self._analyze_temporal_continuity(watermark_regions['time_presence'])
        
        # Check frequency distribution
        frequency_score = self._analyze_frequency_distribution(watermark_regions['freq_presence'])
        
        # Combine scores
        psychoacoustic_confidence = (
            0.3 * consistency_score +
            0.3 * smr_score +
            0.2 * temporal_score +
            0.2 * frequency_score
        )
        
        return float(np.clip(psychoacoustic_confidence, 0.0, 1.0))
    
    def _measure_pattern_consistency(self, watermark_mask: np.ndarray) -> float:
        """Measure consistency of watermark patterns."""
        if np.sum(watermark_mask) == 0:
            return 0.0
        
        # Analyze local patterns in watermark mask
        # Look for regular structures that suggest intentional watermarking
        flattened = watermark_mask.flatten()
        
        # Ensure both arrays have the same length for correlation calculation
        if len(flattened) < 2:
            return 0.0
        
        # Take every other element, but ensure equal lengths
        min_len = len(flattened) // 2
        if min_len == 0:
            return 0.0
        
        even_indices = flattened[::2][:min_len]
        odd_indices = flattened[1::2][:min_len]
        
        # Calculate correlation if we have enough data
        if len(even_indices) != len(odd_indices) or len(even_indices) < 2:
            return 0.0
        
        correlation = np.corrcoef(even_indices, odd_indices)[0, 1]
        
        # Handle NaN correlation (e.g., when variance is zero)
        if np.isnan(correlation):
            correlation = 0.0
        
        return float(np.abs(correlation))
    
    def _analyze_smr_distribution(self, smr: np.ndarray, watermark_mask: np.ndarray) -> float:
        """Analyze signal-to-masking ratio distribution."""
        if np.sum(watermark_mask) == 0:
            return 0.0
        
        # Extract SMR values in watermark regions
        watermark_smr = smr[watermark_mask]
        
        if len(watermark_smr) == 0:
            return 0.0
        
        # Check if SMR distribution is typical for watermarks (narrow range around masking threshold)
        smr_mean = np.mean(watermark_smr)
        smr_std = np.std(watermark_smr)
        
        # Ideal watermark should have SMR slightly above masking threshold with low variance
        ideal_smr = 1.0  # Just above masking threshold
        smr_score = np.exp(-np.abs(smr_mean - ideal_smr)) * np.exp(-smr_std)
        
        return float(np.clip(smr_score, 0.0, 1.0))
    
    def _analyze_temporal_continuity(self, time_presence: np.ndarray) -> float:
        """Analyze temporal continuity of watermark presence."""
        if len(time_presence) == 0:
            return 0.0
        
        # Measure how consistently watermark appears over time
        continuity = 1.0 - np.std(time_presence) / (np.mean(time_presence) + 1e-8)
        return float(np.clip(continuity, 0.0, 1.0))
    
    def _analyze_frequency_distribution(self, freq_presence: np.ndarray) -> float:
        """Analyze frequency distribution of watermark presence."""
        if len(freq_presence) == 0:
            return 0.0
        
        # Check if watermark is concentrated in perceptually important frequency ranges
        # Mid frequencies (500-4000 Hz) are typically preferred for watermarking
        n_freqs = len(freq_presence)
        mid_range = slice(n_freqs//4, 3*n_freqs//4)  # Approximate mid-frequency range
        
        mid_freq_energy = np.mean(freq_presence[mid_range])
        total_energy = np.mean(freq_presence)
        
        if total_energy == 0:
            return 0.0
        
        frequency_score = mid_freq_energy / total_energy
        return float(np.clip(frequency_score, 0.0, 1.0))

    def get_extraction_stats(self) -> Dict[str, any]:
        """Get statistics about the extraction system."""
        stats = {
            "device": str(self.device),
            "model_path": self.model_path,
            "use_phm": self.use_phm,
            "watermark_detector_loaded": hasattr(self, 'watermark_detector')
        }
        
        if self.use_phm:
            stats.update({
                "perceptual_cnn_loaded": hasattr(self, 'perceptual_cnn'),
                "technical_gru_loaded": hasattr(self, 'technical_gru'),
                "technical_conformer_loaded": hasattr(self, 'technical_conformer'),
                "fusion_layer_loaded": hasattr(self, 'fusion_layer')
            })
       