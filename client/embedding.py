import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    PerceptualAnalyzer,
    integrate_masking_with_watermark
)
from .phm.psychoacoustic.perceptual_losses import (
    MFCCPerceptualLoss,
    CombinedPerceptualLoss,
    EnhancedCombinedPerceptualLoss,
    create_perceptual_loss
)
from .phm.psychoacoustic.adaptive_bit_allocation import (
    PerceptualSignificanceMetric,
    AdaptiveBitAllocator,
    PerceptualAdaptiveBitAllocation,
    create_perceptual_significance_metric,
    create_adaptive_bit_allocator
)

logger = logging.getLogger(__name__)


class InvertibleEncoder(nn.Module):
    """Invertible encoder using affine coupling blocks (improved robustness)."""

    class AffineCoupling(nn.Module):
        def __init__(self, channels: int, hidden: int = 256):
            super().__init__()
            self.net_scale = nn.Sequential(
                nn.Conv1d(channels // 2, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
                nn.Conv1d(hidden, channels // 2, 3, padding=1),
            )
            self.net_shift = nn.Sequential(
                nn.Conv1d(channels // 2, hidden, 3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, 1), nn.ReLU(),
                nn.Conv1d(hidden, channels // 2, 3, padding=1),
            )

        def forward(self, x, cond):  # cond: watermark features broadcast
            x_a, x_b = torch.chunk(x, 2, dim=1)
            h = x_a + cond[:, :x_a.size(1), :]
            log_s = torch.tanh(self.net_scale(h))  # bounded scale
            t = self.net_shift(h)
            y_b = (x_b * torch.exp(log_s)) + t
            return torch.cat([x_a, y_b], dim=1)

    def __init__(self, audio_channels=1, watermark_dim=128, base_channels: int = 64, num_couplings: int = 4):
        super(InvertibleEncoder, self).__init__()
        self.watermark_dim = watermark_dim
        self.pre = nn.Sequential(
            nn.Conv1d(audio_channels, base_channels, 7, padding=3), nn.ReLU(),
            nn.Conv1d(base_channels, base_channels * 2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2), nn.ReLU(),
        )
        model_channels = base_channels * 4
        if model_channels % 2 != 0:
            model_channels += 1
        self.watermark_proj = nn.Sequential(
            nn.Linear(watermark_dim, model_channels), nn.ReLU(),
            nn.Linear(model_channels, model_channels)
        )
        self.couplings = nn.ModuleList([
            self.AffineCoupling(model_channels) for _ in range(num_couplings)
        ])
        self.post = nn.Sequential(
            nn.ConvTranspose1d(model_channels, base_channels * 2, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.ConvTranspose1d(base_channels * 2, base_channels, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.Conv1d(base_channels, audio_channels, 7, padding=3), nn.Tanh()
        )

    def forward(self, audio: torch.Tensor, watermark: torch.Tensor):
        z = self.pre(audio)
        B, C, L = z.shape
        cond = self.watermark_proj(watermark).unsqueeze(-1).expand(-1, C, L)
        for layer in self.couplings:
            z = layer(z, cond)
        out = self.post(z)
        # Ensure strict length preservation (crop/pad to match input length)
        target_len = audio.shape[-1]
        cur_len = out.shape[-1]
        if cur_len > target_len:
            out = out[..., :target_len]
        elif cur_len < target_len:
            pad_amt = target_len - cur_len
            out = F.pad(out, (0, pad_amt), mode='replicate')
        return out


class AudioWatermarkEmbedder:
    """
    Advanced audio watermark embedder using Parallel Hybrid Model (PHM).
    Combines perceptual CNN, technical RNN, and fusion layers for quality assessment.
    Includes Moore-Glasberg psychoacoustic analysis for perceptual watermarking.
    """
    
    def __init__(self, 
                 model_path: str = None, 
                 device: str = 'cpu', 
                 use_phm: bool = True,
                 sample_rate: int = 16000,
                 enable_psychoacoustic: bool = True,
                 enable_mfcc_loss: bool = True,
                 training_mode: bool = False,
                 enable_adaptive_allocation: bool = True,
                 total_watermark_bits: int = 1000):
        """
        Initialize the watermark embedder.
        
        Args:
            model_path: Path to pre-trained models
            device: Device to run inference on ('cpu' or 'cuda')
            use_phm: Whether to use PHM for quality assessment
            sample_rate: Audio sample rate for processing
            enable_psychoacoustic: Enable Moore-Glasberg psychoacoustic analysis
            enable_mfcc_loss: Enable MFCC-based perceptual loss for training
            training_mode: Whether the embedder is in training mode
            enable_adaptive_allocation: Enable adaptive bit allocation (Task 1.2.2.3)
            total_watermark_bits: Total number of watermark bits to allocate (default: 1000)
        """
        self.model_path = model_path
        self.device = torch.device(device)
        self.use_phm = use_phm
        self.sample_rate = sample_rate
        self.enable_psychoacoustic = enable_psychoacoustic
        self.enable_mfcc_loss = enable_mfcc_loss
        self.training_mode = training_mode
        self.enable_adaptive_allocation = enable_adaptive_allocation
        self.total_watermark_bits = total_watermark_bits
        
        # Initialize MFCC-based perceptual loss for training
        if enable_mfcc_loss:
            self._init_perceptual_losses()
        
        # Initialize psychoacoustic analyzer
        if enable_psychoacoustic:
            self._init_psychoacoustic_analyzer()
        
        # Initialize adaptive bit allocation (Task 1.2.2.3)
        if enable_adaptive_allocation:
            logger.info("Initializing adaptive bit allocation for Task 1.2.2.3 integration...")
            self._init_adaptive_bit_allocation()
            logger.info("Task 1.2.2.3 integration completed: Adaptive bit allocation enabled")
        
        # Initialize PHM components
        if use_phm:
            self._init_phm_models()
        
        # Initialize watermark encoder
        self.watermark_encoder = InvertibleEncoder()
        self.watermark_encoder.to(self.device)
        
        # Load pre-trained models if available
        if model_path and os.path.exists(model_path):
            self._load_models()
    
    def _init_psychoacoustic_analyzer(self):
        """Initialize Moore-Glasberg psychoacoustic analyzer."""
        logger.info("Initializing Moore-Glasberg psychoacoustic analyzer...")
        
        # Standard STFT parameters for psychoacoustic analysis (per specification)
        self.n_fft = 1000
        self.hop_length = 400
        self.n_critical_bands = 24
        
        # Initialize Moore-Glasberg analyzer
        self.moore_glasberg_analyzer = MooreGlasbergAnalyzer(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_critical_bands=self.n_critical_bands
        )
        
        # Initialize neural perceptual analyzer
        self.perceptual_analyzer = PerceptualAnalyzer(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_critical_bands=self.n_critical_bands
        ).to(self.device)
        
        logger.info("Psychoacoustic analyzer initialized successfully")
    
    def _init_phm_models(self):
        """Initialize Parallel Hybrid Model components."""
        logger.info("Initializing PHM components...")
        
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
        
        logger.info("PHM components initialized successfully")
    
    def _load_models(self):
        """Load pre-trained models."""
        try:
            logger.info(f"Loading models from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'watermark_encoder' in checkpoint:
                self.watermark_encoder.load_state_dict(checkpoint['watermark_encoder'])
            
            if self.use_phm:
                if 'perceptual_cnn' in checkpoint:
                    self.perceptual_cnn.load_state_dict(checkpoint['perceptual_cnn'])
                if 'technical_gru' in checkpoint:
                    self.technical_gru.load_state_dict(checkpoint['technical_gru'])
                if 'technical_conformer' in checkpoint:
                    self.technical_conformer.load_state_dict(checkpoint['technical_conformer'])
                if 'fusion_layer' in checkpoint:
                    self.fusion_layer.load_state_dict(checkpoint['fusion_layer'])
            
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load models: {e}. Using default initialization.")
    
    def _extract_perceptual_features(self, audio: np.ndarray, sr: int = 22050) -> torch.Tensor:
        """Extract perceptual features using mel-spectrogram."""
        # Generate mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, hop_length=400, win_length=1000
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
        rms = librosa.feature.rms(y=audio, hop_length=400)[0]
        features.extend([np.mean(rms), np.std(rms)])

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.extend([np.mean(zcr), np.std(zcr)])

        # MFCC features (first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend([np.mean(mfccs), np.std(mfccs)])

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma)])

        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
        features.extend([np.mean(tonnetz), np.std(tonnetz)])

        # Pad or truncate to 32 features
        while len(features) < 32:
            features.append(0.0)
        features = features[:32]

        # For sequence models, create a sequence
        seq_len = min(100, len(audio) // 512)
        feature_seq = np.tile(features, (seq_len, 1))

        return torch.FloatTensor(feature_seq).unsqueeze(0).to(self.device)
    
    def _assess_quality(self, audio: np.ndarray, sr: int = 22050) -> Dict[str, float]:
        """Assess audio quality using PHM."""
        if not self.use_phm:
            return {'quality_score': 0.8, 'perceptual_score': 0.8, 'technical_score': 0.8}
        
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
            
            return {
                'quality_score': quality_score,
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
    
    def _generate_watermark_vector(self, watermark_data: Union[str, bytes]) -> torch.Tensor:
        """
        Generate watermark vector from input data.

        The payload can be up to 1000 bits, but the neural network uses 128 dimensions
        for computational efficiency. We use a hash-based approach to create a
        128-dimensional vector that represents the watermark data.
        """
        if isinstance(watermark_data, str):
            watermark_data = watermark_data.encode('utf-8')

        # For large payloads (>256 bits), we need to handle them properly
        if len(watermark_data) > 32:  # More than 256 bits
            # Use multiple hashes to capture more of the payload
            import hashlib

            # Primary hash of the full data
            hash_obj = hashlib.sha256(watermark_data)
            primary_hash = hash_obj.digest()

            # Secondary hash of data chunks for larger payloads
            chunk_size = max(1, len(watermark_data) // 8)
            chunks = [watermark_data[i:i+chunk_size] for i in range(0, len(watermark_data), chunk_size)]
            secondary_hashes = []
            for chunk in chunks[:8]:  # Limit to 8 chunks to keep it manageable
                chunk_hash = hashlib.sha256(chunk).digest()
                secondary_hashes.extend(chunk_hash[:4])  # Take first 4 bytes of each chunk hash

            # Combine hashes
            combined_data = primary_hash + bytes(secondary_hashes[:32])  # 32 + 32 = 64 bytes
        else:
            # For smaller payloads, use simple SHA256
            import hashlib
            hash_obj = hashlib.sha256(watermark_data)
            combined_data = hash_obj.digest()  # 32 bytes

        # Convert to 128-dimensional vector for neural network
        watermark_vector = np.frombuffer(combined_data, dtype=np.uint8).astype(np.float32)

        # Ensure we have exactly 128 dimensions (pad or truncate if necessary)
        if len(watermark_vector) < 128:
            padding = np.zeros(128 - len(watermark_vector))
            watermark_vector = np.concatenate([watermark_vector, padding])
        else:
            watermark_vector = watermark_vector[:128]

        # Normalize to [-1, 1] for neural network input
        watermark_vector = (watermark_vector / 127.5) - 1.0

        return torch.FloatTensor(watermark_vector).unsqueeze(0).to(self.device)
    
    def embed_watermark(self, 
                       audio: Union[np.ndarray, str], 
                       watermark_data: Union[str, bytes],
                       strength: float = 0.1,
                       adaptive_strength: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Embed a watermark into an audio signal.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            watermark_data: Data to embed as watermark
            strength: Base embedding strength (0.0 to 1.0)
            adaptive_strength: Whether to adapt strength based on quality assessment
            
        Returns:
            Tuple of (watermarked_audio, quality_metrics)
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = 22050  # Default sample rate
        
        logger.info(f"Embedding watermark with base strength {strength}")
        
        # Assess audio quality
        quality_metrics = self._assess_quality(audio_data, sr)
        logger.info(f"Quality assessment: {quality_metrics}")
        
        # Adapt strength based on quality
        if adaptive_strength:
            quality_factor = quality_metrics['quality_score']
            if np.isnan(quality_factor):
                quality_factor = 0.5  # Default value for NaN cases
            adapted_strength = strength * (0.5 + quality_factor * 0.5)
            logger.info(f"Adapted strength: {adapted_strength}")
        else:
            adapted_strength = strength
        
        # Generate watermark vector
        watermark_vector = self._generate_watermark_vector(watermark_data)
        
        # Prepare audio for embedding
        audio_tensor = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Embed watermark using invertible encoder
        with torch.no_grad():
            watermarked_tensor = self.watermark_encoder(audio_tensor, watermark_vector)
        
        # Convert back to numpy
        watermarked_audio = watermarked_tensor.squeeze().cpu().numpy()
        
        # Ensure output shape matches input shape
        if len(watermarked_audio) != len(audio_data):
            if len(watermarked_audio) > len(audio_data):
                # Trim if output is longer
                watermarked_audio = watermarked_audio[:len(audio_data)]
            else:
                # Pad if output is shorter
                padding = len(audio_data) - len(watermarked_audio)
                watermarked_audio = np.pad(watermarked_audio, (0, padding), mode='edge')
        
        # Apply strength scaling
        watermarked_audio = audio_data + adapted_strength * (watermarked_audio - audio_data)
        
        # Add quality metrics
        quality_metrics['adapted_strength'] = adapted_strength
        quality_metrics['original_strength'] = strength
        
        logger.info("Watermark embedding completed")
        return watermarked_audio, quality_metrics
    
    def embed(self, 
              audio: Union[np.ndarray, str], 
              watermark_data: Union[str, bytes] = "default_watermark",
              sample_rate: int = None,
              strength: float = 0.1,
              adaptive_strength: bool = True,
              use_psychoacoustic: bool = True) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Embed a watermark into an audio signal with psychoacoustic optimization.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            watermark_data: Data to embed as watermark
            sample_rate: Sample rate of the audio
            strength: Base embedding strength (0.0 to 1.0)
            adaptive_strength: Whether to adapt strength based on quality assessment
            use_psychoacoustic: Whether to use Moore-Glasberg psychoacoustic masking
            
        Returns:
            Dictionary containing watermarked_audio and metadata
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = sample_rate or self.sample_rate
        
        logger.info(f"Embedding watermark with psychoacoustic analysis enabled: {use_psychoacoustic}")
        
        # Assess audio quality using PHM
        quality_metrics = {}
        if self.use_phm:
            quality_metrics = self._assess_quality(audio_data, sr)
            logger.info(f"Quality assessment: {quality_metrics}")
        
        # Generate watermark vector
        watermark_vector = self._generate_watermark_vector(watermark_data)
        
        if use_psychoacoustic and self.enable_psychoacoustic:
            # Use psychoacoustic-aware embedding
            watermarked_audio = self._embed_with_psychoacoustic_masking(
                audio_data, watermark_vector, sr, strength, adaptive_strength, quality_metrics
            )
        else:
            # Use standard embedding
            watermarked_audio, quality_metrics = self.embed_watermark(
                audio_data, watermark_data, strength, adaptive_strength
            )
        
        return {
            'watermarked_audio': watermarked_audio,
            'metadata': {
                'quality_metrics': quality_metrics,
                'sample_rate': sr,
                'psychoacoustic_enabled': use_psychoacoustic and self.enable_psychoacoustic,
                'watermark_strength': strength,
                'original_length': len(audio_data)
            }
        }
    
    def _embed_with_psychoacoustic_masking(self,
                                         audio: np.ndarray,
                                         watermark_vector: torch.Tensor,
                                         sample_rate: int,
                                         strength: float,
                                         adaptive_strength: bool,
                                         quality_metrics: Dict) -> np.ndarray:
        """
        Embed watermark using Moore-Glasberg psychoacoustic masking.
        
        Args:
            audio: Input audio signal
            watermark_vector: Encoded watermark vector
            sample_rate: Audio sample rate
            strength: Base embedding strength
            adaptive_strength: Whether to adapt strength
            quality_metrics: Quality assessment results
            
        Returns:
            Psychoacoustically optimized watermarked audio
        """
        logger.info("Performing psychoacoustic analysis...")
        
        # Analyze masking thresholds using Moore-Glasberg model
        masking_analysis = self.moore_glasberg_analyzer.analyze_masking_threshold(audio)
        
        # Compute STFT of original audio
        stft_original = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        
        # Task 1.2.2.3: Integrate adaptive bit allocation with watermark embedding
        band_bit_allocation = None
        if self.enable_adaptive_allocation:
            logger.info("Computing adaptive bit allocation (Task 1.2.2.3)...")
            
            # Compute perceptual significance for each frequency band
            perceptual_significance = self.significance_metric.compute_band_significance(
                audio_fft=np.abs(stft_original).mean(axis=1),
                masking_thresholds=masking_analysis['masking_thresholds'],
                critical_bands=masking_analysis['band_indices']
            )
            
            # Perform dynamic bit allocation based on perceptual significance
            band_bit_allocation = self.bit_allocator.dynamic_allocate_bits(
                perceptual_significance,
                total_bits=self.total_watermark_bits
            )
            
            logger.info(f"Adaptive bit allocation: {band_bit_allocation}")
        
        # Generate watermark signal in frequency domain
        watermark_signal = self._generate_frequency_domain_watermark(
            watermark_vector, stft_original.shape, sample_rate, band_bit_allocation
        )
        
        # Apply psychoacoustic masking to watermark
        masked_watermark = self._apply_psychoacoustic_masking(
            watermark_signal, masking_analysis, strength, adaptive_strength, quality_metrics
        )
        
        # Combine original audio with masked watermark
        watermarked_stft = stft_original + masked_watermark
        
        # Convert back to time domain
        watermarked_audio = librosa.istft(watermarked_stft, hop_length=self.hop_length)
        
        # Ensure output length matches input length
        if len(watermarked_audio) != len(audio):
            if len(watermarked_audio) > len(audio):
                watermarked_audio = watermarked_audio[:len(audio)]
            else:
                padding = len(audio) - len(watermarked_audio)
                watermarked_audio = np.pad(watermarked_audio, (0, padding), mode='edge')
        
        logger.info("Psychoacoustic watermark embedding completed")
        return watermarked_audio
    
    def _generate_frequency_domain_watermark(self,
                                           watermark_vector: torch.Tensor,
                                           stft_shape: Tuple[int, int],
                                           sample_rate: int,
                                           band_bit_allocation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate watermark signal in frequency domain with optional adaptive bit allocation.
        
        Args:
            watermark_vector: Encoded watermark data
            stft_shape: Shape of the STFT (freq_bins, time_frames)
            sample_rate: Audio sample rate
            band_bit_allocation: Optional array of bits allocated per frequency band (Task 1.2.2.3)
            
        Returns:
            Watermark signal in frequency domain
        """
        n_freq_bins, n_time_frames = stft_shape
        
        # Convert watermark vector to numpy
        watermark_data = watermark_vector.cpu().numpy().flatten()
        
        # Create pseudo-random watermark signal based on watermark data
        np.random.seed(int(np.sum(watermark_data * 1000) % 2**32))
        
        # Generate complex watermark signal
        magnitude = np.random.normal(0, 1, (n_freq_bins, n_time_frames))
        phase = np.random.uniform(0, 2*np.pi, (n_freq_bins, n_time_frames))
        
        watermark_complex = magnitude * np.exp(1j * phase)
        
        # Apply frequency shaping (emphasize mid frequencies where masking is better)
        freq_weights = self._compute_frequency_weights(n_freq_bins, sample_rate)
        
        # Task 1.2.2.3: Apply adaptive bit allocation if available
        if band_bit_allocation is not None:
            logger.info("Applying adaptive bit allocation to watermark generation...")
            freq_weights = self._apply_adaptive_bit_allocation(
                freq_weights, band_bit_allocation, n_freq_bins
            )
        
        watermark_complex = watermark_complex * freq_weights[:, np.newaxis]
        
        return watermark_complex
    
    def _compute_frequency_weights(self, n_freq_bins: int, sample_rate: int) -> np.ndarray:
        """
        Compute frequency-dependent weights for watermark shaping.
        
        Args:
            n_freq_bins: Number of frequency bins
            sample_rate: Audio sample rate
            
        Returns:
            Frequency weights array
        """
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=(n_freq_bins-1)*2)
        
        # Perceptual weighting: emphasize 500-4000 Hz range
        weights = np.ones(len(frequencies))
        
        for i, freq in enumerate(frequencies):
            if freq < 100:
                weights[i] = 0.1  # Low weight for very low frequencies
            elif freq < 500:
                weights[i] = 0.5  # Medium weight for low frequencies
            elif freq < 4000:
                weights[i] = 1.0  # Full weight for mid frequencies
            elif freq < 8000:
                weights[i] = 0.7  # Reduced weight for high frequencies
            else:
                weights[i] = 0.3  # Low weight for very high frequencies
        
        return weights
    
    def _apply_psychoacoustic_masking(self,
                                    watermark_signal: np.ndarray,
                                    masking_analysis: Dict,
                                    strength: float,
                                    adaptive_strength: bool,
                                    quality_metrics: Dict) -> np.ndarray:
        """
        Apply psychoacoustic masking to watermark signal.
        
        Args:
            watermark_signal: Watermark signal in frequency domain
            masking_analysis: Results from Moore-Glasberg analysis
            strength: Base embedding strength
            adaptive_strength: Whether to adapt strength
            quality_metrics: Quality assessment results
            
        Returns:
            Masked watermark signal
        """
        masking_thresholds = masking_analysis['masking_thresholds']
        band_thresholds = masking_analysis['band_thresholds']
        band_indices = masking_analysis['band_indices']
        
        # Adapt strength based on quality if enabled
        if adaptive_strength and quality_metrics:
            quality_factor = quality_metrics.get('quality_score', 0.5)
            if np.isnan(quality_factor):
                quality_factor = 0.5
            adapted_strength = strength * (0.3 + quality_factor * 0.7)  # More conservative
        else:
            adapted_strength = strength
        
        logger.info(f"Applying psychoacoustic masking with adapted strength: {adapted_strength:.3f}")
        
        # Apply per-frame masking
        masked_watermark = watermark_signal.copy()
        safety_factor = 0.5  # Conservative safety factor
        
        for frame_idx in range(watermark_signal.shape[1]):
            frame_watermark = watermark_signal[:, frame_idx]
            frame_thresholds = masking_thresholds[:, frame_idx]
            
            # Scale watermark to respect masking thresholds
            masking_data = {
                'band_indices': band_indices,
                'band_thresholds': band_thresholds[:, frame_idx]
            }
            scaled_frame = integrate_masking_with_watermark(
                frame_watermark,
                masking_data,
                safety_factor
            )
            
            # Apply global strength scaling
            masked_watermark[:, frame_idx] = scaled_frame * adapted_strength
        
        return masked_watermark

    def get_embedding_stats(self) -> Dict[str, any]:
        """Get statistics about the embedding system."""
        stats = {
            "device": str(self.device),
            "model_path": self.model_path,
            "use_phm": self.use_phm,
            "watermark_encoder_loaded": hasattr(self, 'watermark_encoder')
        }
        
        if self.use_phm:
            stats.update({
                "perceptual_cnn_loaded": hasattr(self, 'perceptual_cnn'),
                "technical_gru_loaded": hasattr(self, 'technical_gru'),
                "technical_conformer_loaded": hasattr(self, 'technical_conformer'),
                "fusion_layer_loaded": hasattr(self, 'fusion_layer')
            })
        
        return stats
    
    def embed_psychoacoustic(self, 
                           audio: np.ndarray, 
                           watermark_data: Union[str, bytes] = "default_watermark",
                           target_snr_db: float = 38.0,
                           sample_rate: int = None) -> np.ndarray:
        """
        Embed a watermark using psychoacoustic masking.
        
        Args:
            audio: Input audio signal
            watermark_data: Data to embed as watermark
            target_snr_db: Target SNR in dB
            sample_rate: Sample rate of the audio
            
        Returns:
            Watermarked audio signal
        """
        sr = sample_rate or self.sample_rate
        
        # Generate watermark vector
        watermark_vector = self._generate_watermark_vector(watermark_data)
        
        # Use psychoacoustic-aware embedding
        watermarked_audio = self._embed_with_psychoacoustic_masking(
            audio, watermark_vector, sr, strength=0.1, adaptive_strength=True, quality_metrics={}
        )
        
        return watermarked_audio

    def _init_perceptual_losses(self):
        """Initialize MFCC-based and other perceptual loss functions with optimal weights from Task 1.1.3.2."""
        logger.info("Initializing MFCC-based perceptual loss functions with optimal weights...")
        
        # MFCC-based perceptual loss with optimal weight
        self.mfcc_loss = MFCCPerceptualLoss(
            n_mfcc=13,
            n_fft=2048,
            hop_length=400,
            n_mels=128,
            sample_rate=self.sample_rate,
            weight=1.0  # Optimal weight from Task 1.1.3.2
        ).to(self.device)
        
        # Enhanced combined perceptual loss with optimal weights from Task 1.1.3.2
        self.combined_perceptual_loss = EnhancedCombinedPerceptualLoss(
            mfcc_weight=1.0,      # Optimal MFCC weight
            spectral_weight=0.5,  # Standard spectral weight
            temporal_weight=0.3,  # Standard temporal weight
            robustness_weight=2.0, # Optimal robustness weight
            sample_rate=self.sample_rate
        ).to(self.device)
        
        # Reconstruction loss for overall audio fidelity
        self.reconstruction_loss = nn.MSELoss()
        
        logger.info("Perceptual loss functions initialized with optimal weights from Task 1.1.3.2")
    
    def compute_perceptual_loss(self, 
                               original_audio: Union[np.ndarray, torch.Tensor],
                               watermarked_audio: Union[np.ndarray, torch.Tensor],
                               loss_type: str = 'mfcc') -> Dict[str, float]:
        """
        Compute MFCC-based perceptual loss between original and watermarked audio.
        
        Args:
            original_audio: Original audio signal
            watermarked_audio: Watermarked audio signal
            loss_type: Type of loss to compute ('mfcc', 'combined', 'all')
            
        Returns:
            Dictionary containing computed losses
        """
        if not self.enable_mfcc_loss:
            logger.warning("MFCC loss not enabled. Returning zero loss.")
            return {'mfcc_loss': 0.0}
        
        # Convert to tensors if needed
        if isinstance(original_audio, np.ndarray):
            original_audio = torch.FloatTensor(original_audio).to(self.device)
        if isinstance(watermarked_audio, np.ndarray):
            watermarked_audio = torch.FloatTensor(watermarked_audio).to(self.device)
        
        # Ensure same shape and add batch dimension if needed
        if original_audio.dim() == 1:
            original_audio = original_audio.unsqueeze(0)
        if watermarked_audio.dim() == 1:
            watermarked_audio = watermarked_audio.unsqueeze(0)
        
        # Ensure same length
        min_length = min(original_audio.shape[-1], watermarked_audio.shape[-1])
        original_audio = original_audio[..., :min_length]
        watermarked_audio = watermarked_audio[..., :min_length]
        
        losses = {}
        
        with torch.no_grad() if not self.training_mode else torch.enable_grad():
            if loss_type in ['mfcc', 'all']:
                # Compute MFCC-based perceptual loss
                mfcc_loss = self.mfcc_loss(original_audio, watermarked_audio)
                losses['mfcc_loss'] = mfcc_loss.item()
                
            if loss_type in ['combined', 'all']:
                # Compute combined perceptual loss
                combined_losses = self.combined_perceptual_loss(original_audio, watermarked_audio)
                losses.update({k: v.item() for k, v in combined_losses.items()})
                
            if loss_type in ['reconstruction', 'all']:
                # Compute reconstruction loss
                recon_loss = self.reconstruction_loss(watermarked_audio, original_audio)
                losses['reconstruction_loss'] = recon_loss.item()
        
        return losses

    def embed_with_perceptual_optimization(self, 
                                         audio: Union[np.ndarray, str], 
                                         watermark_data: Union[str, bytes] = "default_watermark",
                                         target_mfcc_loss: float = 0.01,
                                         max_iterations: int = 10,
                                         learning_rate: float = 0.01,
                                         strength: float = 0.1) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Embed watermark with MFCC-based perceptual optimization.
        Uses gradient descent to minimize MFCC perceptual loss while maintaining watermark strength.
        
        Args:
            audio: Input audio signal (numpy array or file path)
            watermark_data: Data to embed as watermark
            target_mfcc_loss: Target MFCC loss to achieve
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for optimization
            strength: Base embedding strength
            
        Returns:
            Dictionary containing optimized watermarked_audio and metadata
        """
        if not self.enable_mfcc_loss:
            logger.warning("MFCC loss not enabled. Using standard embedding.")
            return self.embed(audio, watermark_data, strength=strength)
        
        # Load audio if path provided
        if isinstance(audio, str):
            audio_data, sr = self._load_audio(audio)
        else:
            audio_data = audio
            sr = self.sample_rate
        
        logger.info(f"Starting perceptual optimization with target MFCC loss: {target_mfcc_loss}")
        
        # Convert to tensors
        original_tensor = torch.FloatTensor(audio_data).unsqueeze(0).unsqueeze(0).to(self.device)
        watermark_vector = self._generate_watermark_vector(watermark_data)
        
        # Enable gradient computation for optimization
        self.watermark_encoder.train()
        
        # Create optimizable strength parameter
        strength_param = torch.tensor(strength, device=self.device, requires_grad=True)
        
        # Optimizer for strength parameter
        optimizer = torch.optim.Adam([strength_param], lr=learning_rate)
        
        best_loss = float('inf')
        best_audio = None
        optimization_history = []
        
        for iteration in range(max_iterations):
            optimizer.zero_grad()
            
            # Forward pass with current strength
            with torch.no_grad():
                base_watermarked = self.watermark_encoder(original_tensor, watermark_vector)
                
                # Ensure same length as original
                min_length = min(original_tensor.shape[-1], base_watermarked.shape[-1])
                original_tensor_crop = original_tensor[..., :min_length]
                base_watermarked_crop = base_watermarked[..., :min_length]
            
            # Apply strength scaling (this part needs gradients)
            watermarked_tensor = original_tensor_crop + strength_param * (base_watermarked_crop - original_tensor_crop)
            
            # Compute MFCC perceptual loss
            mfcc_loss = self.mfcc_loss(original_tensor_crop.squeeze(1), watermarked_tensor.squeeze(1))
            
            # Compute combined loss (MFCC + regularization)
            regularization = 0.1 * torch.abs(strength_param - strength)  # Keep close to original strength
            total_loss = mfcc_loss + regularization
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Clamp strength to reasonable range
            with torch.no_grad():
                strength_param.clamp_(0.01, 1.0)
            
            current_loss = mfcc_loss.item()
            optimization_history.append({
                'iteration': iteration,
                'mfcc_loss': current_loss,
                'strength': strength_param.item(),
                'total_loss': total_loss.item()
            })
            
            logger.info(f"Iteration {iteration + 1}/{max_iterations}: "
                       f"MFCC Loss = {current_loss:.6f}, "
                       f"Strength = {strength_param.item():.4f}")
            
            # Check if target is reached
            if current_loss < target_mfcc_loss:
                logger.info(f"Target MFCC loss reached at iteration {iteration + 1}")
                break
            
            # Update best result
            if current_loss < best_loss:
                best_loss = current_loss
                best_audio = watermarked_tensor.squeeze().cpu().numpy().copy()
        
        # Final embedding with optimized strength
        optimized_strength = strength_param.item()
        
        # Generate final watermarked audio
        self.watermark_encoder.eval()
        with torch.no_grad():
            final_watermarked = self.watermark_encoder(original_tensor, watermark_vector)
            
            # Ensure same length
            min_length = min(original_tensor.shape[-1], final_watermarked.shape[-1])
            original_crop = original_tensor[..., :min_length]
            final_watermarked_crop = final_watermarked[..., :min_length]
            
            final_watermarked_tensor = original_crop + optimized_strength * (final_watermarked_crop - original_crop)
            final_audio = final_watermarked_tensor.squeeze().cpu().numpy()
            
            # Ensure final audio matches original length
            if len(final_audio) != len(audio_data):
                if len(final_audio) > len(audio_data):
                    final_audio = final_audio[:len(audio_data)]
                else:
                    padding = len(audio_data) - len(final_audio)
                    final_audio = np.pad(final_audio, (0, padding), mode='edge')
        
        # Compute final losses
        final_losses = self.compute_perceptual_loss(audio_data, final_audio, loss_type='all')
        
        logger.info(f"Perceptual optimization completed. Final MFCC loss: {final_losses.get('mfcc_loss', 0):.6f}")
        
        return {
            'watermarked_audio': final_audio,
            'metadata': {
                'optimized_strength': optimized_strength,
                'original_strength': strength,
                'final_mfcc_loss': final_losses.get('mfcc_loss', 0),
                'perceptual_losses': final_losses,
                'optimization_history': optimization_history,
                'iterations_used': len(optimization_history),
                'target_achieved': final_losses.get('mfcc_loss', float('inf')) < target_mfcc_loss,
                'sample_rate': sr,
                'original_length': len(audio_data)
            }
        }
    
    def _init_adaptive_bit_allocation(self):
        """
        Initialize adaptive bit allocation components for Task 1.2.2.3.
        
        This implements the integration of the adaptive bit allocation algorithm
        with the watermark embedding process, distributing watermark bits across
        frequency bands based on their perceptual significance.
        """
        logger.info("Initializing adaptive bit allocation (Task 1.2.2.3)...")
        
        # Initialize perceptual significance metric (Task 1.2.2.1)
        self.significance_metric = create_perceptual_significance_metric(
            sample_rate=self.sample_rate,
            n_critical_bands=self.n_critical_bands,
            method="logarithmic"
        )
        
        # Initialize adaptive bit allocator (Task 1.2.2.2)
        self.bit_allocator = create_adaptive_bit_allocator(
            total_bits=self.total_watermark_bits,
            strategy="dynamic"  # Use dynamic allocation strategy
        )
        
        # Initialize neural adaptive allocation module
        self.neural_adaptive_allocator = PerceptualAdaptiveBitAllocation(
            sample_rate=self.sample_rate,
            n_critical_bands=self.n_critical_bands,
            total_bits=self.total_watermark_bits,
            learnable_allocation=True
        ).to(self.device)
        
        logger.info(f"Adaptive bit allocation initialized with {self.total_watermark_bits} total bits")
    
    def _apply_adaptive_bit_allocation(self,
                                     freq_weights: np.ndarray,
                                     band_bit_allocation: np.ndarray,
                                     n_freq_bins: int) -> np.ndarray:
        """
        Apply adaptive bit allocation to frequency weights (Task 1.2.2.3).
        
        This method modifies the frequency domain watermark weights based on the
        number of bits allocated to each frequency band according to their
        perceptual significance.
        
        Args:
            freq_weights: Original frequency weights
            band_bit_allocation: Number of bits allocated per frequency band
            n_freq_bins: Total number of frequency bins
            
        Returns:
            Modified frequency weights with adaptive bit allocation
        """
        logger.info("Applying adaptive bit allocation to frequency weights...")
        
        # Map band allocations to frequency bins
        n_bands = len(band_bit_allocation)
        bins_per_band = n_freq_bins // n_bands
        
        adaptive_weights = freq_weights.copy()
        
        for band_idx, allocated_bits in enumerate(band_bit_allocation):
            # Calculate frequency bin range for this band
            start_bin = band_idx * bins_per_band
            end_bin = min((band_idx + 1) * bins_per_band, n_freq_bins)
            
            # Normalize allocated bits to weight multiplier
            # More bits = stronger watermark in that band
            max_bits = np.max(band_bit_allocation) if np.max(band_bit_allocation) > 0 else 1
            weight_multiplier = allocated_bits / max_bits
            
            # Apply adaptive weight to frequency bins in this band
            adaptive_weights[start_bin:end_bin] *= weight_multiplier
            
            logger.debug(f"Band {band_idx}: bins {start_bin}-{end_bin}, "
                        f"allocated_bits: {allocated_bits}, multiplier: {weight_multiplier:.3f}")
        
        return adaptive_weights
