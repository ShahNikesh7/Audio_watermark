"""
Perceptual Loss Functions for Audio Watermarking
Implements MFCC-based and other perceptual loss functions for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import sys
import os

# Add project root to path for attacks import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from attacks.simulate import simulate_attack, ATTACK_FUNCTIONS

logger = logging.getLogger(__name__)


class MFCCPerceptualLoss(nn.Module):
    """
    MFCC-based perceptual loss function for audio watermarking.
    Extracts MFCCs from both original and watermarked audio and computes MSE.
    """
    
    def __init__(self, 
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 sample_rate: int = 22050,
                 f_min: float = 0.0,
                 f_max: Optional[float] = None,
                 weight: float = 1.0):
        """
        Initialize MFCC-based perceptual loss.
        
        Args:
            n_mfcc: Number of MFCC coefficients to extract
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bands
            sample_rate: Audio sample rate
            f_min: Minimum frequency for mel filterbank
            f_max: Maximum frequency for mel filterbank (None = sr/2)
            weight: Loss weight factor
        """
        super(MFCCPerceptualLoss, self).__init__()
        
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.weight = weight
        
        # Pre-compute mel filterbank for efficiency
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=self.f_max
        )
        self.mel_basis = torch.FloatTensor(self.mel_basis)
        
        # Pre-compute DCT basis for MFCC computation
        # Use librosa's dct type-II matrix normalized
        dct_matrix = np.zeros((n_mfcc, n_mels))
        for k in range(n_mfcc):
            for n in range(n_mels):
                dct_matrix[k, n] = np.cos(np.pi * k * (2 * n + 1) / (2 * n_mels))
        # Normalize DCT
        dct_matrix[0] *= np.sqrt(1 / n_mels)
        dct_matrix[1:] *= np.sqrt(2 / n_mels)
        self.dct_basis = torch.FloatTensor(dct_matrix)
        
        logger.info(f"MFCC Perceptual Loss initialized: n_mfcc={n_mfcc}, weight={weight}")
    
    def _extract_mfcc_torch(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features using PyTorch operations for differentiability.
        
        Args:
            audio: Audio tensor of shape (batch_size, audio_length)
            
        Returns:
            MFCC features of shape (batch_size, n_mfcc, time_frames)
        """
        device = audio.device
        batch_size = audio.shape[0]
        
        # Move filterbanks to correct device
        if self.mel_basis.device != device:
            self.mel_basis = self.mel_basis.to(device)
            self.dct_basis = self.dct_basis.to(device)
        
        # Compute STFT
        stft = torch.stft(
            audio.view(-1, audio.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=device),
            return_complex=True
        )
        
        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)  # (batch_size, freq_bins, time_frames)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_basis, magnitude)  # (batch_size, n_mels, time_frames)
        
        # Convert to log scale (add small epsilon for numerical stability)
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Apply DCT to get MFCC
        mfcc = torch.matmul(self.dct_basis, log_mel_spec)  # (batch_size, n_mfcc, time_frames)
        
        return mfcc.view(batch_size, self.n_mfcc, -1)
    
    def forward(self, 
                original_audio: torch.Tensor, 
                watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC-based perceptual loss.
        
        Args:
            original_audio: Original audio tensor (batch_size, audio_length)
            watermarked_audio: Watermarked audio tensor (batch_size, audio_length)
            
        Returns:
            MFCC perceptual loss scalar
        """
        # Ensure same length
        min_length = min(original_audio.shape[-1], watermarked_audio.shape[-1])
        original_audio = original_audio[..., :min_length]
        watermarked_audio = watermarked_audio[..., :min_length]
        
        # Extract MFCC features
        mfcc_original = self._extract_mfcc_torch(original_audio)
        mfcc_watermarked = self._extract_mfcc_torch(watermarked_audio)
        
        # Ensure same time dimensions
        min_time_frames = min(mfcc_original.shape[-1], mfcc_watermarked.shape[-1])
        mfcc_original = mfcc_original[..., :min_time_frames]
        mfcc_watermarked = mfcc_watermarked[..., :min_time_frames]
        
        # Compute MSE between MFCC features
        mfcc_loss = F.mse_loss(mfcc_watermarked, mfcc_original)
        
        return self.weight * mfcc_loss


class SpectralPerceptualLoss(nn.Module):
    """
    Spectral-based perceptual loss using mel-spectrogram differences.
    """
    
    def __init__(self, 
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 n_mels: int = 128,
                 sample_rate: int = 22050,
                 weight: float = 1.0):
        """
        Initialize spectral perceptual loss.
        
        Args:
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bands
            sample_rate: Audio sample rate
            weight: Loss weight factor
        """
        super(SpectralPerceptualLoss, self).__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.weight = weight
        
        # Pre-compute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.mel_basis = torch.FloatTensor(self.mel_basis)
        
        logger.info(f"Spectral Perceptual Loss initialized: n_mels={n_mels}, weight={weight}")
    
    def forward(self, 
                original_audio: torch.Tensor, 
                watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral perceptual loss.
        
        Args:
            original_audio: Original audio tensor (batch_size, audio_length)
            watermarked_audio: Watermarked audio tensor (batch_size, audio_length)
            
        Returns:
            Spectral perceptual loss scalar
        """
        device = original_audio.device
        
        # Move mel basis to correct device
        if self.mel_basis.device != device:
            self.mel_basis = self.mel_basis.to(device)
        
        # Ensure same length
        min_length = min(original_audio.shape[-1], watermarked_audio.shape[-1])
        original_audio = original_audio[..., :min_length]
        watermarked_audio = watermarked_audio[..., :min_length]
        
        # Compute STFT
        stft_orig = torch.stft(
            original_audio.view(-1, original_audio.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=device),
            return_complex=True
        )
        
        stft_wm = torch.stft(
            watermarked_audio.view(-1, watermarked_audio.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft, device=device),
            return_complex=True
        )
        
        # Compute magnitude spectrograms
        mag_orig = torch.abs(stft_orig)
        mag_wm = torch.abs(stft_wm)
        
        # Apply mel filterbank
        mel_orig = torch.matmul(self.mel_basis, mag_orig)
        mel_wm = torch.matmul(self.mel_basis, mag_wm)
        
        # Convert to log scale
        log_mel_orig = torch.log(mel_orig + 1e-8)
        log_mel_wm = torch.log(mel_wm + 1e-8)
        
        # Compute MSE
        spectral_loss = F.mse_loss(log_mel_wm, log_mel_orig)
        
        return self.weight * spectral_loss


class CombinedPerceptualLoss(nn.Module):
    """
    Combined perceptual loss that incorporates multiple perceptual metrics.
    """
    
    def __init__(self,
                 mfcc_weight: float = 1.0,
                 spectral_weight: float = 0.5,
                 temporal_weight: float = 0.3,
                 sample_rate: int = 22050,
                 n_fft: int = 2048,
                 hop_length: int = 256):
        """
        Initialize combined perceptual loss.
        
        Args:
            mfcc_weight: Weight for MFCC loss
            spectral_weight: Weight for spectral loss
            temporal_weight: Weight for temporal loss
            sample_rate: Audio sample rate
        """
        super(CombinedPerceptualLoss, self).__init__()
        
        self.mfcc_loss = MFCCPerceptualLoss(
            weight=mfcc_weight,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        self.spectral_loss = SpectralPerceptualLoss(
            weight=spectral_weight,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        self.temporal_weight = temporal_weight
        
        logger.info(f"Combined Perceptual Loss initialized: "
                   f"MFCC={mfcc_weight}, Spectral={spectral_weight}, Temporal={temporal_weight}")
    
    def _temporal_loss(self, 
                      original_audio: torch.Tensor, 
                      watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal domain loss (RMS energy preservation).
        
        Args:
            original_audio: Original audio tensor
            watermarked_audio: Watermarked audio tensor
            
        Returns:
            Temporal loss scalar
        """
        # Compute RMS energy in sliding windows
        window_size = 1000
        hop_size = 400

        def compute_rms_energy(a: torch.Tensor):
            windows = a.unfold(-1, window_size, hop_size)
            return torch.sqrt(torch.mean(windows ** 2, dim=-1))
        
        rms_orig = compute_rms_energy(original_audio)
        rms_wm = compute_rms_energy(watermarked_audio)
        
        # Ensure same number of windows
        min_windows = min(rms_orig.shape[-1], rms_wm.shape[-1])
        rms_orig = rms_orig[..., :min_windows]
        rms_wm = rms_wm[..., :min_windows]
        
        return F.mse_loss(rms_wm, rms_orig)
    
    def forward(self, 
                original_audio: torch.Tensor, 
                watermarked_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined perceptual loss.
        
        Args:
            original_audio: Original audio tensor (batch_size, audio_length)
            watermarked_audio: Watermarked audio tensor (batch_size, audio_length)
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Compute individual losses
        mfcc_loss = self.mfcc_loss(original_audio, watermarked_audio)
        spectral_loss = self.spectral_loss(original_audio, watermarked_audio)
        temporal_loss = self.temporal_weight * self._temporal_loss(original_audio, watermarked_audio)
        
        # Compute total loss
        total_loss = mfcc_loss + spectral_loss + temporal_loss
        
        return {
            'mfcc_loss': mfcc_loss,
            'spectral_loss': spectral_loss,
            'temporal_loss': temporal_loss,
            'total_perceptual_loss': total_loss
        }


class AdversarialPerceptualLoss(nn.Module):
    """
    Adversarial loss for imperceptibility training.
    """
    
    def __init__(self, discriminator: nn.Module, weight: float = 0.1):
        """
        Initialize adversarial perceptual loss.
        
        Args:
            discriminator: Discriminator network
            weight: Loss weight factor
        """
        super(AdversarialPerceptualLoss, self).__init__()
        
        self.discriminator = discriminator
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        logger.info(f"Adversarial Perceptual Loss initialized: weight={weight}")
    
    def forward(self, watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute adversarial loss for generator training.
        
        Args:
            watermarked_audio: Watermarked audio tensor
            
        Returns:
            Adversarial loss scalar
        """
        # Get discriminator prediction for watermarked audio
        disc_pred = self.discriminator(watermarked_audio)
        
        # Generator wants discriminator to classify watermarked audio as real (1)
        target = torch.ones_like(disc_pred)
        
        adv_loss = self.bce_loss(disc_pred, target)
        
        return self.weight * adv_loss


class RobustnessLoss(nn.Module):
    """
    Robustness loss function that applies various attacks during training
    to make the watermark more robust against common audio processing operations.
    """
    
    def __init__(self,
                 sample_rate: int = 22050,
                 attack_types: List[str] = None,
                 attack_probability: float = 0.5,
                 severity_range: Tuple[float, float] = (0.1, 0.7),
                 weight: float = 1.0):
        """
        Initialize robustness loss.
        
        Args:
            sample_rate: Audio sample rate
            attack_types: List of attack types to use for robustness training
            attack_probability: Probability of applying an attack during training
            severity_range: Range of attack severities (min, max)
            weight: Loss weight factor
        """
        super(RobustnessLoss, self).__init__()
        
        self.sample_rate = sample_rate
        self.attack_probability = attack_probability
        self.severity_range = severity_range
        self.weight = weight
        
        # Default attack types for robustness training
        if attack_types is None:
            self.attack_types = [
                "additive_noise", "gaussian_noise", "mp3_compression", 
                "lowpass_filter", "highpass_filter", "clipping",
                "dynamic_range_compression", "time_stretch", "pitch_shift"
            ]
        else:
            self.attack_types = attack_types
        
        # Validate attack types
        valid_attacks = [attack for attack in self.attack_types if attack in ATTACK_FUNCTIONS]
        if len(valid_attacks) != len(self.attack_types):
            invalid = set(self.attack_types) - set(valid_attacks)
            logger.warning(f"Invalid attack types removed: {invalid}")
            self.attack_types = valid_attacks
        
        logger.info(f"Robustness Loss initialized: {len(self.attack_types)} attack types, "
                   f"weight={weight}, attack_prob={attack_probability}")
    
    def _apply_random_attack(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Apply a random attack to audio signal.
        
        Args:
            audio_np: Audio signal as numpy array
            
        Returns:
            Attacked audio signal
        """
        # Randomly decide whether to apply attack
        if np.random.rand() > self.attack_probability:
            return audio_np
        
        # Choose random attack
        attack_type = np.random.choice(self.attack_types)
        
        # Choose random severity
        severity = np.random.uniform(self.severity_range[0], self.severity_range[1])
        
        try:
            # Apply attack
            attacked_audio = simulate_attack(
                audio_np, attack_type, severity, self.sample_rate
            )
            return attacked_audio
        except Exception as e:
            logger.warning(f"Attack {attack_type} failed: {e}. Using original audio.")
            return audio_np
    
    def forward(self,
                original_audio: torch.Tensor,
                watermarked_audio: torch.Tensor,
                watermark_extractor_fn=None) -> torch.Tensor:
        """
        Compute robustness loss by applying attacks and measuring watermark preservation.
        
        Args:
            original_audio: Original audio tensor (batch_size, audio_length)
            watermarked_audio: Watermarked audio tensor (batch_size, audio_length)
            watermark_extractor_fn: Function to extract watermark (optional)
            
        Returns:
            Robustness loss scalar
        """
        batch_size = watermarked_audio.shape[0]
        device = watermarked_audio.device
        
        robustness_losses = []
        
        for i in range(batch_size):
            # Convert to numpy for attack simulation
            watermarked_np = watermarked_audio[i].cpu().numpy()
            if watermarked_np.ndim > 1:
                watermarked_np = watermarked_np[0]  # Remove channel dimension
            
            # Apply random attack
            attacked_audio_np = self._apply_random_attack(watermarked_np)
            
            # Convert back to tensor
            attacked_tensor = torch.FloatTensor(attacked_audio_np).unsqueeze(0).to(device)
            
            # Ensure same length
            min_length = min(watermarked_audio[i].shape[-1], attacked_tensor.shape[-1])
            watermarked_crop = watermarked_audio[i:i+1, :, :min_length]
            attacked_crop = attacked_tensor[:, :min_length].unsqueeze(0)
            
            # Compute loss between watermarked and attacked versions
            # This encourages the watermark to survive attacks
            loss = F.mse_loss(attacked_crop, watermarked_crop)
            robustness_losses.append(loss)
        
        # Average over batch
        total_robustness_loss = torch.stack(robustness_losses).mean()
        
        return self.weight * total_robustness_loss


class EnhancedCombinedPerceptualLoss(nn.Module):
    """
    Enhanced combined perceptual loss that incorporates robustness training.
    """
    
    def __init__(self,
                 mfcc_weight: float = 1.0,  # Optimal weight from Task 1.1.3.2
                 spectral_weight: float = 0.5,
                 temporal_weight: float = 0.3,
                 robustness_weight: float = 2.0,  # Optimal weight from Task 1.1.3.2
                 sample_rate: int = 22050):
        """
        Initialize enhanced combined perceptual loss with optimal weights.
        
        Optimal weights determined from Task 1.1.3.2 experimentation:
        - MFCC Weight: 1.0 (imperceptibility focus)
        - Robustness Weight: 2.0 (enhanced robustness)
        
        Args:
            mfcc_weight: Weight for MFCC loss (optimal: 1.0)
            spectral_weight: Weight for spectral loss
            temporal_weight: Weight for temporal loss
            robustness_weight: Weight for robustness loss (optimal: 2.0)
            sample_rate: Audio sample rate
        """
        super(EnhancedCombinedPerceptualLoss, self).__init__()
        
        self.mfcc_loss = MFCCPerceptualLoss(
            weight=mfcc_weight,
            sample_rate=sample_rate
        )
        
        self.spectral_loss = SpectralPerceptualLoss(
            weight=spectral_weight,
            sample_rate=sample_rate
        )
        
        self.robustness_loss = RobustnessLoss(
            weight=robustness_weight,
            sample_rate=sample_rate
        )
        
        self.temporal_weight = temporal_weight
        
        logger.info(f"Enhanced Combined Perceptual Loss initialized with optimal weights from Task 1.1.3.2: "
                   f"MFCC={mfcc_weight}, Spectral={spectral_weight}, "
                   f"Temporal={temporal_weight}, Robustness={robustness_weight}")
    
    def _temporal_loss(self, 
                      original_audio: torch.Tensor, 
                      watermarked_audio: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal domain loss (RMS energy preservation).
        """
        # Compute RMS energy in sliding windows
        window_size = 1000
        hop_size = 400

        def compute_rms_energy(a: torch.Tensor):
            windows = a.unfold(-1, window_size, hop_size)
            return torch.sqrt(torch.mean(windows ** 2, dim=-1))
        
        rms_orig = compute_rms_energy(original_audio)
        rms_wm = compute_rms_energy(watermarked_audio)
        
        # Ensure same number of windows
        min_windows = min(rms_orig.shape[-1], rms_wm.shape[-1])
        rms_orig = rms_orig[..., :min_windows]
        rms_wm = rms_wm[..., :min_windows]
        
        return F.mse_loss(rms_wm, rms_orig)
    
    def forward(self, 
                original_audio: torch.Tensor, 
                watermarked_audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute enhanced combined perceptual loss with robustness.
        
        Args:
            original_audio: Original audio tensor (batch_size, audio_length)
            watermarked_audio: Watermarked audio tensor (batch_size, audio_length)
            
        Returns:
            Dictionary containing individual and total losses
        """
        # Compute individual losses
        mfcc_loss = self.mfcc_loss(original_audio, watermarked_audio)
        spectral_loss = self.spectral_loss(original_audio, watermarked_audio)
        temporal_loss = self.temporal_weight * self._temporal_loss(original_audio, watermarked_audio)
        robustness_loss = self.robustness_loss(original_audio, watermarked_audio)
        
        # Compute total loss
        total_loss = mfcc_loss + spectral_loss + temporal_loss + robustness_loss
        
        return {
            'mfcc_loss': mfcc_loss,
            'spectral_loss': spectral_loss,
            'temporal_loss': temporal_loss,
            'robustness_loss': robustness_loss,
            'total_perceptual_loss': total_loss
        }


def create_perceptual_loss(loss_config: Dict) -> nn.Module:
    """
    Factory function to create perceptual loss based on configuration.
    
    Args:
        loss_config: Dictionary containing loss configuration
        
    Returns:
        Configured perceptual loss module
    """
    loss_type = loss_config.get('type', 'mfcc')
    
    if loss_type == 'mfcc':
        return MFCCPerceptualLoss(**loss_config.get('params', {}))
    elif loss_type == 'spectral':
        return SpectralPerceptualLoss(**loss_config.get('params', {}))
    elif loss_type == 'combined':
        return CombinedPerceptualLoss(**loss_config.get('params', {}))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
