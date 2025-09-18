"""
Training Module for Audio Watermarking with MFCC-based Perceptual Loss
Implements comprehensive training pipeline with perceptual loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa
import os
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm

# Optional wandb import for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ..perceptual_cnn.mobilenetv3 import MobileNetV3
from ..technical_rnn.gru_module import GRUModule
from ..technical_rnn.conformer_lite import ConformerLite
from ..fusion.fusion_layer import FusionLayer
from ..psychoacoustic.perceptual_losses import MFCCPerceptualLoss, CombinedPerceptualLoss, create_perceptual_loss
from ..psychoacoustic.moore_glasberg import MooreGlasbergAnalyzer, PerceptualAnalyzer
from ...embedding import InvertibleEncoder, AudioWatermarkEmbedder
from ...extraction import WatermarkDetector

logger = logging.getLogger(__name__)


class AudioWatermarkDataset(Dataset):
    """
    Dataset for audio watermarking training.
    """
    
    def __init__(self, 
                 audio_dir: str,
                 sample_rate: int = 22050,
                 duration: float = 3.0,
                 augment: bool = True):
        """
        Initialize audio dataset.
        
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            duration: Audio clip duration in seconds
            augment: Whether to apply data augmentation
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        self.audio_files = self._find_audio_files()
        
        logger.info(f"AudioWatermarkDataset initialized with {len(self.audio_files)} files")
    
    def _find_audio_files(self) -> List[str]:
        """Find all audio files in the directory."""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []
        
        for root, dirs, files in os.walk(self.audio_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_files.append(os.path.join(root, file))
        
        return audio_files
    
    def _load_audio_segment(self, file_path: str) -> np.ndarray:
        """Load a random segment from audio file."""
        try:
            # Load full audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # Calculate target length
            target_length = int(self.duration * self.sample_rate)
            
            if len(audio) >= target_length:
                # Random crop
                start_idx = np.random.randint(0, len(audio) - target_length + 1)
                audio_segment = audio[start_idx:start_idx + target_length]
            else:
                # Pad if too short
                padding = target_length - len(audio)
                audio_segment = np.pad(audio, (0, padding), mode='reflect')
            
            return audio_segment
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            # Return silence as fallback
            return np.zeros(int(self.duration * self.sample_rate))
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """Apply data augmentation to audio."""
        if not self.augment:
            return audio
        
        # Random amplitude scaling (0.7 to 1.3)
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.7, 1.3)
            audio = audio * scale
        
        # Random time shifting (up to 10% of duration)
        if np.random.rand() < 0.3:
            shift_samples = int(0.1 * len(audio))
            shift = np.random.randint(-shift_samples, shift_samples + 1)
            audio = np.roll(audio, shift)
        
        # Add subtle noise (SNR > 40 dB)
        if np.random.rand() < 0.3:
            noise_power = np.var(audio) / (10 ** (40 / 10))  # 40 dB SNR
            noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
            audio = audio + noise
        
        # Ensure audio is in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a training sample."""
        file_path = self.audio_files[idx]
        audio = self._load_audio_segment(file_path)
        audio = self._augment_audio(audio)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # Add channel dimension
        
        # Generate random watermark for training
        watermark_dim = 128
        watermark = torch.randn(watermark_dim) * 0.1  # Small random watermark
        
        return {
            'audio': audio_tensor,
            'watermark': watermark,
            'file_path': file_path
        }


class WatermarkTrainer:
    """
    Comprehensive trainer for audio watermarking with MFCC-based perceptual loss.
    """
    
    def __init__(self,
                 config: Dict,
                 device: str = 'cuda',
                 use_wandb: bool = False):
        """
        Initialize the watermark trainer.
        
        Args:
            config: Training configuration dictionary
            device: Training device ('cuda' or 'cpu')
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb
        # Caches for GPU mel computation in quality metrics
        self._mel_basis = None
        self._hann_window = None
        
        # Initialize models
        self._init_models()
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        # Track the loss value at the moment we last wrote a checkpoint
        self.last_saved_loss = float('inf')
        
        logger.info(f"WatermarkTrainer initialized on device: {self.device}")
    
    def _init_models(self):
        """Initialize all models."""
        model_config = self.config['model']
        
        # Main watermark encoder
        self.watermark_encoder = InvertibleEncoder(
            audio_channels=1,
            watermark_dim=128  # Optimal neural network dimension
        ).to(self.device)
        
        # PHM components for quality assessment
        if model_config.get('use_phm', True):
            self.perceptual_cnn = MobileNetV3(
                num_classes=128,
                input_channels=1,
                variant='small'
            ).to(self.device)
            
            self.technical_gru = GRUModule(
                input_size=32,
                hidden_size=64,
                num_layers=2
            ).to(self.device)
            
            self.technical_conformer = ConformerLite(
                input_dim=32,
                num_heads=4,
                ff_dim=128,
                num_layers=2
            ).to(self.device)
            
            self.fusion_layer = FusionLayer(
                perceptual_dim=128,
                technical_dim=1,
                fusion_dim=128,
                num_heads=4
            ).to(self.device)
        
        # Psychoacoustic analyzer
        if model_config.get('use_psychoacoustic', True):
            self.perceptual_analyzer = PerceptualAnalyzer(
                sample_rate=self.config['data']['sample_rate'],
                n_fft=1000,
                hop_length=400,
                n_critical_bands=24
            ).to(self.device)

        # Watermark detector for BER metric (inference only)
        self.watermark_detector = WatermarkDetector(
            audio_channels=1,
            watermark_dim=128
        ).to(self.device)
        self.watermark_detector.eval()
        
        logger.info("Models initialized successfully")
    
    def _init_loss_functions(self):
        """Initialize loss functions."""
        loss_config = self.config['loss']
        
        # MFCC-based perceptual loss
        self.mfcc_loss = MFCCPerceptualLoss(
            n_mfcc=loss_config.get('n_mfcc', 13),
            n_fft=loss_config.get('n_fft', 2048),
            hop_length=loss_config.get('hop_length', 512),
            sample_rate=self.config['data']['sample_rate'],
            weight=loss_config.get('mfcc_weight', 1.0)
        ).to(self.device)
        
        # Combined perceptual loss
        self.perceptual_loss = CombinedPerceptualLoss(
            mfcc_weight=loss_config.get('mfcc_weight', 1.0),
            spectral_weight=loss_config.get('spectral_weight', 0.5),
            temporal_weight=loss_config.get('temporal_weight', 0.3),
            sample_rate=self.config['data']['sample_rate']
        ).to(self.device)
        
        # Reconstruction loss
        self.reconstruction_loss = nn.MSELoss()
        
        # Quality preservation loss
        self.quality_loss = nn.MSELoss()
        
        logger.info("Loss functions initialized successfully")
    
    def _init_optimizers(self):
        """Initialize optimizers."""
        opt_config = self.config['optimizer']
        
        # Collect all trainable parameters
        params = []
        params.extend(self.watermark_encoder.parameters())
        
        if hasattr(self, 'perceptual_cnn'):
            params.extend(self.perceptual_cnn.parameters())
            params.extend(self.technical_gru.parameters())
            params.extend(self.technical_conformer.parameters())
            params.extend(self.fusion_layer.parameters())
        
        if hasattr(self, 'perceptual_analyzer'):
            params.extend(self.perceptual_analyzer.parameters())
        
        # Main optimizer
        self.optimizer = optim.AdamW(
            params,
            lr=opt_config.get('learning_rate', 1e-4),
            weight_decay=opt_config.get('weight_decay', 1e-5),
            betas=opt_config.get('betas', (0.9, 0.999))
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=opt_config.get('scheduler_t0', 10),
            T_mult=opt_config.get('scheduler_tmult', 2),
            eta_min=opt_config.get('min_lr', 1e-6)
        )
        
        logger.info("Optimizers initialized successfully")
    
    def _init_data_loaders(self):
        """Initialize data loaders."""
        data_config = self.config['data']
        
        # Training dataset
        train_dataset = AudioWatermarkDataset(
            audio_dir=data_config['train_dir'],
            sample_rate=data_config['sample_rate'],
            duration=data_config.get('duration', 3.0),
            augment=data_config.get('augment', True)
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 8),
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True
        )
        
        # Validation dataset
        if data_config.get('val_dir'):
            val_dataset = AudioWatermarkDataset(
                audio_dir=data_config['val_dir'],
                sample_rate=data_config['sample_rate'],
                duration=data_config.get('duration', 3.0),
                augment=False  # No augmentation for validation
            )
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=data_config.get('batch_size', 8),
                shuffle=False,
                num_workers=data_config.get('num_workers', 4),
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        logger.info(f"Data loaders initialized: train_size={len(train_dataset)}")
    
    def _compute_quality_metrics(self, original_audio: torch.Tensor) -> torch.Tensor:
        """Compute quality metrics using PHM with GPU-accelerated mel path."""
        if not hasattr(self, 'perceptual_cnn'):
            return torch.tensor(0.8, device=self.device).expand(original_audio.shape[0])

        with torch.no_grad():
            device = original_audio.device
            sr = int(self.config['data']['sample_rate'])
            n_fft = 1000
            hop = 400
            n_mels = 128

            # Cache window and mel filterbank on device
            if self._hann_window is None or self._hann_window.device != device:
                self._hann_window = torch.hann_window(n_fft, device=device)
            if self._mel_basis is None or self._mel_basis.device != device:
                mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
                self._mel_basis = torch.tensor(mel_fb, dtype=torch.float32, device=device)

            # [B, 1, T] -> [B, T] float32
            audio_bt = original_audio[:, 0, :].float()

            # STFT on GPU (keep in float32; no autocast)
            stft = torch.stft(
                audio_bt,
                n_fft=n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=self._hann_window,
                return_complex=True
            )
            power_spec = (stft.abs() ** 2).float()

            # Mel projection: [n_mels, F] @ [B, F, T] -> [B, n_mels, T]
            mel_spec = torch.matmul(self._mel_basis, power_spec)

            # Convert to dB-like scale and normalize per sample
            mel_db = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
            mel_min = mel_db.amin(dim=(1, 2), keepdim=True)
            mel_max = mel_db.amax(dim=(1, 2), keepdim=True)
            mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

            # Resize to CNN input [B, 1, 128, 128]
            mel_img = F.interpolate(mel_norm.unsqueeze(1), size=(128, 128), mode='bilinear', align_corners=False)

            # Pass through CNN and produce quality score
            perceptual_features = self.perceptual_cnn(mel_img)
            quality_scores = torch.sigmoid(perceptual_features.mean(dim=1))
            return quality_scores
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one training step."""
        self.optimizer.zero_grad()
        
        original_audio = batch['audio'].to(self.device)
        watermark = batch['watermark'].to(self.device)
        
        # Forward pass: embed watermark
        watermarked_audio = self.watermark_encoder(original_audio, watermark)
        
        # Compute losses
        losses = {}
        
        # 1. MFCC-based perceptual loss (primary)
        mfcc_loss = self.mfcc_loss(original_audio.squeeze(1), watermarked_audio.squeeze(1))
        losses['mfcc_loss'] = mfcc_loss.item()
        
        # 2. Combined perceptual loss
        perceptual_losses = self.perceptual_loss(original_audio.squeeze(1), watermarked_audio.squeeze(1))
        losses.update({k: v.item() for k, v in perceptual_losses.items()})
        
        # 3. Reconstruction loss (with adaptive weighting)
        reconstruction_loss = self.reconstruction_loss(watermarked_audio, original_audio)
        losses['reconstruction_loss'] = reconstruction_loss.item()
        
        # 4. Quality preservation loss
        original_quality = self._compute_quality_metrics(original_audio)
        watermarked_quality = self._compute_quality_metrics(watermarked_audio)
        quality_loss = self.quality_loss(watermarked_quality, original_quality)
        losses['quality_loss'] = quality_loss.item()
        
        # Combine losses with weights
        loss_weights = self.config['loss']
        total_loss = (
            loss_weights.get('mfcc_weight', 1.0) * mfcc_loss +
            loss_weights.get('perceptual_weight', 1.0) * perceptual_losses['total_perceptual_loss'] +
            loss_weights.get('reconstruction_weight', 0.1) * reconstruction_loss +
            loss_weights.get('quality_weight', 0.1) * quality_loss
        )
        
        losses['total_loss'] = total_loss.item()

        # Compute BER (bit error rate) via detector (no grad, metric only)
        try:
            with torch.no_grad():
                presence_prob, predicted_vector = self.watermark_detector(watermarked_audio.detach())
                # Threshold both ground truth and predictions at 0 to form bits
                gt_bits = (watermark > 0).float()
                pred_bits = (predicted_vector > 0).float()
                # Align shapes if needed
                min_len = min(gt_bits.shape[-1], pred_bits.shape[-1])
                gt_bits = gt_bits[..., :min_len]
                pred_bits = pred_bits[..., :min_len]
                ber = (pred_bits.ne(gt_bits)).float().mean()
                losses['ber'] = ber.item()
        except Exception:
            # If detector not available or any error, skip BER
            pass
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for model in [self.watermark_encoder] for p in model.parameters()],
            max_norm=self.config['training'].get('grad_clip', 1.0)
        )
        
        self.optimizer.step()
        self.global_step += 1
        
        return losses
    
    def validate(self) -> Dict[str, float]:
        """Perform validation."""
        if self.val_loader is None:
            return {}
        
        self.watermark_encoder.eval()
        if hasattr(self, 'perceptual_cnn'):
            self.perceptual_cnn.eval()
            self.technical_gru.eval()
            self.technical_conformer.eval()
            self.fusion_layer.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                original_audio = batch['audio'].to(self.device)
                watermark = batch['watermark'].to(self.device)
                
                # Forward pass
                watermarked_audio = self.watermark_encoder(original_audio, watermark)
                
                # Compute MFCC loss
                mfcc_loss = self.mfcc_loss(original_audio.squeeze(1), watermarked_audio.squeeze(1))
                
                # Compute perceptual losses
                perceptual_losses = self.perceptual_loss(original_audio.squeeze(1), watermarked_audio.squeeze(1))
                
                val_losses.append({
                    'mfcc_loss': mfcc_loss.item(),
                    'total_perceptual_loss': perceptual_losses['total_perceptual_loss'].item()
                })
        
        # Compute average validation metrics
        avg_val_loss = {}
        if val_losses:
            for key in val_losses[0].keys():
                avg_val_loss[f'val_{key}'] = np.mean([loss[key] for loss in val_losses])
        
        # Set models back to training mode
        self.watermark_encoder.train()
        if hasattr(self, 'perceptual_cnn'):
            self.perceptual_cnn.train()
            self.technical_gru.train()
            self.technical_conformer.train()
            self.fusion_layer.train()
        
        return avg_val_loss
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.watermark_encoder.train()
        if hasattr(self, 'perceptual_cnn'):
            self.perceptual_cnn.train()
            self.technical_gru.train()
            self.technical_conformer.train()
            self.fusion_layer.train()
        
        epoch_losses = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}')
        for batch in pbar:
            losses = self.train_step(batch)
            epoch_losses.append(losses)
            
            # Update progress bar
            pbar.set_postfix({
                'MFCC': f"{losses.get('mfcc_loss', 0.0):.4f}",
                'Total': f"{losses.get('total_loss', 0.0):.4f}",
                'BER': f"{losses.get('ber', 0.0):.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Compute average epoch metrics
        avg_epoch_loss = {}
        if epoch_losses:
            for key in epoch_losses[0].keys():
                avg_epoch_loss[key] = np.mean([loss[key] for loss in epoch_losses])
        
        return avg_epoch_loss
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'watermark_encoder': self.watermark_encoder.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config
        }
        
        # Add PHM components if available
        if hasattr(self, 'perceptual_cnn'):
            checkpoint.update({
                'perceptual_cnn': self.perceptual_cnn.state_dict(),
                'technical_gru': self.technical_gru.state_dict(),
                'technical_conformer': self.technical_conformer.state_dict(),
                'fusion_layer': self.fusion_layer.state_dict()
            })
        
        if hasattr(self, 'perceptual_analyzer'):
            checkpoint['perceptual_analyzer'] = self.perceptual_analyzer.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = filepath.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_filepath)
        
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.watermark_encoder.load_state_dict(checkpoint['watermark_encoder'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        # Load PHM components if available
        if hasattr(self, 'perceptual_cnn') and 'perceptual_cnn' in checkpoint:
            self.perceptual_cnn.load_state_dict(checkpoint['perceptual_cnn'])
            self.technical_gru.load_state_dict(checkpoint['technical_gru'])
            self.technical_conformer.load_state_dict(checkpoint['technical_conformer'])
            self.fusion_layer.load_state_dict(checkpoint['fusion_layer'])
        
        if hasattr(self, 'perceptual_analyzer') and 'perceptual_analyzer' in checkpoint:
            self.perceptual_analyzer.load_state_dict(checkpoint['perceptual_analyzer'])
        
        logger.info(f"Checkpoint loaded: {filepath}")
    
    def train(self, num_epochs: int, save_dir: str):
        """Main training loop."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize wandb if enabled and available
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project="audio-watermarking",
                config=self.config,
                name=f"mfcc_perceptual_training_{self.config.get('experiment_name', 'default')}"
            )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']
            metrics['epoch'] = epoch
            
            # Log metrics
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - " + 
                       " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
            
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log(metrics, step=self.global_step)
            
            # Save checkpoint strictly every 5 epochs
            if ((epoch + 1) % 5) == 0:
                current_loss = val_metrics.get('val_mfcc_loss', train_metrics.get('mfcc_loss', float('inf')))
                is_best = current_loss < self.best_loss
                if is_best:
                    self.best_loss = current_loss
                checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_path, is_best)
                self.last_saved_loss = current_loss
            
            # Save config
            config_path = os.path.join(save_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        
        logger.info("Training completed!")
        
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def create_training_config(
    train_dir: str,
    val_dir: Optional[str] = None,
    sample_rate: int = 22050,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    epochs: int = 10,
    enable_mfcc_loss: bool = True,
    mfcc_loss_weight: float = 1.0,
    experiment_name: str = "mfcc_perceptual_training"
) -> Dict:
    """
    Create a default training configuration.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        sample_rate: Audio sample rate
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Number of training epochs
        enable_mfcc_loss: Whether to enable MFCC loss
        mfcc_loss_weight: Weight for MFCC loss
        experiment_name: Experiment name
        
    Returns:
        Training configuration dictionary
    """
    return {
        'experiment_name': experiment_name,
        'data': {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'sample_rate': sample_rate,
            'duration': 3.0,
            'batch_size': batch_size,
            'num_workers': 4,
            'augment': True
        },
        'model': {
            'watermark_dim': 128,
            'use_phm': True,
            'use_psychoacoustic': True
        },
        'loss': {
            'enable_mfcc_loss': enable_mfcc_loss,
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'mfcc_loss_weight': mfcc_loss_weight,
            'spectral_weight': 0.5,
            'temporal_weight': 0.3,
            'perceptual_weight': 1.0,
            'reconstruction_weight': 0.1,
            'quality_weight': 0.1
        },
        'optimizer': {
            'learning_rate': learning_rate,
            'weight_decay': 1e-5,
            'betas': (0.9, 0.999),
            'scheduler_t0': 10,
            'scheduler_tmult': 2,
            'min_lr': 1e-6
        },
        'training': {
            'epochs': epochs,
            'grad_clip': 1.0
        }
    }
