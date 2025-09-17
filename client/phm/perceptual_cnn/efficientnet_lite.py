"""
EfficientNet Lite implementation for perceptual quality assessment.
Lightweight version optimized for mobile deployment.

Task 1.2.1: BCH Error Correction Integration
Task 1.2.2.1: Perceptual Significance Metric for Adaptive Bit Allocation
- Enhanced with BCH error correction for robust watermarking
- Optimized BCH parameters for EfficientNet architecture
- Integrated encoding/decoding for watermark protection
- Perceptual significance-based adaptive bit allocation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from .bch_error_correction import CNNBCHWatermarkProtector, optimize_bch_parameters_for_cnn
from ..psychoacoustic.adaptive_bit_allocation import (
    PerceptualSignificanceMetric, 
    AdaptiveBitAllocator,
    create_perceptual_significance_metric,
    create_adaptive_bit_allocator
)

logger = logging.getLogger(__name__)


class EfficientNetLite(nn.Module):
    """EfficientNet Lite model for perceptual quality assessment with BCH error correction and adaptive bit allocation."""
    
    def __init__(self, num_classes=1, input_channels=1,
                 enable_bch=True, message_length=1000, robustness_level="medium",
                 enable_adaptive_allocation=True, sample_rate=16000):
        super(EfficientNetLite, self).__init__()
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.enable_bch = enable_bch
        self.message_length = message_length
        self.robustness_level = robustness_level
        self.enable_adaptive_allocation = enable_adaptive_allocation
        self.sample_rate = sample_rate
        
        # Initialize BCH watermark protector
        if self.enable_bch:
            self.bch_protector = CNNBCHWatermarkProtector(
                message_length=message_length,
                robustness_level=robustness_level,
                cnn_model_type="efficientnet"
            )
            logger.info(f"BCH protection enabled for EfficientNetLite: "
                       f"msg_len={message_length}, robustness={robustness_level}")
        
        # Initialize perceptual significance metric and adaptive bit allocator (Task 1.2.2.1)
        if self.enable_adaptive_allocation:
            self.significance_metric = create_perceptual_significance_metric(
                sample_rate=sample_rate,
                n_critical_bands=24,
                method="logarithmic"
            )
            
            # Determine total bits available for allocation
            total_bits = self.bch_protector.get_codeword_length() if self.enable_bch else message_length
            
            self.bit_allocator = create_adaptive_bit_allocator(
                total_bits=total_bits,
                strategy="optimal"
            )
            
            # Learnable adaptation parameters for CNN-specific optimization
            self.allocation_adaptation = nn.Parameter(torch.ones(24))  # 24 critical bands
            self.band_weight_bias = nn.Parameter(torch.zeros(24))
            self.allocation_temperature = nn.Parameter(torch.tensor(1.0))
            
            logger.info(f"Adaptive bit allocation enabled: {total_bits} total bits across 24 bands")
        
        # Basic EfficientNet architecture
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # MBConv blocks would go here
        self.blocks = nn.Sequential(
            self._make_block(32, 64, 2, 1),
            self._make_block(64, 128, 4, 2),
            self._make_block(128, 256, 4, 2),
        )
        
        self.head = nn.Sequential(
            nn.Conv2d(256, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier for perceptual quality
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Watermark embedding head (if BCH enabled)
        if self.enable_bch:
            codeword_length = self.bch_protector.get_codeword_length()
            self.watermark_embedder = nn.Sequential(
                nn.Linear(1280, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, codeword_length),
                nn.Tanh()  # Output between -1 and 1
            )
            
            # Watermark extraction head
            self.watermark_extractor = nn.Sequential(
                nn.Linear(1280, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, codeword_length),
                nn.Sigmoid()  # Output between 0 and 1 for bits
            )
            
        # Band-specific embedding heads for adaptive allocation
        if self.enable_adaptive_allocation:
            self.band_embedders = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1280, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 8),  # Max 8 bits per band
                    nn.Tanh()
                ) for _ in range(24)  # 24 critical bands
            ])
            
            self.band_extractors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(1280, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 8),  # Max 8 bits per band
                    nn.Sigmoid()
                ) for _ in range(24)  # 24 critical bands
            ])
    
    def _make_block(self, in_channels, out_channels, num_blocks, stride):
        """Create a block of MBConv layers."""
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(self._mbconv_block(in_channels, out_channels, s))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _mbconv_block(self, in_channels, out_channels, stride):
        """Mobile Inverted Bottleneck Conv block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, message_bits=None, audio=None, mode="quality"):
        """
        Forward pass with multiple modes including adaptive allocation.
        
        Args:
            x: Input tensor
            message_bits: Message bits for watermarking [batch_size, message_length]
            audio: Original audio signal [batch_size, samples] (for adaptive allocation)
            mode: "quality" for quality assessment, "embed" for watermark embedding,
                  "extract" for watermark extraction, "adaptive_embed" for adaptive embedding,
                  "adaptive_extract" for adaptive extraction
        
        Returns:
            Output tensor or dictionary depending on mode
        """
        # Common feature extraction
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        features = x.view(x.size(0), -1)
        
        if mode == "quality":
            return self.classifier(features)
        elif mode == "embed" and self.enable_bch:
            return self.watermark_embedder(features)
        elif mode == "extract" and self.enable_bch:
            return self.watermark_extractor(features)
        elif mode == "adaptive_embed" and self.enable_adaptive_allocation:
            if message_bits is None or audio is None:
                raise ValueError("message_bits and audio required for adaptive embedding")
            return self.embed_watermark_with_adaptive_allocation(x, audio, message_bits)
        elif mode == "adaptive_extract" and self.enable_adaptive_allocation:
            if audio is None:
                raise ValueError("audio required for adaptive extraction")
            return self.extract_watermark_with_adaptive_allocation(x, audio)
        else:
            return self.classifier(features)
    
    def embed_watermark_with_bch(self, audio_features: torch.Tensor, 
                                message_bits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed watermark with BCH error correction.
        
        Args:
            audio_features: Audio feature tensor for CNN processing
            message_bits: Original message bits [batch_size, message_length]
            
        Returns:
            Tuple of (watermarked_features, embedding_info)
        """
        if not self.enable_bch:
            raise ValueError("BCH protection not enabled")
        
        # Encode message with BCH
        encoded_bits = self.bch_protector.encode_watermark(message_bits)
        
        # Get embedding signals from CNN
        embedding_signals = self.forward(audio_features, mode="embed")
        
        # Apply BCH-encoded bits to embedding signals
        # Scale encoded bits to match embedding signal range
        scaled_bits = (encoded_bits * 2.0 - 1.0) * 0.1  # Scale to [-0.1, 0.1]
        
        # Modulate embedding signals with encoded bits
        watermarked_signals = embedding_signals + scaled_bits
        
        embedding_info = {
            "original_message_length": torch.tensor(self.message_length),
            "encoded_length": torch.tensor(self.bch_protector.get_codeword_length()),
            "code_rate": torch.tensor(self.bch_protector.get_code_rate()),
            "embedding_strength": torch.mean(torch.abs(scaled_bits))
        }
        
        return watermarked_signals, embedding_info
    
    def extract_watermark_with_bch(self, audio_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract and decode watermark with BCH error correction.
        
        Args:
            audio_features: Audio feature tensor for CNN processing
            
        Returns:
            Tuple of (decoded_message_bits, extraction_info)
        """
        if not self.enable_bch:
            raise ValueError("BCH protection not enabled")
        
        # Extract bits using CNN
        extracted_signals = self.forward(audio_features, mode="extract")
        
        # Convert signals to bits (0 or 1)
        extracted_bits = (extracted_signals > 0.5).float()
        
        # Decode with BCH error correction
        decoded_message, quality_metrics = self.bch_protector.decode_watermark(extracted_bits)
        
        extraction_info = {
            "extracted_length": torch.tensor(extracted_bits.shape[-1]),
            "decoded_length": torch.tensor(decoded_message.shape[-1]),
            **quality_metrics
        }
        
        return decoded_message, extraction_info
    
    def evaluate_bch_robustness(self, audio_features: torch.Tensor,
                               message_bits: torch.Tensor,
                               noise_levels: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate BCH robustness under different conditions.
        
        Args:
            audio_features: Audio feature tensor
            message_bits: Original message bits
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary of robustness metrics
        """
        if not self.enable_bch:
            raise ValueError("BCH protection not enabled")
        
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3]
        
        return self.bch_protector.evaluate_robustness(message_bits, noise_levels)
    
    def optimize_bch_parameters(self, robustness_requirements: Dict[str, float],
                               capacity_constraints: Dict[str, int]) -> Dict[str, any]:
        """
        Optimize BCH parameters for current model configuration.
        
        Args:
            robustness_requirements: Target BER, SNR requirements, etc.
            capacity_constraints: Max codeword length, min message length, etc.
            
        Returns:
            Dictionary of optimal BCH parameters
        """
        return optimize_bch_parameters_for_cnn(
            cnn_model_type="efficientnet",
            robustness_requirements=robustness_requirements,
            capacity_constraints=capacity_constraints
        )
    
    def get_watermark_capacity(self) -> Dict[str, int]:
        """
        Get watermark capacity information.
        
        Returns:
            Dictionary with capacity metrics
        """
        if not self.enable_bch:
            return {"message_length": 0, "codeword_length": 0, "parity_length": 0}
        
        codeword_length = self.bch_protector.get_codeword_length()
        message_length = self.bch_protector.get_message_length()
        
        return {
            "message_length": message_length,
            "codeword_length": codeword_length,
            "parity_length": codeword_length - message_length,
            "code_rate": self.bch_protector.get_code_rate(),
            "redundancy_ratio": (codeword_length - message_length) / message_length
        }
    
    def compute_perceptual_significance(self, audio: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute perceptual significance for adaptive bit allocation (Task 1.2.2.1).
        
        Args:
            audio: Input audio tensor [batch_size, samples]
            
        Returns:
            Dictionary with perceptual significance metrics
        """
        if not self.enable_adaptive_allocation:
            raise ValueError("Adaptive allocation not enabled")
        
        batch_size = audio.shape[0]
        device = audio.device
        
        batch_significance = []
        batch_allocations = []
        
        for i in range(batch_size):
            audio_np = audio[i].detach().cpu().numpy()
            
            # Compute perceptual significance using psychoacoustic model
            significance_result = self.significance_metric.compute_band_significance(audio_np)
            significance = significance_result['band_significance']
            
            # Apply learnable adaptations
            significance_tensor = torch.from_numpy(significance).float().to(device)
            adapted_significance = significance_tensor * self.allocation_adaptation
            adapted_significance = adapted_significance + self.band_weight_bias
            adapted_significance = torch.clamp(adapted_significance, min=0.01)  # Ensure positive
            
            # Temperature scaling for smooth allocation
            scaled_significance = adapted_significance / self.allocation_temperature
            
            batch_significance.append(scaled_significance)
            
            # Compute bit allocation
            allocation_result = self.bit_allocator.allocate_bits(scaled_significance.detach().cpu().numpy())
            allocation = torch.from_numpy(allocation_result['bit_allocation']).float().to(device)
            batch_allocations.append(allocation)
        
        significance_batch = torch.stack(batch_significance)
        allocation_batch = torch.stack(batch_allocations)
        
        return {
            'band_significance': significance_batch,
            'bit_allocation': allocation_batch,
            'total_allocated_bits': torch.sum(allocation_batch, dim=1),
            'allocation_entropy': self._compute_entropy(allocation_batch),
            'significance_weights': self.allocation_adaptation,
            'allocation_bias': self.band_weight_bias,
            'temperature': self.allocation_temperature
        }
    
    def embed_watermark_with_adaptive_allocation(self, 
                                               audio_features: torch.Tensor,
                                               audio: torch.Tensor,
                                               message_bits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed watermark using adaptive bit allocation based on perceptual significance.
        
        Args:
            audio_features: Audio feature tensor for CNN processing
            audio: Original audio signal for psychoacoustic analysis
            message_bits: Original message bits [batch_size, message_length]
            
        Returns:
            Tuple of (watermarked_features, embedding_info)
        """
        if not (self.enable_bch and self.enable_adaptive_allocation):
            raise ValueError("BCH and adaptive allocation must be enabled")
        
        # Compute perceptual significance and bit allocation
        significance_info = self.compute_perceptual_significance(audio)
        bit_allocations = significance_info['bit_allocation']
        
        # Get CNN features
        features = self._extract_features(audio_features)
        
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        # Process each sample in batch
        watermarked_features = []
        embedding_info_batch = []
        
        for i in range(batch_size):
            sample_features = features[i:i+1]
            sample_message = message_bits[i]
            sample_allocation = bit_allocations[i]
            
            # Encode message with BCH
            encoded_message = self.bch_protector.encode_watermark(sample_message.unsqueeze(0))
            
            # Distribute bits across bands according to allocation
            band_embeddings = []
            bit_idx = 0
            
            for band_idx in range(24):  # 24 critical bands
                num_bits = int(sample_allocation[band_idx].item())
                
                if num_bits > 0 and bit_idx < encoded_message.shape[1]:
                    # Extract bits for this band
                    band_bits = encoded_message[0, bit_idx:bit_idx + num_bits]
                    bit_idx += num_bits
                    
                    # Pad to 8 bits if necessary
                    if len(band_bits) < 8:
                        padded_bits = torch.zeros(8, device=device)
                        padded_bits[:len(band_bits)] = band_bits
                        band_bits = padded_bits
                    else:
                        band_bits = band_bits[:8]  # Truncate if too long
                    
                    # Embed using band-specific embedder
                    band_embedding = self.band_embedders[band_idx](sample_features.view(1, -1))
                    
                    # Scale embedding based on bit content
                    bit_scaling = (band_bits * 2.0 - 1.0) * 0.1  # Scale to [-0.1, 0.1]
                    scaled_embedding = band_embedding * bit_scaling.unsqueeze(0)
                    
                    band_embeddings.append(scaled_embedding)
                else:
                    # No bits for this band
                    band_embeddings.append(torch.zeros(1, 8, device=device))
            
            # Combine band embeddings
            combined_embedding = torch.cat(band_embeddings, dim=1)  # [1, 24*8]
            
            # Add to original features
            watermarked_sample = sample_features.view(1, -1) + torch.mean(combined_embedding, dim=1, keepdim=True)
            watermarked_features.append(watermarked_sample)
            
            # Store embedding info
            embedding_info_batch.append({
                'bits_used': bit_idx,
                'band_allocation': sample_allocation,
                'encoding_efficiency': bit_idx / max(1, len(encoded_message[0]))
            })
        
        watermarked_tensor = torch.cat(watermarked_features, dim=0)
        
        # Aggregate embedding info
        total_bits_used = sum(info['bits_used'] for info in embedding_info_batch)
        mean_efficiency = np.mean([info['encoding_efficiency'] for info in embedding_info_batch])
        
        embedding_info = {
            'total_bits_used': torch.tensor(total_bits_used),
            'mean_encoding_efficiency': torch.tensor(mean_efficiency),
            'significance_info': significance_info,
            'per_sample_info': embedding_info_batch
        }
        
        return watermarked_tensor, embedding_info
    
    def extract_watermark_with_adaptive_allocation(self,
                                                  audio_features: torch.Tensor,
                                                  audio: torch.Tensor,
                                                  expected_allocation: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract watermark using adaptive bit allocation.
        
        Args:
            audio_features: Audio feature tensor for CNN processing
            audio: Original audio signal for psychoacoustic analysis (if allocation unknown)
            expected_allocation: Expected bit allocation if known
            
        Returns:
            Tuple of (decoded_message_bits, extraction_info)
        """
        if not (self.enable_bch and self.enable_adaptive_allocation):
            raise ValueError("BCH and adaptive allocation must be enabled")
        
        # Determine bit allocation
        if expected_allocation is None:
            # Compute from audio psychoacoustic analysis
            significance_info = self.compute_perceptual_significance(audio)
            bit_allocations = significance_info['bit_allocation']
        else:
            bit_allocations = expected_allocation
        
        # Get CNN features
        features = self._extract_features(audio_features)
        
        batch_size = audio_features.shape[0]
        device = audio_features.device
        
        decoded_messages = []
        extraction_info_batch = []
        
        for i in range(batch_size):
            sample_features = features[i:i+1]
            sample_allocation = bit_allocations[i]
            
            # Extract bits from each band
            extracted_bits = []
            
            for band_idx in range(24):  # 24 critical bands
                num_bits = int(sample_allocation[band_idx].item())
                
                if num_bits > 0:
                    # Extract using band-specific extractor
                    band_extraction = self.band_extractors[band_idx](sample_features.view(1, -1))
                    
                    # Convert to bits and take only needed amount
                    extracted_band_bits = (band_extraction > 0.5).float()[0, :num_bits]
                    extracted_bits.append(extracted_band_bits)
            
            if extracted_bits:
                # Concatenate all extracted bits
                full_extracted = torch.cat(extracted_bits)
                
                # Decode with BCH
                decoded_message, quality_metrics = self.bch_protector.decode_watermark(
                    full_extracted.unsqueeze(0)
                )
                
                decoded_messages.append(decoded_message[0])
                extraction_info_batch.append({
                    'extracted_bits': len(full_extracted),
                    'quality_metrics': quality_metrics
                })
            else:
                # No bits extracted
                decoded_messages.append(torch.zeros(self.message_length, device=device))
                extraction_info_batch.append({
                    'extracted_bits': 0,
                    'quality_metrics': {'error_rate': torch.tensor(1.0), 'reliability': torch.tensor(0.0)}
                })
        
        decoded_batch = torch.stack(decoded_messages)
        
        # Aggregate extraction info
        total_extracted_bits = sum(info['extracted_bits'] for info in extraction_info_batch)
        mean_error_rate = np.mean([info['quality_metrics']['error_rate'].item() 
                                 for info in extraction_info_batch])
        
        extraction_info = {
            'total_extracted_bits': torch.tensor(total_extracted_bits),
            'mean_error_rate': torch.tensor(mean_error_rate),
            'bit_allocations': bit_allocations,
            'per_sample_info': extraction_info_batch
        }
        
        return decoded_batch, extraction_info
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features for watermark processing."""
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.view(x.size(0), -1)
    
    def _compute_entropy(self, allocations: torch.Tensor) -> torch.Tensor:
        """Compute entropy of bit allocation distribution."""
        # Normalize to probabilities
        probs = allocations / (torch.sum(allocations, dim=1, keepdim=True) + 1e-10)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=1)
        return entropy
    
    def evaluate_adaptive_allocation_performance(self,
                                               audio_features: torch.Tensor,
                                               audio: torch.Tensor,
                                               message_bits: torch.Tensor,
                                               noise_levels: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate performance of adaptive bit allocation under different conditions.
        
        Args:
            audio_features: Audio feature tensor
            audio: Original audio signal
            message_bits: Original message bits
            noise_levels: List of noise levels to test
            
        Returns:
            Performance evaluation metrics
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        performance_results = {}
        
        for noise_level in noise_levels:
            # Add noise to audio features
            noisy_features = audio_features + torch.randn_like(audio_features) * noise_level
            
            # Embed and extract with noise
            watermarked, embed_info = self.embed_watermark_with_adaptive_allocation(
                noisy_features, audio, message_bits
            )
            decoded, extract_info = self.extract_watermark_with_adaptive_allocation(
                watermarked, audio
            )
            
            # Calculate bit accuracy
            bit_accuracy = torch.mean((decoded == message_bits).float())
            
            performance_results[f'noise_{noise_level}'] = {
                'bit_accuracy': bit_accuracy,
                'encoding_efficiency': embed_info['mean_encoding_efficiency'],
                'extraction_error_rate': extract_info['mean_error_rate'],
                'bits_used': embed_info['total_bits_used'],
                'allocation_entropy': torch.mean(embed_info['significance_info']['allocation_entropy'])
            }
        
        return performance_results
    
    def get_adaptive_allocation_summary(self) -> Dict[str, any]:
        """
        Get summary of adaptive allocation configuration.
        
        Returns:
            Configuration summary
        """
        if not self.enable_adaptive_allocation:
            return {"adaptive_allocation": "disabled"}
        
        total_bits = self.bch_protector.get_codeword_length() if self.enable_bch else self.message_length
        
        return {
            "adaptive_allocation": "enabled",
            "total_bits": total_bits,
            "critical_bands": 24,
            "allocation_strategy": self.bit_allocator.allocation_strategy,
            "min_bits_per_band": self.bit_allocator.min_bits_per_band,
            "max_bits_per_band": self.bit_allocator.max_bits_per_band,
            "significance_method": self.significance_metric.significance_method,
            "psychoacoustic_model": "Moore-Glasberg",
            "learnable_parameters": {
                "allocation_adaptation": self.allocation_adaptation.numel(),
                "band_weight_bias": self.band_weight_bias.numel(),
                "allocation_temperature": 1
            }
        }
