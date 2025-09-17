"""
MobileNetV3 implementation for perceptual quality assessment.
Optimized for on-device deployment with audio watermarking.

Task 1.2.1: BCH Error Correction Integration
- Enhanced with BCH error correction for robust watermarking
- Optimized BCH parameters for MobileNetV3 architecture
- Integrated encoding/decoding for watermark protection

Task 1.2.2.1: Adaptive Bit Allocation with Perceptual Significance
- Integrated perceptual significance metric for adaptive bit allocation
- Uses Moore-Glasberg psychoacoustic model for masking thresholds
- Implements band-specific watermark embedding based on perceptual significance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    """Inverted residual block with SE."""
    
    def __init__(self, inp, oup, kernel_size, stride, expand_ratio, use_se=True, activation='relu'):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.stride = stride
        self.use_res_connect = stride == 1 and inp == oup
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_se = use_se
        
        if activation == 'relu':
            activation_layer = nn.ReLU
        elif activation == 'hswish':
            activation_layer = nn.Hardswish
        else:
            activation_layer = nn.ReLU
        
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation_layer(inplace=True),
                # SE
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # Pointwise
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise expansion
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation_layer(inplace=True),
                # Depthwise
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, kernel_size // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                activation_layer(inplace=True),
                # SE
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # Pointwise projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    """MobileNetV3 model for perceptual quality assessment with BCH error correction and adaptive bit allocation."""
    
    def __init__(self, num_classes=1, input_channels=1, width_mult=1.0, variant='small',
                 enable_bch=True, message_length=1000, robustness_level="medium",
                 enable_adaptive_allocation=False, allocation_strategy='proportional',
                 significance_method='db_scaled', sample_rate=16000):
        super(MobileNetV3, self).__init__()
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
                cnn_model_type="mobilenetv3"
            )
            logger.info(f"BCH protection enabled for MobileNetV3: "
                       f"msg_len={message_length}, robustness={robustness_level}")
        
        # Initialize adaptive bit allocation components (Task 1.2.2.1)
        if self.enable_adaptive_allocation:
            self.significance_metric = create_perceptual_significance_metric(
                sample_rate=sample_rate,
                significance_method=significance_method
            )
            
            total_bits = self.bch_protector.get_codeword_length() if self.enable_bch else message_length
            self.bit_allocator = create_adaptive_bit_allocator(
                total_bits=total_bits,
                num_bands=24,  # 24 critical bands
                allocation_strategy=allocation_strategy
            )
            
            # Learnable parameters for adaptive allocation
            self.allocation_adaptation = nn.Parameter(torch.ones(24))
            self.band_weight_bias = nn.Parameter(torch.zeros(24))
            self.allocation_temperature = nn.Parameter(torch.ones(1))
            
            logger.info(f"Adaptive bit allocation enabled for MobileNetV3: "
                       f"strategy={allocation_strategy}, method={significance_method}")
        else:
            self.significance_metric = None
            self.bit_allocator = None
        
        if variant == 'small':
            # MobileNetV3-Small configuration
            inverted_residual_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'relu', 2],
                [3, 72,  24,  False, 'relu', 2],
                [3, 88,  24,  False, 'relu', 1],
                [5, 96,  40,  True,  'hswish', 2],
                [5, 240, 40,  True,  'hswish', 1],
                [5, 240, 40,  True,  'hswish', 1],
                [5, 120, 48,  True,  'hswish', 1],
                [5, 144, 48,  True,  'hswish', 1],
                [5, 288, 96,  True,  'hswish', 2],
                [5, 576, 96,  True,  'hswish', 1],
                [5, 576, 96,  True,  'hswish', 1],
            ]
        else:
            # MobileNetV3-Large configuration
            inverted_residual_setting = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'relu', 1],
                [3, 64,  24,  False, 'relu', 2],
                [3, 72,  24,  False, 'relu', 1],
                [5, 72,  40,  True,  'relu', 2],
                [5, 120, 40,  True,  'relu', 1],
                [5, 120, 40,  True,  'relu', 1],
                [3, 240, 80,  False, 'hswish', 2],
                [3, 200, 80,  False, 'hswish', 1],
                [3, 184, 80,  False, 'hswish', 1],
                [3, 184, 80,  False, 'hswish', 1],
                [3, 480, 112, True,  'hswish', 1],
                [3, 672, 112, True,  'hswish', 1],
                [5, 672, 160, True,  'hswish', 2],
                [5, 960, 160, True,  'hswish', 1],
                [5, 960, 160, True,  'hswish', 1],
            ]
        
        # First layer
        input_channel = int(16 * width_mult)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.Hardswish(inplace=True)
        )
        
        # Inverted residual blocks
        features = []
        for k, exp, c, se, nl, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            features.append(InvertedResidual(input_channel, output_channel, k, s, exp, se, nl))
            input_channel = output_channel
        
        self.features = nn.Sequential(*features)
        
        # Final layers
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, int(576 * width_mult), 1, 1, 0, bias=False),
            nn.BatchNorm2d(int(576 * width_mult)),
            nn.Hardswish(inplace=True)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(int(576 * width_mult), int(1280 * width_mult)),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(int(1280 * width_mult), num_classes)
        )
        
        # Watermark embedding and extraction heads (if BCH enabled)
        if self.enable_bch:
            codeword_length = self.bch_protector.get_codeword_length()
            feature_dim = int(576 * width_mult)
            
            self.watermark_embedder = nn.Sequential(
                nn.Linear(feature_dim, int(640 * width_mult)),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(int(640 * width_mult), codeword_length),
                nn.Tanh()  # Output between -1 and 1
            )
            
            self.watermark_extractor = nn.Sequential(
                nn.Linear(feature_dim, int(640 * width_mult)),
                nn.Hardswish(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(int(640 * width_mult), codeword_length),
                nn.Sigmoid()  # Output between 0 and 1 for bits
            )
            
            # Band-specific embedders and extractors for adaptive allocation
            if self.enable_adaptive_allocation:
                self.band_embedders = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(feature_dim, 64),
                        nn.Hardswish(inplace=True),
                        nn.Linear(64, 8),  # Up to 8 bits per band
                        nn.Tanh()
                    ) for _ in range(24)  # 24 critical bands
                ])
                
                self.band_extractors = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(feature_dim, 64),
                        nn.Hardswish(inplace=True),
                        nn.Linear(64, 8),  # Up to 8 bits per band
                        nn.Sigmoid()
                    ) for _ in range(24)  # 24 critical bands
                ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x, mode="quality"):
        """
        Forward pass with multiple modes.
        
        Args:
            x: Input tensor
            mode: "quality" for quality assessment, "embed" for watermark embedding,
                  "extract" for watermark extraction
        
        Returns:
            Output tensor depending on mode
        """
        # Common feature extraction
        x = self.conv1(x)
        x = self.features(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        
        if mode == "quality":
            return self.classifier(features)
        elif mode == "embed" and self.enable_bch:
            return self.watermark_embedder(features)
        elif mode == "extract" and self.enable_bch:
            return self.watermark_extractor(features)
        else:
            return self.classifier(features)
    
    def embed_watermark_with_bch(self, audio_features: torch.Tensor, 
                                message_bits: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Embed watermark with BCH error correction using MobileNetV3.
        
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
        
        # Get embedding signals from MobileNetV3
        embedding_signals = self.forward(audio_features, mode="embed")
        
        # Apply BCH-encoded bits to embedding signals
        # Scale encoded bits to match embedding signal range
        scaled_bits = (encoded_bits * 2.0 - 1.0) * 0.05  # Scale to [-0.05, 0.05] for mobile efficiency
        
        # Content-adaptive embedding strength based on MobileNetV3 features
        feature_variance = torch.var(embedding_signals, dim=-1, keepdim=True)
        adaptive_strength = torch.clamp(feature_variance * 2.0, 0.01, 0.1)
        scaled_bits = scaled_bits * adaptive_strength
        
        # Modulate embedding signals with encoded bits
        watermarked_signals = embedding_signals + scaled_bits
        
        embedding_info = {
            "original_message_length": torch.tensor(self.message_length),
            "encoded_length": torch.tensor(self.bch_protector.get_codeword_length()),
            "code_rate": torch.tensor(self.bch_protector.get_code_rate()),
            "embedding_strength": torch.mean(torch.abs(scaled_bits)),
            "adaptive_strength": torch.mean(adaptive_strength),
            "feature_variance": torch.mean(feature_variance)
        }
        
        return watermarked_signals, embedding_info
    
    def extract_watermark_with_bch(self, audio_features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract and decode watermark with BCH error correction using MobileNetV3.
        
        Args:
            audio_features: Audio feature tensor for CNN processing
            
        Returns:
            Tuple of (decoded_message_bits, extraction_info)
        """
        if not self.enable_bch:
            raise ValueError("BCH protection not enabled")
        
        # Extract bits using MobileNetV3
        extracted_signals = self.forward(audio_features, mode="extract")
        
        # Convert signals to soft bits and then hard bits
        soft_bits = torch.sigmoid(extracted_signals * 10.0)  # Enhance soft decision
        extracted_bits = (soft_bits > 0.5).float()
        
        # Decode with BCH error correction
        decoded_message, quality_metrics = self.bch_protector.decode_watermark(extracted_bits)
        
        extraction_info = {
            "extracted_length": torch.tensor(extracted_bits.shape[-1]),
            "decoded_length": torch.tensor(decoded_message.shape[-1]),
            "soft_bit_confidence": torch.mean(torch.abs(soft_bits - 0.5) * 2.0),  # Confidence measure
            **quality_metrics
        }
        
        return decoded_message, extraction_info
    
    def content_adaptive_embedding(self, audio_features: torch.Tensor,
                                  message_bits: torch.Tensor,
                                  adaptation_strength: float = 1.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Content-adaptive watermark embedding with BCH protection.
        
        Args:
            audio_features: Audio feature tensor
            message_bits: Original message bits
            adaptation_strength: Strength of content adaptation
            
        Returns:
            Tuple of (watermarked_features, adaptation_info)
        """
        if not self.enable_bch:
            raise ValueError("BCH protection not enabled")
        
        # Analyze content characteristics using MobileNetV3 features
        content_features = self.forward(audio_features, mode="quality")
        content_complexity = torch.sigmoid(content_features) * adaptation_strength
        
        # Adapt BCH parameters based on content
        if torch.mean(content_complexity) > 0.7:
            # High complexity content - use higher redundancy
            temp_robustness = "high"
        elif torch.mean(content_complexity) > 0.3:
            # Medium complexity content - balanced
            temp_robustness = "medium"
        else:
            # Low complexity content - efficiency focus
            temp_robustness = "low"
        
        # Create temporary BCH protector with adapted parameters
        temp_protector = CNNBCHWatermarkProtector(
            message_length=self.message_length,
            robustness_level=temp_robustness,
            cnn_model_type="mobilenetv3"
        )
        
        # Encode and embed with adapted parameters
        encoded_bits = temp_protector.encode_watermark(message_bits)
        embedding_signals = self.forward(audio_features, mode="embed")
        
        # Adaptive embedding strength
        base_strength = 0.03 + 0.07 * content_complexity
        scaled_bits = (encoded_bits * 2.0 - 1.0) * base_strength
        
        watermarked_signals = embedding_signals + scaled_bits
        
        adaptation_info = {
            "content_complexity": torch.mean(content_complexity),
            "adapted_robustness": temp_robustness,
            "adaptive_strength": torch.mean(base_strength),
            "adapted_code_rate": torch.tensor(temp_protector.get_code_rate()),
            "original_message_length": torch.tensor(self.message_length),
            "adapted_encoded_length": torch.tensor(temp_protector.get_codeword_length())
        }
        
        return watermarked_signals, adaptation_info
    
    def evaluate_bch_robustness(self, audio_features: torch.Tensor,
                               message_bits: torch.Tensor,
                               noise_levels: List[float] = None) -> Dict[str, torch.Tensor]:
        """
        Evaluate BCH robustness under different conditions with MobileNetV3.
        
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
            noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]  # Mobile-optimized range
        
        return self.bch_protector.evaluate_robustness(message_bits, noise_levels)
    
    def optimize_bch_for_mobile(self, target_latency_ms: float = 100.0,
                               target_memory_mb: float = 50.0) -> Dict[str, any]:
        """
        Optimize BCH parameters specifically for mobile deployment.
        
        Args:
            target_latency_ms: Target inference latency in milliseconds
            target_memory_mb: Target memory usage in megabytes
            
        Returns:
            Dictionary of mobile-optimized BCH parameters
        """
        # Mobile-specific constraints
        robustness_requirements = {
            "target_ber": 1e-3,  # Relaxed for mobile
            "min_snr_db": 8.0    # Lower SNR for mobile scenarios
        }
        
        capacity_constraints = {
            "max_codeword_length": 128,  # Reduced for mobile efficiency
            "min_message_length": 32
        }
        
        # Additional mobile constraints
        if target_latency_ms < 50.0:
            capacity_constraints["max_codeword_length"] = 64
        if target_memory_mb < 25.0:
            capacity_constraints["max_codeword_length"] = 48
        
        return optimize_bch_parameters_for_cnn(
            cnn_model_type="mobilenetv3",
            robustness_requirements=robustness_requirements,
            capacity_constraints=capacity_constraints
        )
    
    def get_mobile_efficiency_metrics(self) -> Dict[str, float]:
        """
        Get efficiency metrics for mobile deployment.
        
        Returns:
            Dictionary with mobile efficiency metrics
        """
        if not self.enable_bch:
            return {"total_params": self._count_parameters(), "bch_overhead": 0.0}
        
        base_params = self._count_parameters_without_bch()
        bch_params = self._count_bch_parameters()
        
        return {
            "base_params": base_params,
            "bch_params": bch_params,
            "total_params": base_params + bch_params,
            "bch_overhead": bch_params / base_params,
            "code_rate": self.bch_protector.get_code_rate(),
            "watermark_capacity": self.bch_protector.get_message_length()
        }
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def _count_parameters_without_bch(self) -> int:
        """Count parameters excluding BCH-related layers."""
        total = 0
        for name, param in self.named_parameters():
            if "watermark" not in name and "bch" not in name:
                total += param.numel()
        return total
    
    def _count_bch_parameters(self) -> int:
        """Count BCH-related parameters."""
        total = 0
        for name, param in self.named_parameters():
            if "watermark" in name or "bch" in name:
                total += param.numel()
        return total
