"""
Conformer Lite implementation for technical quality assessment with BCH integration.
Lightweight version of Conformer architecture for mobile deployment and robust watermark processing.

Enhanced for Task 1.2.1.3: Integration of BCH encoder/decoder into watermark processes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple
from .bch_error_correction import BCHWatermarkProtector, create_optimal_bch_protector

logger = logging.getLogger(__name__)


class ConformerLite(nn.Module):
    """
    Lightweight Conformer model for technical quality assessment with BCH error correction.
    
    Enhanced features:
    - Efficient attention mechanism for audio sequence processing
    - Integrated BCH error correction for robust watermark handling
    - Adaptive watermark embedding/extraction based on content analysis
    """
    
    def __init__(self, 
                 input_dim, 
                 num_heads=4, 
                 ff_dim=128, 
                 num_layers=2, 
                 dropout=0.1,
                 enable_bch_protection=True,
                 watermark_length=1000,
                 bch_robustness='medium'):
        """
        Initialize Conformer Lite with BCH error correction capabilities.
        
        Args:
            input_dim: Input feature dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward dimension
            num_layers: Number of Conformer blocks
            dropout: Dropout rate
            enable_bch_protection: Enable BCH error correction
            watermark_length: Length of watermark bits
            bch_robustness: BCH robustness level
        """
        super(ConformerLite, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.enable_bch_protection = enable_bch_protection
        self.watermark_length = watermark_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, ff_dim)
        
        # Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(ff_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_projection = nn.Linear(ff_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # BCH error correction integration (Task 1.2.1.3)
        if enable_bch_protection:
            try:
                self.bch_protector = create_optimal_bch_protector(
                    watermark_length=watermark_length,
                    expected_attack_strength=bch_robustness
                )

                # Watermark processing layers
                self.watermark_generator = nn.Linear(ff_dim, watermark_length)
                self.watermark_attention = nn.MultiheadAttention(
                    ff_dim, num_heads=min(num_heads, 4), dropout=dropout, batch_first=True
                )
                self.watermark_fusion = nn.Linear(ff_dim + self.bch_protector.get_protected_length(), ff_dim)

                logger.info(f"Conformer Lite initialized with BCH protection: "
                           f"expansion_factor={self.bch_protector.get_expansion_factor():.2f}")
            except Exception as e:
                logger.warning(f"BCH initialization failed: {e}. Continuing without BCH protection.")
                self.bch_protector = None
                enable_bch_protection = False
        else:
            self.bch_protector = None

        # Final status logging is handled above in the conditional blocks
    
    def forward(self, x, mode='quality_assessment', watermark_data=None):
        """
        Forward pass with multiple operation modes.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mode: Operation mode ('quality_assessment', 'watermark_embed', 'watermark_extract')
            watermark_data: Optional watermark data for embedding
            
        Returns:
            Output depending on mode
        """
        # Input projection
        x = self.input_projection(x)
        
        # Process through Conformer blocks
        for block in self.conformer_blocks:
            x = block(x)
        
        if mode == 'quality_assessment':
            return self._quality_assessment_forward(x)
        elif mode == 'watermark_embed' and self.enable_bch_protection:
            return self._watermark_embed_forward(x, watermark_data)
        elif mode == 'watermark_extract' and self.enable_bch_protection:
            return self._watermark_extract_forward(x)
        else:
            return self._quality_assessment_forward(x)
    
    def _quality_assessment_forward(self, x):
        """Standard quality assessment forward pass."""
        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.dropout(x)
        x = self.output_projection(x)
        return x
    
    def _watermark_embed_forward(self, x, watermark_data=None):
        """
        Forward pass for watermark embedding with BCH protection.
        
        Args:
            x: Conformer features
            watermark_data: Optional external watermark data
            
        Returns:
            Dictionary with embedding information
        """
        # Generate or use provided watermark
        if watermark_data is None:
            # Generate watermark from content
            pooled_features = torch.mean(x, dim=1)
            raw_watermark = torch.sigmoid(self.watermark_generator(pooled_features))
            watermark_bits = (raw_watermark > 0.5).float()
        else:
            # Use provided watermark
            watermark_bits = watermark_data
        
        # Apply BCH protection
        protected_watermark = self.bch_protector.protect_watermark(watermark_bits)
        
        # Content-aware watermark adaptation using attention
        batch_size, seq_len, feat_dim = x.shape
        watermark_query = protected_watermark.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Pad watermark query to match feature dimension
        if watermark_query.shape[-1] < feat_dim:
            padding = torch.zeros(batch_size, seq_len, feat_dim - watermark_query.shape[-1], 
                                device=x.device)
            watermark_query = torch.cat([watermark_query, padding], dim=-1)
        elif watermark_query.shape[-1] > feat_dim:
            watermark_query = watermark_query[..., :feat_dim]
        
        # Apply attention for adaptive embedding
        adapted_features, attention_weights = self.watermark_attention(
            watermark_query, x, x
        )
        
        # Quality assessment
        quality_score = torch.sigmoid(self.output_projection(torch.mean(x, dim=1)))
        
        return {
            'original_watermark': watermark_bits,
            'protected_watermark': protected_watermark,
            'adapted_features': adapted_features,
            'attention_weights': attention_weights,
            'quality_score': quality_score,
            'protection_info': self.bch_protector.encoding_info
        }
    
    def _watermark_extract_forward(self, x):
        """
        Forward pass for watermark extraction with BCH decoding.
        
        Args:
            x: Conformer features from potentially watermarked audio
            
        Returns:
            Dictionary with extraction information
        """
        # Global feature pooling
        pooled_features = torch.mean(x, dim=1)
        
        # Estimate quality for extraction confidence
        quality_score = torch.sigmoid(self.output_projection(pooled_features))
        
        # Prepare for watermark extraction
        # (The actual received bits would come from the extraction process)
        extraction_features = {
            'content_features': pooled_features,
            'sequence_features': x,
            'quality_score': quality_score,
            'protected_length': self.bch_protector.get_protected_length()
        }
        
        return extraction_features
    
    def embed_watermark_with_bch(self, audio_features, watermark_data):
        """
        Embed watermark with BCH error correction and content adaptation.
        
        Args:
            audio_features: Audio feature tensor
            watermark_data: Watermark bits to embed
            
        Returns:
            Enhanced embedding result with BCH protection
        """
        if not self.enable_bch_protection:
            raise ValueError("BCH protection not enabled")
        
        # Process through Conformer with embedding mode
        embedding_result = self.forward(
            audio_features, 
            mode='watermark_embed', 
            watermark_data=watermark_data
        )
        
        # Add content adaptation analysis
        attention_weights = embedding_result['attention_weights']
        content_complexity = torch.std(attention_weights, dim=-1).mean()
        
        embedding_result.update({
            'content_complexity': content_complexity,
            'recommended_strength': self._calculate_embedding_strength(
                embedding_result['quality_score'],
                content_complexity
            )
        })
        
        return embedding_result
    
    def extract_watermark_with_bch(self, received_bits, audio_features):
        """
        Extract watermark with BCH error correction and quality assessment.
        
        Args:
            received_bits: Received (possibly corrupted) watermark bits
            audio_features: Audio features for quality assessment
            
        Returns:
            Comprehensive extraction result
        """
        if not self.enable_bch_protection:
            raise ValueError("BCH protection not enabled")
        
        # Get extraction features
        extraction_features = self.forward(audio_features, mode='watermark_extract')
        
        # Apply BCH decoding
        recovery_result = self.bch_protector.recover_watermark(received_bits)
        
        # Combine results
        extraction_result = {
            **recovery_result,
            'extraction_quality': extraction_features['quality_score'],
            'content_features': extraction_features['content_features'],
            'extraction_confidence': self._calculate_extraction_confidence(
                recovery_result['success_rate'],
                extraction_features['quality_score']
            )
        }
        
        return extraction_result
    
    def _calculate_embedding_strength(self, quality_score, content_complexity):
        """Calculate optimal embedding strength based on quality and complexity."""
        base_strength = 0.1
        quality_factor = 1.0 - quality_score.mean().item()  # Lower quality = higher strength
        complexity_factor = min(1.0, content_complexity.item())  # Higher complexity = higher strength
        
        return base_strength * (1.0 + quality_factor + complexity_factor)
    
    def _calculate_extraction_confidence(self, bch_success_rate, quality_score):
        """Calculate extraction confidence based on BCH success and quality."""
        return (bch_success_rate + quality_score.mean().item()) / 2.0
    
    def evaluate_embedding_capacity(self, audio_features):
        """
        Evaluate watermark embedding capacity for given audio content.
        
        Args:
            audio_features: Audio content features
            
        Returns:
            Capacity analysis
        """
        if not self.enable_bch_protection:
            return {'capacity': 0, 'bch_enabled': False}
        
        # Analyze content for embedding capacity
        extraction_features = self.forward(audio_features, mode='watermark_extract')
        quality = extraction_features['quality_score'].mean().item()
        
        # Calculate capacity based on quality and BCH expansion
        base_capacity = self.watermark_length
        protected_capacity = self.bch_protector.get_protected_length()
        expansion_factor = self.bch_protector.get_expansion_factor()
        
        return {
            'base_watermark_length': base_capacity,
            'protected_watermark_length': protected_capacity,
            'expansion_factor': expansion_factor,
            'estimated_quality': quality,
            'recommended_capacity': int(base_capacity * quality),
            'bch_info': self.bch_protector.encoding_info
        }


class ConformerBlock(nn.Module):
    """Single Conformer block."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(ConformerBlock, self).__init__()
        self.ff1 = FeedForward(d_model, dropout)
        self.mhsa = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, dropout)
        self.ff2 = FeedForward(d_model, dropout)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Conformer block: FF -> MHSA -> Conv -> FF
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        x = self.norm(x)
        return x


class FeedForward(nn.Module):
    """Feed-forward module."""
    
    def __init__(self, d_model, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.mhsa = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x_norm = self.norm(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm)
        return attn_out


class ConvolutionModule(nn.Module):
    """Convolution module for Conformer."""
    
    def __init__(self, d_model, dropout=0.1):
        super(ConvolutionModule, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv1 = nn.Conv1d(d_model, d_model * 2, kernel_size=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = self.norm(x)
        x = x.transpose(1, 2)  # (batch_size, d_model, seq_len)
        
        x = self.conv1(x)
        x = F.glu(x, dim=1)  # Gate Linear Unit
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)
        
        x = x.transpose(1, 2)  # Back to (batch_size, seq_len, d_model)
        return x
