"""
GRU Module for technical quality assessment with BCH error correction integration.
Processes temporal sequences for audio watermarking quality evaluation and robust bit processing.

Enhanced for Task 1.2.1.3: Integration of BCH encoder/decoder into watermark processes.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from .bch_error_correction import BCHWatermarkProtector, create_optimal_bch_protector

logger = logging.getLogger(__name__)


class GRUModule(nn.Module):
    """
    GRU-based module for technical quality assessment with BCH error correction.
    
    Enhanced to include:
    - Robust watermark bit processing using BCH codes
    - Error correction integration for embedding/extraction
    - Adaptive error correction based on channel quality assessment
    """
    
    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=2, 
                 dropout=0.1,
                 enable_bch_protection=True,
                 watermark_length=1000,
                 bch_robustness='medium'):
        """
        Initialize GRU module with optional BCH error correction.
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden state size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            enable_bch_protection: Enable BCH error correction
            watermark_length: Length of watermark bits to protect
            bch_robustness: BCH robustness level ('low', 'medium', 'high')
        """
        super(GRUModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.enable_bch_protection = enable_bch_protection
        self.watermark_length = watermark_length
        
        # Main GRU for sequence processing
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        # Quality assessment classifier
        self.classifier = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        
        # BCH error correction integration (Task 1.2.1.3)
        if enable_bch_protection:
            try:
                self.bch_protector = create_optimal_bch_protector(
                    watermark_length=watermark_length,
                    expected_attack_strength=bch_robustness
                )

                # Additional layers for watermark bit processing
                self.watermark_encoder = nn.Linear(hidden_size * 2, watermark_length)
                self.watermark_decoder = nn.Linear(
                    self.bch_protector.get_protected_length(),
                    hidden_size
                )
            except Exception as e:
                logger.warning(f"BCH initialization failed: {e}. Continuing without BCH protection.")
                self.bch_protector = None
                enable_bch_protection = False
        else:
            self.bch_protector = None

        if self.bch_protector is not None:
            logger.info(f"GRU Module initialized with BCH protection: "
                       f"expansion_factor={self.bch_protector.get_expansion_factor():.2f}")
        else:
            logger.info("GRU Module initialized without BCH protection")
    
    def forward(self, x, mode='quality_assessment'):
        """
        Forward pass with multiple operation modes.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            mode: Operation mode ('quality_assessment', 'watermark_embed', 'watermark_extract')
            
        Returns:
            Output depending on mode
        """
        # Process sequence through GRU
        gru_out, hidden = self.gru(x)
        
        # Use the last output from both directions
        last_output = gru_out[:, -1, :]  # Take last timestep
        
        if mode == 'quality_assessment':
            return self._quality_assessment_forward(last_output)
        elif mode == 'watermark_embed' and self.enable_bch_protection:
            return self._watermark_embed_forward(last_output, gru_out)
        elif mode == 'watermark_extract' and self.enable_bch_protection:
            return self._watermark_extract_forward(last_output, gru_out)
        else:
            # Default quality assessment
            return self._quality_assessment_forward(last_output)
    
    def _quality_assessment_forward(self, last_output):
        """Standard quality assessment forward pass."""
        output = self.dropout(last_output)
        output = self.classifier(output)
        return output
    
    def _watermark_embed_forward(self, last_output, sequence_output):
        """
        Forward pass for watermark embedding with BCH protection.
        
        Args:
            last_output: Final GRU output
            sequence_output: Full sequence output
            
        Returns:
            Dictionary with embedding information
        """
        # Generate watermark bits from GRU features
        raw_watermark = torch.sigmoid(self.watermark_encoder(last_output))
        watermark_bits = (raw_watermark > 0.5).float()
        
        # Apply BCH protection
        protected_watermark = self.bch_protector.protect_watermark(watermark_bits)
        
        # Channel quality assessment for adaptive protection
        quality_score = torch.sigmoid(self.classifier(last_output))
        
        return {
            'original_watermark': watermark_bits,
            'protected_watermark': protected_watermark,
            'quality_score': quality_score,
            'protection_info': self.bch_protector.encoding_info,
            'embedding_features': last_output
        }
    
    def _watermark_extract_forward(self, last_output, sequence_output):
        """
        Forward pass for watermark extraction with BCH decoding.
        
        Args:
            last_output: Final GRU output
            sequence_output: Full sequence output
            
        Returns:
            Dictionary with extraction information
        """
        # Estimate received watermark bits from features
        # This would typically come from the extraction process
        protected_length = self.bch_protector.get_protected_length()
        
        # Generate features for watermark recovery
        recovery_features = self.watermark_decoder(last_output[:, :protected_length])
        
        # Quality-based confidence estimation
        quality_score = torch.sigmoid(self.classifier(last_output))
        
        return {
            'recovery_features': recovery_features,
            'quality_score': quality_score,
            'protected_length': protected_length
        }
    
    def embed_watermark_with_bch(self, audio_features, watermark_data):
        """
        Embed watermark with BCH error correction protection.
        
        Args:
            audio_features: Audio feature tensor
            watermark_data: Watermark bits to embed
            
        Returns:
            Protected watermark for embedding
        """
        if not self.enable_bch_protection:
            raise ValueError("BCH protection not enabled")
        
        # Ensure watermark is proper tensor
        if isinstance(watermark_data, (list, np.ndarray)):
            watermark_data = torch.tensor(watermark_data, dtype=torch.float32)
        
        if watermark_data.dim() == 1:
            watermark_data = watermark_data.unsqueeze(0)
        
        # Apply BCH protection
        protected_watermark = self.bch_protector.protect_watermark(watermark_data)
        
        # Process through GRU for adaptation
        embedding_result = self.forward(audio_features, mode='watermark_embed')
        
        return {
            'protected_bits': protected_watermark,
            'embedding_strength': embedding_result['quality_score'],
            'bch_info': self.bch_protector.encoding_info
        }
    
    def extract_watermark_with_bch(self, received_bits, audio_features=None):
        """
        Extract watermark with BCH error correction.
        
        Args:
            received_bits: Received (possibly corrupted) watermark bits
            audio_features: Optional audio features for quality assessment
            
        Returns:
            Recovered watermark and correction information
        """
        if not self.enable_bch_protection:
            raise ValueError("BCH protection not enabled")
        
        # Apply BCH decoding
        recovery_result = self.bch_protector.recover_watermark(received_bits)
        
        # Optional quality assessment if audio features provided
        if audio_features is not None:
            extraction_result = self.forward(audio_features, mode='watermark_extract')
            recovery_result['quality_assessment'] = extraction_result['quality_score']
        
        return recovery_result
    
    def evaluate_channel_robustness(self, original_watermark, received_bits):
        """
        Evaluate robustness of BCH protection against channel effects.
        
        Args:
            original_watermark: Original watermark bits
            received_bits: Received bits after channel effects
            
        Returns:
            Robustness evaluation metrics
        """
        if not self.enable_bch_protection:
            raise ValueError("BCH protection not enabled")
        
        return self.bch_protector.evaluate_robustness(original_watermark, received_bits)
    
    def get_bch_info(self):
        """Get BCH encoding information."""
        if self.enable_bch_protection:
            return self.bch_protector.encoding_info
        else:
            return None
