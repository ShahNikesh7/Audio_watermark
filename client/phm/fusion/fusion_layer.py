"""
Multi-head attention fusion layer.
Combines perceptual and technical quality metrics for final quality assessment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionLayer(nn.Module):
    """Multi-head attention fusion layer for combining quality metrics."""
    
    def __init__(self, perceptual_dim, technical_dim, fusion_dim=128, num_heads=4, dropout=0.1):
        super(FusionLayer, self).__init__()
        self.perceptual_dim = perceptual_dim
        self.technical_dim = technical_dim
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        
        # Project inputs to common dimension
        self.perceptual_proj = nn.Linear(perceptual_dim, fusion_dim)
        self.technical_proj = nn.Linear(technical_dim, fusion_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1)
        )
    
    def forward(self, perceptual_features, technical_features):
        """
        Args:
            perceptual_features: (batch_size, perceptual_dim)
            technical_features: (batch_size, technical_dim)
        
        Returns:
            quality_score: (batch_size, 1)
        """
        # Project to common dimension
        perc_proj = self.perceptual_proj(perceptual_features)  # (batch_size, fusion_dim)
        tech_proj = self.technical_proj(technical_features)    # (batch_size, fusion_dim)
        
        # Add sequence dimension for attention
        perc_proj = perc_proj.unsqueeze(1)  # (batch_size, 1, fusion_dim)
        tech_proj = tech_proj.unsqueeze(1)  # (batch_size, 1, fusion_dim)
        
        # Concatenate for cross-attention
        combined = torch.cat([perc_proj, tech_proj], dim=1)  # (batch_size, 2, fusion_dim)
        
        # Self-attention
        attn_out, _ = self.attention(combined, combined, combined)
        attn_out = self.norm1(attn_out + combined)
        
        # Feed-forward
        ff_out = self.ff(attn_out)
        ff_out = self.norm2(ff_out + attn_out)
        
        # Flatten and classify
        ff_out = ff_out.view(ff_out.size(0), -1)  # (batch_size, fusion_dim * 2)
        quality_score = self.classifier(ff_out)
        
        return quality_score
