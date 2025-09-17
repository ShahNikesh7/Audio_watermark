"""
Psychoacoustic Package Initialization

Task 1.2.2: Design Adaptive Bit Allocation
- Perceptual significance metric for frequency bands
- Adaptive bit allocation based on masking thresholds
- Neural network integration for end-to-end optimization
"""

from .moore_glasberg import (
    MooreGlasbergAnalyzer,
    PerceptualAnalyzer,
    integrate_masking_with_watermark
)

from .band_thresholds import (
    BandThresholdCalculator,
    calculate_per_band_threshold
)

from .adaptive_bit_allocation import (
    PerceptualSignificanceMetric,
    AdaptiveBitAllocator,
    PerceptualAdaptiveBitAllocation,
    create_perceptual_significance_metric,
    create_adaptive_bit_allocator
)

from .perceptual_losses import (
    MFCCPerceptualLoss,
    CombinedPerceptualLoss,
    EnhancedCombinedPerceptualLoss,
    create_perceptual_loss
)

__all__ = [
    # Core psychoacoustic analysis
    'MooreGlasbergAnalyzer',
    'PerceptualAnalyzer', 
    'integrate_masking_with_watermark',
    
    # Band threshold calculation
    'BandThresholdCalculator',
    'calculate_per_band_threshold',
    
    # Adaptive bit allocation (Task 1.2.2.1)
    'PerceptualSignificanceMetric',
    'AdaptiveBitAllocator',
    'PerceptualAdaptiveBitAllocation',
    'create_perceptual_significance_metric',
    'create_adaptive_bit_allocator',
    
    # Perceptual losses
    'MFCCPerceptualLoss',
    'CombinedPerceptualLoss',
    'EnhancedCombinedPerceptualLoss',
    'create_perceptual_loss'
]
