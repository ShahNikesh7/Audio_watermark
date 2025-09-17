"""
Perceptual CNN Models Package with BCH Error Correction
Includes MobileNetV3 and EfficientNet implementations for perceptual quality assessment.

Task 1.2.1: BCH Error Correction Integration
- Enhanced CNN models with BCH error correction capabilities
- Robust watermark embedding and extraction with error correction
- Optimized parameters for different robustness requirements
"""

from .mobilenetv3 import MobileNetV3, SqueezeExcitation, InvertedResidual
from .efficientnet_lite import EfficientNetLite
from .bch_error_correction import (
    CNNBCHEncoder, 
    CNNBCHDecoder, 
    CNNBCHWatermarkProtector, 
    optimize_bch_parameters_for_cnn
)

__all__ = [
    'MobileNetV3', 
    'EfficientNetLite',
    'SqueezeExcitation',
    'InvertedResidual',
    'CNNBCHEncoder',
    'CNNBCHDecoder', 
    'CNNBCHWatermarkProtector',
    'optimize_bch_parameters_for_cnn'
]
