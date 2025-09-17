"""
Demo script for BCH Error Correction in CNN-based Audio Watermarking

This script demonstrates the implementation of Task 1.2.1:
- BCH codes implementation using reedsolo library
- Optimal BCH parameter determination
- Integration of BCH encoder/decoder into CNN watermark processes

Usage example for the enhanced CNN models with BCH error correction.
"""

import torch
import numpy as np
import logging
from typing import Dict, List

# Import the BCH-enhanced CNN models
from client.phm.perceptual_cnn import (
    MobileNetV3, 
    EfficientNetLite, 
    CNNBCHWatermarkProtector,
    optimize_bch_parameters_for_cnn
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_mobilenetv3_bch():
    """Demonstrate MobileNetV3 with BCH error correction."""
    print("\n" + "="*60)
    print("MobileNetV3 with BCH Error Correction Demo")
    print("="*60)
    
    # Initialize MobileNetV3 with BCH protection
    model = MobileNetV3(
        num_classes=1,
        input_channels=1,
        variant='small',
        enable_bch=True,
        message_length=64,
        robustness_level="medium"
    )
    
    # Create sample data
    batch_size = 4
    audio_features = torch.randn(batch_size, 1, 128, 128)  # Sample mel-spectrogram
    message_bits = torch.randint(0, 2, (batch_size, 64), dtype=torch.float32)
    
    print(f"Input audio features shape: {audio_features.shape}")
    print(f"Original message bits shape: {message_bits.shape}")
    
    # Embed watermark with BCH protection
    watermarked_features, embedding_info = model.embed_watermark_with_bch(
        audio_features, message_bits
    )
    
    print(f"\nEmbedding Results:")
    print(f"Watermarked features shape: {watermarked_features.shape}")
    print(f"BCH code rate: {embedding_info['code_rate']:.3f}")
    print(f"Original message length: {embedding_info['original_message_length']}")
    print(f"Encoded length: {embedding_info['encoded_length']}")
    print(f"Embedding strength: {embedding_info['embedding_strength']:.6f}")
    
    # Extract watermark with BCH decoding
    decoded_message, extraction_info = model.extract_watermark_with_bch(audio_features)
    
    print(f"\nExtraction Results:")
    print(f"Decoded message shape: {decoded_message.shape}")
    print(f"Error rate: {torch.mean(extraction_info['error_rate']):.6f}")
    print(f"Reliability: {torch.mean(extraction_info['reliability']):.3f}")
    print(f"Soft bit confidence: {extraction_info['soft_bit_confidence']:.3f}")
    
    # Calculate bit accuracy
    bit_accuracy = torch.mean((decoded_message == message_bits).float())
    print(f"Bit accuracy: {bit_accuracy:.3f}")
    
    # Demonstrate content-adaptive embedding
    print(f"\nContent-Adaptive Embedding:")
    adaptive_features, adaptation_info = model.content_adaptive_embedding(
        audio_features, message_bits, adaptation_strength=1.5
    )
    print(f"Content complexity: {adaptation_info['content_complexity']:.3f}")
    print(f"Adapted robustness level: {adaptation_info['adapted_robustness']}")
    print(f"Adapted code rate: {adaptation_info['adapted_code_rate']:.3f}")
    
    # Mobile efficiency metrics
    efficiency_metrics = model.get_mobile_efficiency_metrics()
    print(f"\nMobile Efficiency Metrics:")
    print(f"Base parameters: {efficiency_metrics['base_params']:,}")
    print(f"BCH parameters: {efficiency_metrics['bch_params']:,}")
    print(f"BCH overhead: {efficiency_metrics['bch_overhead']:.1%}")
    
    return model, bit_accuracy


def demo_efficientnet_bch():
    """Demonstrate EfficientNetLite with BCH error correction."""
    print("\n" + "="*60)
    print("EfficientNetLite with BCH Error Correction Demo")
    print("="*60)
    
    # Initialize EfficientNetLite with BCH protection
    model = EfficientNetLite(
        num_classes=1,
        input_channels=1,
        enable_bch=True,
        message_length=32,  # Smaller message for efficiency
        robustness_level="high"  # Higher robustness
    )
    
    # Create sample data
    batch_size = 2
    audio_features = torch.randn(batch_size, 1, 64, 64)  # Smaller features for efficiency
    message_bits = torch.randint(0, 2, (batch_size, 32), dtype=torch.float32)
    
    print(f"Input audio features shape: {audio_features.shape}")
    print(f"Original message bits shape: {message_bits.shape}")
    
    # Get watermark capacity info
    capacity_info = model.get_watermark_capacity()
    print(f"\nWatermark Capacity:")
    print(f"Message length: {capacity_info['message_length']}")
    print(f"Codeword length: {capacity_info['codeword_length']}")
    print(f"Parity length: {capacity_info['parity_length']}")
    print(f"Code rate: {capacity_info['code_rate']:.3f}")
    print(f"Redundancy ratio: {capacity_info['redundancy_ratio']:.2f}")
    
    # Embed and extract watermark
    watermarked_features, embedding_info = model.embed_watermark_with_bch(
        audio_features, message_bits
    )
    decoded_message, extraction_info = model.extract_watermark_with_bch(audio_features)
    
    print(f"\nEmbedding & Extraction:")
    print(f"Embedding strength: {embedding_info['embedding_strength']:.6f}")
    print(f"Error rate: {torch.mean(extraction_info['error_rate']):.6f}")
    print(f"Reliability: {torch.mean(extraction_info['reliability']):.3f}")
    
    # Calculate bit accuracy
    bit_accuracy = torch.mean((decoded_message == message_bits).float())
    print(f"Bit accuracy: {bit_accuracy:.3f}")
    
    return model, bit_accuracy


def demo_bch_robustness_evaluation():
    """Demonstrate BCH robustness evaluation under various noise conditions."""
    print("\n" + "="*60)
    print("BCH Robustness Evaluation Demo")
    print("="*60)
    
    # Create a standalone BCH protector
    bch_protector = CNNBCHWatermarkProtector(
        message_length=48,
        robustness_level="medium",
        cnn_model_type="mobilenetv3"
    )
    
    # Generate test message
    test_message = torch.randint(0, 2, (1, 48), dtype=torch.float32)
    
    # Evaluate robustness under different noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    robustness_results = bch_protector.evaluate_robustness(test_message, noise_levels)
    
    print(f"Robustness Evaluation Results:")
    print(f"Message length: {bch_protector.get_message_length()}")
    print(f"Codeword length: {bch_protector.get_codeword_length()}")
    print(f"Code rate: {bch_protector.get_code_rate():.3f}")
    
    print(f"\nNoise Level vs Performance:")
    for i, noise_level in enumerate(noise_levels):
        ber = robustness_results["bit_error_rates"][i].item()
        success_rate = robustness_results["decode_success_rates"][i].item()
        quality_score = robustness_results["quality_scores"][i].item()
        
        print(f"Noise: {noise_level:4.2f} | BER: {ber:6.4f} | "
              f"Success: {success_rate:5.1%} | Quality: {quality_score:5.3f}")
    
    return robustness_results


def demo_parameter_optimization():
    """Demonstrate BCH parameter optimization for different scenarios."""
    print("\n" + "="*60)
    print("BCH Parameter Optimization Demo")
    print("="*60)
    
    # Define different scenarios
    scenarios = [
        {
            "name": "High Quality Audio",
            "robustness_req": {"target_ber": 1e-5, "min_snr_db": 15.0},
            "capacity_constraints": {"max_codeword_length": 256, "min_message_length": 64}
        },
        {
            "name": "Mobile Streaming",
            "robustness_req": {"target_ber": 1e-3, "min_snr_db": 8.0},
            "capacity_constraints": {"max_codeword_length": 128, "min_message_length": 32}
        },
        {
            "name": "Noisy Environment",
            "robustness_req": {"target_ber": 1e-4, "min_snr_db": 5.0},
            "capacity_constraints": {"max_codeword_length": 512, "min_message_length": 64}
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"Target BER: {scenario['robustness_req']['target_ber']}")
        print(f"Min SNR: {scenario['robustness_req']['min_snr_db']} dB")
        
        # Optimize for MobileNetV3
        mobile_params = optimize_bch_parameters_for_cnn(
            "mobilenetv3", 
            scenario['robustness_req'], 
            scenario['capacity_constraints']
        )
        
        # Optimize for EfficientNet
        efficient_params = optimize_bch_parameters_for_cnn(
            "efficientnet", 
            scenario['robustness_req'], 
            scenario['capacity_constraints']
        )
        
        print(f"MobileNetV3 - Rate: {mobile_params['code_rate']:.3f}, "
              f"Msg: {mobile_params['message_length']}, "
              f"Code: {mobile_params['codeword_length']}")
        print(f"EfficientNet - Rate: {efficient_params['code_rate']:.3f}, "
              f"Msg: {efficient_params['message_length']}, "
              f"Code: {efficient_params['codeword_length']}")


def main():
    """Run all BCH-CNN integration demos."""
    print("BCH Error Correction in CNN-based Audio Watermarking")
    print("Task 1.2.1 Implementation Demo")
    print("="*60)
    
    try:
        # Demo 1: MobileNetV3 with BCH
        mobile_model, mobile_accuracy = demo_mobilenetv3_bch()
        
        # Demo 2: EfficientNetLite with BCH  
        efficient_model, efficient_accuracy = demo_efficientnet_bch()
        
        # Demo 3: Robustness evaluation
        robustness_results = demo_bch_robustness_evaluation()
        
        # Demo 4: Parameter optimization
        demo_parameter_optimization()
        
        # Summary
        print("\n" + "="*60)
        print("TASK 1.2.1 IMPLEMENTATION SUMMARY")
        print("="*60)
        print("✓ 1.2.1.1: BCH codes implemented using reedsolo library")
        print("✓ 1.2.1.2: Optimal BCH parameters determined based on robustness requirements")
        print("✓ 1.2.1.3: BCH encoder/decoder integrated into CNN watermark processes")
        print(f"\nPerformance Summary:")
        print(f"MobileNetV3 bit accuracy: {mobile_accuracy:.1%}")
        print(f"EfficientNetLite bit accuracy: {efficient_accuracy:.1%}")
        print(f"BCH protection: ENABLED in both CNN models")
        print(f"Parameter optimization: AVAILABLE for different scenarios")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ BCH-CNN integration demo completed successfully!")
    else:
        print("\n❌ Demo encountered errors.")
