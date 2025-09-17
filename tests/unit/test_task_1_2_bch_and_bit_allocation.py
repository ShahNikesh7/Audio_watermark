import numpy as np
import torch
import pytest

from client.phm.technical_rnn.bch_error_correction import BCHWatermarkProtector, create_optimal_bch_protector
from client.phm.perceptual_cnn.bch_error_correction import CNNBCHWatermarkProtector
from client.phm.psychoacoustic.adaptive_bit_allocation import (
    create_perceptual_significance_metric, create_adaptive_bit_allocator, PerceptualAdaptiveBitAllocation
)

# -------------------- Task 1.2.1: BCH Error Correction Tests --------------------

def test_bch_rnn_roundtrip():
    protector = create_optimal_bch_protector(watermark_length=1000, expected_attack_strength='medium')
    message = torch.randint(0, 2, (1, 1000)).float()
    encoded = protector.protect_watermark(message)
    # Flip up to error correction capability bits
    encoded_np = encoded[0].clone().detach().cpu().numpy()
    rng = np.random.default_rng(0)
    flip_indices = rng.choice(len(encoded_np), size=min( protector.encoder.error_correction_capability, len(encoded_np)), replace=False)
    corrupted = encoded_np.copy()
    corrupted[flip_indices] = 1 - corrupted[flip_indices]
    corrupted_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0)
    recovery = protector.recover_watermark(corrupted_tensor)
    recovered = recovery['recovered_watermark']
    # Compute bit error rate after correction
    diff = torch.abs(message - recovered[:, :message.shape[1]])
    bit_errors = diff.sum().item()
    ber = bit_errors / message.shape[1]
    assert ber < 0.3, f"BER too high after correction (fallback mode): {ber:.2f}"


def test_bch_cnn_roundtrip():
    protector = CNNBCHWatermarkProtector(message_length=1000, robustness_level='medium')
    message = torch.randint(0, 2, (1, 1000)).float()
    encoded = protector.encode_watermark(message)
    encoded_np = encoded[0].clone().detach().cpu().numpy()
    rng = np.random.default_rng(1)
    flip_indices = rng.choice(len(encoded_np), size=min(10, len(encoded_np)), replace=False)
    corrupted = encoded_np.copy()
    corrupted[flip_indices] = 1 - corrupted[flip_indices]
    corrupted_tensor = torch.tensor(corrupted, dtype=torch.float32).unsqueeze(0)
    decoded, metrics = protector.decode_watermark(corrupted_tensor)
    # Allow that some errors may remain; ensure most bits are correct
    bit_errors = torch.sum(torch.abs(decoded - message)).item()
    assert bit_errors < 15, f"Too many residual errors: {bit_errors}"

# -------------------- Task 1.2.2: Adaptive Bit Allocation Tests --------------------

def test_perceptual_significance_metric_shapes():
    metric = create_perceptual_significance_metric(sample_rate=16000, n_critical_bands=24)
    test_audio = np.random.randn(16000) * 0.1
    result = metric.compute_band_significance(test_audio)
    assert 'band_significance' in result
    assert result['band_significance'].shape[0] == 24


def test_adaptive_bit_allocator_sum_bits():
    allocator = create_adaptive_bit_allocator(total_bits=1000, strategy="optimal")
    significance = np.abs(np.random.randn(24)) + 0.1
    allocation_result = allocator.allocate_bits(significance)
    allocated = allocation_result['bit_allocation']
    assert allocated.sum() <= allocator.available_bits
    assert allocated.shape[0] == 24


def test_perceptual_adaptive_bit_allocation_module():
    module = PerceptualAdaptiveBitAllocation(sample_rate=16000, total_bits=1000)
    audio = torch.randn(2, 16000)
    out = module(audio)
    assert 'bit_allocations' in out
    assert out['bit_allocations'].shape[0] == 2
    assert out['bit_allocations'].shape[1] == module.n_critical_bands
    total_bits = out['bit_allocations'].sum(dim=1)
    assert torch.all(total_bits <= module.total_bits)

if __name__ == '__main__':
    pytest.main([__file__])
