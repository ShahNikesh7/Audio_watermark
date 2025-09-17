import numpy as np
import torch
import pytest

from attacks.simulate import ATTACK_FUNCTIONS, simulate_attack
from client.embedding import AudioWatermarkEmbedder
from client.extraction import AudioWatermarkExtractor

# ---------------- Attack Function Sanity Tests ----------------
@pytest.mark.parametrize("attack_type", list(ATTACK_FUNCTIONS.keys()))
def test_attack_output_shape_and_energy(attack_type):
    sr = 16000
    dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), False)
    audio = 0.2 * np.sin(2 * np.pi * 440 * t)
    attacked = simulate_attack(audio, attack_type, severity=0.4, sample_rate=sr)
    assert attacked.shape == audio.shape, f"Attack {attack_type} changed shape"
    orig_energy = np.mean(audio ** 2)
    new_energy = np.mean(attacked ** 2)
    # Energy should remain finite and within a reasonable multiple (allow heavy distortion)
    assert np.isfinite(new_energy)
    assert new_energy < orig_energy * 25, f"Attack {attack_type} energy exploded"

# ---------------- Embed → Attack → (Mock) Extract Pipeline ----------------

def test_pipeline_bit_allocation_stability():
    embedder = AudioWatermarkEmbedder(device='cpu', enable_psychoacoustic=True, enable_adaptive_allocation=True)
    extractor = AudioWatermarkExtractor(device='cpu', enable_psychoacoustic=True)
    sr = 16000
    dur = 1.0
    t = np.linspace(0, dur, int(sr * dur), False)
    base_audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    watermark_data = "TEST123"
    audio_tensor = torch.FloatTensor(base_audio).unsqueeze(0).unsqueeze(0)

    # Embed (simplified path using encoder directly)
    wm_vec = embedder._generate_watermark_vector(watermark_data).unsqueeze(0)
    watermarked = embedder.watermark_encoder(audio_tensor, wm_vec).squeeze().detach().cpu().numpy()

    # Choose subset of attacks for speed
    subset_attacks = [a for a in ["gaussian_noise", "lowpass_filter", "sample_suppression", "echo_addition"] if a in ATTACK_FUNCTIONS]
    alloc_deltas = []
    for attack in subset_attacks:
        attacked = simulate_attack(watermarked, attack, severity=0.5, sample_rate=sr)
        # Re-run significance metric for attacked audio
        metric = embedder.perceptual_significance_metric
        sig = metric.compute_band_significance(attacked)
        alloc = embedder.adaptive_bit_allocator.allocate_bits(sig['band_significance'])['bit_allocation']
        alloc_deltas.append(alloc)
    # Compare variance across allocations
    stacked = np.vstack(alloc_deltas)
    # Ensure allocations are not degenerate (not all identical zeros)
    assert np.any(stacked.sum(axis=1) > 0), "All allocations zero after attacks"
    # Ensure variability but bounded: max total bits shouldn't exceed configured total
    assert np.max(stacked.sum(axis=1)) <= embedder.total_watermark_bits

# ---------------- BCH Reedsolo Path Tightened Test ----------------

def test_bch_reedsolo_path_if_available():
    try:
        import reedsolo  # noqa
    except ImportError:
        pytest.skip("reedsolo not available")
    # Dynamically create a small protector favoring reedsolo path
    from client.phm.technical_rnn.bch_error_correction import create_optimal_bch_protector
    protector = create_optimal_bch_protector(32, 'low')
    if not protector.encoder.use_reedsolo:
        pytest.skip("reedsolo path not active")
    msg = torch.randint(0,2,(1,32)).float()
    encoded = protector.protect_watermark(msg)
    # Introduce fewer errors than capability
    flip = protector.encoder.error_correction_capability - 1
    encoded_np = encoded[0].numpy()
    idx = np.random.choice(len(encoded_np), size=flip, replace=False)
    corrupted = encoded_np.copy()
    corrupted[idx] = 1 - corrupted[idx]
    recovered = protector.recover_watermark(torch.tensor(corrupted).unsqueeze(0))
    rec_bits = recovered['recovered_watermark'][:, :32]
    ber = torch.sum(torch.abs(rec_bits - msg)).item() / 32
    assert ber == 0.0, f"Reedsolo path failed to perfectly correct within capability (BER={ber})"
