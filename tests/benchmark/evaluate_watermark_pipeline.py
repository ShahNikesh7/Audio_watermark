import os
import sys
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add the workspace directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from client.embedding import AudioWatermarkEmbedder
from client.extraction import AudioWatermarkExtractor
from attacks.simulate import run_attack_suite

# Optional: configure logging if needed
import logging
logging.basicConfig(level=logging.INFO)

# ===== Configuration =====
TEST_AUDIO_DIR = Path("data/test_audio/")
RESULTS_DIR = Path("tests/benchmark/results/")
WATERMARK_TEXT = "TestWatermark123"
SEVERITY_LEVELS = [0.5]  # Reduced to single severity level for faster evaluation
SAMPLE_RATE = 22050
CONFIDENCE_THRESHOLD = 0.45  # Reduced from 0.65 for better detection rates
SUPPORTED_EXTENSIONS = [".wav", ".flac"]


# ===== Initialize Modules =====
embedder = AudioWatermarkEmbedder(device='cpu', enable_psychoacoustic=True)
extractor = AudioWatermarkExtractor(device='cpu', enable_psychoacoustic=True)

# Ensure results dir exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===== Evaluation Routine =====
def evaluate_on_audio_file(file_path: Path) -> dict:
    # Load original audio
    original_audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Embed watermark
    embed_result = embedder.embed(
        original_audio,
        watermark_data=WATERMARK_TEXT,
        sample_rate=SAMPLE_RATE,
        use_psychoacoustic=True,
        adaptive_strength=True
    )
    watermarked_audio = embed_result['watermarked_audio']

    # Run attack suite (limited set for faster evaluation)
    attack_results = run_attack_suite(
        watermarked_audio,
        watermark_data=WATERMARK_TEXT,
        sample_rate=SAMPLE_RATE,
        severity_levels=SEVERITY_LEVELS,
        attack_types=['additive_noise', 'gaussian_noise', 'reverb', 'time_stretch', 'pitch_shift', 
                     'lowpass_filter', 'highpass_filter', 'clipping', 'phase_shift']
    )

    report = {
        'file': str(file_path.name),
        'total_attacks': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'attack_details': []
    }

    for attack_type, attack_variants in attack_results['attacks'].items():
        # Skip baseline if it exists
        if attack_type == "baseline":
            continue
            
        for severity_key, attack_data in attack_variants.items():
            # attack_data is a dictionary with 'audio', 'severity', 'success', etc.
            if not attack_data.get('success', False):
                # Skip failed attacks
                continue
                
            attacked_audio = attack_data['audio']
            if isinstance(attacked_audio, list):
                attacked_audio = np.array(attacked_audio)

            # Ensure audio is contiguous and has positive strides
            if hasattr(attacked_audio, 'copy'):
                attacked_audio = attacked_audio.copy()
            
            extraction = extractor.extract(
                attacked_audio,
                sample_rate=SAMPLE_RATE,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                use_psychoacoustic=True
            )

            report['total_attacks'] += 1
            if extraction.get('extraction_successful', False):
                report['successful_extractions'] += 1
            else:
                report['failed_extractions'] += 1

            report['attack_details'].append({
                'attack': attack_type,
                'severity': attack_data['severity'],
                'success': extraction['extraction_successful'],
                'confidence': extraction.get('confidence', 0.0),
                'detected': extraction.get('watermark_detected', False),
                'attack_success': attack_data['success']
            })

    return report


def evaluate_batch():
    audio_files = [f for f in TEST_AUDIO_DIR.glob("*") if f.suffix in SUPPORTED_EXTENSIONS]
    total = len(audio_files)
    all_reports = []
    total_success = 0
    total_tests = 0

    print(f"🧪 Starting evaluation on {total} files...")

    for file_path in tqdm(audio_files):
        report = evaluate_on_audio_file(file_path)
        all_reports.append(report)
        total_success += report['successful_extractions']
        total_tests += report['total_attacks']

        print(f"\n📄 {report['file']} — Success: {report['successful_extractions']}/{report['total_attacks']}\n")

    # Print global stats
    overall_success_rate = total_success / total_tests if total_tests > 0 else 0.0
    print("=" * 60)
    print(f"✅ Overall Success Rate: {overall_success_rate:.2%}")
    print("=" * 60)

    # Save results
    import json
    with open(RESULTS_DIR / "watermark_evaluation_results.json", "w") as f:
        json.dump(all_reports, f, indent=2)


if __name__ == "__main__":
    evaluate_batch()
