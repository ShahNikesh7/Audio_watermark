"""
Run inference with the AudioWatermarkEmbedder.

Usage example:
python Audio_watermark/scripts/run_embed_inference.py \
  --model path/to/checkpoint.pth \
  --input path/to/input.wav \
  --output path/to/output_watermarked.wav \
  --watermark-data "hello world" \
  --strength 0.1 \
  --device cuda
"""

import argparse
import sys
from pathlib import Path
import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt

# Add project root to path so we can import client modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from client.embedding import AudioWatermarkEmbedder  # noqa: E402


def plot_and_save_spectrograms(original_audio: np.ndarray,
                               watermarked_audio: np.ndarray,
                               sample_rate: int,
                               out_path: Path) -> Path:
    """
    Create a before/after spectrogram figure and save it next to the output audio.
    Returns the saved image path.
    """
    n_fft = 2048
    hop = 256

    # Compute magnitude spectrograms in dB
    orig_spec = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio, n_fft=n_fft, hop_length=hop)) + 1e-8,
                                        ref=np.max)
    wm_spec = librosa.amplitude_to_db(np.abs(librosa.stft(watermarked_audio, n_fft=n_fft, hop_length=hop)) + 1e-8,
                                      ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    im0 = axes[0].imshow(orig_spec, aspect='auto', origin='lower', cmap='magma')
    axes[0].set_title('Original Audio - Spectrogram (dB)')
    axes[0].set_ylabel('Frequency bins')
    fig.colorbar(im0, ax=axes[0], format='%+2.0f dB')

    im1 = axes[1].imshow(wm_spec, aspect='auto', origin='lower', cmap='magma')
    axes[1].set_title('Watermarked Audio - Spectrogram (dB)')
    axes[1].set_xlabel('Time frames')
    axes[1].set_ylabel('Frequency bins')
    fig.colorbar(im1, ax=axes[1], format='%+2.0f dB')

    fig.tight_layout()

    img_path = out_path.with_suffix('.spectrogram.png')
    fig.savefig(img_path)
    plt.close(fig)
    return img_path


def main():
    parser = argparse.ArgumentParser(description='Run watermark embedding inference and visualize spectrograms')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to .pth/.pt checkpoint containing watermark_encoder')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input audio file (.wav, .mp3, etc.)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save watermarked audio (.wav). Defaults to <input>_watermarked.wav')
    parser.add_argument('--watermark-data', type=str, default='soundsafe',
                        help='Watermark payload string')
    parser.add_argument('--strength', type=float, default=0.1,
                        help='Embedding strength (0.0 - 1.0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda or cpu)')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Optional target sample rate; defaults to file SR')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input audio not found: {input_path}")
        return 1

    # Derive output path if not provided
    if args.output is None:
        out_path = input_path.with_name(f"{input_path.stem}_watermarked.wav")
    else:
        out_path = Path(args.output)
        os.makedirs(out_path.parent, exist_ok=True)

    # Load audio (preserve original SR unless user overrides)
    audio, sr = librosa.load(str(input_path), sr=args.sample_rate, mono=True)

    # Initialize embedder and load model
    embedder = AudioWatermarkEmbedder(model_path=args.model, device=args.device, use_phm=False,
                                      sample_rate=sr, enable_psychoacoustic=True,
                                      enable_mfcc_loss=False, training_mode=False)

    # Run embedding
    result = embedder.embed(audio, watermark_data=args.watermark_data, sample_rate=sr,
                            strength=args.strength, adaptive_strength=True, use_psychoacoustic=False)
    watermarked_audio = result['watermarked_audio']

    # Save watermarked audio
    sf.write(str(out_path), watermarked_audio, sr)

    # Plot and save spectrograms
    img_path = plot_and_save_spectrograms(audio, watermarked_audio, sr, out_path)

    print("Inference completed successfully.")
    print(f"Input:  {input_path}")
    print(f"Model:  {args.model}")
    print(f"Output: {out_path}")
    print(f"Spectrograms saved to: {img_path}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


