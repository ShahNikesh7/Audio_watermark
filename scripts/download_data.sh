#!/bin/bash

# Download dataset script for SoundSafeAI

set -e

echo "Starting dataset download..."

# Create data directories
mkdir -p data/raw
mkdir -p data/processed

# Download sample datasets (placeholder URLs)
echo "Downloading sample audio datasets..."

# LibriSpeech sample
if [ ! -f "data/raw/librispeech_sample.tar.gz" ]; then
    echo "Downloading LibriSpeech sample..."
    # wget -O data/raw/librispeech_sample.tar.gz "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    echo "LibriSpeech sample download would go here"
fi

# Music datasets
if [ ! -f "data/raw/music_sample.tar.gz" ]; then
    echo "Downloading music sample dataset..."
    # wget -O data/raw/music_sample.tar.gz "https://example.com/music_dataset.tar.gz"
    echo "Music sample download would go here"
fi

# Environmental sounds
if [ ! -f "data/raw/environmental_sounds.tar.gz" ]; then
    echo "Downloading environmental sounds..."
    # wget -O data/raw/environmental_sounds.tar.gz "https://example.com/environmental_sounds.tar.gz"
    echo "Environmental sounds download would go here"
fi

# Create dummy datasets for development
echo "Creating dummy datasets for development..."

# Create dummy audio files
python3 -c "
import numpy as np
import os
from scipy.io import wavfile

# Ensure data directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# Create dummy audio files
sample_rate = 44100
duration = 5  # 5 seconds

for i in range(10):
    # Generate dummy audio (sine wave with noise)
    t = np.linspace(0, duration, duration * sample_rate)
    frequency = 440 + i * 110  # A4 + harmonics
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    noise = np.random.normal(0, 0.05, len(audio))
    audio = audio + noise
    
    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save as WAV file
    filename = f'data/raw/dummy_audio_{i:03d}.wav'
    wavfile.write(filename, sample_rate, audio_int16)
    print(f'Created {filename}')

print('Dummy dataset creation complete!')
"

echo "Dataset download and setup complete!"
echo "Raw data location: data/raw/"
echo "Processed data location: data/processed/"

# Extract archives if they exist
echo "Extracting downloaded archives..."
for archive in data/raw/*.tar.gz; do
    if [ -f "$archive" ]; then
        echo "Extracting $archive..."
        tar -xzf "$archive" -C data/raw/
    fi
done

echo "Dataset preparation complete!"
echo "Next steps:"
echo "1. Run preprocessing: python scripts/preprocess.py"
echo "2. Start training: bash scripts/launch_training.sh"
