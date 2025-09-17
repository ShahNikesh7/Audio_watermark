# SoundSafeAI Documentation

Welcome to the SoundSafeAI documentation!

## Overview

SoundSafeAI is an advanced audio watermarking system that uses the Parallel Hybrid Model (PHM) for robust watermark embedding and extraction.

## Quick Start

### Installation

```bash
pip install soundsafeai
```

### Basic Usage

```python
from client.embedding import PHMEmbedder
from client.extraction import PHMExtractor
import numpy as np

# Initialize embedder and extractor
embedder = PHMEmbedder()
extractor = PHMExtractor()

# Create sample audio data
sample_rate = 16000
audio_data = np.random.randn(sample_rate * 2).astype(np.float32)

# Embed watermark
result = embedder.embed(audio_data, sample_rate)
watermarked_audio = result['watermarked_audio']

# Extract watermark
extraction_result = extractor.extract(watermarked_audio, sample_rate)
print(f"Watermark detected: {extraction_result['watermark_detected']}")
print(f"Confidence: {extraction_result['confidence']}")
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Contents:

api
architecture
installation
usage
examples
```

## Indices and tables

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
