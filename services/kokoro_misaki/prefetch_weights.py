#!/usr/bin/env python3
"""Pre-download Kokoro weights into the image layer."""
from huggingface_hub import snapshot_download

snapshot_download("hexgrad/Kokoro-82M")
print("weights cached")
