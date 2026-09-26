#!/usr/bin/env python3
"""Pre-download stock Kokoro-82M weights into the image layer."""
from huggingface_hub import snapshot_download

snapshot_download("hexgrad/Kokoro-82M")
print("weights cached")
