#!/usr/bin/env python3
"""
check_dataset_files.py

Compares metadata.json entries with actual audio files on disk.
Reports:
 - Files listed in metadata but missing on disk
 - Files present on disk but missing in metadata
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "datasets" / "Drone_Audio_Dataset"
AUDIO_DIR = DATASET_DIR / "audio"
META_FILE = DATASET_DIR / "metadata.json"

# Load metadata
with open(META_FILE) as f:
    metadata = json.load(f)

metadata_files = set(d["filename"] for d in metadata)

# Gather all audio files in the dataset
all_audio_files = set()
for folder in AUDIO_DIR.rglob("*"):
    if folder.is_file() and folder.suffix.lower() == ".wav":
        all_audio_files.add(folder.name)

# Compare
missing_on_disk = metadata_files - all_audio_files
extra_on_disk = all_audio_files - metadata_files

print("=== Dataset consistency report ===\n")

if missing_on_disk:
    print("⚠️  Files listed in metadata but missing on disk:")
    for f in sorted(missing_on_disk):
        print("  -", f)
else:
    print("✅ All metadata files exist on disk.")

if extra_on_disk:
    print("\n⚠️  Files present on disk but missing in metadata:")
    for f in sorted(extra_on_disk):
        print("  -", f)
else:
    print("\n✅ No extra files found on disk.")

print("\nScan complete.")
