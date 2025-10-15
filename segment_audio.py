#!/usr/bin/env python3
"""
segment_audio.py
Segments audio into 2-second normalized clips and updates metadata.json.
Supports local files or YouTube URLs.
Includes audio preview, validated labeling, and manual quality scoring (1–5).
"""

import os
import sys
import json
import tempfile
import subprocess
import shutil
import random
import numpy as np
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from pydub import AudioSegment
from pydub.effects import normalize

# ---------- CONFIG ----------
SEG_LEN = 2000       # ms
STEP = 1500          # ms overlap 0.5 s
BASE_OUT_DIR = Path("datasets/Drone_Audio_Dataset/audio")
META_FILE = Path("datasets/Drone_Audio_Dataset/metadata.json")
random.seed(42)
# ----------------------------

BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Detect yt-dlp in PATH
YTDLP = shutil.which("yt-dlp")
if not YTDLP:
    print("Warning: yt-dlp not found. You can only process local files.")


# ---------- Helper Functions ----------

def is_url(path: str):
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except:
        return False


def extract_youtube_id(url: str) -> str:
    """Extract the YouTube video ID from any standard or Shorts URL."""
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        # Shortened youtu.be/VIDEO_ID
        return parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        if "watch" in parsed.path:
            # Normal link
            return parse_qs(parsed.query).get("v", ["ytclip"])[0]
        elif "shorts" in parsed.path:
            # Shorts link
            return parsed.path.split("/")[-1]
    return "ytclip"  # Fallback


def download_youtube_audio(url: str) -> tuple[Path, str]:
    """Download YouTube audio (standard or Shorts) and convert to WAV (16kHz mono)."""
    vid_id = extract_youtube_id(url)
    tmp_wav = Path(tempfile.gettempdir()) / f"yt_audio_{vid_id}.wav"
    print(f"Downloading audio from YouTube: {url}")
    subprocess.run([
        YTDLP, "-x", "--audio-format", "wav",
        "-o", str(tmp_wav),
        url
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return tmp_wav, vid_id



def convert_to_wav(src_path: Path) -> Path:
    """Ensure file is WAV mono 16kHz."""
    if src_path.suffix.lower() != ".wav":
        tmp_wav = Path(tempfile.gettempdir()) / f"temp_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src_path),
            "-ac", "1", "-ar", "16000",
            str(tmp_wav)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_wav
    return src_path


def segment_wav(wav_path: Path, out_dir: Path, prefix: str) -> list:
    """Segment, normalize, export segments."""
    audio = AudioSegment.from_wav(wav_path)
    segments = []
    for i, start in enumerate(range(0, len(audio) - SEG_LEN, STEP)):
        seg = audio[start:start + SEG_LEN]
        seg = normalize(seg)
        fname = f"{prefix}_{i:03d}.wav"
        fpath = out_dir / fname
        seg.export(fpath, format="wav")
        segments.append(fpath)
    return segments


def play_audio(file_path: Path):
    """Play audio segment in terminal using ffplay."""
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(file_path)]
        )
    except Exception:
        print("(Audio preview unavailable — ffplay not found)")


def safe_load_metadata():
    """Safely load metadata JSON (create if missing or invalid)."""
    if META_FILE.exists() and META_FILE.stat().st_size > 0:
        try:
            with open(META_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: metadata.json corrupted; starting fresh.")
            return []
    return []


def update_metadata(entries):
    """Append new entries safely."""
    data = safe_load_metadata()
    data.extend(entries)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------- Main ----------

def main():
    # Input source (local file or YouTube)
    if len(sys.argv) > 1:
        src = sys.argv[1].strip()
    else:
        src = input("Enter path to a local file or YouTube URL: ").strip()

    source_type = "youtube" if is_url(src) else "local"
    metadata = safe_load_metadata()

    # Skip duplicate YouTube URLs
    if source_type == "youtube":
        if any(d.get("youtube_url") == src for d in metadata):
            print("⚠️  This video has already been processed. Skipping.")
            return

        wav_path, vid_id = download_youtube_audio(src)
        motor_count = input("Enter motor count (2/4/6/8 or unknown): ").strip()
        motor_label = f"{motor_count}_motors" if motor_count.isdigit() else "unknown"
        out_dir = BASE_OUT_DIR / motor_label
        prefix = f"{vid_id}_{motor_label}"

    else:
        src_path = Path(src)
        if not src_path.exists():
            print("File not found.")
            return
        wav_path = convert_to_wav(src_path)
        motor_label = input("Enter motor count (2/4/6/8 or unknown): ").strip()
        out_dir = BASE_OUT_DIR / (f"{motor_label}_motors" if motor_label.isdigit() else "no_label")
        prefix = f"{src_path.stem}_{motor_label}"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Segment ----------
    seg_paths = segment_wav(wav_path, out_dir, prefix)

    # ---------- Label manually ----------
    entries = []
    print("\nLabel each segment:")
    for p in seg_paths:
        print(f"\nFile: {p.name}")
        play_audio(p)  # 🎧 Audio preview before labeling

        # ✅ Binary label (drone / no_drone)
        while True:
            drone_present = input("  Drone present? (y/n): ").strip().lower()
            if drone_present in ("y", "n"):
                break
            print("  ⚠️  Please type only 'y' or 'n'.")

        binary_label = "drone" if drone_present == "y" else "no_drone"
        motor_lbl = motor_label if drone_present == "y" else None

        # ✅ Quality label (1–5)
        while True:
            q = input("  Quality (1–5): ").strip()
            if q in ("1", "2", "3", "4", "5"):
                quality = int(q)
                break
            print("  ⚠️  Please type a number 1–5.")

        entries.append({
            "filename": p.name,
            "binary_label": binary_label,
            "motor_label": motor_lbl,
            "quality": quality,
            "source": source_type,
            "youtube_url": src if source_type == "youtube" else None,
            "duration": SEG_LEN / 1000
        })

    update_metadata(entries)
    print(f"\n✅ Created {len(entries)} segments and updated metadata.json")

if __name__ == "__main__":
    main()
