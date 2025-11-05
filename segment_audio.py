#!/usr/bin/env python3
"""
segment_audio.py — v6
Changes:
 - Adds support for subcategories under "not_a_drone"
   (wind, cars, motors, airplanes, birds, crowd, electronics, random)
 - Displays short key-based menu for non-drone subtypes
 - Stores "subtype" field in metadata.json
 - Keeps same functionality for drone labeling and quality scoring
"""

import os
import sys
import json
import tempfile
import subprocess
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from pydub import AudioSegment
from pydub.effects import normalize
import random

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
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    elif "youtube.com" in parsed.netloc:
        if "watch" in parsed.path:
            return parse_qs(parsed.query).get("v", ["ytclip"])[0]
        elif "shorts" in parsed.path:
            return parsed.path.split("/")[-1]
    return "ytclip"

def download_youtube_audio(url: str) -> tuple[Path, str]:
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
    if src_path.suffix.lower() != ".wav":
        tmp_wav = Path(tempfile.gettempdir()) / "temp_audio.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(src_path),
            "-ac", "1", "-ar", "16000",
            str(tmp_wav)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_wav
    return src_path

def segment_wav(wav_path: Path, out_dir: Path, prefix: str) -> list:
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
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(file_path)]
        )
    except Exception:
        print("(Audio preview unavailable — ffplay not found)")

def safe_load_metadata():
    if META_FILE.exists() and META_FILE.stat().st_size > 0:
        try:
            with open(META_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("Warning: metadata.json corrupted; starting fresh.")
            return []
    return []

def update_metadata(entries):
    data = safe_load_metadata()
    data.extend(entries)
    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w") as f:
        json.dump(data, f, indent=2)

def delete_existing_entries(url: str):
    """Remove all metadata and audio files for a given YouTube URL."""
    vid_id = extract_youtube_id(url)
    data = safe_load_metadata()
    new_data = [d for d in data if d.get("youtube_url") != url]
    for folder in BASE_OUT_DIR.rglob("*"):
        if folder.is_file() and folder.name.startswith(vid_id):
            try:
                folder.unlink()
            except Exception as e:
                print(f"Could not delete {folder}: {e}")
    with open(META_FILE, "w") as f:
        json.dump(new_data, f, indent=2)
    print(f"🗑️  Deleted old segments and metadata for {url}")

# ---------- Main ----------

def main():
    if len(sys.argv) > 1:
        src = sys.argv[1].strip()
    else:
        src = input("Enter path to a local file or YouTube URL: ").strip()

    source_type = "youtube" if is_url(src) else "local"
    metadata = safe_load_metadata()

    if source_type == "youtube" and any(d.get("youtube_url") == src for d in metadata):
        ans = input("⚠️  This video has already been processed. Delete previous segments and reprocess? (y/n): ").strip().lower()
        if ans == "y":
            delete_existing_entries(src)
        else:
            print("Skipping reprocessing.")
            return

    # --- Prepare WAV ---
    if source_type == "youtube":
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

    # --- Segment audio ---
    seg_paths = segment_wav(wav_path, out_dir, prefix)
    entries = []

    print("\nLabel each segment:")
    for p in seg_paths:
        print(f"\nFile: {p.name}")
        play_audio(p)

        while True:
            drone_present = input("  Drone present? (y/n): ").strip().lower()
            if drone_present in ("y", "n"):
                break
            print("  ⚠️  Please type only 'y' or 'n'.")

        if drone_present == "y":
            binary_label = "drone"
            motor_lbl = motor_label

            while True:
                q = input("  Quality (1–5): ").strip()
                if q in ("1", "2", "3", "4", "5"):
                    quality = int(q)
                    break
                print("  ⚠️  Please type a number 1–5.")

            final_dir = out_dir
            subtype = None

        else:
            binary_label = "no_drone"
            motor_lbl = None
            quality = None

            # --- subtype selection with short keys ---
            options = {
                "a": "airplanes",
                "b": "birds",
                "ca": "cars",
                "cr": "crowd",
                "e": "electronics",
                "m": "motors",
                "r": "random",
                "w": "wind"
            }

            print("\nSelect subtype for non-drone sound:")
            for k, v in options.items():
                print(f"  {k} → {v}")

            prompt = "Enter your choice (e.g., a, b, ca, cr, e, m, r, w): "
            while True:
                choice = input(prompt).strip().lower()
                if choice in options:
                    subtype = options[choice]
                    break
                print("⚠️  Invalid choice. Use one of the initials shown above.")

            final_dir = BASE_OUT_DIR / "not_a_drone" / subtype
            final_dir.mkdir(parents=True, exist_ok=True)

            new_path = final_dir / p.name
            shutil.move(p, new_path)
            p = new_path

        entry = {
            "filename": p.name,
            "binary_label": binary_label,
            "motor_label": motor_lbl,
            "source": source_type,
            "youtube_url": src if source_type == "youtube" else None,
            "duration": SEG_LEN / 1000
        }
        if quality is not None:
            entry["quality"] = quality
        if binary_label == "no_drone":
            entry["subtype"] = subtype

        entries.append(entry)

    update_metadata(entries)
    print(f"\n✅ Created {len(entries)} segments and updated metadata.json")

if __name__ == "__main__":
    main()
