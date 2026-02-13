#!/usr/bin/env python3
"""
segment_audio.py — v8
Changes from v7:
 - Added global subtype option for no_drone segments
 - Drone quality: 1-5; 0 means no_drone and asks for subtype
 - Segment playback before quality/subtype input
 - Preserves CLI parameters and metadata.json format
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

def segment_wav(wav_path: Path, out_dir: Path, prefix: str, start_ms: int = 0, end_ms: int = None, start_index: int = 0) -> list:
    audio = AudioSegment.from_wav(wav_path)
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)
    segments = []
    for i, start in enumerate(range(start_ms, end_ms - SEG_LEN, STEP), start_index):
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

def parse_time_arg(arg: str) -> int:
    """Convert minutes.seconds string to milliseconds"""
    try:
        if '.' in arg:
            mins, secs = arg.split('.')
            total_sec = int(mins)*60 + int(secs)
        else:
            total_sec = int(arg)*60
        return total_sec * 1000
    except:
        return None

def is_overlap(new_start, new_end, processed_parts):
    """Check if new part overlaps any previous part"""
    for s,e,_ in processed_parts:
        if not (new_end <= s or new_start >= e):
            return True
    return False

# ---------- Main ----------

def main():
    args = sys.argv[1:]
    if len(args) == 0:
        src = input("Enter path to a local file or YouTube URL: ").strip()
        start_ms = None
        end_ms = None
    else:
        src = args[0].strip()
        if len(args) >= 3:
            start_ms = parse_time_arg(args[1])
            end_ms = parse_time_arg(args[2])
            if start_ms is None or end_ms is None:
                print("⚠️ Invalid time format. Use minutes.seconds (e.g., 1.20 for 1min20s).")
                return
            if start_ms >= end_ms:
                print("⚠️ Start time must be smaller than end time.")
                return
        else:
            start_ms = None
            end_ms = None

    source_type = "youtube" if is_url(src) else "local"
    metadata = safe_load_metadata()
    vid_id = extract_youtube_id(src)

    # --- ORIGINAL URL check ---
    if source_type == "youtube" and any(d.get("youtube_url") == src for d in metadata):
        ans = input("⚠️  This video has already been processed. Delete previous segments and reprocess? (y/n): ").strip().lower()
        if ans == "y":
            delete_existing_entries(src)
        else:
            print("Skipping reprocessing.")
            return

    processed_parts = []

    # --- Prepare WAV and determine motor_label ---
    if source_type == "youtube":
        wav_path, vid_id = download_youtube_audio(src)
        motor_count = input("Enter motor count (2/4/6/8 or leave empty for no_drone segments): ").strip()
        if motor_count.isdigit():
            motor_label = f"{motor_count}_motors"
            out_dir = BASE_OUT_DIR / motor_label
            is_no_drone_global = False
        else:
            motor_label = None
            out_dir = BASE_OUT_DIR / "not_a_drone"
            is_no_drone_global = True
        prefix_base = f"{vid_id}_{motor_label}" if motor_label else f"{vid_id}_no_label"
    else:
        src_path = Path(src)
        if not src_path.exists():
            print("File not found.")
            return
        wav_path = convert_to_wav(src_path)
        motor_count = input("Enter motor count (2/4/6/8 or leave empty for no_drone segments): ").strip()
        if motor_count.isdigit():
            motor_label = f"{motor_count}_motors"
            out_dir = BASE_OUT_DIR / motor_label
            is_no_drone_global = False
        else:
            motor_label = None
            out_dir = BASE_OUT_DIR / "not_a_drone"
            is_no_drone_global = True
        prefix_base = f"{src_path.stem}_{motor_label}" if motor_label else f"{src_path.stem}_no_label"

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Global subtype for all no_drone segments ---
    global_subtype = None
    if is_no_drone_global:
        ans = input("Do you want to set a global subtype for all no_drone segments? (y/n): ").strip().lower()
        if ans == "y":
            options = {"a":"airplanes","b":"birds","ca":"cars","cr":"crowd","e":"electronics","m":"motors","r":"random","w":"wind"}
            print("\nSelect global subtype for no_drone segments:")
            for k,v in options.items():
                print(f"  {k} → {v}")
            while True:
                choice = input("Enter your choice: ").strip().lower()
                if choice in options:
                    global_subtype = options[choice]
                    break
                print("⚠️ Invalid choice. Use one of the initials shown above.")

    # --- Loop for multiple parts ---
    while True:
        if start_ms is None and end_ms is None:
            part_start = 0
            part_end = None
            process_whole_video = True
        else:
            part_start = start_ms
            part_end = end_ms
            process_whole_video = False

        # --- Check for overlap ---
        if not process_whole_video and is_overlap(part_start, part_end, processed_parts):
            print("⚠️ This part overlaps a previously processed part. Please enter start/end again.")
            while True:
                s_new = input("Enter new start time (minutes.seconds): ").strip()
                e_new = input("Enter new end time (minutes.seconds): ").strip()
                s_ms = parse_time_arg(s_new)
                e_ms = parse_time_arg(e_new)
                if s_ms is None or e_ms is None or s_ms >= e_ms or is_overlap(s_ms,e_ms,processed_parts):
                    print("⚠️ Invalid or overlapping times, try again.")
                    continue
                part_start = s_ms
                part_end = e_ms
                break

        # --- Determine start index for segment numbering ---
        existing_files = list(out_dir.glob(f"{prefix_base}_*.wav"))
        start_index = max([int(f.stem.split('_')[-1]) for f in existing_files], default=-1) + 1

        # --- Segment audio ---
        seg_paths = segment_wav(wav_path, out_dir, prefix_base, part_start, part_end, start_index=start_index)
        processed_parts.append((part_start if part_start else 0, part_end if part_end else len(AudioSegment.from_wav(wav_path)), len(seg_paths)))

        entries = []
        # --- if all segments are no_drone with global subtype ---
        if is_no_drone_global and global_subtype:
            for p in seg_paths:
                final_dir = BASE_OUT_DIR / "not_a_drone" / global_subtype
                final_dir.mkdir(parents=True, exist_ok=True)
                new_path = final_dir / p.name
                shutil.move(p, new_path)
                p = new_path
                entry = {
                    "filename": p.name,
                    "binary_label": "no_drone",
                    "motor_label": None,
                    "source": source_type,
                    "youtube_url": src if source_type=="youtube" else None,
                    "duration": SEG_LEN/1000,
                    "subtype": global_subtype
                }
                entries.append(entry)
            update_metadata(entries)
            print(f"\n✅ Created {len(entries)} no_drone segments with global subtype '{global_subtype}' and updated metadata.json")
            return  # End script immediately

        # --- else normal per-segment labeling (drone or manual no_drone) ---
        for p in seg_paths:
            play_audio(p)
            while True:
                q = input(f"Quality (0 for no drone, 1–5) for {p.name}: ").strip()
                if q in ("0","1","2","3","4","5"):
                    quality = int(q)
                    break
                print("⚠️ Please enter 0–5.")

            if quality == 0:
                binary_label = "no_drone"
                motor_lbl = None
                # Decide subtype
                options = {"a":"airplanes","b":"birds","ca":"cars","cr":"crowd","e":"electronics","m":"motors","r":"random","w":"wind"}
                print("\nSelect subtype for this no_drone segment:")
                for k,v in options.items():
                    print(f"  {k} → {v}")
                while True:
                    choice = input("Enter your choice: ").strip().lower()
                    if choice in options:
                        subtype = options[choice]
                        break
                    print("⚠️ Invalid choice.")
                final_dir = BASE_OUT_DIR / "not_a_drone" / subtype
                final_dir.mkdir(parents=True, exist_ok=True)
                new_path = final_dir / p.name
                shutil.move(p,new_path)
                p = new_path
                entry = {
                    "filename": p.name,
                    "binary_label": binary_label,
                    "motor_label": motor_lbl,
                    "source": source_type,
                    "youtube_url": src if source_type=="youtube" else None,
                    "duration": SEG_LEN/1000,
                    "subtype": subtype
                }
            else:
                binary_label = "drone"
                motor_lbl = motor_label
                entry = {
                    "filename": p.name,
                    "binary_label": binary_label,
                    "motor_label": motor_lbl,
                    "source": source_type,
                    "youtube_url": src if source_type=="youtube" else None,
                    "duration": SEG_LEN/1000,
                    "quality": quality
                }
            entries.append(entry)

        update_metadata(entries)
        print(f"\n✅ Created {len(entries)} segments and updated metadata.json")

        if process_whole_video or source_type!="youtube":
            break



if __name__=="__main__":
    main()
