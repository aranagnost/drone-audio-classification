#!/usr/bin/env python3
"""
discover_drone_audio.py

Automated drone audio discovery pipeline:
  1. search  — Interactive YouTube search: find videos, inspect links, then
               selectively download (full or time-ranged), segment, and
               pre-classify with the Stage 1 model.
  2. fetch   — Non-interactive: process a single YouTube URL or local file
               (with optional time ranges).
  3. review  — Interactive CLI to listen, label (quality 0-5), and ingest
               queued segments into the main dataset.
  4. status  — Show current queue stats.

Usage:
  python discover_drone_audio.py search "fpv drone flight audio" --count 5 --motor 4
  python discover_drone_audio.py fetch https://youtube.com/watch?v=XYZ --motor 4 2:15-3:02
  python discover_drone_audio.py fetch recording.wav --motor 6
  python discover_drone_audio.py review
  python discover_drone_audio.py status
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import torch
import torch.nn as nn
import torchaudio
from pydub import AudioSegment
from pydub.effects import normalize

# ──────────────────── paths ────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUEUE_DIR = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "review_queue"
QUEUE_META = QUEUE_DIR / "queue.json"
DATASET_AUDIO = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "audio"
DATASET_META = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "metadata.json"
CHECKPOINT = PROJECT_ROOT / "artifacts" / "checkpoints" / "stage1_smallcnn.pt"

# ──────────────────── audio config (must match training) ────────────────────
SEG_LEN_MS = 2000
STEP_MS = 1500
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 400
F_MIN = 20
CLIP_SAMPLES = int(SAMPLE_RATE * (SEG_LEN_MS / 1000))

YTDLP = shutil.which("yt-dlp")

# ──────────────────── model (self-contained copy of SmallCNN) ────────────────
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.classifier(z)


def load_model() -> SmallCNN:
    if not CHECKPOINT.exists():
        print(f"Checkpoint not found: {CHECKPOINT}")
        sys.exit(1)
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = SmallCNN(num_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def wav_to_logmel(wav_path: Path) -> torch.Tensor:
    """Load a 2-second wav and return a (1, 1, n_mels, time) log-mel tensor."""
    waveform, sr = torchaudio.load(str(wav_path))
    # resample if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # pad or trim to exact clip length
    if waveform.shape[1] < CLIP_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, CLIP_SAMPLES - waveform.shape[1]))
    else:
        waveform = waveform[:, :CLIP_SAMPLES]
    # mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        f_min=F_MIN,
        n_mels=N_MELS,
    )
    mel = mel_transform(waveform)
    log_mel = torch.log(mel + 1e-9)
    return log_mel.unsqueeze(0)  # (1, 1, n_mels, time)


# ──────────────────── queue helpers ────────────────────
def load_queue() -> list[dict]:
    if QUEUE_META.exists() and QUEUE_META.stat().st_size > 0:
        try:
            with open(QUEUE_META) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_queue(data: list[dict]):
    QUEUE_META.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_META, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset_metadata() -> list[dict]:
    if DATASET_META.exists() and DATASET_META.stat().st_size > 0:
        try:
            with open(DATASET_META) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []


def save_dataset_metadata(data: list[dict]):
    DATASET_META.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_META, "w") as f:
        json.dump(data, f, indent=2)


def build_known_sources() -> set[str]:
    """Collect all source URLs/paths already in the queue or dataset."""
    known: set[str] = set()
    for e in load_queue():
        known.add(e.get("source_ref", ""))
    for e in load_dataset_metadata():
        known.add(e.get("youtube_url", ""))
        known.add(e.get("local_path", ""))
    known.discard("")
    return known


# ──────────────────── YouTube helpers ────────────────────
def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme, result.netloc])
    except Exception:
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


def search_youtube(query: str, max_results: int) -> list[str]:
    """Use yt-dlp to search YouTube and return video URLs."""
    if not YTDLP:
        print("yt-dlp not found. Install it to search YouTube.")
        sys.exit(1)
    result = subprocess.run(
        [YTDLP, f"ytsearch{max_results}:{query}",
         "--get-url", "--get-id", "--flat-playlist",
         "--no-warnings"],
        capture_output=True, text=True,
    )
    ids = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
    urls = [f"https://www.youtube.com/watch?v={vid_id}" for vid_id in ids if len(vid_id) == 11]
    return urls


def download_youtube_audio(url: str) -> tuple[Path, str]:
    vid_id = extract_youtube_id(url)
    tmp_wav = Path(tempfile.gettempdir()) / f"yt_audio_{vid_id}.wav"
    if tmp_wav.exists():
        tmp_wav.unlink()
    print(f"  Downloading: {url}")
    subprocess.run(
        [YTDLP, "-x", "--audio-format", "wav", "-o", str(tmp_wav), url],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp_wav, vid_id


def convert_to_wav(src_path: Path) -> Path:
    if src_path.suffix.lower() != ".wav":
        tmp_wav = Path(tempfile.gettempdir()) / "discover_temp_audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(tmp_wav)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return tmp_wav
    return src_path


def format_ms(ms: int) -> str:
    total_sec = ms // 1000
    m, s = divmod(total_sec, 60)
    return f"{m}m{s:02d}s"


# ──────────────────── time-range parsing ────────────────────
def _parse_time(s: str) -> int:
    """Parse 'MM:SS' or 'H:MM:SS' into milliseconds."""
    parts = s.split(":")
    if len(parts) == 2:
        return (int(parts[0]) * 60 + int(parts[1])) * 1000
    elif len(parts) == 3:
        return (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
    raise ValueError(f"Invalid time: {s}")


def parse_time_range(s: str) -> tuple[int, int]:
    """Parse '2:15-3:02' or '1:14:48-1:17:50' into (start_ms, end_ms)."""
    left, sep, right = s.partition("-")
    if not sep:
        raise ValueError(f"Invalid range (no '-'): {s}")
    return _parse_time(left), _parse_time(right)


# ──────────────────── segmentation ────────────────────
def segment_wav_to_dir(wav_path: Path, out_dir: Path, prefix: str,
                       start_ms: int = 0, end_ms: int | None = None) -> list[Path]:
    audio = AudioSegment.from_wav(str(wav_path))
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, start in enumerate(range(start_ms, end_ms - SEG_LEN_MS, STEP_MS)):
        seg = audio[start:start + SEG_LEN_MS]
        seg = normalize(seg)
        fname = f"{prefix}_{i:03d}.wav"
        fpath = out_dir / fname
        seg.export(str(fpath), format="wav")
        paths.append(fpath)
    return paths


# ──────────────────── process_source (shared by search & fetch) ──────────────
def process_source(source: str, ranges: list[tuple[int, int]],
                   model: SmallCNN, threshold: float,
                   motor_label: str | None) -> int:
    """Download, segment, classify, and queue a single source. Returns segments queued."""
    try:
        if is_url(source):
            wav_path, source_id = download_youtube_audio(source)
            source_type = "youtube"
            source_ref = source
        else:
            src_path = Path(source)
            wav_path = convert_to_wav(src_path)
            source_id = src_path.stem
            source_type = "local"
            source_ref = str(src_path.resolve())
    except Exception as e:
        print(f"  Failed to get audio from {source}: {e}")
        return 0

    audio_obj = AudioSegment.from_wav(str(wav_path))
    print(f"  Duration: {format_ms(len(audio_obj))}")

    tmp_seg_dir = Path(tempfile.mkdtemp(prefix="drone_discover_"))

    if ranges:
        seg_paths: list[Path] = []
        for ri, (start_ms, end_ms) in enumerate(ranges):
            range_prefix = f"{source_id}_r{ri}"
            paths = segment_wav_to_dir(wav_path, tmp_seg_dir, range_prefix,
                                       start_ms=start_ms, end_ms=end_ms)
            seg_paths.extend(paths)
            print(f"  Range {format_ms(start_ms)}-{format_ms(end_ms)}: {len(paths)} segments")
    else:
        seg_paths = segment_wav_to_dir(wav_path, tmp_seg_dir, source_id)

    print(f"  {len(seg_paths)} chunks total. Running pre-classifier...")

    queue = load_queue()
    kept = 0
    for seg_path in seg_paths:
        logmel = wav_to_logmel(seg_path)
        with torch.no_grad():
            logits = model(logmel)
            probs = torch.softmax(logits, dim=1)
            drone_prob = probs[0, 1].item()

        if drone_prob >= threshold:
            queue_subdir = QUEUE_DIR / source_id
            queue_subdir.mkdir(parents=True, exist_ok=True)
            dest = queue_subdir / seg_path.name
            shutil.move(str(seg_path), str(dest))
            queue.append({
                "filename": seg_path.name,
                "source_id": source_id,
                "queue_path": str(dest),
                "drone_prob": round(drone_prob, 4),
                "source_type": source_type,
                "source_ref": source_ref,
                "motor_hint": motor_label,
            })
            kept += 1
        else:
            seg_path.unlink()

    shutil.rmtree(tmp_seg_dir, ignore_errors=True)
    save_queue(queue)
    print(f"  Queued {kept}/{len(seg_paths)} segments (threshold={threshold})\n")
    return kept


# ──────────────────── SEARCH command (interactive) ────────────────────
def cmd_search(args):
    query = args.query
    count = args.count
    threshold = args.threshold
    motor_label = f"{args.motor}_motors" if args.motor else None

    if not YTDLP:
        print("yt-dlp not found. Install it to search YouTube.")
        sys.exit(1)

    # Search for extra results to have room after filtering known sources
    print(f"Searching YouTube for: \"{query}\"")
    known = build_known_sources()
    raw_urls = search_youtube(query, count * 3)
    new_urls = [u for u in raw_urls if u not in known][:count]

    if not new_urls:
        print("No new videos found.")
        return

    print(f"\nFound {len(new_urls)} new videos:")
    for i, url in enumerate(new_urls, 1):
        print(f"  [{i}] {url}")

    print("\nInspect the videos, then enter commands below.")
    print("  <number>                          download full video")
    print("  <number> 2:15-3:02 14:48-17:50    download specific ranges")
    print("  skip <number>                     skip a video")
    print("  done                              finish\n")

    model = load_model()
    processed: set[int] = set()
    total_queued = 0

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in ("done", "q"):
            break

        parts = line.split()

        # skip command
        if parts[0].lower() == "skip":
            try:
                idx = int(parts[1]) - 1
                if 0 <= idx < len(new_urls):
                    processed.add(idx)
                    print(f"  Skipped [{idx + 1}]")
                else:
                    print(f"  Invalid number. Use 1-{len(new_urls)}.")
            except (ValueError, IndexError):
                print("  Usage: skip <number>")
            continue

        # video number (+ optional time ranges)
        try:
            idx = int(parts[0]) - 1
        except ValueError:
            print("  Unknown command. Type 'done' to finish.")
            continue

        if not (0 <= idx < len(new_urls)):
            print(f"  Invalid number. Use 1-{len(new_urls)}.")
            continue

        if idx in processed:
            print(f"  [{idx + 1}] already processed.")
            continue

        # Parse optional time ranges
        ranges: list[tuple[int, int]] = []
        valid = True
        for token in parts[1:]:
            try:
                ranges.append(parse_time_range(token))
            except ValueError:
                print(f"  Invalid time range: {token}  (format: MM:SS-MM:SS or H:MM:SS-H:MM:SS)")
                valid = False
                break

        if not valid:
            continue

        queued = process_source(new_urls[idx], ranges, model, threshold, motor_label)
        total_queued += queued
        processed.add(idx)

    print(f"Done. {total_queued} new segments in review queue.")


# ──────────────────── FETCH command (non-interactive) ────────────────────
def cmd_fetch(args):
    target = args.target
    threshold = args.threshold
    motor_label = f"{args.motor}_motors" if args.motor else None

    # Validate target
    if not is_url(target) and not Path(target).exists():
        print(f"Target not found: {target}")
        print("Provide a URL or path to an existing file.")
        sys.exit(1)

    # Check if already processed
    known = build_known_sources()
    ref = target if is_url(target) else str(Path(target).resolve())
    if ref in known:
        print(f"Already processed: {target}")
        print("Remove from queue/dataset first to reprocess.")
        return

    # Parse time ranges from positional args
    ranges: list[tuple[int, int]] = []
    for token in (args.ranges or []):
        try:
            ranges.append(parse_time_range(token))
        except ValueError:
            print(f"Invalid time range: {token}  (format: MM:SS-MM:SS or H:MM:SS-H:MM:SS)")
            sys.exit(1)

    model = load_model()
    queued = process_source(target, ranges, model, threshold, motor_label)
    print(f"Done. {queued} new segments in review queue.")


# ──────────────────── REVIEW command ────────────────────
def play_audio(file_path: Path):
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(file_path)]
        )
    except Exception:
        print("(Audio preview unavailable - ffplay not found)")


def cmd_review(args):
    queue = load_queue()
    if not queue:
        print("Review queue is empty. Run 'search' first.")
        return

    # sort by confidence descending so best candidates come first
    queue.reverse()  # LIFO: most recently added first

    print(f"\n{len(queue)} segments to review.\n")
    print("Controls per segment:")
    print("  1-5  = drone quality (accept into dataset)")
    print("  0    = not a drone (reclassify as no_drone with subtype)")
    print("  s    = skip (keep in queue for later)")
    print("  d    = discard (delete permanently)")
    print("  q    = quit review\n")

    nodrone_options = {
        "a": "airplanes", "b": "birds", "ca": "cars", "cr": "crowd",
        "e": "electronics", "m": "motors", "r": "random", "sp": "speech", "w": "wind",
    }

    remaining = []
    accepted = 0
    discarded = 0
    dataset_entries = []

    for i, entry in enumerate(queue):
        seg_path = Path(entry["queue_path"])
        if not seg_path.exists():
            # stale entry, skip
            continue

        motor_hint = entry.get("motor_hint") or "unknown"
        prob = entry["drone_prob"]
        src = entry.get("source_ref", "?")
        print(f"[{i+1}/{len(queue)}] {entry['filename']}  "
              f"P(drone)={prob:.2f}  hint={motor_hint}  src={src}")

        play_audio(seg_path)

        while True:
            choice = input("  Label> ").strip().lower()
            if choice in ("1", "2", "3", "4", "5"):
                quality = int(choice)
                # determine motor label
                if motor_hint and motor_hint != "unknown":
                    motor_label = motor_hint
                else:
                    mc = input("  Motor count (2/4/6/8): ").strip()
                    if mc in ("2", "4", "6", "8"):
                        motor_label = f"{mc}_motors"
                    else:
                        print("  Invalid motor count, skipping.")
                        remaining.append(entry)
                        break
                # move file to dataset
                dest_dir = DATASET_AUDIO / motor_label
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / seg_path.name
                shutil.move(str(seg_path), str(dest))
                # build metadata entry
                meta_entry = {
                    "filename": seg_path.name,
                    "binary_label": "drone",
                    "motor_label": motor_label,
                    "quality": quality,
                    "source": entry["source_type"],
                    "duration": SEG_LEN_MS / 1000,
                }
                if entry["source_type"] == "youtube":
                    meta_entry["youtube_url"] = entry["source_ref"]
                else:
                    meta_entry["local_path"] = entry["source_ref"]
                dataset_entries.append(meta_entry)
                accepted += 1
                print(f"  -> Accepted as {motor_label} q={quality}")
                break

            elif choice == "0":
                # reclassify as no_drone
                print("  Subtype:", "  ".join(f"{k}={v}" for k, v in nodrone_options.items()))
                while True:
                    sc = input("  Subtype> ").strip().lower()
                    if sc in nodrone_options:
                        subtype = nodrone_options[sc]
                        break
                    print("  Invalid choice.")
                dest_dir = DATASET_AUDIO / "not_a_drone" / subtype
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / seg_path.name
                shutil.move(str(seg_path), str(dest))
                meta_entry = {
                    "filename": seg_path.name,
                    "binary_label": "no_drone",
                    "motor_label": None,
                    "subtype": subtype,
                    "source": entry["source_type"],
                    "duration": SEG_LEN_MS / 1000,
                }
                if entry["source_type"] == "youtube":
                    meta_entry["youtube_url"] = entry["source_ref"]
                else:
                    meta_entry["local_path"] = entry["source_ref"]
                dataset_entries.append(meta_entry)
                accepted += 1
                print(f"  -> Reclassified as no_drone/{subtype}")
                break

            elif choice == "s":
                remaining.append(entry)
                print("  -> Skipped")
                break

            elif choice == "d":
                seg_path.unlink(missing_ok=True)
                discarded += 1
                print("  -> Discarded")
                break

            elif choice == "q":
                # put this and all remaining back
                remaining.append(entry)
                remaining.extend(queue[i+1:])
                print("\nQuitting review.")
                break

            else:
                print("  Invalid input. Use 0-5, s, d, or q.")
                continue
        else:
            # inner while broke normally, continue outer for
            continue
        # if we hit 'q', break out of outer loop too
        if choice == "q":
            break

    # save remaining queue
    save_queue(remaining)

    # update dataset metadata
    if dataset_entries:
        ds_meta = load_dataset_metadata()
        ds_meta.extend(dataset_entries)
        save_dataset_metadata(ds_meta)

    # clean up empty queue subdirs
    if QUEUE_DIR.exists():
        for d in QUEUE_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    print(f"\nReview complete: {accepted} accepted, {discarded} discarded, "
          f"{len(remaining)} still in queue.")


# ──────────────────── STATUS command ────────────────────
def cmd_status(args):
    queue = load_queue()
    if not queue:
        print("Review queue is empty.")
        return

    print(f"\nQueue: {len(queue)} segments pending review\n")

    # group by source
    by_source: dict[str, list] = {}
    for e in queue:
        key = e.get("source_id", "unknown")
        by_source.setdefault(key, []).append(e)

    print(f"{'Source':<25} {'Count':>6}  {'Avg P(drone)':>12}  {'Motor hint':<12}")
    print("-" * 60)
    for src_id, entries in sorted(by_source.items()):
        avg_p = sum(e["drone_prob"] for e in entries) / len(entries)
        hint = entries[0].get("motor_hint") or "-"
        print(f"{src_id:<25} {len(entries):>6}  {avg_p:>12.3f}  {hint:<12}")
    print()


# ──────────────────── CLI ────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Automated drone audio discovery and review pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # search — interactive YouTube search
    sp = sub.add_parser("search", help="Interactive YouTube search: find, inspect, then process")
    sp.add_argument("query", help="YouTube search query string")
    sp.add_argument("--count", type=int, default=5,
                    help="Number of new videos to find (default: 5)")
    sp.add_argument("--motor", type=int, choices=[2, 4, 6, 8], default=None,
                    help="Expected motor count (hint for review)")
    sp.add_argument("--threshold", type=float, default=0.3,
                    help="Minimum P(drone) to queue for review (default: 0.3)")

    # fetch — process a single URL or local file (non-interactive)
    sp = sub.add_parser("fetch", help="Process a YouTube URL or local audio file")
    sp.add_argument("target", help="YouTube URL or local file path")
    sp.add_argument("ranges", nargs="*", default=[],
                    help="Time ranges to extract (e.g. 2:15-3:02 14:48-17:50)")
    sp.add_argument("--motor", type=int, choices=[2, 4, 6, 8], default=None,
                    help="Expected motor count (hint for review)")
    sp.add_argument("--threshold", type=float, default=0.3,
                    help="Minimum P(drone) to queue for review (default: 0.3)")

    # review
    sub.add_parser("review", help="Interactively review and label queued segments")

    # status
    sub.add_parser("status", help="Show review queue stats")

    args = parser.parse_args()
    if args.command == "search":
        cmd_search(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "review":
        cmd_review(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
