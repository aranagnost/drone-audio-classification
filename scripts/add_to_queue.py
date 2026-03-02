#!/usr/bin/env python3
"""
add_to_queue.py

Batch-adds pre-cut 2-second WAV files from
    datasets/Drone_Audio_Dataset/add_queue/
into the review queue so you can label them in the review app.

Usage:
    python scripts/add_to_queue.py

Drop your clips into the appropriate subfolder:

  add_queue/
    2_motors/   <- clips with 2-motor drones
    4_motors/   <- clips with 4-motor drones
    6_motors/   <- clips with 6-motor drones
    8_motors/   <- clips with 8-motor drones

The motor hint is taken from the subfolder name and pre-selected in
the review app — you can still change it before accepting each segment.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PROJECT_ROOT    = Path(__file__).resolve().parent.parent
ADD_QUEUE_DIR   = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "add_queue"
REVIEW_QUEUE_DIR = PROJECT_ROOT / "datasets" / "Drone_Audio_Dataset" / "review_queue"
QUEUE_META      = REVIEW_QUEUE_DIR / "queue.json"

MOTOR_LABELS = {"2_motors", "4_motors", "6_motors", "8_motors"}


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


def unique_dest(dest_dir: Path, filename: str) -> Path:
    """Return a non-colliding path inside dest_dir."""
    dest = dest_dir / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while dest.exists():
        dest = dest_dir / f"{stem}_{i}{suffix}"
        i += 1
    return dest


def main():
    ADD_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect WAV files from the 4 known motor-class subfolders only.
    groups: dict[str, list[Path]] = {}
    for wav in sorted(ADD_QUEUE_DIR.rglob("*.wav")):
        rel = wav.relative_to(ADD_QUEUE_DIR)
        if len(rel.parts) < 2:
            print(f"  [skip] {wav.name} — place files inside a class subfolder "
                  f"(2_motors / 4_motors / 6_motors / 8_motors)")
            continue
        group_name = rel.parts[0]
        if group_name not in MOTOR_LABELS:
            print(f"  [skip] {wav.name} — unknown subfolder '{group_name}'")
            continue
        groups.setdefault(group_name, []).append(wav)

    if not groups:
        print(f"No WAV files found in:\n  {ADD_QUEUE_DIR}")
        print("Drop clips into 2_motors/, 4_motors/, 6_motors/, or 8_motors/ and re-run.")
        return

    total = sum(len(v) for v in groups.values())
    print(f"Found {total} WAV file(s) across {len(groups)} group(s).\n")

    queue = load_queue()
    added = 0

    for group_name, wav_paths in sorted(groups.items()):
        label = group_name  # subfolder name IS the motor label
        print(f"  {label}: {len(wav_paths)} file(s)")

        dest_dir = REVIEW_QUEUE_DIR / group_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for wav_path in wav_paths:
            dest = unique_dest(dest_dir, wav_path.name)
            original_path = str(wav_path.resolve())
            shutil.move(str(wav_path), str(dest))
            queue.append({
                "filename": dest.name,
                "source_id": group_name,
                "queue_path": str(dest),
                "drone_prob": 1.0,
                "source_type": "local",
                "source_ref": original_path,
                "motor_hint": label,
            })
            added += 1

    save_queue(queue)
    print(f"Done. Added {added} segment(s) to the review queue.")
    print(f"Start the review app:  python scripts/review_app.py")

    # Remove empty leftover folders in add_queue
    for d in sorted(ADD_QUEUE_DIR.iterdir(), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()


if __name__ == "__main__":
    main()
