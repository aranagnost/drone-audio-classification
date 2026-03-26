#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import csv

def build_row(item, audio_root: Path):
    binary = item["binary_label"]  # "drone" or "no_drone"
    fname = item["filename"]

    if binary == "drone":
        motor = item["motor_label"]  # "2_motors" ...
        quality = item.get("quality", None)  # 1..5
        subtype = ""
        rel_path = Path(motor) / fname
    else:
        motor = ""
        quality = ""
        subtype = item.get("subtype", "")
        rel_path = Path("not_a_drone") / subtype / fname

    full_path = (audio_root / rel_path).resolve()

    return {
        "filepath": str(full_path),
        "relpath": str(rel_path),
        "binary_label": binary,
        "motor_label": motor,          # empty for no_drone
        "subtype": subtype,            # empty for drone
        "quality": quality,            # empty for no_drone
        "youtube_url": item.get("youtube_url") or item.get("local_path", ""),
        "source": item.get("source", ""),
        "duration": item.get("duration", ""),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="datasets/Drone_Audio_Dataset",
                    help="Path to dataset root containing metadata.json and audio/")
    ap.add_argument("--out_csv", type=str, default="data/prep/manifest.csv")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    meta_path = dataset_root / "metadata.json"
    audio_root = dataset_root / "audio"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found at: {meta_path}")
    if not audio_root.exists():
        raise FileNotFoundError(f"audio folder not found at: {audio_root}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["filepath", "relpath", "binary_label", "motor_label", "subtype",
                  "quality", "youtube_url", "source", "duration"]

    missing_files = 0
    rows = []
    for item in meta:
        row = build_row(item, audio_root)
        if not Path(row["filepath"]).exists():
            missing_files += 1
        rows.append(row)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows -> {out_path}")
    if missing_files:
        print(f"[WARN] {missing_files} files referenced in metadata but not found on disk.")

if __name__ == "__main__":
    main()
