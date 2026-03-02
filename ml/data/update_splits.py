#!/usr/bin/env python3
"""
update_splits.py

One-command rebuild of all data splits from the current dataset.

Runs the full pipeline:
  1. make_manifest.py   — metadata.json  →  manifest.csv
  2. split_by_youtube_url.py  — manifest  →  ml/splits/{train,val,test}.csv
  3. make_stage2_splits.py    — manifest  →  ml/splits_stage2/{train,val,test}.csv

Usage (from project root):
    python ml/data/update_splits.py
    python ml/data/update_splits.py --seed 123        # different seed
    python ml/data/update_splits.py --train 0.7 --val 0.15 --test 0.15
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]):
    print(f"\n{'─'*60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="Rebuild all train/val/test splits from the current dataset.")
    ap.add_argument("--dataset_root", default="datasets/Drone_Audio_Dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    args = ap.parse_args()

    scripts_dir = Path(__file__).resolve().parent

    # 1. Manifest
    run([
        str(scripts_dir / "make_manifest.py"),
        "--dataset_root", args.dataset_root,
        "--out_csv", "ml/data/manifest.csv",
    ])

    # 2. Stage-1 splits (all samples, grouped by youtube_url)
    run([
        str(scripts_dir / "split_by_youtube_url.py"),
        "--manifest", "ml/data/manifest.csv",
        "--out_dir", "ml/splits",
        "--seed", str(args.seed),
        "--train", str(args.train),
        "--val", str(args.val),
        "--test", str(args.test),
    ])

    # 3. Stage-2 splits (drone-only, re-split by url)
    run([
        str(scripts_dir / "make_stage2_splits.py"),
        "--in_csv", "ml/data/manifest.csv",
        "--out_dir", "ml/splits_stage2",
        "--seed", str(args.seed),
        "--train", str(args.train),
        "--val", str(args.val),
        "--test", str(args.test),
    ])

    print(f"\n{'═'*60}")
    print("  All splits updated.")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
