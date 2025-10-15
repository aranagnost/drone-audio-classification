#!/usr/bin/env python3
"""
dataset_stats.py
Reads metadata.json and prints/saves dataset statistics:
- Counts per binary_label and motor_label
- Quality distribution (1–5)
- Totals per class
- Optionally saves summary CSV
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter

META_FILE = Path("datasets/Drone_Audio_Dataset/metadata.json")
CSV_OUT = Path("datasets/Drone_Audio_Dataset/dataset_summary.csv")

def main():
    if not META_FILE.exists():
        print("❌ metadata.json not found.")
        return

    with open(META_FILE) as f:
        data = json.load(f)
    if not data:
        print("⚠️ metadata.json is empty.")
        return

    df = pd.DataFrame(data)

    # Normalize fields
    df["binary_label"] = df["binary_label"].fillna("unknown")
    df["motor_label"] = df["motor_label"].fillna("none")
    df["quality"] = df["quality"].fillna(0).astype(int)

    # Count totals per label
    summary = []
    for (binary_label, motor_label), group in df.groupby(["binary_label", "motor_label"]):
        counts = Counter(group["quality"])
        total = len(group)
        row = {
            "binary_label": binary_label,
            "motor_label": motor_label,
            "total": total,
            **{f"q{i}": counts.get(i, 0) for i in range(1, 6)}
        }
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values(["binary_label", "motor_label"]).reset_index(drop=True)

    # Print summary
    print("\n=== Dataset Summary ===\n")
    print(summary_df.to_string(index=False))
    print("\nTotal segments:", len(df))

    # Optional: totals by quality
    quality_counts = df["quality"].value_counts().sort_index()
    print("\n=== Global Quality Distribution ===")
    for q, c in quality_counts.items():
        print(f"Quality {q}: {c}")

    # Save to CSV
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(CSV_OUT, index=False)
    print(f"\n✅ Summary saved to {CSV_OUT}")

if __name__ == "__main__":
    main()
