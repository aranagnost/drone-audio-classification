#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path
from collections import defaultdict

def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def write_rows(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="ml/data/manifest.csv")
    ap.add_argument("--out_dir", type=str, default="ml/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-9

    manifest_path = Path(args.manifest)
    rows = read_manifest(manifest_path)
    fieldnames = list(rows[0].keys())

    # Group rows by youtube_url (this prevents segment leakage)
    groups = defaultdict(list)
    for row in rows:
        url = row.get("youtube_url", "")
        groups[url].append(row)

    urls = list(groups.keys())
    random.Random(args.seed).shuffle(urls)

    n = len(urls)
    n_train = int(n * args.train)
    n_val = int(n * args.val)

    train_urls = set(urls[:n_train])
    val_urls = set(urls[n_train:n_train + n_val])
    test_urls = set(urls[n_train + n_val:])

    train_rows, val_rows, test_rows = [], [], []
    for url, items in groups.items():
        if url in train_urls:
            train_rows.extend(items)
        elif url in val_urls:
            val_rows.extend(items)
        else:
            test_rows.extend(items)

    out_dir = Path(args.out_dir)
    write_rows(out_dir / "train.csv", train_rows, fieldnames)
    write_rows(out_dir / "val.csv", val_rows, fieldnames)
    write_rows(out_dir / "test.csv", test_rows, fieldnames)

    print("[OK] Split by youtube_url")
    print(f"  train: {len(train_rows)} rows ({len(train_urls)} urls)")
    print(f"  val:   {len(val_rows)} rows ({len(val_urls)} urls)")
    print(f"  test:  {len(test_rows)} rows ({len(test_urls)} urls)")

if __name__ == "__main__":
    main()
