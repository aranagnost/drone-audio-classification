# ml/data/make_stage3_splits.py
"""
Create train/val/test splits specifically for stage 3 (4_motors vs 6_motors),
grouped by youtube_url to prevent leakage and optimised for 4/6 balance.

The stage2 splits are not suitable for stage 3 because they balance all four
motor classes globally — val and test can end up with opposite 4/6 ratios,
causing the model to learn the wrong class prior.
"""
import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path

FINE_CLASSES = ["4_motors", "6_motors"]


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(out_path: Path, rows, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def balance_score(counter: Counter) -> float:
    """Lower is better: max/min ratio across 4/6 classes (1.0 = perfectly balanced)."""
    vals = [counter.get(k, 0) for k in FINE_CLASSES]
    if min(vals) == 0:
        return float("inf")
    return max(vals) / min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="ml/data/manifest.csv")
    ap.add_argument("--out_dir", default="ml/splits_stage3")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--max_tries", type=int, default=10000)
    args = ap.parse_args()

    rows = read_rows(Path(args.in_csv))
    fieldnames = list(rows[0].keys())

    fine_rows = [r for r in rows
                 if r["binary_label"] == "drone" and r["motor_label"] in FINE_CLASSES]

    groups = defaultdict(list)
    for r in fine_rows:
        groups[r["youtube_url"]].append(r)
    urls = list(groups.keys())

    def counts_for(url_list):
        c = Counter()
        for u in url_list:
            for r in groups[u]:
                c[r["motor_label"]] += 1
        return c

    best = None
    best_score = float("inf")

    for k in range(args.max_tries):
        rnd = random.Random(args.seed + k)
        shuffled = urls[:]
        rnd.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * args.train)
        n_val = int(n * args.val)

        train_urls = shuffled[:n_train]
        val_urls = shuffled[n_train:n_train + n_val]
        test_urls = shuffled[n_train + n_val:]

        c_tr = counts_for(train_urls)
        c_va = counts_for(val_urls)
        c_te = counts_for(test_urls)

        if not all(c.get(cls, 0) > 0
                   for c in [c_tr, c_va, c_te]
                   for cls in FINE_CLASSES):
            continue

        score = max(balance_score(c_va), balance_score(c_te))
        if score < best_score:
            best_score = score
            best = (train_urls, val_urls, test_urls, c_tr, c_va, c_te)

    if best is None:
        print("[ERROR] Could not find a valid split. Try increasing --max_tries.")
        return

    train_urls, val_urls, test_urls, c_tr, c_va, c_te = best

    train_rows = [r for u in train_urls for r in groups[u]]
    val_rows = [r for u in val_urls for r in groups[u]]
    test_rows = [r for u in test_urls for r in groups[u]]

    out_dir = Path(args.out_dir)
    write_rows(out_dir / "train.csv", train_rows, fieldnames)
    write_rows(out_dir / "val.csv", val_rows, fieldnames)
    write_rows(out_dir / "test.csv", test_rows, fieldnames)

    print(f"[INFO] Best balance score: {best_score:.2f}x (1.0=perfect, tried {args.max_tries} seeds)")
    print(f"Train: {len(train_urls)} urls  {len(train_rows)} rows  {dict(c_tr)}")
    print(f"Val:   {len(val_urls)} urls  {len(val_rows)} rows  {dict(c_va)}")
    print(f"Test:  {len(test_urls)} urls  {len(test_rows)} rows  {dict(c_te)}")


if __name__ == "__main__":
    main()
