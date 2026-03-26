# ml/data/make_stage2_splits.py
import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path


MOTOR_CLASSES = ["2_motors", "4_motors", "6_motors", "8_motors"]


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(out_path: Path, rows, fieldnames):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def group_by_url(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["youtube_url"]].append(r)
    return g


def url_motor_counts(url_rows):
    c = Counter()
    for r in url_rows:
        c[r["motor_label"]] += 1
    return c


def split_urls(urls, train_ratio, val_ratio, seed):
    rnd = random.Random(seed)
    rnd.shuffle(urls)
    n = len(urls)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = urls[:n_train]
    val = urls[n_train:n_train + n_val]
    test = urls[n_train + n_val:]
    return train, val, test


def counts_in_split(groups, split_urls_list):
    c = Counter()
    for u in split_urls_list:
        c.update(url_motor_counts(groups[u]))
    return c


def all_classes_present(counter: Counter):
    return all(counter.get(k, 0) > 0 for k in MOTOR_CLASSES)


def balance_score(counter: Counter) -> float:
    """Lower is better. Returns max/min ratio across classes (1.0 = perfectly balanced)."""
    vals = [counter.get(k, 0) for k in MOTOR_CLASSES]
    if min(vals) == 0:
        return float("inf")
    return max(vals) / min(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/prep/manifest.csv")
    ap.add_argument("--out_dir", default="data/splits_stage2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--max_tries", type=int, default=5000)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    rows = read_rows(in_csv)
    fieldnames = list(rows[0].keys())

    # Keep only drone rows
    drone_rows = [r for r in rows if r["binary_label"] == "drone"]
    groups = group_by_url(drone_rows)
    urls = list(groups.keys())

    # Try many random splits, pick the one with the most balanced val+test
    best = None
    best_score = float("inf")
    for k in range(args.max_tries):
        train_urls, val_urls, test_urls = split_urls(urls, args.train, args.val, args.seed + k)

        c_tr = counts_in_split(groups, train_urls)
        c_va = counts_in_split(groups, val_urls)
        c_te = counts_in_split(groups, test_urls)

        if not (all_classes_present(c_tr) and all_classes_present(c_va) and all_classes_present(c_te)):
            continue

        # Score: worst imbalance across val and test (lower = better)
        score = max(balance_score(c_va), balance_score(c_te))
        if score < best_score:
            best_score = score
            best = (train_urls, val_urls, test_urls, c_tr, c_va, c_te)

    if best is None:
        print("[ERROR] Could not find a split where all motor classes exist in train/val/test.")
        print("Try increasing --val/--test ratios, or increase --max_tries.")
        return

    print(f"[INFO] Best balance score: {best_score:.2f}x (1.0 = perfect, tried {args.max_tries} seeds)")
    train_urls, val_urls, test_urls, c_tr, c_va, c_te = best

    # Materialize rows
    train_rows, val_rows, test_rows = [], [], []
    for u in train_urls:
        train_rows.extend(groups[u])
    for u in val_urls:
        val_rows.extend(groups[u])
    for u in test_urls:
        test_rows.extend(groups[u])

    out_dir = Path(args.out_dir)
    write_rows(out_dir / "train.csv", train_rows, fieldnames)
    write_rows(out_dir / "val.csv", val_rows, fieldnames)
    write_rows(out_dir / "test.csv", test_rows, fieldnames)

    print("[OK] Wrote stage2 splits (drone-only, grouped by url, all classes present)")
    print("Train urls:", len(train_urls), "rows:", len(train_rows), "counts:", dict(c_tr))
    print("Val   urls:", len(val_urls), "rows:", len(val_rows), "counts:", dict(c_va))
    print("Test  urls:", len(test_urls), "rows:", len(test_rows), "counts:", dict(c_te))


if __name__ == "__main__":
    main()
