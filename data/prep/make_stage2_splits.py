# ml/data/make_stage2_splits.py
import argparse
import csv
import random
from collections import defaultdict, Counter
from pathlib import Path


MOTOR_CLASSES = ["2_motors", "4_motors", "6_motors", "8_motors"]
QUALITY_BINS = [(1, 2), (3, 3), (4, 5)]  # low (q1-q2), mid (q3), high (q4-q5)


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


def quality_bin(q: int) -> int:
    """Map quality score to bin index (0=low, 1=mid, 2=high)."""
    for i, (lo, hi) in enumerate(QUALITY_BINS):
        if lo <= q <= hi:
            return i
    return 0


def url_stratum(url_rows) -> str:
    """Assign each URL a stratum key = (dominant_motor, dominant_quality_bin)."""
    motor_counts = Counter()
    qbin_counts = Counter()
    for r in url_rows:
        motor_counts[r["motor_label"]] += 1
        try:
            qbin_counts[quality_bin(int(r.get("quality", "1")))] += 1
        except ValueError:
            qbin_counts[0] += 1
    dominant_motor = motor_counts.most_common(1)[0][0]
    dominant_qbin = qbin_counts.most_common(1)[0][0]
    return f"{dominant_motor}__q{dominant_qbin}"


def stratified_split(groups, train_ratio, val_ratio, seed):
    """Split URLs stratified by (motor_label, quality_bin) stratum."""
    rnd = random.Random(seed)

    # Group URLs by stratum
    by_stratum: dict[str, list] = defaultdict(list)
    for url, url_rows in groups.items():
        stratum = url_stratum(url_rows)
        by_stratum[stratum].append(url)

    train_urls, val_urls, test_urls = [], [], []
    for stratum, stratum_urls in by_stratum.items():
        rnd.shuffle(stratum_urls)
        n = len(stratum_urls)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Ensure we don't exceed n
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_urls.extend(stratum_urls[:n_train])
        val_urls.extend(stratum_urls[n_train:n_train + n_val])
        test_urls.extend(stratum_urls[n_train + n_val:])

    return train_urls, val_urls, test_urls


def counts_in_split(groups, split_urls_list):
    c = Counter()
    for u in split_urls_list:
        c.update(url_motor_counts(groups[u]))
    return c


def motor_quality_counts(groups, split_urls_list):
    """Return {motor: {qbin: count}} for all rows."""
    result = defaultdict(Counter)
    for u in split_urls_list:
        for r in groups[u]:
            try:
                qb = quality_bin(int(r.get("quality", "1")))
            except ValueError:
                qb = 0
            result[r["motor_label"]][qb] += 1
    return result


def all_classes_present(counter: Counter):
    return all(counter.get(k, 0) > 0 for k in MOTOR_CLASSES)


def balance_score(counter: Counter) -> float:
    """Lower is better. Returns max/min ratio across classes (1.0 = perfectly balanced)."""
    vals = [counter.get(k, 0) for k in MOTOR_CLASSES]
    if min(vals) == 0:
        return float("inf")
    return max(vals) / min(vals)


def quality_skew_score(groups, val_urls, test_urls) -> float:
    """
    Penalize splits where val and test have very different quality distributions
    within each motor class. Lower is better.
    """
    val_mq = motor_quality_counts(groups, val_urls)
    test_mq = motor_quality_counts(groups, test_urls)
    total_diff = 0.0
    for motor in MOTOR_CLASSES:
        v_total = sum(val_mq[motor].values()) or 1
        t_total = sum(test_mq[motor].values()) or 1
        for qb in range(len(QUALITY_BINS)):
            v_frac = val_mq[motor].get(qb, 0) / v_total
            t_frac = test_mq[motor].get(qb, 0) / t_total
            total_diff += abs(v_frac - t_frac)
    return total_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="data/prep/manifest.csv")
    ap.add_argument("--out_dir", default="data/splits_stage2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--max_tries", type=int, default=500)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    rows = read_rows(in_csv)
    fieldnames = list(rows[0].keys())

    # Keep only drone rows
    drone_rows = [r for r in rows if r["binary_label"] == "drone"]
    groups = group_by_url(drone_rows)

    # Try many seeds, pick the split with the most balanced val+test
    # AND the most similar quality distributions across val and test
    best = None
    best_score = float("inf")
    for k in range(args.max_tries):
        train_urls, val_urls, test_urls = stratified_split(groups, args.train, args.val, args.seed + k)

        c_tr = counts_in_split(groups, train_urls)
        c_va = counts_in_split(groups, val_urls)
        c_te = counts_in_split(groups, test_urls)

        if not (all_classes_present(c_tr) and all_classes_present(c_va) and all_classes_present(c_te)):
            continue

        # Combined score: class balance + quality distribution similarity between val/test
        score = (max(balance_score(c_va), balance_score(c_te))
                 + quality_skew_score(groups, val_urls, test_urls))
        if score < best_score:
            best_score = score
            best = (train_urls, val_urls, test_urls, c_tr, c_va, c_te)

    if best is None:
        print("[ERROR] Could not find a valid split. Try increasing --val/--test ratios.")
        return

    print(f"[INFO] Best score: {best_score:.3f} (tried {args.max_tries} seeds)")
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

    print("[OK] Wrote stage2 splits (stratified by motor x quality, grouped by url)")
    print("Train urls:", len(train_urls), "rows:", len(train_rows), "counts:", dict(c_tr))
    print("Val   urls:", len(val_urls), "rows:", len(val_rows), "counts:", dict(c_va))
    print("Test  urls:", len(test_urls), "rows:", len(test_rows), "counts:", dict(c_te))

    # Print quality-within-class breakdown
    for split_name, split_urls_list in [("Val", val_urls), ("Test", test_urls)]:
        mq = motor_quality_counts(groups, split_urls_list)
        bin_labels = [f"q{i}({lo}-{hi})" for i, (lo, hi) in enumerate(QUALITY_BINS)]
        print(f"\n{split_name} quality breakdown (all qualities, bins: {bin_labels}):")
        for motor in MOTOR_CLASSES:
            total = sum(mq[motor].values()) or 1
            fracs = {f"q{b}": f"{mq[motor].get(b,0)/total:.0%}" for b in range(len(QUALITY_BINS))}
            print(f"  {motor}: {dict(fracs)} ({sum(mq[motor].values())} samples)")


if __name__ == "__main__":
    main()
