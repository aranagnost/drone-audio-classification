import csv
from collections import Counter, defaultdict
from pathlib import Path

def count(csv_path, min_q=1):
    counts = Counter()
    by_motor = Counter()
    by_q = Counter()

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["binary_label"] != "drone":
                continue
            q = int(row.get("quality", 5))
            if q < min_q:
                continue
            motor = row["motor_label"]
            by_motor[motor] += 1
            by_q[q] += 1
            counts["total_drone"] += 1
    return counts, by_motor, by_q

def show(name, path, min_q):
    c, m, q = count(path, min_q=min_q)
    print(f"\n=== {name} (min_quality={min_q}) ===")
    print("Motor counts:", dict(m))
    print("Quality counts:", dict(q))
    print("Total drone:", c["total_drone"])

if __name__ == "__main__":
    base = Path("ml/splits")
    for min_q in [1,2,3,4]:
        show("TRAIN", base/"train.csv", min_q)
        show("VAL", base/"val.csv", min_q)
        show("TEST", base/"test.csv", min_q)
