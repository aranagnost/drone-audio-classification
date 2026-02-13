import json
import pandas as pd
from pathlib import Path
from collections import Counter

META_FILE = Path("datasets/Drone_Audio_Dataset/metadata.json")
CSV_OUT = Path("datasets/Drone_Audio_Dataset/dataset_segments_per_motor.csv")

with open(META_FILE) as f:
    data = json.load(f)

df = pd.DataFrame(data)

# --- Drone sections ---
df["motor_label"] = df["motor_label"].fillna("none")
df["quality"] = df.get("quality", pd.Series([0]*len(df))).fillna(0).astype(int)

drone_df = df[df["binary_label"] == "drone"]
nodrone_df = df[df["binary_label"] == "no_drone"]

# ---------- DRONE SUMMARY ----------
summary = []
for motor_label, group in drone_df.groupby("motor_label"):
    counts = Counter(group["quality"])
    row = {
        "motor_label": motor_label,
        **{f"q{i}": counts.get(i, 0) for i in range(1, 6)},
        "total": len(group)
    }
    summary.append(row)

summary_df = pd.DataFrame(summary).sort_values("motor_label").reset_index(drop=True)

print("\n========================================================================")
print("=== Drone Segments Summary ===\n")
print(summary_df)

summary_df.to_csv(CSV_OUT, index=False)


# ---------- NO-DRONE SUBTYPE SUMMARY ----------
print("\n\n\n=== Non-Drone Segments by Subtype ===\n")

if not nodrone_df.empty and "subtype" in nodrone_df.columns:
    subtype_counts = nodrone_df["subtype"].value_counts().reset_index()
    subtype_counts.columns = ["subtype", "count"]
    print(subtype_counts)
else:
    print("No non-drone segments found.")

print("========================================================================\n")