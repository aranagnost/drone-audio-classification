import json
import pandas as pd
from pathlib import Path
from collections import Counter

META_FILE  = Path("datasets/Drone_Audio_Dataset/metadata.json")
AUDIO_ROOT = Path("datasets/Drone_Audio_Dataset/audio")
CSV_OUT    = Path("datasets/Drone_Audio_Dataset/dataset_segments_per_motor.csv")

with open(META_FILE) as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["motor_label"] = df["motor_label"].fillna("none")
df["quality"]     = df.get("quality", pd.Series([0] * len(df))).fillna(0).astype(int)
df["subtype"]     = df.get("subtype", pd.Series([None] * len(df)))

drone_df   = df[df["binary_label"] == "drone"]
nodrone_df = df[df["binary_label"] == "no_drone"]

SEP = "=" * 72

# ---------- DRONE SUMMARY ----------
quality_levels = sorted(drone_df["quality"].unique())

yt_urls_per_motor  = drone_df[drone_df["source"] == "youtube"].groupby("motor_label")["youtube_url"].nunique()
local_segs_per_motor = drone_df[drone_df["source"] == "local"].groupby("motor_label").size()

summary = []
for motor_label, group in drone_df.groupby("motor_label"):
    counts = Counter(group["quality"])
    row = {
        "motor_label": motor_label,
        **{f"q{q}": counts.get(q, 0) for q in quality_levels},
        "total":      len(group),
        "yt_urls":    int(yt_urls_per_motor.get(motor_label, 0)),
        "local_segs": int(local_segs_per_motor.get(motor_label, 0)),
    }
    summary.append(row)

summary_df = pd.DataFrame(summary).sort_values("motor_label").reset_index(drop=True)
summary_df.loc[len(summary_df)] = {
    "motor_label": "TOTAL",
    **{f"q{q}": summary_df[f"q{q}"].sum() for q in quality_levels},
    "total":      summary_df["total"].sum(),
    "yt_urls":    summary_df["yt_urls"].sum(),
    "local_segs": summary_df["local_segs"].sum(),
}

print(f"\n{SEP}")
print("=== Drone Segments ===\n")
print(summary_df.to_string(index=False))
summary_df[summary_df["motor_label"] != "TOTAL"].to_csv(CSV_OUT, index=False)

# ---------- NO-DRONE SUBTYPE SUMMARY ----------
# Discover all subtypes from both metadata and filesystem folders
meta_subtypes   = set(nodrone_df["subtype"].dropna().unique())
folder_subtypes = {p.name for p in (AUDIO_ROOT / "not_a_drone").iterdir() if p.is_dir()} \
                  if (AUDIO_ROOT / "not_a_drone").exists() else set()
all_subtypes    = sorted(meta_subtypes | folder_subtypes)

subtype_counts = nodrone_df["subtype"].value_counts()
url_col        = "youtube_url" if "youtube_url" in nodrone_df.columns else None
subtype_urls   = nodrone_df.groupby("subtype")[url_col].nunique() if url_col else {}

rows = [{
    "subtype": s,
    "count":   int(subtype_counts.get(s, 0)),
    "urls":    int(subtype_urls.get(s, 0)) if url_col else 0,
} for s in all_subtypes]

nodrone_df_out = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
nodrone_df_out.loc[len(nodrone_df_out)] = {
    "subtype": "TOTAL",
    "count":   nodrone_df_out["count"].sum(),
    "urls":    nodrone_df_out["urls"].sum(),
}

print(f"\n\n=== Non-Drone Segments by Subtype ===\n")
print(nodrone_df_out.to_string(index=False))

# ---------- GRAND TOTAL ----------
print(f"\n\n=== Grand Total ===\n")
print(f"  Drone    : {len(drone_df):>6}")
print(f"  No-Drone : {len(nodrone_df):>6}")
print(f"  Total    : {len(df):>6}")
print(f"\n{SEP}\n")
