"""
Extract 10-second handcrafted features by stitching consecutive 2-second segments.

Stitching logic (matches scripts/segment_audio.py settings: SEG_LEN=2000 ms, STEP=1500 ms):
  - Group clips by (motor_label, source_group) where source_group is the
    relpath with its trailing segment index stripped. For YouTube clips that
    means the `<videoID>_r<N>` prefix (treating different time-ranges from the
    same video as separate groups). For local clips, the file stem.
  - Sort each group by segment index.
  - Stitch by taking the full first segment, then only the *novel* 1.5 s of
    each subsequent segment (i.e., drop the 500 ms overlap with the previous
    segment).
  - Re-normalize the stitched recording to peak=1.0 — fixes the per-segment
    peak-normalisation discontinuities introduced at segmentation time.

For every 2 s clip in the input CSV(s), we extract a 10 s window *centered*
on that clip's position within the stitched recording. If the stitched
recording is shorter than 10 s, we loop it up to 10 s. This keeps the row
count of the output parquet identical to the input, so the rest of the
pipeline (train/val/test splits, XGB training, cascade eval) does not
change.

Feature set is identical to `features/extract_features.py` — same 92 columns
and same meta columns — so it is a drop-in replacement for the XGB pipeline.
The only change is that the HPS/sub-band FFTs use a larger N_FFT to exploit
the longer audio (~0.5 Hz frequency resolution vs ~8 Hz at 2 s).

Usage:
    python features/extract_features_10s.py \\
        --train_csv data/splits_stage2/train.csv \\
        --val_csv   data/splits_stage2/val.csv \\
        --test_csv  data/splits_stage2/test.csv \\
        --dataset_root datasets/Drone_Audio_Dataset/audio \\
        --out features/stage2_features_10s.parquet \\
        --n_jobs 4
"""
from __future__ import annotations

import argparse
import os
import re
import traceback
import warnings
from pathlib import Path

# Disable numba JIT before librosa is imported — chroma_stft calls
# estimate_tuning() which uses numba and segfaults inside parallel workers.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# Must match scripts/segment_audio.py
TARGET_SR  = 16_000
SEG_LEN_MS = 2000
STEP_MS    = 1500
SEG_LEN    = int(TARGET_SR * SEG_LEN_MS / 1000)   # 32_000 samples
STEP       = int(TARGET_SR * STEP_MS    / 1000)   # 24_000 samples
OVERLAP    = SEG_LEN - STEP                       # 8_000 samples (0.5 s)

# Target output duration
DURATION   = 10.0
TARGET_LEN = int(TARGET_SR * DURATION)             # 160_000 samples

# N_FFT choices. 2048 for frame-based features (kept from 2 s script).
# 32768 for global HPS / sub-band energy — gives ~0.5 Hz resolution,
# useful for motor-fundamental discrimination.
N_FFT_FRAME  = 2048
N_FFT_GLOBAL = 32768

META_COLS = ["filepath", "relpath", "motor_label", "binary_label",
             "quality", "youtube_url", "split"]

SEG_INDEX_RE = re.compile(r'_(\d+)_?\.wav$')


# ---------------------------------------------------------------------------
# Stitch-group key parsing
# ---------------------------------------------------------------------------

def parse_group_key(relpath: str) -> tuple[str, int]:
    """
    Strip the segment index suffix from relpath to get a stitch group key.
    Examples:
        2_motors/Ot3Em2lI9y8_r0_031.wav -> ("2_motors/Ot3Em2lI9y8_r0", 31)
        4_motors/mixed_2-bebop_001_.wav -> ("4_motors/mixed_2-bebop",  1)
    Clips with no index match are treated as their own singleton group.
    """
    m = SEG_INDEX_RE.search(relpath)
    if not m:
        return relpath, 0
    return relpath[: m.start()], int(m.group(1))


# ---------------------------------------------------------------------------
# Audio loading + stitching
# ---------------------------------------------------------------------------

def resolve_path(row: dict, dataset_root: str | None) -> str:
    if dataset_root:
        return os.path.join(dataset_root, row["relpath"])
    return row["filepath"]


def load_2s_clip(fp: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(fp, sr=TARGET_SR, mono=True,
                        duration=(SEG_LEN_MS / 1000) + 0.1)
    if len(y) < SEG_LEN:
        y = np.pad(y, (0, SEG_LEN - len(y)))
    else:
        y = y[:SEG_LEN]
    return y.astype(np.float32)


def stitch_rows(rows_sorted: list[dict], dataset_root: str | None
                ) -> tuple[np.ndarray, list[int]]:
    """
    Stitch consecutive 2 s clips (already sorted by seg index) into a single
    waveform using overlap removal. Re-normalise to peak 1.0.

    Returns:
        stitched           : (N_samples,) float32
        clip_start_samples : list of length len(rows_sorted); clip_start_samples[i]
                             is the sample-index at which clip i's *content*
                             begins in the stitched waveform. (For i==0 it is 0;
                             for i>0 it is where the novel 1.5 s starts.)
    """
    parts: list[np.ndarray] = []
    starts: list[int] = []
    cum = 0
    for i, row in enumerate(rows_sorted):
        y = load_2s_clip(resolve_path(row, dataset_root))
        if i == 0:
            parts.append(y)
            starts.append(0)
            cum += len(y)
        else:
            parts.append(y[OVERLAP:])   # drop the 0.5 s overlap with previous
            starts.append(cum)          # novel content begins here
            cum += len(y) - OVERLAP

    stitched = np.concatenate(parts).astype(np.float32)
    peak = float(np.max(np.abs(stitched))) if len(stitched) else 0.0
    if peak > 1e-6:
        stitched = stitched / peak
    return stitched, starts


def window_around(stitched: np.ndarray, clip_start: int) -> np.ndarray:
    """
    Extract a TARGET_LEN (10 s) window centred on the 2 s clip at position
    clip_start inside the stitched waveform. If the stitched recording is
    shorter than TARGET_LEN, loop it until it is long enough.
    """
    if len(stitched) >= TARGET_LEN:
        clip_centre = clip_start + SEG_LEN // 2
        start = clip_centre - TARGET_LEN // 2
        if start < 0:
            start = 0
        if start + TARGET_LEN > len(stitched):
            start = len(stitched) - TARGET_LEN
        return stitched[start:start + TARGET_LEN].astype(np.float32)
    # Short recording: loop until long enough, then trim
    n_repeats = int(np.ceil(TARGET_LEN / max(len(stitched), 1)))
    looped = np.tile(stitched, n_repeats)[:TARGET_LEN]
    return looped.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extractors (same set as features/extract_features.py,
#  with bumped FFT size where it helps)
# ---------------------------------------------------------------------------

def _mfcc_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=N_FFT_FRAME)
    feats = {}
    for i in range(20):
        feats[f"mfcc_mean_{i:02d}"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_std_{i:02d}"]  = float(np.std(mfcc[i]))
    return feats


def _spectral_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    feats = {}
    for name, fn in [
        ("centroid",  librosa.feature.spectral_centroid),
        ("rolloff",   librosa.feature.spectral_rolloff),
        ("bandwidth", librosa.feature.spectral_bandwidth),
    ]:
        arr = fn(y=y, sr=sr, n_fft=N_FFT_FRAME)
        feats[f"spec_{name}_mean"] = float(np.mean(arr))
        feats[f"spec_{name}_std"]  = float(np.std(arr))

    flat = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT_FRAME)
    feats["spec_flatness_mean"] = float(np.mean(flat))
    feats["spec_flatness_std"]  = float(np.std(flat))
    return feats


def _zcr_features(y: np.ndarray) -> dict:
    import librosa
    zcr = librosa.feature.zero_crossing_rate(y=y)
    return {"zcr_mean": float(np.mean(zcr)),
            "zcr_std":  float(np.std(zcr))}


def _hnr_feature(y: np.ndarray) -> dict:
    import librosa
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_harm ** 2))
    perc_energy = float(np.mean(y_perc ** 2))
    hnr = harm_energy / (perc_energy + 1e-8)
    return {"hnr_mean": float(np.log1p(hnr))}


def _chroma_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0, n_fft=N_FFT_FRAME)
    return {f"chroma_mean_{i:02d}": float(np.mean(chroma[i])) for i in range(12)}


def _contrast_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT_FRAME)
    return {f"spec_contrast_mean_{i}": float(np.mean(contrast[i])) for i in range(7)}


def _tonnetz_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    y_harm, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harm)) < 1e-6:
        y_harm = y
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, tuning=0.0, n_fft=N_FFT_FRAME)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    return {f"tonnetz_mean_{i}": float(np.mean(tonnetz[i])) for i in range(6)}


def _hps_features(y: np.ndarray, sr: int, n_harmonics: int = 5) -> dict:
    """Harmonic Product Spectrum — bumped to N_FFT_GLOBAL for finer resolution."""
    from scipy.signal import find_peaks

    D = np.abs(np.fft.rfft(y, n=N_FFT_GLOBAL)).astype(np.float64)
    freqs = np.fft.rfftfreq(N_FFT_GLOBAL, d=1.0 / sr)

    hps = D.copy()
    for h in range(2, n_harmonics + 1):
        downsampled = D[::h]
        length = min(len(hps), len(downsampled))
        hps[:length] *= downsampled[:length]

    f0_lo = int(np.searchsorted(freqs, 50))
    f0_hi = int(np.searchsorted(freqs, 1500))
    f0_hi = max(f0_hi, f0_lo + 1)
    f0_idx = np.argmax(hps[f0_lo:f0_hi]) + f0_lo
    f0 = float(freqs[f0_idx])

    min_height = np.max(D) * 0.01
    peaks, _ = find_peaks(D, height=min_height, distance=5)
    if len(peaks) == 0:
        peaks = np.argsort(D)[::-1][:20]

    sorted_peaks = peaks[np.argsort(D[peaks])[::-1]]
    top5_idx = sorted_peaks[:5]
    while len(top5_idx) < 5:
        top5_idx = np.append(top5_idx, 0)
    top5_freqs = freqs[top5_idx.astype(int)]

    f1 = top5_freqs[0] if top5_freqs[0] > 0 else 1.0
    ratios = [float(top5_freqs[i] / f1) if f1 > 0 else 0.0 for i in range(1, 5)]

    valid = np.sort(top5_freqs[top5_freqs > 0])
    if len(valid) > 1:
        spacings = np.diff(valid)
        mean_spacing = float(np.mean(spacings))
        std_spacing  = float(np.std(spacings))
    else:
        mean_spacing, std_spacing = 0.0, 0.0

    return {
        "hps_f0":               f0,
        "hps_peak_freq_1":      float(top5_freqs[0]),
        "hps_peak_freq_2":      float(top5_freqs[1]),
        "hps_peak_freq_3":      float(top5_freqs[2]),
        "hps_peak_freq_4":      float(top5_freqs[3]),
        "hps_peak_freq_5":      float(top5_freqs[4]),
        "hps_ratio_f2_f1":      ratios[0],
        "hps_ratio_f3_f1":      ratios[1],
        "hps_ratio_f4_f1":      ratios[2],
        "hps_ratio_f5_f1":      ratios[3],
        "hps_spacing_mean":     mean_spacing,
        "hps_spacing_std":      std_spacing,
    }


def _subband_energy(y: np.ndarray, n_bands: int = 4) -> dict:
    D = np.abs(np.fft.rfft(y, n=N_FFT_GLOBAL)) ** 2
    total = float(np.sum(D)) + 1e-8
    band_size = len(D) // n_bands
    feats = {}
    for i in range(n_bands):
        start = i * band_size
        end   = (i + 1) * band_size if i < n_bands - 1 else len(D)
        feats[f"subband_energy_ratio_{i}"] = float(np.sum(D[start:end]) / total)
    return feats


def extract_all_features(y: np.ndarray, sr: int) -> dict:
    feats: dict = {}
    feats.update(_mfcc_features(y, sr))
    feats.update(_spectral_features(y, sr))
    feats.update(_zcr_features(y))
    feats.update(_hnr_feature(y))
    feats.update(_chroma_features(y, sr))
    feats.update(_contrast_features(y, sr))
    feats.update(_tonnetz_features(y, sr))
    feats.update(_hps_features(y, sr))
    feats.update(_subband_energy(y))
    return feats


# ---------------------------------------------------------------------------
# Per-group processing (one stitch + many feature extractions)
# ---------------------------------------------------------------------------

def process_group(group_rows: list[dict], dataset_root: str | None) -> list[dict]:
    # Ensure deterministic order: sort by parsed segment index
    rows_sorted = sorted(
        group_rows,
        key=lambda r: parse_group_key(r["relpath"])[1],
    )
    results: list[dict] = []

    try:
        stitched, starts = stitch_rows(rows_sorted, dataset_root)
    except Exception:
        err = traceback.format_exc(limit=3)
        for row in rows_sorted:
            r = {col: row.get(col, "") for col in META_COLS}
            r["filepath"] = resolve_path(row, dataset_root)
            r["_ok"] = False
            r["_error"] = err
            results.append(r)
        return results

    for row, clip_start in zip(rows_sorted, starts):
        r = {col: row.get(col, "") for col in META_COLS}
        r["filepath"] = resolve_path(row, dataset_root)
        try:
            window = window_around(stitched, clip_start)
            feats = extract_all_features(window, TARGET_SR)
            r.update(feats)
            r["_ok"] = True
        except Exception:
            r["_ok"] = False
            r["_error"] = traceback.format_exc(limit=2)
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_split_csv(path: str, split_name: str) -> list[dict]:
    df = pd.read_csv(path)
    df["split"] = split_name
    return df.to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser(description="Extract 10 s stitched-context features for Stage B")
    ap.add_argument("--train_csv",    default="data/splits_stage2/train.csv")
    ap.add_argument("--val_csv",      default="data/splits_stage2/val.csv")
    ap.add_argument("--test_csv",     default="data/splits_stage2/test.csv")
    ap.add_argument("--dataset_root", default=None,
                    help="Root of Drone_Audio_Dataset/audio (uses relpath column).")
    ap.add_argument("--out",          default="features/stage2_features_10s.parquet")
    ap.add_argument("--n_jobs",       type=int, default=4)
    ap.add_argument("--splits",       default="train,val,test")
    args = ap.parse_args()

    requested = {s.strip() for s in args.splits.split(",")}
    split_map = {
        "train": args.train_csv,
        "val":   args.val_csv,
        "test":  args.test_csv,
    }

    all_rows: list[dict] = []
    for split_name, csv_path in split_map.items():
        if split_name not in requested:
            continue
        print(f"[INFO] Loading {split_name}: {csv_path}")
        rows = load_split_csv(csv_path, split_name)
        rows = [r for r in rows if r.get("binary_label") == "drone"]
        print(f"         {len(rows)} drone clips")
        all_rows.extend(rows)

    # Group by (motor_label, group_key) so we never stitch across motor classes
    # (even if two different classes happened to produce similar group_keys).
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in all_rows:
        gk, _ = parse_group_key(r["relpath"])
        key = (r.get("motor_label", ""), gk)
        groups.setdefault(key, []).append(r)

    print(f"\n[INFO] Total clips: {len(all_rows)}  over {len(groups)} stitch groups")
    multi = sum(1 for g in groups.values() if len(g) > 1)
    print(f"[INFO] Groups with >1 clip: {multi} / {len(groups)}")
    print(f"[INFO] Parallel workers: {args.n_jobs}")

    group_lists = list(groups.values())
    results_per_group = Parallel(n_jobs=args.n_jobs, verbose=5, prefer="threads")(
        delayed(process_group)(group, args.dataset_root) for group in group_lists
    )
    results = [row for group in results_per_group for row in group]

    df = pd.DataFrame(results)
    n_ok   = int(df["_ok"].sum())
    n_fail = int((~df["_ok"]).sum())
    print(f"\n[INFO] Succeeded: {n_ok}  Failed: {n_fail}")
    if n_fail > 0:
        for fp in df.loc[~df["_ok"], "filepath"].head(5).tolist():
            print(f"[WARN]   {fp}")

    df = df[df["_ok"]].drop(columns=["_ok", "_error"], errors="ignore").reset_index(drop=True)
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce").fillna(0).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\n[DONE] Saved {len(df)} rows x {len(df.columns)} cols -> {out_path}")
    print(f"       Feature cols: {len(df.columns) - len(META_COLS)}")
    print(f"       Split counts:\n{df['split'].value_counts().to_string()}")
    print(f"       Motor counts:\n{df['motor_label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
