"""
Extract handcrafted acoustic features from audio clips.

Produces a single parquet (or CSV) file with one row per clip containing:
  - 40  MFCCs (20 coefficients × mean + std)
  - 8   Spectral: centroid, rolloff, bandwidth, flatness (mean + std each)
  - 2   Zero-crossing rate (mean + std)
  - 1   Harmonic-to-noise ratio (mean)
  - 12  Chroma (12 bins × mean)
  - 7   Spectral contrast (7 bands × mean)
  - 6   Tonnetz (6 dims × mean)
  - 12  HPS: f0, top-5 peak freqs, 4 ratios (fi/f1), spacing mean + std
  - 4   Sub-band energy ratios (4 equal bands)
  Total: 92 acoustic features

  Plus metadata columns: filepath, relpath, motor_label, binary_label,
                         quality, youtube_url, split

Usage:
    python features/extract_features.py \\
        --train_csv data/splits_stage2/train.csv \\
        --val_csv   data/splits_stage2/val.csv \\
        --test_csv  data/splits_stage2/test.csv \\
        --dataset_root /path/to/Drone_Audio_Dataset/audio \\
        --out features/stage2_features.parquet \\
        --n_jobs 4
"""
from __future__ import annotations

import argparse
import os
import traceback
import warnings
from pathlib import Path

# Disable numba JIT before any librosa import — chroma_stft calls estimate_tuning()
# which uses numba and segfaults inside parallel workers (both loky and threading).
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

TARGET_SR = 16_000
DURATION  = 2.0
TARGET_LEN = int(TARGET_SR * DURATION)   # 32 000 samples
N_FFT     = 2048

META_COLS = ["filepath", "relpath", "motor_label", "binary_label",
             "quality", "youtube_url", "split"]


# ---------------------------------------------------------------------------
# Path resolution (matches existing audio_dataset.py convention)
# ---------------------------------------------------------------------------

def resolve_path(row: dict, dataset_root: str | None) -> str:
    if dataset_root:
        return os.path.join(dataset_root, row["relpath"])
    return row["filepath"]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(fp: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(fp, sr=TARGET_SR, mono=True, duration=DURATION + 0.1)
    if len(y) < TARGET_LEN:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    else:
        y = y[:TARGET_LEN]
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Individual feature extractors
# ---------------------------------------------------------------------------

def _mfcc_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
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
        arr = fn(y=y, sr=sr)
        feats[f"spec_{name}_mean"] = float(np.mean(arr))
        feats[f"spec_{name}_std"]  = float(np.std(arr))

    flat = librosa.feature.spectral_flatness(y=y)
    feats["spec_flatness_mean"] = float(np.mean(flat))
    feats["spec_flatness_std"]  = float(np.std(flat))
    return feats


def _zcr_features(y: np.ndarray) -> dict:
    import librosa
    zcr = librosa.feature.zero_crossing_rate(y=y)
    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std":  float(np.std(zcr)),
    }


def _hnr_feature(y: np.ndarray) -> dict:
    import librosa
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = float(np.mean(y_harm ** 2))
    perc_energy = float(np.mean(y_perc ** 2))
    hnr = harm_energy / (perc_energy + 1e-8)
    return {"hnr_mean": float(np.log1p(hnr))}   # log scale for stability


def _chroma_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    # tuning=0.0 skips estimate_tuning() which uses numba and segfaults in forked workers
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, tuning=0.0)
    return {f"chroma_mean_{i:02d}": float(np.mean(chroma[i])) for i in range(12)}


def _contrast_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return {f"spec_contrast_mean_{i}": float(np.mean(contrast[i])) for i in range(7)}


def _tonnetz_features(y: np.ndarray, sr: int) -> dict:
    import librosa
    y_harm, _ = librosa.effects.hpss(y)
    if np.max(np.abs(y_harm)) < 1e-6:
        y_harm = y     # fallback if harmonic component is silent
    # Pass pre-computed chroma to avoid tonnetz calling CQT internally (uses numba → segfault)
    chroma = librosa.feature.chroma_stft(y=y_harm, sr=sr, tuning=0.0)
    tonnetz = librosa.feature.tonnetz(chroma=chroma)
    return {f"tonnetz_mean_{i}": float(np.mean(tonnetz[i])) for i in range(6)}


def _hps_features(y: np.ndarray, sr: int, n_harmonics: int = 5) -> dict:
    """Harmonic Product Spectrum: estimate f0, extract top-5 peaks and their ratios."""
    from scipy.signal import find_peaks

    # Magnitude spectrum
    D = np.abs(np.fft.rfft(y, n=N_FFT)).astype(np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)

    # Build HPS by downsampling and multiplying
    hps = D.copy()
    for h in range(2, n_harmonics + 1):
        downsampled = D[::h]
        length = min(len(hps), len(downsampled))
        hps[:length] *= downsampled[:length]

    # f0: search between 50 Hz and 1500 Hz (covers drone motor fundamentals)
    f0_lo = int(np.searchsorted(freqs, 50))
    f0_hi = int(np.searchsorted(freqs, 1500))
    f0_hi = max(f0_hi, f0_lo + 1)
    f0_idx = np.argmax(hps[f0_lo:f0_hi]) + f0_lo
    f0 = float(freqs[f0_idx])

    # Top-5 peaks from the raw magnitude spectrum
    min_height = np.max(D) * 0.01
    peaks, _ = find_peaks(D, height=min_height, distance=5)
    if len(peaks) == 0:
        peaks = np.argsort(D)[::-1][:20]

    sorted_peaks = peaks[np.argsort(D[peaks])[::-1]]
    top5_idx = sorted_peaks[:5]
    # Pad to 5 if fewer peaks found
    while len(top5_idx) < 5:
        top5_idx = np.append(top5_idx, 0)
    top5_freqs = freqs[top5_idx.astype(int)]

    # Frequency ratios fi / f1
    f1 = top5_freqs[0] if top5_freqs[0] > 0 else 1.0
    ratios = [float(top5_freqs[i] / f1) if f1 > 0 else 0.0 for i in range(1, 5)]

    # Peak spacing statistics (sorted unique positive frequencies)
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
    D = np.abs(np.fft.rfft(y, n=N_FFT)) ** 2
    total = float(np.sum(D)) + 1e-8
    band_size = len(D) // n_bands
    feats = {}
    for i in range(n_bands):
        start = i * band_size
        end   = (i + 1) * band_size if i < n_bands - 1 else len(D)
        feats[f"subband_energy_ratio_{i}"] = float(np.sum(D[start:end]) / total)
    return feats


# ---------------------------------------------------------------------------
# Per-row extraction (called in parallel)
# ---------------------------------------------------------------------------

def extract_row(row: dict, dataset_root: str | None) -> dict:
    fp = resolve_path(row, dataset_root)
    result = {col: row.get(col, "") for col in META_COLS}
    result["filepath"] = fp

    try:
        y = load_audio(fp)
        sr = TARGET_SR

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
        result.update(feats)
        result["_ok"] = True

    except Exception:
        result["_ok"] = False
        result["_error"] = traceback.format_exc(limit=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_split_csv(path: str, split_name: str) -> list[dict]:
    df = pd.read_csv(path)
    df["split"] = split_name
    return df.to_dict(orient="records")


def main():
    ap = argparse.ArgumentParser(description="Extract acoustic features for Stage B classifier")
    ap.add_argument("--train_csv",    default="data/splits_stage2/train.csv")
    ap.add_argument("--val_csv",      default="data/splits_stage2/val.csv")
    ap.add_argument("--test_csv",     default="data/splits_stage2/test.csv")
    ap.add_argument("--dataset_root", default=None,
                    help="Root of Drone_Audio_Dataset/audio (uses relpath column). "
                         "If omitted, uses the absolute filepath column.")
    ap.add_argument("--out",          default="features/stage2_features.parquet",
                    help="Output path (.parquet or .csv)")
    ap.add_argument("--n_jobs",       type=int, default=4,
                    help="Parallel workers (joblib). Use 1 to disable parallelism.")
    ap.add_argument("--splits",       default="train,val,test",
                    help="Comma-separated splits to process (default: train,val,test)")
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
        # Stage 2: keep only drone rows
        rows = [r for r in rows if r.get("binary_label") == "drone"]
        print(f"         {len(rows)} drone clips")
        all_rows.extend(rows)

    print(f"\n[INFO] Total clips to process: {len(all_rows)}")
    print(f"[INFO] Parallel workers: {args.n_jobs}")

    results = Parallel(n_jobs=args.n_jobs, verbose=5, prefer="threads")(
        delayed(extract_row)(row, args.dataset_root)
        for row in all_rows
    )

    df = pd.DataFrame(results)
    n_ok   = df["_ok"].sum()
    n_fail = (~df["_ok"]).sum()
    print(f"\n[INFO] Succeeded: {n_ok}  Failed: {n_fail}")

    if n_fail > 0:
        failed_paths = df.loc[~df["_ok"], "filepath"].tolist()
        print(f"[WARN] First 5 failures:")
        for fp in failed_paths[:5]:
            print(f"       {fp}")

    # Drop internal columns, keep only successful rows
    df = df[df["_ok"]].drop(columns=["_ok", "_error"], errors="ignore")
    df = df.reset_index(drop=True)

    # Ensure quality is numeric
    df["quality"] = pd.to_numeric(df["quality"], errors="coerce").fillna(0).astype(int)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    print(f"\n[DONE] Saved {len(df)} rows × {len(df.columns)} columns -> {out_path}")
    print(f"       Feature columns: {len(df.columns) - len(META_COLS)}")
    print(f"       Split counts:\n{df['split'].value_counts().to_string()}")
    print(f"       Motor counts:\n{df['motor_label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
