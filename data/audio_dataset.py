# data/audio_dataset.py
from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
import torchaudio


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    clip_seconds: float = 2.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 160  # 10ms at 16k
    win_length: int = 400  # 25ms at 16k
    f_min: int = 20
    f_max: Optional[int] = None  # if None, sr//2


def _pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    """wav: (1, T)"""
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T > target_len:
        return wav[..., :target_len]
    # pad
    pad = target_len - T
    return torch.nn.functional.pad(wav, (0, pad))


class AudioDataset(Dataset):
    """
    Reads a split CSV (train/val/test) produced earlier and returns:
      x: (1, n_mels, time) log-mel spectrogram
      y: integer label (binary or motor-count)
      meta: dict with filepath, quality, subtype, etc.
    """

    def __init__(
        self,
        csv_path: str | Path,
        task: str,
        cfg: AudioConfig = AudioConfig(),
        quality_weighting: bool = False,
        min_quality: Optional[int] = None,
        max_no_drone_per_subtype: Optional[int] = None,
        exclude_subtypes: Optional[List[str]] = None,
        dataset_root: Optional[str] = None,
    ):
        """
        task:
          - "stage1": binary_label -> {no_drone:0, drone:1}
          - "stage2": motor_label -> {2_motors:0, 4_motors:1, 6_motors:2, 8_motors:3}
                    (rows with binary_label != drone are ignored)

        quality_weighting:
          If True, returns an extra per-sample weight in meta["weight"] based on quality (q1..q5).
          Only meaningful for stage2 (drone clips).
        min_quality:
          If set (e.g., 3), keep only drone clips with quality >= min_quality (stage2 only).
        """
        self.csv_path = Path(csv_path)
        self.task = task
        self.cfg = cfg
        self.quality_weighting = quality_weighting
        self.min_quality = min_quality
        self.max_no_drone_per_subtype = max_no_drone_per_subtype
        self.exclude_subtypes = set(exclude_subtypes) if exclude_subtypes else None
        self.dataset_root = Path(dataset_root) if dataset_root else None

        if task not in {"stage1", "stage2", "stage2_coarse", "stage3"}:
            raise ValueError("task must be 'stage1', 'stage2', 'stage2_coarse', or 'stage3'")

        self.rows: List[Dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.task in {"stage2", "stage2_coarse"}:
                    if row["binary_label"] != "drone":
                        continue
                    q = row.get("quality", "")
                    if self.min_quality is not None:
                        try:
                            qi = int(q)
                        except Exception:
                            continue
                        if qi < self.min_quality:
                            continue
                elif self.task == "stage3":
                    if row["binary_label"] != "drone":
                        continue
                    if row.get("motor_label", "") not in {"4_motors", "6_motors"}:
                        continue
                    q = row.get("quality", "")
                    if self.min_quality is not None:
                        try:
                            qi = int(q)
                        except Exception:
                            continue
                        if qi < self.min_quality:
                            continue
                elif self.task == "stage1":
                    if self.min_quality is not None and row.get("binary_label") == "drone":
                        try:
                            if int(row.get("quality", "0")) < self.min_quality:
                                continue
                        except (ValueError, TypeError):
                            pass
                    if (self.exclude_subtypes is not None
                            and row.get("binary_label") == "no_drone"
                            and row.get("subtype", "") in self.exclude_subtypes):
                        continue
                self.rows.append(row)

        if self.task == "stage1" and self.max_no_drone_per_subtype is not None:
            drone_rows = [r for r in self.rows if r.get("binary_label") == "drone"]
            no_drone_rows = [r for r in self.rows if r.get("binary_label") == "no_drone"]
            by_subtype: Dict[str, list] = {}
            for r in no_drone_rows:
                sub = r.get("subtype", "")
                by_subtype.setdefault(sub, []).append(r)
            sampled: List[Dict[str, Any]] = []
            for sub, rows in by_subtype.items():
                if len(rows) > self.max_no_drone_per_subtype:
                    sampled.extend(random.sample(rows, self.max_no_drone_per_subtype))
                else:
                    sampled.extend(rows)
            self.rows = drone_rows + sampled
            random.shuffle(self.rows)

        # Pre-create transforms (fast & consistent)
        self.resampler_cache: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            n_mels=cfg.n_mels,
            power=2.0,
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

        self.target_len = int(cfg.sample_rate * cfg.clip_seconds)

        # Label maps
        self.stage1_map = {"no_drone": 0, "drone": 1}
        self.stage2_map = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}
        # stage2_coarse: merge 4+6 into a single class
        self.stage2_coarse_map = {"2_motors": 0, "4_motors": 1, "6_motors": 1, "8_motors": 2}
        # stage3: binary fine-grained 4 vs 6
        self.stage3_map = {"4_motors": 0, "6_motors": 1}

    def __len__(self) -> int:
        return len(self.rows)

    def _load_wav(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load(filepath)  # (C,T)
        if wav.ndim != 2:
            raise ValueError(f"Unexpected wav shape: {wav.shape} for {filepath}")

        # convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # resample if needed
        if sr != self.cfg.sample_rate:
            key = (sr, self.cfg.sample_rate)
            if key not in self.resampler_cache:
                self.resampler_cache[key] = torchaudio.transforms.Resample(sr, self.cfg.sample_rate)
            wav = self.resampler_cache[key](wav)

        # fix length
        wav = _pad_or_trim(wav, self.target_len)
        return wav

    def _wav_to_logmel(self, wav: torch.Tensor) -> torch.Tensor:
        mel = self.mel(wav)             # (1, n_mels, time)
        logmel = self.amp_to_db(mel)    # (1, n_mels, time)
        # Per-sample normalization (helps stability)
        mean = logmel.mean()
        std = logmel.std().clamp_min(1e-6)
        logmel = (logmel - mean) / std
        return logmel

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fp = str(self.dataset_root / row["relpath"]) if self.dataset_root else row["filepath"]
        wav = self._load_wav(fp)
        x = self._wav_to_logmel(wav)

        if self.task == "stage1":
            y = self.stage1_map[row["binary_label"]]
            weight = 1.0
        elif self.task == "stage2":
            y = self.stage2_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0
        elif self.task == "stage2_coarse":
            y = self.stage2_coarse_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0
        else:  # stage3
            y = self.stage3_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0

        meta = {
            "filepath": fp,
            "binary_label": row.get("binary_label", ""),
            "motor_label": row.get("motor_label", ""),
            "subtype": row.get("subtype", ""),
            "quality": row.get("quality", ""),
            "youtube_url": row.get("youtube_url", ""),
            "weight": float(weight),
        }
        return x, int(y), meta


# ──────────────────── noise-mixing utilities ──────────────────────────────────

# Default weights for sampling noise subcategories.
# Speech and wind are upweighted because they are the most common real-world
# contaminants in drone recordings (commentary, outdoor wind).
DEFAULT_NOISE_WEIGHTS: Dict[str, float] = {
    "speech": 3.0,
    "wind": 3.0,
    "crowd": 4.0,      # hard FP subtype — upweighted to compensate for small pool
    "airplanes": 3.0,  # hard FP subtype
    "insects": 4.0,    # hard FP subtype — spectrally close to drones
    "cars": 1.0,
    "electronics": 1.0,
    "motors": 1.0,
    "random": 2.0,     # hard FP subtype
    "birds": 1.0,
}


def _mix_at_snr(signal: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix *noise* into *signal* at the given SNR (in dB).  Both are (1, T)."""
    sig_power = signal.pow(2).mean().clamp_min(1e-10)
    noise_power = noise.pow(2).mean().clamp_min(1e-10)
    # scale noise so that 10*log10(sig_power / scaled_noise_power) == snr_db
    scale = (sig_power / (noise_power * 10 ** (snr_db / 10))).sqrt()
    return signal + scale * noise


def _build_noise_pool(
    csv_path: str | Path,
    subtypes: Optional[List[str]] = None,
    exclude_subtypes: Optional[set] = None,
    noise_weights: Optional[Dict[str, float]] = None,
    dataset_root: Optional[Path] = None,
) -> Tuple[List[str], List[float]]:
    """Scan a split CSV and return (filepaths, sampling_weights) for not_a_drone rows."""
    weights_map = noise_weights or DEFAULT_NOISE_WEIGHTS
    paths: List[str] = []
    weights: List[float] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "no_drone":
                continue
            sub = row.get("subtype", "")
            if subtypes and sub not in subtypes:
                continue
            if exclude_subtypes and sub in exclude_subtypes:
                continue
            fp = str(dataset_root / row["relpath"]) if dataset_root else row["filepath"]
            if not Path(fp).exists():
                continue
            paths.append(fp)
            weights.append(weights_map.get(sub, 1.0))
    return paths, weights


class AugmentedAudioDataset(AudioDataset):
    """AudioDataset with waveform-level augmentations for noise-robust training.

    Augmentations (applied only when ``augment=True``):
      1. **Noise mixing** – overlay a random not_a_drone sample at a random SNR.
         Speech and wind are sampled more often (configurable weights).
      2. **Gain jitter** – random volume perturbation in dB.

    The augmentations happen *before* mel-spectrogram computation, so the model
    learns to extract drone features from noisy spectrograms.
    """

    def __init__(
        self,
        csv_path: str | Path,
        task: str,
        cfg: AudioConfig = AudioConfig(),
        *,
        augment: bool = True,
        noise_csv: Optional[str | Path] = None,
        noise_mix_prob: float = 0.7,
        snr_range: Tuple[float, float] = (0.0, 20.0),
        noise_subtypes: Optional[List[str]] = None,
        noise_weights: Optional[Dict[str, float]] = None,
        gain_jitter_db: float = 6.0,
        **kwargs,
    ):
        super().__init__(csv_path, task, cfg, **kwargs)
        self.augment = augment
        self.noise_mix_prob = noise_mix_prob
        self.snr_range = snr_range
        self.gain_jitter_db = gain_jitter_db

        # Build noise pool.  Use noise_csv if provided (important when
        # csv_path is a drone-only split like splits_stage2/ that has no
        # not_a_drone rows).  Falls back to csv_path otherwise.
        self.noise_paths: List[str] = []
        self.noise_sample_weights: List[float] = []
        if augment:
            noise_src = noise_csv or csv_path
            self.noise_paths, self.noise_sample_weights = _build_noise_pool(
                noise_src, subtypes=noise_subtypes, noise_weights=noise_weights,
                exclude_subtypes=self.exclude_subtypes,
                dataset_root=self.dataset_root,
            )

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fp = str(self.dataset_root / row["relpath"]) if self.dataset_root else row["filepath"]
        wav = self._load_wav(fp)

        # ── waveform augmentations (training only) ──
        if self.augment:
            # Noise mixing: only for drone samples (don't add drone noise to no_drone)
            is_drone = row.get("binary_label") == "drone"
            if is_drone and self.noise_paths and random.random() < self.noise_mix_prob:
                noise_fp = random.choices(self.noise_paths, weights=self.noise_sample_weights, k=1)[0]
                try:
                    noise_wav = self._load_wav(noise_fp)
                    snr_db = random.uniform(*self.snr_range)
                    wav = _mix_at_snr(wav, noise_wav, snr_db)
                except Exception:
                    pass  # skip augmentation on load failure

            # Gain jitter (apply to all samples)
            if self.gain_jitter_db > 0:
                gain_db = random.uniform(-self.gain_jitter_db, self.gain_jitter_db)
                wav = wav * (10 ** (gain_db / 20))

        x = self._wav_to_logmel(wav)

        if self.task == "stage1":
            y = self.stage1_map[row["binary_label"]]
            weight = 1.0
        elif self.task == "stage2":
            y = self.stage2_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0
        elif self.task == "stage2_coarse":
            y = self.stage2_coarse_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0
        else:  # stage3
            y = self.stage3_map[row["motor_label"]]
            if self.quality_weighting:
                try:
                    q = int(row.get("quality", 5))
                    q = max(1, min(5, q))
                    weight = q / 5.0
                except Exception:
                    weight = 1.0
            else:
                weight = 1.0

        meta = {
            "filepath": fp,
            "binary_label": row.get("binary_label", ""),
            "motor_label": row.get("motor_label", ""),
            "subtype": row.get("subtype", ""),
            "quality": row.get("quality", ""),
            "youtube_url": row.get("youtube_url", ""),
            "weight": float(weight),
        }
        return x, int(y), meta
