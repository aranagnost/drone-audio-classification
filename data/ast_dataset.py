from __future__ import annotations

import csv
import hashlib
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import ASTFeatureExtractor

# Stitching constants - must match scripts/segment_audio.py
_BASE_SEG_SECONDS = 2.0
_STEP_SECONDS     = 1.5
_OVERLAP_SECONDS  = _BASE_SEG_SECONDS - _STEP_SECONDS    # 0.5 s
_SEG_INDEX_RE     = re.compile(r'_(\d+)_?\.wav$')


def _pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T > target_len:
        return wav[..., :target_len]
    return torch.nn.functional.pad(wav, (0, target_len - T))


def _parse_group_key(relpath: str) -> Tuple[str, int]:
    """
    Strip the segment index suffix from relpath to get a stitch-group key.
        2_motors/Ot3Em2lI9y8_r0_031.wav -> ("2_motors/Ot3Em2lI9y8_r0", 31)
        4_motors/mixed_2-bebop_001_.wav -> ("4_motors/mixed_2-bebop",  1)
    Clips with no index match become singleton groups.
    """
    m = _SEG_INDEX_RE.search(relpath)
    if not m:
        return relpath, 0
    return relpath[: m.start()], int(m.group(1))


class ASTAudioDataset(Dataset):
    """
    Dataset for AST fine-tuning. Uses HuggingFace ASTFeatureExtractor
    instead of the custom mel pipeline in AudioDataset.

    Returns (input_values, y, meta) where input_values has shape
    (time_frames, num_mel_bins) as expected by ASTForAudioClassification.

    NOTE: Do NOT use with eval.py - AST checkpoints have no 'cfg' key.
          Use train_ast.py --eval instead.
    """

    STAGE1_MAP = {"no_drone": 0, "drone": 1}
    STAGE2_MAP = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}

    def __init__(
        self,
        csv_path: str | Path,
        task: str,
        extractor: ASTFeatureExtractor,
        dataset_root: Optional[str] = None,
        min_quality: Optional[int] = None,
        max_no_drone_per_subtype: Optional[int] = None,
        exclude_subtypes: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        augment: bool = False,
        time_mask_max: int = 40,
        freq_mask_max: int = 20,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        context_seconds: float = 2.0,
    ):
        if task not in {"stage1", "stage2"}:
            raise ValueError("task must be 'stage1' or 'stage2'")
        if context_seconds < _BASE_SEG_SECONDS:
            raise ValueError(
                f"context_seconds must be >= {_BASE_SEG_SECONDS} (got {context_seconds})"
            )

        self.csv_path = Path(csv_path)
        self.task = task
        self.extractor = extractor
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.min_quality = min_quality
        self.exclude_subtypes = set(exclude_subtypes) if exclude_subtypes else None
        self.sample_rate = extractor.sampling_rate  # 16000
        self.context_seconds = float(context_seconds)
        self.target_len = int(self.sample_rate * self.context_seconds)
        self.base_seg_len = int(self.sample_rate * _BASE_SEG_SECONDS)
        self.overlap_len  = int(self.sample_rate * _OVERLAP_SECONDS)

        self.rows: List[Dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if task == "stage2":
                    if row["binary_label"] != "drone":
                        continue
                    if min_quality is not None:
                        try:
                            if int(row.get("quality", "0")) < min_quality:
                                continue
                        except ValueError:
                            continue
                elif task == "stage1":
                    if min_quality is not None and row.get("binary_label") == "drone":
                        try:
                            if int(row.get("quality", "0")) < min_quality:
                                continue
                        except (ValueError, TypeError):
                            pass
                    if (self.exclude_subtypes is not None
                            and row.get("binary_label") == "no_drone"
                            and row.get("subtype", "") in self.exclude_subtypes):
                        continue
                self.rows.append(row)

        if task == "stage1" and max_no_drone_per_subtype is not None:
            drone_rows = [r for r in self.rows if r.get("binary_label") == "drone"]
            no_drone_rows = [r for r in self.rows if r.get("binary_label") == "no_drone"]
            by_subtype: Dict[str, list] = {}
            for r in no_drone_rows:
                by_subtype.setdefault(r.get("subtype", ""), []).append(r)
            sampled: List[Dict[str, Any]] = []
            for sub, rows in by_subtype.items():
                sampled.extend(random.sample(rows, min(len(rows), max_no_drone_per_subtype)))
            self.rows = drone_rows + sampled
            random.shuffle(self.rows)

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        self.augment = augment
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks

        self.resampler_cache: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

        # Stitch-group setup - only needed when we want > 2 s context.
        # For stage2 we group by (motor_label, group_key); for stage1 drone
        # rows we also group (so we never stitch across classes). no_drone
        # rows are always left as singleton groups.
        self._stitch_enabled = self.context_seconds > _BASE_SEG_SECONDS
        self._row_to_group: List[Optional[Tuple[Tuple[str, str, str], int]]] = []
        self._group_rows:   Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
        self._stitched_cache_dir: Optional[Path] = None
        if self._stitch_enabled:
            self._build_stitch_groups()
            if self.cache_dir is not None:
                self._stitched_cache_dir = self.cache_dir / f"stitched_{self.context_seconds:g}s"
                self._stitched_cache_dir.mkdir(parents=True, exist_ok=True)

        # TTA state: when non-zero, every __getitem__ shifts the 10 s window
        # by this many samples. Feature cache is bypassed for non-zero offsets
        # (the cache key doesn't know about the offset, so reusing it would
        # silently give wrong data).
        self._tta_offset_samples = 0

    def _build_stitch_groups(self) -> None:
        """
        Populate _group_rows and _row_to_group.

        The group key is (motor_label, youtube_url, group_key_root):
        - motor_label keeps classes apart even if URLs somehow collide.
        - youtube_url prevents cross-split collisions - our splits are
          URL-disjoint (verified), so train/val/test can never share a
          stitch group or a cache file.
        - group_key_root lets us split a single long recording into
          separate time-ranges (e.g., `<videoID>_r0_...` vs `<videoID>_r1_...`)
          so we never stitch across unrelated sections.

        Local files (non-YouTube) have a unique youtube_url per file in
        this dataset (it stores the absolute local path), so they
        automatically become singleton groups - the stitched waveform
        is a single 2 s clip that will be looped to fill the window.
        """
        tmp: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = defaultdict(list)

        for i, row in enumerate(self.rows):
            relpath = row.get("relpath", "")
            binary  = row.get("binary_label", "")
            motor   = row.get("motor_label", "") or binary
            url     = row.get("youtube_url", "") or f"__no_url__::{relpath}"
            if binary == "no_drone":
                # Singleton group - stitching across noise clips is meaningless.
                key = (f"no_drone::{relpath}", "", "")
                self._group_rows[key] = [(0, i)]
                continue
            group_key_root, seg_idx = _parse_group_key(relpath)
            key = (motor, url, group_key_root)
            tmp[key].append((seg_idx, i))

        for key, lst in tmp.items():
            lst.sort()                                # sort by seg_idx
            self._group_rows[key] = lst

        self._row_to_group = [None] * len(self.rows)
        for key, lst in self._group_rows.items():
            for pos, (_seg_idx, row_idx) in enumerate(lst):
                self._row_to_group[row_idx] = (key, pos)

    def __len__(self) -> int:
        return len(self.rows)

    def _row_filepath(self, row: Dict[str, Any]) -> str:
        return (str(self.dataset_root / row["relpath"])
                if self.dataset_root else row["filepath"])

    def _load_wav(self, filepath: str) -> torch.Tensor:
        wav, sr = torchaudio.load(filepath)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self.resampler_cache:
                self.resampler_cache[key] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self.resampler_cache[key](wav)
        return _pad_or_trim(wav, self.target_len)

    def _load_wav_fixed(self, filepath: str, target_len: int) -> torch.Tensor:
        """Like _load_wav but pads/trims to an explicit length."""
        wav, sr = torchaudio.load(filepath)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            key = (sr, self.sample_rate)
            if key not in self.resampler_cache:
                self.resampler_cache[key] = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = self.resampler_cache[key](wav)
        return _pad_or_trim(wav, target_len)

    def _stitched_cache_path(self, key: Tuple[str, str, str]) -> Path:
        h = hashlib.md5("|".join(key).encode()).hexdigest()
        return self._stitched_cache_dir / f"{h}.pt"   # caller checks cache dir != None

    def _compute_clip_starts(self, group_size: int) -> List[int]:
        """
        For a group of N consecutive 2-second clips with 1.5 s step (0.5 s
        overlap), return the sample-index at which each clip's content
        begins in the stitched waveform.
        """
        starts = [0]
        cum = self.base_seg_len
        for _ in range(1, group_size):
            starts.append(cum)
            cum += self.base_seg_len - self.overlap_len
        return starts

    def _build_stitched_waveform(self, key: Tuple[str, str, str]) -> np.ndarray:
        """Load all siblings of `key`, stitch with overlap removal, peak-normalise."""
        members = self._group_rows[key]
        parts: List[np.ndarray] = []
        for i, (_seg_idx, row_idx) in enumerate(members):
            row = self.rows[row_idx]
            wav = self._load_wav_fixed(self._row_filepath(row),
                                       target_len=self.base_seg_len)
            y = wav.squeeze(0).numpy()
            if i == 0:
                parts.append(y)
            else:
                parts.append(y[self.overlap_len:])
        stitched = (np.concatenate(parts).astype(np.float32)
                    if parts else np.zeros(0, dtype=np.float32))
        peak = float(np.max(np.abs(stitched))) if len(stitched) else 0.0
        if peak > 1e-6:
            stitched = stitched / peak
        return stitched

    def _get_stitched_waveform(self, key: Tuple[str, str, str]) -> np.ndarray:
        """Load stitched waveform from disk cache if present; else build + save."""
        if self._stitched_cache_dir is not None:
            cache_path = self._stitched_cache_path(key)
            if cache_path.exists():
                try:
                    return torch.load(cache_path, weights_only=True).numpy().astype(np.float32)
                except Exception:
                    pass    # fall through to rebuild

        stitched = self._build_stitched_waveform(key)

        if self._stitched_cache_dir is not None:
            cache_path = self._stitched_cache_path(key)
            tmp = cache_path.with_suffix(".tmp")
            try:
                torch.save(torch.from_numpy(stitched).half(), tmp)
                os.replace(tmp, cache_path)
            except (OSError, RuntimeError):
                # Disk-full or other I/O failure - drop the cache write but
                # still return the in-memory waveform. Cleanup partial tmp.
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        return stitched

    def prestitch_groups(self, verbose: bool = False,
                         progress_every: int = 50) -> None:
        """
        Pre-build (and cache) the stitched waveform for every group. Must be
        called BEFORE the DataLoader spawns workers so that workers only ever
        read from the cache - avoids both redundant work and write races.

        Cheap no-op if the cache is already fully populated.
        """
        if not self._stitch_enabled or self._stitched_cache_dir is None:
            return
        keys = list(self._group_rows.keys())
        n_cached = n_built = 0
        for i, key in enumerate(keys, start=1):
            if self._stitched_cache_path(key).exists():
                n_cached += 1
            else:
                self._get_stitched_waveform(key)
                n_built += 1
            if verbose and (i % progress_every == 0 or i == len(keys)):
                size = len(self._group_rows[key])
                print(f"  prestitch {i:>5}/{len(keys)}  "
                      f"(cached={n_cached}, built={n_built}, "
                      f"latest_group_size={size})", flush=True)
        if verbose:
            print(f"[prestitch] done: {len(keys)} groups "
                  f"({n_cached} already cached, {n_built} newly built)",
                  flush=True)

    def _stitched_wav_for_idx(self, idx: int,
                              offset_samples: int = 0) -> torch.Tensor:
        """
        Read the cached stitched waveform for row `idx`'s group and extract a
        context_seconds window centred on the clip's position. Optional
        `offset_samples` shifts the window by that many samples (useful for
        test-time augmentation). If the stitched recording is shorter than
        target_len, loop it; the offset is applied as a circular roll on the
        looped sequence.
        """
        group_info = self._row_to_group[idx]
        if group_info is None:
            row = self.rows[idx]
            return self._load_wav(self._row_filepath(row))

        key, pos = group_info
        stitched = self._get_stitched_waveform(key)
        group_size = len(self._group_rows[key])
        clip_start = self._compute_clip_starts(group_size)[pos]

        if len(stitched) >= self.target_len:
            clip_centre = clip_start + self.base_seg_len // 2
            start = clip_centre - self.target_len // 2 + offset_samples
            start = max(0, min(start, len(stitched) - self.target_len))
            window = stitched[start:start + self.target_len]
        elif len(stitched) == 0:
            window = np.zeros(self.target_len, dtype=np.float32)
        else:
            n_repeats = int(np.ceil(self.target_len / len(stitched)))
            looped = np.tile(stitched, n_repeats)
            if offset_samples != 0:
                looped = np.roll(looped, offset_samples)
            window = looped[: self.target_len]

        return torch.from_numpy(window.astype(np.float32)).unsqueeze(0)

    def _specaugment(self, x: torch.Tensor) -> torch.Tensor:
        """SpecAugment: random time and frequency masking (in-place on a clone)."""
        T, F = x.shape
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_max, T - 1))
            t0 = random.randint(0, T - t)
            x[t0:t0 + t, :] = 0.0
        for _ in range(self.n_freq_masks):
            f = random.randint(0, min(self.freq_mask_max, F - 1))
            f0 = random.randint(0, F - f)
            x[:, f0:f0 + f] = 0.0
        return x

    def _compute_features(self, idx: int) -> torch.Tensor:
        if self._stitch_enabled:
            wav = self._stitched_wav_for_idx(idx, offset_samples=self._tta_offset_samples)
        else:
            row = self.rows[idx]
            wav = self._load_wav(self._row_filepath(row))
        inputs = self.extractor(
            wav.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=False,
        )
        return inputs.input_values.squeeze(0)  # (time_frames, num_mel_bins)

    def _try_save_feature_cache(self, x: torch.Tensor, cache_file: Path) -> None:
        """Write per-clip AST features to disk. On I/O failure (typically
        disk-full on Kaggle), drop the cache silently and mark this worker's
        feature cache as disabled so we don't hammer the disk for the rest
        of the run. Subsequent items still load any features that were
        already cached before the failure."""
        tmp = cache_file.with_suffix(".tmp")
        try:
            torch.save(x.half(), tmp)
            os.replace(tmp, cache_file)
        except (OSError, RuntimeError):
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            self._feature_cache_full = True

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fp = self._row_filepath(row)

        # Only use the AST-feature cache when the TTA offset is zero. The
        # cache key does not include the offset, so reusing cached features
        # with a non-zero offset would silently return wrong data.
        use_cache = (self.cache_dir is not None
                     and self._tta_offset_samples == 0
                     and not getattr(self, "_feature_cache_full", False))

        if use_cache:
            # Include context_seconds in the cache key so 2 s and 10 s features
            # do not collide when the same directory is reused.
            cache_key = f"{fp}|ctx={self.context_seconds}"
            h = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = self.cache_dir / f"{h}.pt"
            if cache_file.exists():
                try:
                    x = torch.load(cache_file, weights_only=True).float()
                except Exception:
                    x = self._compute_features(idx)
                    self._try_save_feature_cache(x, cache_file)
            else:
                x = self._compute_features(idx)
                self._try_save_feature_cache(x, cache_file)
        else:
            x = self._compute_features(idx)

        if self.augment:
            x = self._specaugment(x.clone())

        if self.task == "stage1":
            y = self.STAGE1_MAP[row["binary_label"]]
        else:
            y = self.STAGE2_MAP[row["motor_label"]]

        meta = {
            "filepath": fp,
            "relpath": row.get("relpath", ""),
            "binary_label": row.get("binary_label", ""),
            "motor_label": row.get("motor_label", ""),
            "subtype": row.get("subtype", ""),
            "quality": row.get("quality", ""),
            "youtube_url": row.get("youtube_url", ""),
        }
        return x, int(y), meta
