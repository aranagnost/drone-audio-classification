# data/ast_dataset.py
from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import torch
from torch.utils.data import Dataset
import torchaudio
from transformers import ASTFeatureExtractor


def _pad_or_trim(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    T = wav.shape[-1]
    if T == target_len:
        return wav
    if T > target_len:
        return wav[..., :target_len]
    return torch.nn.functional.pad(wav, (0, target_len - T))


class ASTAudioDataset(Dataset):
    """
    Dataset for AST fine-tuning. Uses HuggingFace ASTFeatureExtractor
    instead of the custom mel pipeline in AudioDataset.

    Returns (input_values, y, meta) where input_values has shape
    (time_frames, num_mel_bins) as expected by ASTForAudioClassification.

    NOTE: Do NOT use with eval.py — AST checkpoints have no 'cfg' key.
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
    ):
        if task not in {"stage1", "stage2"}:
            raise ValueError("task must be 'stage1' or 'stage2'")

        self.csv_path = Path(csv_path)
        self.task = task
        self.extractor = extractor
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.min_quality = min_quality
        self.exclude_subtypes = set(exclude_subtypes) if exclude_subtypes else None
        self.sample_rate = extractor.sampling_rate  # 16000
        self.target_len = int(self.sample_rate * 2.0)  # 2-second clips

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

        self.resampler_cache: Dict[Tuple[int, int], torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.rows)

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

    def _compute_features(self, fp: str) -> torch.Tensor:
        wav = self._load_wav(fp)
        inputs = self.extractor(
            wav.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=False,
        )
        return inputs.input_values.squeeze(0)  # (time_frames, num_mel_bins)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        fp = str(self.dataset_root / row["relpath"]) if self.dataset_root else row["filepath"]

        if self.cache_dir is not None:
            h = hashlib.md5(fp.encode()).hexdigest()
            cache_file = self.cache_dir / f"{h}.pt"
            if cache_file.exists():
                try:
                    x = torch.load(cache_file, weights_only=True).float()
                except Exception:
                    x = self._compute_features(fp)
                    tmp = cache_file.with_suffix(".tmp")
                    torch.save(x.half(), tmp)
                    os.replace(tmp, cache_file)
            else:
                x = self._compute_features(fp)
                tmp = cache_file.with_suffix(".tmp")
                torch.save(x.half(), tmp)
                os.replace(tmp, cache_file)
        else:
            x = self._compute_features(fp)

        if self.task == "stage1":
            y = self.STAGE1_MAP[row["binary_label"]]
        else:
            y = self.STAGE2_MAP[row["motor_label"]]

        meta = {
            "filepath": fp,
            "binary_label": row.get("binary_label", ""),
            "motor_label": row.get("motor_label", ""),
            "subtype": row.get("subtype", ""),
            "quality": row.get("quality", ""),
            "youtube_url": row.get("youtube_url", ""),
        }
        return x, int(y), meta
