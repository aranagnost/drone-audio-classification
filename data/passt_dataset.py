"""PaSST dataset adapter: reuses ASTAudioDataset's stitching and group cache,
but outputs raw 32 kHz waveform tensors (PaSST does its own mel extraction).

The parent class internally works at 16 kHz (the stitched-waveform cache is
saved at 16 kHz). We resample to 32 kHz at __getitem__ time so the cache is
shared across AST and PaSST runs.
"""
from __future__ import annotations

import torch
import torchaudio

from data.ast_dataset import ASTAudioDataset

PASST_SAMPLE_RATE = 32000  # PaSST's native input rate


class PaSSTAudioDataset(ASTAudioDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Resampler from internal 16 kHz to PaSST's 32 kHz.
        self._passt_resampler = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate,
            new_freq=PASST_SAMPLE_RATE,
        )

    def _compute_features(self, idx: int) -> torch.Tensor:
        """Return a 32 kHz waveform of length context_seconds * 32000."""
        if self._stitch_enabled:
            wav = self._stitched_wav_for_idx(
                idx, offset_samples=self._tta_offset_samples
            )
        else:
            row = self.rows[idx]
            wav = self._load_wav(self._row_filepath(row))
        # wav: (1, samples_at_16kHz), float32
        wav_32k = self._passt_resampler(wav)  # (1, samples_at_32kHz)
        return wav_32k.squeeze(0)             # (samples_at_32kHz,)

    def __getitem__(self, idx: int):
        # Bypass the AST mel-feature cache entirely: PaSST consumes waveforms,
        # and resampling 10 s of audio is cheap.
        row = self.rows[idx]
        x = self._compute_features(idx)

        if self.task == "stage1":
            y = self.STAGE1_MAP[row["binary_label"]]
        else:
            y = self.STAGE2_MAP[row["motor_label"]]

        meta = {
            "filepath":     self._row_filepath(row),
            "relpath":      row.get("relpath", ""),
            "binary_label": row.get("binary_label", ""),
            "motor_label":  row.get("motor_label", ""),
            "subtype":      row.get("subtype", ""),
            "quality":      row.get("quality", ""),
            "youtube_url":  row.get("youtube_url", ""),
        }
        return x, int(y), meta
