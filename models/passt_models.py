"""PaSST classifier: patch-out fast spectrogram transformer (kkoutini).

Wraps hear21passt.base.get_basic_model(mode="embed_only") and adds a small
classification head over the 768-dim PaSST embedding. Mirrors the ASTClassifier
interface (num_classes, dropout, mlp_head) so it slots into the same scripts.

Input  : (batch, samples) raw waveform at 32 kHz
Output : (batch, num_classes) raw logits
"""
from __future__ import annotations

import torch
from torch import nn


class _MLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, 256)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(self.norm(x)))))


class PaSSTClassifier(nn.Module):
    """PaSST backbone + linear/MLP head for drone audio classification."""

    EMBED_DIM = 768  # passt_s_swa_p16_128_ap476 hidden size

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        pretrained_model: str = "passt_s_swa_p16_128_ap476",  # ignored, kept for parity
        mlp_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        from hear21passt.base import get_basic_model
        # mode="embed_only" returns (B, 768): we drop PaSST's internal 527-class
        # head and replace it with our own num_classes head.
        self.backbone = get_basic_model(mode="embed_only")

        if mlp_head:
            self.classifier = _MLPHead(self.EMBED_DIM, num_classes, dropout)
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.EMBED_DIM),
                nn.Dropout(dropout),
                nn.Linear(self.EMBED_DIM, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, samples) or (B, 1, samples)
        if x.dim() == 3:
            x = x.squeeze(1)
        emb = self.backbone(x)
        if isinstance(emb, (tuple, list)):
            emb = emb[0]
        return self.classifier(emb)


# Alias kept for import compatibility.
PaSSTv1 = PaSSTClassifier
