# models/ast_v1/model.py
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers import ASTForAudioClassification


class _MLPHead(nn.Module):
    """2-layer MLP head to replace the default linear classifier."""
    def __init__(self, hidden_size: int, num_classes: int, dropout: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size)
        self.fc1     = nn.Linear(hidden_size, 256)
        self.act     = nn.GELU()
        self.drop    = nn.Dropout(dropout)
        self.fc2     = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


class ASTClassifier(nn.Module):
    """
    Fine-tunable AST (Audio Spectrogram Transformer) wrapper.

    Wraps HuggingFace ASTForAudioClassification with a constructor signature
    compatible with the existing training scripts:
        ModelClass(num_classes=N, dropout=D)

    Input:  (batch, time_frames, num_mel_bins)  — from ASTFeatureExtractor
    Output: (batch, num_classes)                 — raw logits
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.0,
        pretrained_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        mlp_head: bool = False,
    ):
        super().__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ast = ASTForAudioClassification.from_pretrained(
                pretrained_model,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
        if dropout > 0.0:
            self.ast.config.hidden_dropout_prob = dropout
            self.ast.config.attention_probs_dropout_prob = dropout

        if mlp_head:
            hidden_size = self.ast.config.hidden_size  # 768
            self.ast.classifier = _MLPHead(hidden_size, num_classes, dropout)

        # Expose classifier so freeze logic and inference_app work unchanged
        self.classifier = self.ast.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ast(input_values=x).logits
