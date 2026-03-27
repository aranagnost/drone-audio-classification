# models/ast_v1/model.py
from __future__ import annotations

import warnings

import torch
import torch.nn as nn
from transformers import ASTForAudioClassification


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

        # Expose classifier so inference_app can read num_classes from state dict
        self.classifier = self.ast.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ast(input_values=x).logits
