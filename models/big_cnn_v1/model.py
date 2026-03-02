import torch
import torch.nn as nn


class _SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation: learns per-channel importance weights."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class _FreqAttention(nn.Module):
    """Frequency-band attention: learns which mel bands are most discriminative.

    Pools over time to get a per-channel frequency profile, then uses a small
    conv stack to produce a (B, 1, F, 1) attention map that reweights the
    feature map along the frequency axis.  Complementary to channel SE — SE
    asks "which channels matter?", this asks "which frequency bands matter?".
    Works with any spatial size so it is safe across all pool depths.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.time_pool = nn.AdaptiveAvgPool2d((None, 1))  # collapse time
        self.attn = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, F, T)
        pooled = self.time_pool(x)      # (B, C, F, 1)
        attn = self.attn(pooled)        # (B, 1, F, 1)
        return x * attn                 # (B, C, F, T)


class _ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + ChannelSE + residual shortcut, then ReLU + pool."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = _SEBlock(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class BigCNNv1(nn.Module):
    """
    Stage 2: direct 4-class motor count classifier (2 / 4 / 6 / 8 motors).

    Architecture improvements over SmallCNNv2 (~80K params):
      - Wider backbone: 1→32→64→128 channels  (vs 1→16→32→64)
      - FrequencyAttention after the backbone: highlights the mel bands that
        carry harmonic information for each motor count class
      - Deeper 3-layer MLP head with BatchNorm for better generalisation
      - ~325K parameters (~4× larger than SmallCNNv2)

    Input:  (B, 1, n_mels, time_frames)   — log-mel spectrogram
    Output: (B, num_classes)              — raw logits
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            _ResBlock(1,   32,  dropout=dropout),   # → (B, 32,  F/2,  T/2)
            _ResBlock(32,  64,  dropout=dropout),   # → (B, 64,  F/4,  T/4)
            _ResBlock(64,  128, dropout=dropout),   # → (B, 128, F/8,  T/8)
        )
        self.freq_attn = _FreqAttention(channels=128, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))    # → (B, 128, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Named `classifier` so inference tooling can read num_classes from
        # checkpoint["model_state"]["classifier.weight"].shape[0]
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)     # (B, 128, F//8, T//8)
        x = self.freq_attn(x)   # (B, 128, F//8, T//8) — frequency-weighted
        x = self.pool(x)         # (B, 128, 1, 1)
        x = self.head(x)         # (B, 32)
        return self.classifier(x)
