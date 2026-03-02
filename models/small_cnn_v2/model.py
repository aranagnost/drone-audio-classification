import torch
import torch.nn as nn


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation: learns per-channel importance weights."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
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


class _ResBlock(nn.Module):
    """Conv-BN-ReLU-Conv-BN + SE + residual shortcut, then ReLU + pool."""

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

        # 1x1 projection when channel count changes
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


class SmallCNNv2(nn.Module):
    """
    Improved CNN for log-mel inputs: (B, 1, n_mels, time).

    Same channel widths as v1 (16 -> 32 -> 64) to stay small (~80K params),
    but with architectural improvements:
      - Residual connections for better gradient flow
      - Squeeze-and-Excitation attention (per-channel weighting)
      - Spatial dropout for regularization
      - Two-layer classifier head with dropout
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            _ResBlock(1, 16, dropout=dropout),
            _ResBlock(16, 32, dropout=dropout),
            _ResBlock(32, 64, dropout=dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        # Named `classifier` so inference_app can read num_classes from
        # checkpoint["model_state"]["classifier.weight"].shape[0]
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        z = self.head(z)
        return self.classifier(z)
