import torch
import torch.nn as nn

class _SEBlock(nn.Module):
    """Channel Squeeze-and-Excitation: learns per-channel importance weights."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        # Ensure minimum mid-channels to support smaller networks like SmallCNNv2
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


class _FreqAttention(nn.Module):
    """Frequency-band attention: learns which mel bands are most discriminative.
    
    Pools over time to get a per-channel frequency profile, then uses a small
    conv stack to produce a (B, 1, F, 1) attention map that reweights the
    feature map along the frequency axis.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.time_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.attn = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.time_pool(x)
        attn = self.attn(pooled)
        return x * attn


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


class SmallCNNv1(nn.Module):
    """
    Baseline CNN for log-mel inputs: (B, 1, n_mels, time).
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        return self.classifier(z)


class SmallCNNv2(nn.Module):
    """
    Improved CNN (~80K params) incorporating Residual connections, 
    Squeeze-and-Excitation attention, and Spatial Dropout.
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
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = z.flatten(1)
        z = self.head(z)
        return self.classifier(z)


class BigCNNv1(nn.Module):
    """
    Direct multi-class motor count classifier (~325K params).
    
    Architecture characteristics:
      - Wider backbone (32 -> 64 -> 128 channels).
      - FrequencyAttention after the backbone to highlight harmonic mel bands.
      - Deeper 3-layer MLP head with BatchNorm.
    """

    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.backbone = nn.Sequential(
            _ResBlock(1, 32, dropout=dropout),
            _ResBlock(32, 64, dropout=dropout),
            _ResBlock(64, 128, dropout=dropout),
        )
        self.freq_attn = _FreqAttention(channels=128, reduction=8)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

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
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.freq_attn(x)
        x = self.pool(x)
        x = self.head(x)
        return self.classifier(x)
