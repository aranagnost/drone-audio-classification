# models/small_cnn_v2/train_stage2.py
"""
Train SmallCNNv2 stage 2 (motor classification, 4-class).

Improvements over v1 training:
  - AdamW optimizer (proper weight decay)
  - Cosine-annealing LR schedule
  - Label smoothing (0.05)
  - SpecAugment (time + frequency masking)

Usage (from project root):
    python models/small_cnn_v2/train_stage2.py \
        --train_csv ml/splits/train.csv \
        --val_csv   ml/splits/val.csv \
        --use_weighted_sampler \
        --out artifacts/checkpoints/stage2_smallcnnv2.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from ml.utils.audio_dataset import AudioDataset, AudioConfig
from ml.utils.train_utils import confusion_matrix, macro_f1, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

_MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_DIR))
from model import SmallCNNv2  # noqa: E402


def make_weighted_sampler_stage2_from_csv(csv_path: str, min_quality: int = 1, use_quality: bool = False):
    label_map = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}
    labels, qweights, counts = [], [], [0, 0, 0, 0]

    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            try:
                qi = int(row.get("quality", ""))
            except (ValueError, TypeError):
                qi = 5
            if qi < min_quality:
                continue
            y = label_map[row["motor_label"]]
            counts[y] += 1
            labels.append(y)
            qweights.append(max(1, min(5, qi)) / 5.0 if use_quality else 1.0)

    base = [1.0 / counts[y] for y in labels]
    weights = [b * qw for b, qw in zip(base, qweights)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.as_tensor(ys, dtype=torch.long), list(metas)


def run_epoch(model, loader, optimizer=None, device="cpu",
              use_quality_loss=False, label_smoothing=0.0, spec_augment=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []

    for x, y, meta in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        if train and spec_augment is not None:
            for aug in spec_augment:
                x = aug(x)

        logits = model(x)

        if use_quality_loss:
            w = torch.as_tensor([m.get("weight", 1.0) for m in meta], device=device, dtype=torch.float32)
            per = F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)
            loss = (per * w).mean()
        else:
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        ps.append(logits.argmax(1).detach().cpu())
        ys.append(y.detach().cpu())

    return total_loss / len(loader.dataset), torch.cat(ys), torch.cat(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="ml/splits/train.csv")
    ap.add_argument("--val_csv", default="ml/splits/val.csv")
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/checkpoints/stage2_smallcnnv2.pt")
    ap.add_argument("--min_quality", type=int, default=3)
    ap.add_argument("--use_quality_weighting", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--use_quality_loss", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freq_mask", type=int, default=8,
                    help="SpecAugment frequency mask width (0 to disable)")
    ap.add_argument("--time_mask", type=int, default=16,
                    help="SpecAugment time mask width (0 to disable)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model=SmallCNNv2  dropout={args.dropout}")

    cfg = AudioConfig()
    mq = args.min_quality

    train_ds = AudioDataset(args.train_csv, task="stage2", cfg=cfg,
                            quality_weighting=args.use_quality_weighting, min_quality=mq)
    val_ds = AudioDataset(args.val_csv, task="stage2", cfg=cfg,
                          quality_weighting=args.use_quality_weighting, min_quality=mq)

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_stage2_from_csv(
            args.train_csv, min_quality=mq, use_quality=args.use_quality_weighting)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)
        print(f"[INFO] Stage2 class counts: {counts} (2,4,6,8 motors)")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_with_meta)

    # SpecAugment transforms
    spec_augment = []
    if args.freq_mask > 0:
        spec_augment.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=args.freq_mask))
    if args.time_mask > 0:
        spec_augment.append(torchaudio.transforms.TimeMasking(time_mask_param=args.time_mask))
    if spec_augment:
        print(f"[INFO] SpecAugment: freq_mask={args.freq_mask}, time_mask={args.time_mask}")
    else:
        spec_augment = None

    model = SmallCNNv2(num_classes=4, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_mf1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device,
                                  use_quality_loss=args.use_quality_loss,
                                  label_smoothing=args.label_smoothing,
                                  spec_augment=spec_augment)
        va_loss, va_y, va_p = run_epoch(model, val_loader, device=device)
        scheduler.step()

        cm = confusion_matrix(4, va_y, va_p)
        mf1 = macro_f1(cm)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        print("  CM:", cm.tolist())

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "best_macro_f1": best_mf1,
                "min_quality": mq,
                "use_quality_weighting": args.use_quality_weighting,
                "use_quality_loss": args.use_quality_loss,
            }, out_path)
            print(f"  ** Saved best -> {out_path} (macroF1={best_mf1:.4f})")

    print(f"[DONE] Best macroF1 = {best_mf1:.4f}")


if __name__ == "__main__":
    main()
