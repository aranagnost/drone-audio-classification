# ml/train/train_stage2.py
from __future__ import annotations

import argparse
from pathlib import Path
import csv
import warnings

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from ml.utils.audio_dataset import AudioDataset, AudioConfig
from ml.models.small_cnn import SmallCNN
from ml.utils.train_utils import confusion_matrix, macro_f1, set_seed

# Silence torchaudio backend deprecation spam (optional, safe)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.io")


def make_weighted_sampler_stage2_from_csv(csv_path: str, min_quality: int = 1, use_quality: bool = False):
    """
    Build a WeightedRandomSampler for stage2 using ONLY CSV labels (no audio loading).
    Keeps only drone rows and applies min_quality filtering.
    Optionally multiplies sample weights by (quality/5).
    """
    label_map = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}

    labels = []
    qweights = []
    counts = [0, 0, 0, 0]

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["binary_label"] != "drone":
                continue

            # quality filter
            q = row.get("quality", "")
            try:
                qi = int(q)
            except Exception:
                qi = 5
            if qi < min_quality:
                continue

            y = label_map[row["motor_label"]]
            counts[y] += 1
            labels.append(y)

            if use_quality:
                qweights.append(max(1, min(5, qi)) / 5.0)
            else:
                qweights.append(1.0)

    base = [1.0 / counts[y] for y in labels]
    weights = [b * qw for b, qw in zip(base, qweights)]

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler, counts


def collate_with_meta(batch):
    """
    Default PyTorch collate turns list-of-dicts into dict-of-lists.
    We want meta as list-of-dicts so we can read meta[i]["weight"].
    """
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.as_tensor(ys, dtype=torch.long)
    meta = list(metas)
    return x, y, meta


def run_epoch(model, loader, optimizer=None, device="cpu", use_quality_loss=False):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    ys, ps = [], []

    for x, y, meta in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        logits = model(x)

        if use_quality_loss:
            # meta is list-of-dicts due to custom collate
            w = torch.as_tensor([m.get("weight", 1.0) for m in meta], device=device, dtype=torch.float32)
            per = F.cross_entropy(logits, y, reduction="none")
            loss = (per * w).mean()
        else:
            loss = F.cross_entropy(logits, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1).detach().cpu()
        ys.append(y.detach().cpu())
        ps.append(pred)

    ys = torch.cat(ys)
    ps = torch.cat(ps)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="ml/splits/train.csv")
    ap.add_argument("--val_csv", default="ml/splits/val.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ml/models/stage2_smallcnn.pt")

    ap.add_argument("--min_quality", type=int, default=3, help="Keep only q>=this (stage2). Use 1 to keep all.")
    ap.add_argument("--use_quality_weighting", action="store_true", help="Use quality as sample weight (q/5).")
    ap.add_argument("--use_weighted_sampler", action="store_true", help="Balance motor classes via sampler.")
    ap.add_argument("--use_quality_loss", action="store_true", help="Multiply CE loss by quality weight (q/5).")
    ap.add_argument("--num_workers", type=int, default=4)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cpu"
    cfg = AudioConfig()

    mq = args.min_quality if args.min_quality is not None else 1

    train_ds = AudioDataset(
        args.train_csv,
        task="stage2",
        cfg=cfg,
        quality_weighting=args.use_quality_weighting,
        min_quality=mq,
    )
    val_ds = AudioDataset(
        args.val_csv,
        task="stage2",
        cfg=cfg,
        quality_weighting=args.use_quality_weighting,
        min_quality=mq,
    )

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_stage2_from_csv(
            args.train_csv, min_quality=mq, use_quality=args.use_quality_weighting
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_with_meta,
        )
        print(f"[INFO] Stage2 class counts (train): {counts} (2,4,6,8 motors)")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_with_meta,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_with_meta,
    )

    model = SmallCNN(num_classes=4).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_mf1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p = run_epoch(
            model, train_loader, optimizer=optim, device=device, use_quality_loss=args.use_quality_loss
        )
        va_loss, va_y, va_p = run_epoch(model, val_loader, optimizer=None, device=device, use_quality_loss=False)

        cm = confusion_matrix(4, va_y, va_p)
        mf1 = macro_f1(cm)

        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_macroF1 {mf1:.4f}")
        print("Val confusion (rows=true, cols=pred):\n", cm)

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch,
                    "best_macro_f1": best_mf1,
                    "min_quality": mq,
                    "use_quality_weighting": args.use_quality_weighting,
                    "use_quality_loss": args.use_quality_loss,
                },
                out_path,
            )
            print(f"[OK] Saved best -> {out_path} (macroF1={best_mf1:.4f})")

    print("[DONE]")


if __name__ == "__main__":
    main()
