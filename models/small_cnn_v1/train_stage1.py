# ml/train/train_stage1.py
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
from ml.utils.train_utils import confusion_matrix, f1_from_cm, set_seed

# Silence torchaudio backend deprecation spam (optional, safe)
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.io")


def make_weighted_sampler_from_csv(csv_path: str):
    """
    Build a WeightedRandomSampler for stage1 using ONLY the CSV labels
    (no audio loading / feature computation).
    """
    labels = []
    counts = {"no_drone": 0, "drone": 0}

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            lbl = row["binary_label"]
            counts[lbl] += 1
            labels.append(0 if lbl == "no_drone" else 1)

    class_counts = [counts["no_drone"], counts["drone"]]
    weights = [1.0 / class_counts[y] for y in labels]

    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler, class_counts


def run_epoch(model, loader, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    ys, ps = [], []

    for x, y, meta in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)  # <-- fixed warning

        logits = model(x)
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
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ml/models/stage1_smallcnn.pt")
    ap.add_argument("--use_weighted_sampler", action="store_true", help="Balance classes via sampler")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cpu"

    cfg = AudioConfig()
    train_ds = AudioDataset(args.train_csv, task="stage1", cfg=cfg)
    val_ds = AudioDataset(args.val_csv, task="stage1", cfg=cfg)

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_from_csv(args.train_csv)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.num_workers)
        print(f"[INFO] Stage1 class counts (train): no_drone={counts[0]} drone={counts[1]}")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    model = SmallCNN(num_classes=2).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, optimizer=optim, device=device)
        va_loss, va_y, va_p = run_epoch(model, val_loader, optimizer=None, device=device)

        cm = confusion_matrix(2, va_y, va_p)
        f1_drone = f1_from_cm(cm, class_index=1)

        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_F1(drone) {f1_drone:.4f}")
        print("Val confusion (rows=true, cols=pred):\n", cm)

        if f1_drone > best_f1:
            best_f1 = f1_drone
            torch.save(
                {"model_state": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "best_f1_drone": best_f1},
                out_path
            )
            print(f"[OK] Saved best -> {out_path} (F1_drone={best_f1:.4f})")

    print("[DONE]")


if __name__ == "__main__":
    main()
