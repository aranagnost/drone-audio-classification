# models/3_stages_cnn_v1/train_stage2.py
"""
Train ThreeStagesCNNv1 stage 2 (coarse motor classification, 3-class).

Classes:
  0 — 2_motors
  1 — 4or6_motors  (4_motors and 6_motors merged)
  2 — 8_motors

Merging 4/6 into one class removes the hardest acoustic boundary from this
stage, letting the model focus on the clear 2 vs 4/6 vs 8 separation.
Stage 3 then handles the fine 4 vs 6 distinction separately.

Usage (from project root):
    python models/3_stages_cnn_v1/train_stage2.py \
        --train_csv ml/splits_stage2/train.csv \
        --val_csv   ml/splits_stage2/val.csv \
        --use_weighted_sampler \
        --out artifacts/checkpoints/stage2_3stages_cnnv1.pt
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

from ml.utils.audio_dataset import AugmentedAudioDataset, AudioDataset, AudioConfig
from ml.utils.train_utils import confusion_matrix, macro_f1, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

_MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_DIR))
from model import ThreeStagesCNNv1  # noqa: E402

# Class labels for display
COARSE_LABELS = ["2_motors", "4or6_motors", "8_motors"]
COARSE_MAP = {"2_motors": 0, "4_motors": 1, "6_motors": 1, "8_motors": 2}


def make_weighted_sampler_stage2_coarse_from_csv(csv_path: str, min_quality: int = 1):
    """Class-balanced sampler with quality weighting.

    Each sample's draw probability = (quality/5) / sum(quality/5 for class).
    Keeps all three coarse classes equally represented while drawing
    high-quality samples more often within each class.
    """
    from collections import defaultdict
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            ml = row.get("motor_label", "")
            if ml not in COARSE_MAP:
                continue
            try:
                qi = int(row.get("quality", ""))
            except (ValueError, TypeError):
                qi = 3  # treat unknown quality as mid-range
            if qi < min_quality:
                continue
            records.append((ml, qi, COARSE_MAP[ml]))

    class_quality_sum = defaultdict(float)
    for ml, qi, _ in records:
        class_quality_sum[ml] += qi / 5.0

    labels, weights = [], []
    for ml, qi, y in records:
        labels.append(y)
        weights.append((qi / 5.0) / class_quality_sum[ml])

    counts = [sum(1 for ml, _, _ in records if COARSE_MAP[ml] == i) for i in range(3)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.as_tensor(ys, dtype=torch.long), list(metas)


def run_epoch(model, loader, optimizer=None, device="cpu",
              label_smoothing=0.0, spec_augment=None):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []

    for x, y, _meta in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)

        if train and spec_augment is not None:
            for aug in spec_augment:
                x = aug(x)

        logits = model(x)
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
    ap.add_argument("--train_csv", default="ml/splits_stage2/train.csv")
    ap.add_argument("--val_csv", default="ml/splits_stage2/val.csv")
    ap.add_argument("--noise_csv", default="ml/splits/train.csv",
                    help="CSV with not_a_drone rows for noise mixing")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/checkpoints/stage2_3stages_cnnv1.pt")
    ap.add_argument("--min_quality", type=int, default=3)
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)

    # SpecAugment
    ap.add_argument("--freq_mask", type=int, default=8)
    ap.add_argument("--time_mask", type=int, default=16)

    # Noise mixing
    ap.add_argument("--noise_mix_prob", type=float, default=0.7)
    ap.add_argument("--snr_low", type=float, default=5.0)
    ap.add_argument("--snr_high", type=float, default=20.0)
    ap.add_argument("--gain_jitter_db", type=float, default=6.0)

    # Eval mode
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate checkpoint on --test_csv (no training)")
    ap.add_argument("--test_csv", default="ml/splits_stage2/test.csv")

    args = ap.parse_args()

    if args.eval:
        from ml.utils.train_utils import f1_from_cm
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.out, map_location=device)
        cfg = AudioConfig(**ckpt["cfg"])
        mq = ckpt.get("min_quality", 1)
        model = ThreeStagesCNNv1(num_classes=3, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ds = AudioDataset(args.test_csv, task="stage2_coarse", cfg=cfg, min_quality=mq)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_with_meta)
        ys, ps = [], []
        with torch.no_grad():
            for x, y, _meta in loader:
                x = x.to(device)
                ps.append(model(x).argmax(1).cpu())
                ys.append(torch.as_tensor(y))
        ys = torch.cat(ys)
        ps = torch.cat(ps)
        cm = confusion_matrix(3, ys, ps)
        mf1 = macro_f1(cm)
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Evaluation on : {args.test_csv}")
        print(f"  Checkpoint    : {args.out}  (min_quality>={mq})")
        print(f"{sep}")
        print(f"  macroF1   : {mf1:.4f}")
        for i, lbl in enumerate(COARSE_LABELS):
            print(f"  F1({lbl}): {f1_from_cm(cm, i):.4f}")
        print(f"  CM [{'/'.join(COARSE_LABELS)}]: {cm.tolist()}")
        print()
        return


    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model=ThreeStagesCNNv1  task=stage2_coarse  dropout={args.dropout}")
    print(f"[INFO] Classes: {COARSE_LABELS}")
    print(f"[INFO] Noise mixing: prob={args.noise_mix_prob}  SNR=[{args.snr_low}, {args.snr_high}] dB  "
          f"gain_jitter={args.gain_jitter_db} dB")

    cfg = AudioConfig()
    mq = args.min_quality

    train_ds = AugmentedAudioDataset(
        args.train_csv, task="stage2_coarse", cfg=cfg,
        augment=True,
        noise_csv=args.noise_csv,
        noise_mix_prob=args.noise_mix_prob,
        snr_range=(args.snr_low, args.snr_high),
        gain_jitter_db=args.gain_jitter_db,
        min_quality=mq,
    )
    val_ds = AudioDataset(args.val_csv, task="stage2_coarse", cfg=cfg, min_quality=mq)

    noise_count = len(train_ds.noise_paths)
    print(f"[INFO] Noise pool: {noise_count} samples from not_a_drone")
    print(f"[INFO] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_stage2_coarse_from_csv(
            args.train_csv, min_quality=mq)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)
        print(f"[INFO] Coarse class counts: 2_motors={counts[0]}  4or6_motors={counts[1]}  8_motors={counts[2]}")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_with_meta)

    spec_augment = []
    if args.freq_mask > 0:
        spec_augment.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=args.freq_mask))
    if args.time_mask > 0:
        spec_augment.append(torchaudio.transforms.TimeMasking(time_mask_param=args.time_mask))
    if spec_augment:
        print(f"[INFO] SpecAugment: freq_mask={args.freq_mask}, time_mask={args.time_mask}")
    else:
        spec_augment = None

    model = ThreeStagesCNNv1(num_classes=3, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_mf1 = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device,
                                  label_smoothing=args.label_smoothing,
                                  spec_augment=spec_augment)
        va_loss, va_y, va_p = run_epoch(model, val_loader, device=device)
        scheduler.step()

        cm = confusion_matrix(3, va_y, va_p)
        mf1 = macro_f1(cm)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        print(f"  CM [{'/'.join(COARSE_LABELS)}]:", cm.tolist())

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "best_macro_f1": best_mf1,
                "num_classes": 3,
                "labels": COARSE_LABELS,
                "min_quality": mq,
                "noise_mix_prob": args.noise_mix_prob,
                "snr_range": [args.snr_low, args.snr_high],
                "gain_jitter_db": args.gain_jitter_db,
            }, out_path)
            print(f"  ** Saved best -> {out_path} (macroF1={best_mf1:.4f})")

    print(f"[DONE] Best macroF1 = {best_mf1:.4f}")


if __name__ == "__main__":
    main()
