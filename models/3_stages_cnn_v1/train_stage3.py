# models/3_stages_cnn_v1/train_stage3.py
"""
Train ThreeStagesCNNv1 stage 3 (fine motor classification, binary: 4 vs 6 motors).

This stage only runs during inference when stage 2 is confident that the input
belongs to the "4or6_motors" class.  Training uses only 4_motors and 6_motors
clips — a focused binary classifier that can dedicate all its capacity to the
single hard acoustic boundary between these two classes.

Default min_quality=4 keeps only the clearest recordings, where the harmonic
difference between quadcopters and hexacopters is most likely to be audible.

Usage (from project root):
    python models/3_stages_cnn_v1/train_stage3.py \
        --train_csv ml/splits_stage2/train.csv \
        --val_csv   ml/splits_stage2/val.csv \
        --use_weighted_sampler \
        --out artifacts/checkpoints/stage3_3stages_cnnv1.pt
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
from ml.utils.train_utils import confusion_matrix, macro_f1, f1_from_cm, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

_MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_DIR))
from model import ThreeStagesCNNv1  # noqa: E402

FINE_LABELS = ["4_motors", "6_motors"]
FINE_MAP = {"4_motors": 0, "6_motors": 1}


def make_weighted_sampler_stage3_from_csv(csv_path: str, min_quality: int = 1):
    """Class-balanced sampler with quality weighting.

    Each sample's draw probability = (quality/5) / sum(quality/5 for class).
    This keeps both classes equally represented while drawing high-quality
    samples more often within each class.
    """
    from collections import defaultdict
    records = []  # (motor_label, quality_int, class_index)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            ml = row.get("motor_label", "")
            if ml not in FINE_MAP:
                continue
            try:
                qi = int(row.get("quality", ""))
            except (ValueError, TypeError):
                qi = 3  # treat unknown quality as mid-range
            if qi < min_quality:
                continue
            records.append((ml, qi, FINE_MAP[ml]))

    # Sum of quality weights per class — used to normalise so both classes
    # contribute equally to training regardless of their quality distributions.
    class_quality_sum = defaultdict(float)
    for ml, qi, _ in records:
        class_quality_sum[ml] += qi / 5.0

    labels, weights = [], []
    for ml, qi, y in records:
        labels.append(y)
        weights.append((qi / 5.0) / class_quality_sum[ml])

    counts = [sum(1 for ml, _, _ in records if ml == "4_motors"),
              sum(1 for ml, _, _ in records if ml == "6_motors")]
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
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-3)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/checkpoints/stage3_3stages_cnnv1.pt")
    ap.add_argument("--min_quality", type=int, default=1,
                    help="Keep only clips with quality >= this (default 1, all samples; high-quality drawn more often via sampler)")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)

    # SpecAugment
    ap.add_argument("--freq_mask", type=int, default=8)
    ap.add_argument("--time_mask", type=int, default=16)

    # Noise mixing
    ap.add_argument("--noise_mix_prob", type=float, default=0.5,
                    help="Lower than stage2: cleaner inputs help the fine distinction")
    ap.add_argument("--snr_low", type=float, default=10.0,
                    help="Higher min SNR: keep the drone signal dominant")
    ap.add_argument("--snr_high", type=float, default=25.0)
    ap.add_argument("--gain_jitter_db", type=float, default=4.0)

    # Eval mode
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate checkpoint on --test_csv (no training)")
    ap.add_argument("--test_csv", default="ml/splits_stage2/test.csv")

    args = ap.parse_args()

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.out, map_location=device)
        cfg = AudioConfig(**ckpt["cfg"])
        mq = ckpt.get("min_quality", 1)
        model = ThreeStagesCNNv1(num_classes=2, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ds = AudioDataset(args.test_csv, task="stage3", cfg=cfg, min_quality=mq)
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
        cm = confusion_matrix(2, ys, ps)
        mf1 = macro_f1(cm)
        f1_4 = f1_from_cm(cm, 0)
        f1_6 = f1_from_cm(cm, 1)
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Evaluation on : {args.test_csv}")
        print(f"  Checkpoint    : {args.out}  (min_quality>={mq})")
        print(f"{sep}")
        print(f"  macroF1   : {mf1:.4f}")
        print(f"  F1(4_mot) : {f1_4:.4f}")
        print(f"  F1(6_mot) : {f1_6:.4f}")
        print(f"  CM [4_motors/6_motors]: {cm.tolist()}")
        print()
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model=ThreeStagesCNNv1  task=stage3  dropout={args.dropout}")
    print(f"[INFO] Classes: {FINE_LABELS}  min_quality={args.min_quality}")
    print(f"[INFO] Noise mixing: prob={args.noise_mix_prob}  SNR=[{args.snr_low}, {args.snr_high}] dB  "
          f"gain_jitter={args.gain_jitter_db} dB")

    cfg = AudioConfig()
    mq = args.min_quality

    train_ds = AugmentedAudioDataset(
        args.train_csv, task="stage3", cfg=cfg,
        augment=True,
        noise_csv=args.noise_csv,
        noise_mix_prob=args.noise_mix_prob,
        snr_range=(args.snr_low, args.snr_high),
        gain_jitter_db=args.gain_jitter_db,
        min_quality=mq,
    )
    val_ds = AudioDataset(args.val_csv, task="stage3", cfg=cfg, min_quality=mq)

    noise_count = len(train_ds.noise_paths)
    print(f"[INFO] Noise pool: {noise_count} samples from not_a_drone")
    print(f"[INFO] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    if len(train_ds) == 0:
        print("[ERROR] No training samples found. Check --train_csv and --min_quality.")
        return
    if len(val_ds) == 0:
        print("[ERROR] No validation samples found. Check --val_csv and --min_quality.")
        return

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_stage3_from_csv(
            args.train_csv, min_quality=mq)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)
        print(f"[INFO] Fine class counts: 4_motors={counts[0]}  6_motors={counts[1]}")
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

    model = ThreeStagesCNNv1(num_classes=2, dropout=args.dropout).to(device)
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

        cm = confusion_matrix(2, va_y, va_p)
        mf1 = macro_f1(cm)
        f1_4 = f1_from_cm(cm, class_index=0)
        f1_6 = f1_from_cm(cm, class_index=1)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | F1(4) {f1_4:.4f} | F1(6) {f1_6:.4f} | lr {lr_now:.2e}")
        print("  CM [4_motors/6_motors]:", cm.tolist())

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "best_macro_f1": best_mf1,
                "num_classes": 2,
                "labels": FINE_LABELS,
                "min_quality": mq,
                "noise_mix_prob": args.noise_mix_prob,
                "snr_range": [args.snr_low, args.snr_high],
                "gain_jitter_db": args.gain_jitter_db,
            }, out_path)
            print(f"  ** Saved best -> {out_path} (macroF1={best_mf1:.4f})")

    print(f"[DONE] Best macroF1 = {best_mf1:.4f}")


if __name__ == "__main__":
    main()
