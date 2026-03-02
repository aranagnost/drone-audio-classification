# models/3_stages_cnn_v1/train_stage1.py
"""
Train ThreeStagesCNNv1 stage 1 (drone detection, binary).

Identical training pipeline to small_cnn_v3 stage 1:
  - Waveform-level noise mixing (speech/wind upweighted 3x)
  - Gain jitter, SpecAugment, label smoothing, AdamW + cosine LR

Usage (from project root):
    python models/3_stages_cnn_v1/train_stage1.py \
        --train_csv ml/splits/train.csv \
        --val_csv   ml/splits/val.csv \
        --use_weighted_sampler \
        --out artifacts/checkpoints/stage1_3stages_cnnv1.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from ml.utils.audio_dataset import AugmentedAudioDataset, AudioDataset, AudioConfig
from ml.utils.train_utils import confusion_matrix, f1_from_cm, macro_f1, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

_MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_MODEL_DIR))
from model import ThreeStagesCNNv1  # noqa: E402


def make_weighted_sampler_from_csv(csv_path: str):
    labels = []
    counts = {"no_drone": 0, "drone": 0}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lbl = row["binary_label"]
            counts[lbl] += 1
            labels.append(0 if lbl == "no_drone" else 1)
    class_counts = [counts["no_drone"], counts["drone"]]
    weights = [1.0 / class_counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), class_counts


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


def run_eval(checkpoint_path: str, csv_path: str, device: str, batch: int, num_workers: int):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = AudioConfig(**ckpt["cfg"])

    model = ThreeStagesCNNv1(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = AudioDataset(csv_path, task="stage1", cfg=cfg)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers)

    all_y, all_p, all_meta = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            all_y.extend(y if isinstance(y, list) else y.tolist())
            all_p.extend(preds.tolist())
            # meta is a dict of lists when default-collated
            for i in range(len(preds)):
                all_meta.append({k: v[i] if isinstance(v, list) else v[i].item()
                                  for k, v in meta.items()})

    y_t = torch.tensor(all_y)
    p_t = torch.tensor(all_p)
    cm = confusion_matrix(2, y_t, p_t)
    mf1 = macro_f1(cm)
    f1_drone = f1_from_cm(cm, class_index=1)
    f1_nodrone = f1_from_cm(cm, class_index=0)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  Evaluation on : {csv_path}")
    print(f"  Checkpoint    : {checkpoint_path}")
    print(f"{sep}")
    print(f"  macroF1      : {mf1:.4f}")
    print(f"  F1(drone)    : {f1_drone:.4f}")
    print(f"  F1(no_drone) : {f1_nodrone:.4f}")
    print(f"  CM [[TN,FP],[FN,TP]]: {cm.tolist()}")

    # False Positives: no_drone predicted as drone — grouped by subtype
    fp_by_sub: dict = defaultdict(int)
    total_by_sub: dict = defaultdict(int)
    for y_i, p_i, m in zip(all_y, all_p, all_meta):
        if m.get("binary_label") == "no_drone":
            sub = m.get("subtype") or "unknown"
            total_by_sub[sub] += 1
            if p_i == 1:
                fp_by_sub[sub] += 1

    print(f"\n  False Positives by no_drone subtype (predicted as drone):")
    print(f"  {'Subtype':<22} {'FP':>5} {'Total':>7} {'FP rate':>9}")
    print(f"  {'-'*47}")
    for sub in sorted(total_by_sub, key=lambda s: -fp_by_sub.get(s, 0)):
        fp = fp_by_sub.get(sub, 0)
        tot = total_by_sub[sub]
        print(f"  {sub:<22} {fp:>5} {tot:>7} {fp/tot:>8.1%}")

    # False Negatives: drone predicted as no_drone — grouped by motor class
    fn_by_motor: dict = defaultdict(int)
    total_by_motor: dict = defaultdict(int)
    for y_i, p_i, m in zip(all_y, all_p, all_meta):
        if m.get("binary_label") == "drone":
            motor = m.get("motor_label") or "unknown"
            total_by_motor[motor] += 1
            if p_i == 0:
                fn_by_motor[motor] += 1

    print(f"\n  False Negatives by drone motor class (missed drones):")
    print(f"  {'Motor class':<22} {'FN':>5} {'Total':>7} {'FN rate':>9}")
    print(f"  {'-'*47}")
    for motor in sorted(total_by_motor, key=lambda s: -fn_by_motor.get(s, 0)):
        fn = fn_by_motor.get(motor, 0)
        tot = total_by_motor[motor]
        print(f"  {motor:<22} {fn:>5} {tot:>7} {fn/tot:>8.1%}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", default="ml/splits/train.csv")
    ap.add_argument("--val_csv", default="ml/splits/val.csv")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="artifacts/checkpoints/stage1_3stages_cnnv1.pt")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)

    # SpecAugment
    ap.add_argument("--freq_mask", type=int, default=8)
    ap.add_argument("--time_mask", type=int, default=16)

    # Noise mixing
    ap.add_argument("--noise_mix_prob", type=float, default=0.7)
    ap.add_argument("--snr_low", type=float, default=0.0)
    ap.add_argument("--snr_high", type=float, default=20.0)
    ap.add_argument("--gain_jitter_db", type=float, default=6.0)

    # Eval mode
    ap.add_argument("--eval", action="store_true",
                    help="Run evaluation on --test_csv using the saved checkpoint (no training)")
    ap.add_argument("--test_csv", default="ml/splits/test.csv")

    args = ap.parse_args()

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_eval(args.out, args.test_csv, device, args.batch, args.num_workers)
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model=ThreeStagesCNNv1  dropout={args.dropout}")
    print(f"[INFO] Noise mixing: prob={args.noise_mix_prob}  SNR=[{args.snr_low}, {args.snr_high}] dB  "
          f"gain_jitter={args.gain_jitter_db} dB")

    cfg = AudioConfig()

    train_ds = AugmentedAudioDataset(
        args.train_csv, task="stage1", cfg=cfg,
        augment=True,
        noise_mix_prob=args.noise_mix_prob,
        snr_range=(args.snr_low, args.snr_high),
        gain_jitter_db=args.gain_jitter_db,
    )
    val_ds = AudioDataset(args.val_csv, task="stage1", cfg=cfg)

    noise_count = len(train_ds.noise_paths)
    print(f"[INFO] Noise pool: {noise_count} samples from not_a_drone")

    if args.use_weighted_sampler:
        sampler, counts = make_weighted_sampler_from_csv(args.train_csv)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers)
        print(f"[INFO] Stage1 class counts: no_drone={counts[0]} drone={counts[1]}")
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

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
        f1_drone = f1_from_cm(cm, class_index=1)
        f1_nodrone = f1_from_cm(cm, class_index=0)
        lr_now = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | F1(drone) {f1_drone:.4f} | F1(no_drone) {f1_nodrone:.4f} | lr {lr_now:.2e}")
        print("  CM:", cm.tolist())

        if mf1 > best_mf1:
            best_mf1 = mf1
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "best_macro_f1": best_mf1,
                "noise_mix_prob": args.noise_mix_prob,
                "snr_range": [args.snr_low, args.snr_high],
                "gain_jitter_db": args.gain_jitter_db,
            }, out_path)
            print(f"  ** Saved best -> {out_path} (macroF1={best_mf1:.4f})")

    print(f"[DONE] Best macroF1 = {best_mf1:.4f}")


if __name__ == "__main__":
    main()
