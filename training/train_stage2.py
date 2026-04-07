# training/train_stage2.py
"""
Consolidated stage-2 training script (motor classification).

Handles:
  - small_cnn_v1/v2/v3: 4-class (2/4/6/8 motors) using "stage2" task
  - big_cnn_v1: 4-class with BigCNNv1 backbone + noise mixing
  - 3_stages_cnn_v1: 3-class coarse (2 / 4or6 / 8) using "stage2_coarse" task

Usage:
    python training/train_stage2.py --model small_cnn_v2
    python training/train_stage2.py --model 3_stages_cnn_v1 --use_weighted_sampler
    python training/train_stage2.py --model big_cnn_v1 --eval
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torchaudio
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

from data.audio_dataset import AugmentedAudioDataset, AudioDataset, AudioConfig
from data.train_utils import confusion_matrix, macro_f1, f1_from_cm, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Coarse label mapping for 3_stages_cnn_v1
COARSE_LABELS = ["2_motors", "4or6_motors", "8_motors"]
COARSE_MAP = {"2_motors": 0, "4_motors": 1, "6_motors": 1, "8_motors": 2}

# Standard 4-class motor labels
MOTOR_LABELS = ["2_motors", "4_motors", "6_motors", "8_motors"]
MOTOR_MAP = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}


def load_model_class(model_name: str, model_source: str | None = None):
    src = model_source or model_name
    model_py = MODELS_DIR / src / "model.py"
    spec = importlib.util.spec_from_file_location(f"{src}_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train_config(model_name: str, stage: str = "stage2") -> dict:
    cfg_path = MODELS_DIR / model_name / "train_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        full = json.load(f)
    return full.get(stage, {})


def collate_with_meta(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.as_tensor(ys, dtype=torch.long), list(metas)


# ── Weighted sampler helpers ──────────────────────────────────────────────────

def make_weighted_sampler_4class(csv_path: str, min_quality: int = 1, use_quality: bool = False):
    """Class-balanced sampler for 4-class stage2."""
    labels, qweights, counts = [], [], [0, 0, 0, 0]
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            ml = row.get("motor_label", "")
            if ml not in MOTOR_MAP:
                continue
            try:
                qi = int(row.get("quality", ""))
            except (ValueError, TypeError):
                qi = 5
            if qi < min_quality:
                continue
            y = MOTOR_MAP[ml]
            counts[y] += 1
            labels.append(y)
            qweights.append(max(1, min(5, qi)) / 5.0 if use_quality else 1.0)
    base = [1.0 / counts[y] for y in labels]
    weights = [b * qw for b, qw in zip(base, qweights)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


def make_weighted_sampler_bigcnn(csv_path: str, min_quality: int = 1):
    """Quality-weighted class-balanced sampler as used in big_cnn_v1."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            ml = row.get("motor_label", "")
            if ml not in MOTOR_MAP:
                continue
            try:
                qi = int(row.get("quality", ""))
            except (ValueError, TypeError):
                qi = 3
            if qi < min_quality:
                continue
            records.append((ml, qi, MOTOR_MAP[ml]))

    class_quality_sum: dict = defaultdict(float)
    for ml, qi, _ in records:
        class_quality_sum[ml] += qi / 5.0

    labels, weights = [], []
    for ml, qi, y in records:
        labels.append(y)
        weights.append((qi / 5.0) / class_quality_sum[ml])

    counts = [sum(1 for ml, _, _ in records if MOTOR_MAP[ml] == i) for i in range(4)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


def make_weighted_sampler_coarse(csv_path: str, min_quality: int = 1):
    """Quality-weighted class-balanced sampler for 3-class coarse stage2."""
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
                qi = 3
            if qi < min_quality:
                continue
            records.append((ml, qi, COARSE_MAP[ml]))

    class_quality_sum: dict = defaultdict(float)
    for ml, qi, _ in records:
        class_quality_sum[ml] += qi / 5.0

    labels, weights = [], []
    for ml, qi, y in records:
        labels.append(y)
        weights.append((qi / 5.0) / class_quality_sum[ml])

    counts = [sum(1 for ml, _, _ in records if COARSE_MAP[ml] == i) for i in range(3)]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


# ── Training loop ─────────────────────────────────────────────────────────────

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

        if use_quality_loss and isinstance(meta, list):
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
    # --- Pass 1: parse --model only ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", required=True)
    pre_args, _ = pre.parse_known_args()
    model_name = pre_args.model

    cfg_dict = load_train_config(model_name, "stage2")

    # Detect coarse mode (3_stages_cnn_v1 uses stage2_coarse task)
    is_coarse = cfg_dict.get("task", "stage2") == "stage2_coarse"
    num_classes_default = 3 if is_coarse else 4

    # --- Pass 2: full parser ---
    ap = argparse.ArgumentParser(description="Stage-2 training: motor classification")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_csv", default="data/splits_stage2/train.csv")
    ap.add_argument("--val_csv",   default="data/splits_stage2/val.csv")
    ap.add_argument("--test_csv",  default="data/splits_stage2/test.csv")
    ap.add_argument("--noise_csv", default="data/splits/train.csv",
                    help="CSV with not_a_drone rows for noise mixing")
    ap.add_argument("--epochs",     type=int,   default=35)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=8e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout",    type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out",        default=None)
    ap.add_argument("--min_quality", type=int, default=3)
    ap.add_argument("--use_quality_weighting", action="store_true")
    ap.add_argument("--use_weighted_sampler",  action="store_true")
    ap.add_argument("--use_quality_loss",      action="store_true")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freq_mask",  type=int,  default=8)
    ap.add_argument("--time_mask",  type=int,  default=16)
    ap.add_argument("--noise_mix_prob", type=float, default=0.0)
    ap.add_argument("--snr_low",    type=float, default=0.0)
    ap.add_argument("--snr_high",   type=float, default=20.0)
    ap.add_argument("--gain_jitter_db", type=float, default=0.0)
    ap.add_argument("--optimizer",  default="adamw", choices=["adam", "adamw"])
    ap.add_argument("--task",       default="stage2",
                    choices=["stage2", "stage2_coarse"],
                    help="Dataset task mode (stage2_coarse for 3_stages_cnn_v1)")
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate checkpoint on --test_csv (no training)")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early stopping: stop if macroF1 does not improve for N epochs (0 = disabled)")
    ap.add_argument("--dataset_root", default=None,
                    help="Override absolute filepaths in CSVs using relpath column "
                         "(e.g. /content/drive/MyDrive/drone-audio/datasets/Drone_Audio_Dataset/audio)")

    ap.set_defaults(**cfg_dict)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"artifacts/checkpoints/stage2_{model_name}.pt"

    model_source = cfg_dict.get("model_source", None)
    model_mod = load_model_class(model_name, model_source)
    model_class_name = cfg_dict.get("model_class", "")
    if not model_class_name:
        raise ValueError(
            f"train_config.json for '{model_name}' is missing 'model_class' in stage2 section."
        )
    ModelClass = getattr(model_mod, model_class_name)

    task = args.task
    is_coarse = (task == "stage2_coarse")
    num_classes = 3 if is_coarse else 4
    labels_list = COARSE_LABELS if is_coarse else MOTOR_LABELS

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        cfg = AudioConfig(**ckpt["cfg"])
        mq = ckpt.get("min_quality", 1)
        model = ModelClass(num_classes=num_classes, dropout=0.0).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ds = AudioDataset(args.test_csv, task=task, cfg=cfg, min_quality=mq, dataset_root=args.dataset_root)
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
        cm = confusion_matrix(num_classes, ys, ps)
        mf1 = macro_f1(cm)
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Evaluation on : {args.test_csv}")
        print(f"  Checkpoint    : {args.out}  (min_quality>={mq})")
        print(f"{sep}")
        print(f"  macroF1   : {mf1:.4f}")
        for i, lbl in enumerate(labels_list):
            print(f"  F1({lbl}): {f1_from_cm(cm, i):.4f}")
        print(f"  CM [{'/'.join(labels_list)}]: {cm.tolist()}")
        print()
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model={model_name} ({model_class_name})  task={task}  dropout={args.dropout}")
    print(f"[INFO] Classes: {labels_list}")

    cfg = AudioConfig()
    mq = args.min_quality
    use_augmentation = args.noise_mix_prob > 0

    if use_augmentation:
        print(f"[INFO] Noise mixing: prob={args.noise_mix_prob}  SNR=[{args.snr_low}, {args.snr_high}] dB  "
              f"gain_jitter={args.gain_jitter_db} dB")
        train_ds = AugmentedAudioDataset(
            args.train_csv, task=task, cfg=cfg,
            augment=True,
            noise_csv=args.noise_csv,
            noise_mix_prob=args.noise_mix_prob,
            snr_range=(args.snr_low, args.snr_high),
            gain_jitter_db=args.gain_jitter_db,
            quality_weighting=args.use_quality_weighting,
            min_quality=mq,
            dataset_root=args.dataset_root,
        )
        print(f"[INFO] Noise pool: {len(train_ds.noise_paths)} samples from not_a_drone")
    else:
        train_ds = AudioDataset(args.train_csv, task=task, cfg=cfg,
                                quality_weighting=args.use_quality_weighting, min_quality=mq,
                                dataset_root=args.dataset_root)

    val_ds = AudioDataset(args.val_csv, task=task, cfg=cfg,
                          quality_weighting=args.use_quality_weighting, min_quality=mq,
                          dataset_root=args.dataset_root)

    print(f"[INFO] Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    if args.use_weighted_sampler:
        if is_coarse:
            sampler, counts = make_weighted_sampler_coarse(args.train_csv, min_quality=mq)
            print(f"[INFO] Coarse class counts: 2_motors={counts[0]}  4or6_motors={counts[1]}  8_motors={counts[2]}")
        elif model_name == "big_cnn_v1":
            sampler, counts = make_weighted_sampler_bigcnn(args.train_csv, min_quality=mq)
            print(f"[INFO] Class counts: " + "  ".join(f"{lbl}={counts[i]}" for i, lbl in enumerate(MOTOR_LABELS)))
        else:
            sampler, counts = make_weighted_sampler_4class(
                args.train_csv, min_quality=mq, use_quality=args.use_quality_weighting)
            print(f"[INFO] Stage2 class counts: {counts} (2,4,6,8 motors)")
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_with_meta)
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

    model = ModelClass(num_classes=num_classes, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_params:,}")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    best_mf1 = -1.0
    no_improve = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device,
                                  use_quality_loss=args.use_quality_loss,
                                  label_smoothing=args.label_smoothing,
                                  spec_augment=spec_augment)
        va_loss, va_y, va_p = run_epoch(model, val_loader, device=device)
        if scheduler is not None:
            scheduler.step()

        cm = confusion_matrix(num_classes, va_y, va_p)
        mf1 = macro_f1(cm)
        lr_now = scheduler.get_last_lr()[0] if scheduler is not None else args.lr

        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        print(f"  CM [{'/'.join(labels_list)}]:", cm.tolist())

        if mf1 > best_mf1:
            best_mf1 = mf1
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "best_macro_f1": best_mf1,
                "num_classes": num_classes,
                "labels": labels_list,
                "min_quality": mq,
                "use_quality_weighting": args.use_quality_weighting,
                "use_quality_loss": args.use_quality_loss,
                "noise_mix_prob": args.noise_mix_prob,
                "snr_range": [args.snr_low, args.snr_high],
                "gain_jitter_db": args.gain_jitter_db,
            }, out_path)
            print(f"  ** Saved best -> {out_path} (macroF1={best_mf1:.4f})")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  [Early stop] No improvement for {args.patience} epochs.")
                break

    print(f"[DONE] Best macroF1 = {best_mf1:.4f}")


if __name__ == "__main__":
    main()
