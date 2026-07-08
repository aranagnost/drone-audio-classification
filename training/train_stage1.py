# training/train_stage1.py
"""
Consolidated stage-1 training script (binary drone detection).

Usage:
    python training/train_stage1.py --model small_cnn_v2
    python training/train_stage1.py --model small_cnn_v2 --lr 1e-3
    python training/train_stage1.py --model big_cnn_v1 --eval --out artifacts/checkpoints/stage1_bigcnnv1.pt

Two-pass argparse: --model is parsed first, train_config.json["stage1"] is loaded,
then parser.set_defaults(**config) so CLI args always win.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
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
from data.train_utils import confusion_matrix, f1_from_cm, macro_f1, set_seed

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_model_class(model_name: str, model_source: str | None = None):
    """Dynamically import model.py from models/<model_source or model_name>/."""
    src = model_source or model_name
    model_py = MODELS_DIR / src / "model.py"
    spec = importlib.util.spec_from_file_location(f"{src}_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train_config(model_name: str, stage: str = "stage1") -> dict:
    cfg_path = MODELS_DIR / model_name / "train_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        full = json.load(f)
    return full.get(stage, {})


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
              label_smoothing=0.0, spec_augment=None, collect_meta=False):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    ys, ps, all_meta = [], [], []

    for x, y, meta in loader:
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

        if collect_meta:
            for i in range(x.size(0)):
                all_meta.append({k: (v[i] if isinstance(v, list) else v[i].item())
                                  for k, v in meta.items()})

    return total_loss / len(loader.dataset), torch.cat(ys), torch.cat(ps), all_meta


def print_error_breakdown(va_y, va_p, all_meta):
    """Print per-epoch FN quality and FP subtype breakdowns (compact single lines)."""
    fn_quality: dict = defaultdict(lambda: [0, 0])   # quality -> [fn, total]
    fp_subtype: dict = defaultdict(lambda: [0, 0])   # subtype -> [fp, total]

    for y_i, p_i, m in zip(va_y.tolist(), va_p.tolist(), all_meta):
        if m.get("binary_label") == "drone":
            q = m.get("quality") or "?"
            fn_quality[q][1] += 1
            if p_i == 0:
                fn_quality[q][0] += 1
        elif m.get("binary_label") == "no_drone":
            sub = m.get("subtype") or "?"
            fp_subtype[sub][1] += 1
            if p_i == 1:
                fp_subtype[sub][0] += 1

    fn_parts = "  ".join(
        f"q{q}={v[0]}/{v[1]}" for q, v in sorted(fn_quality.items()) if v[0] > 0
    ) or "none"
    fp_parts = "  ".join(
        f"{s}={v[0]}/{v[1]}" for s, v in sorted(fp_subtype.items(), key=lambda x: -x[1][0]) if v[0] > 0
    ) or "none"
    print(f"  FN(drone→no_drone) by quality: {fn_parts}")
    print(f"  FP(no_drone→drone) by subtype: {fp_parts}")


def run_tune_threshold(args, model_mod, cfg_dict):
    """Sweep drone confidence thresholds on val set, then compare 0.50 vs best on test."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    cfg = AudioConfig(**ckpt["cfg"])
    model_class_name = cfg_dict.get("model_class", "")
    ModelClass = getattr(model_mod, model_class_name)
    model = ModelClass(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def collect_probs(csv_path):
        ds = AudioDataset(csv_path, task="stage1", cfg=cfg, dataset_root=args.dataset_root)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
        ys, probs = [], []
        with torch.no_grad():
            for x, y, _ in loader:
                p = torch.softmax(model(x.to(device)), dim=1)[:, 1].cpu()
                ys.extend(y.tolist())
                probs.extend(p.tolist())
        return torch.tensor(ys), torch.tensor(probs)

    print(f"\n[Threshold sweep on val: {args.val_csv}]")
    val_y, val_probs = collect_probs(args.val_csv)

    best_thresh, best_mf1 = 0.5, -1.0
    print(f"  {'thresh':>7}  {'macroF1':>8}  {'F1(drone)':>9}  {'F1(nodrone)':>11}")
    print(f"  {'-'*43}")
    for ti in range(10, 91, 5):
        t = ti / 100
        preds = (val_probs >= t).long()
        cm = confusion_matrix(2, val_y, preds)
        mf1 = macro_f1(cm)
        f1d = f1_from_cm(cm, class_index=1)
        f1n = f1_from_cm(cm, class_index=0)
        marker = " <--" if mf1 > best_mf1 else ""
        print(f"  {t:>7.2f}  {mf1:>8.4f}  {f1d:>9.4f}  {f1n:>11.4f}{marker}")
        if mf1 > best_mf1:
            best_mf1, best_thresh = mf1, t

    print(f"\n  Best val threshold: {best_thresh:.2f}  (macroF1={best_mf1:.4f})")

    print(f"\n[Test set: {args.test_csv}]")
    test_y, test_probs = collect_probs(args.test_csv)
    sep = "═" * 60
    print(sep)
    for t, label in [(0.5, f"thresh=0.50 (default)"),
                     (best_thresh, f"thresh={best_thresh:.2f} (tuned) ")]:
        preds = (test_probs >= t).long()
        cm = confusion_matrix(2, test_y, preds)
        print(f"  {label}  macroF1={macro_f1(cm):.4f}  "
              f"F1(drone)={f1_from_cm(cm,1):.4f}  F1(nodrone)={f1_from_cm(cm,0):.4f}")
        print(f"    CM: {cm.tolist()}")
    print(sep)


def run_eval(args, model_mod, cfg_dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.out, map_location=device, weights_only=False)
    cfg = AudioConfig(**ckpt["cfg"])

    model_class_name = cfg_dict.get("model_class", "")
    ModelClass = getattr(model_mod, model_class_name)
    model = ModelClass(num_classes=2, dropout=0.0).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = AudioDataset(args.test_csv, task="stage1", cfg=cfg, dataset_root=args.dataset_root)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    all_y, all_p, all_meta = [], [], []
    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            preds = model(x).argmax(1).cpu()
            all_y.extend(y if isinstance(y, list) else y.tolist())
            all_p.extend(preds.tolist())
            for i in range(len(preds)):
                all_meta.append({k: v[i] if isinstance(v, list) else v[i].item()
                                  for k, v in meta.items()})

    y_t = torch.tensor(all_y)
    p_t = torch.tensor(all_p)
    cm = confusion_matrix(2, y_t, p_t)
    mf1 = macro_f1(cm)
    f1_drone   = f1_from_cm(cm, class_index=1)
    f1_nodrone = f1_from_cm(cm, class_index=0)

    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  Evaluation on : {args.test_csv}")
    print(f"  Checkpoint    : {args.out}")
    print(f"{sep}")
    print(f"  macroF1      : {mf1:.4f}")
    print(f"  F1(drone)    : {f1_drone:.4f}")
    print(f"  F1(no_drone) : {f1_nodrone:.4f}")
    print(f"  CM [[TN,FP],[FN,TP]]: {cm.tolist()}")

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
        fp  = fp_by_sub.get(sub, 0)
        tot = total_by_sub[sub]
        print(f"  {sub:<22} {fp:>5} {tot:>7} {fp/tot:>8.1%}")

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
        fn  = fn_by_motor.get(motor, 0)
        tot = total_by_motor[motor]
        print(f"  {motor:<22} {fn:>5} {tot:>7} {fn/tot:>8.1%}")
    print()


def main():
    # --- Pass 1: parse --model only ---
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", required=True)
    pre_args, _ = pre.parse_known_args()
    model_name = pre_args.model

    # Load config defaults from train_config.json["stage1"]
    cfg_dict = load_train_config(model_name, "stage1")

    # --- Pass 2: full parser with config-driven defaults ---
    ap = argparse.ArgumentParser(
        description="Stage-1 training: binary drone detection"
    )
    ap.add_argument("--model", required=True,
                    help="Model variant name (subdirectory of models/)")
    ap.add_argument("--train_csv", default=f"data/splits/train.csv")
    ap.add_argument("--val_csv",   default=f"data/splits/val.csv")
    ap.add_argument("--test_csv",  default=f"data/splits/test.csv")
    ap.add_argument("--epochs",     type=int,   default=15)
    ap.add_argument("--batch",      type=int,   default=32)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--dropout",    type=float, default=0.0)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out",        default=None,
                    help="Output checkpoint path (default: artifacts/checkpoints/stage1_<model>.pt)")
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--num_workers", type=int,  default=4)
    ap.add_argument("--freq_mask",  type=int,   default=0,
                    help="SpecAugment frequency mask width (0 to disable)")
    ap.add_argument("--time_mask",  type=int,   default=0,
                    help="SpecAugment time mask width (0 to disable)")
    ap.add_argument("--noise_mix_prob", type=float, default=0.0,
                    help="Probability of mixing noise into drone samples (0 to disable)")
    ap.add_argument("--snr_low",    type=float, default=0.0)
    ap.add_argument("--snr_high",   type=float, default=20.0)
    ap.add_argument("--gain_jitter_db", type=float, default=0.0)
    ap.add_argument("--optimizer",  default="adam", choices=["adam", "adamw"])
    ap.add_argument("--min_quality", type=int, default=None,
                    help="Skip drone training clips below this quality (1-5)")
    ap.add_argument("--max_no_drone_per_subtype", type=int, default=None,
                    help="Cap no_drone training clips per subtype (stage1 only)")
    ap.add_argument("--exclude_subtypes", default=None,
                    help="Comma-separated no_drone subtypes to exclude from training and noise pool "
                         "(e.g. 'insects' or 'insects,lawnmowers')")
    ap.add_argument("--use_macro_f1", action="store_true",
                    help="Track macro-F1 (instead of F1(drone)) for checkpoint selection")
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate checkpoint on --test_csv (no training)")
    ap.add_argument("--tune_threshold", action="store_true",
                    help="Sweep drone confidence thresholds on val, compare 0.5 vs best on test")
    ap.add_argument("--dataset_root", default=None,
                    help="Override absolute filepaths in CSVs using relpath column "
                         "(e.g. /content/drive/MyDrive/drone-audio/datasets/Drone_Audio_Dataset/audio)")

    # Apply config-file defaults before parsing CLI
    ap.set_defaults(**cfg_dict)

    args = ap.parse_args()

    # Resolve output path
    if args.out is None:
        args.out = f"artifacts/checkpoints/stage1_{model_name}.pt"

    # Load model module
    model_source = cfg_dict.get("model_source", None)
    model_mod = load_model_class(model_name, model_source)
    model_class_name = cfg_dict.get("model_class", "")
    if not model_class_name:
        # Fallback: try to guess from module attributes
        raise ValueError(
            f"train_config.json for '{model_name}' is missing 'model_class' in stage1 section."
        )
    ModelClass = getattr(model_mod, model_class_name)

    if args.tune_threshold:
        run_tune_threshold(args, model_mod, cfg_dict)
        return

    if args.eval:
        run_eval(args, model_mod, cfg_dict)
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model={model_name} ({model_class_name})  dropout={args.dropout}")

    exclude_list = [s.strip() for s in args.exclude_subtypes.split(",")] if args.exclude_subtypes else None

    use_augmentation = args.noise_mix_prob > 0

    cfg = AudioConfig()

    if use_augmentation:
        print(f"[INFO] Noise mixing: prob={args.noise_mix_prob}  SNR=[{args.snr_low}, {args.snr_high}] dB  "
              f"gain_jitter={args.gain_jitter_db} dB")
        train_ds = AugmentedAudioDataset(
            args.train_csv, task="stage1", cfg=cfg,
            augment=True,
            noise_mix_prob=args.noise_mix_prob,
            snr_range=(args.snr_low, args.snr_high),
            gain_jitter_db=args.gain_jitter_db,
            min_quality=args.min_quality,
            max_no_drone_per_subtype=args.max_no_drone_per_subtype,
            exclude_subtypes=exclude_list,
            dataset_root=args.dataset_root,
        )
        print(f"[INFO] Noise pool: {len(train_ds.noise_paths)} samples from not_a_drone")
    else:
        train_ds = AudioDataset(args.train_csv, task="stage1", cfg=cfg,
                                min_quality=args.min_quality,
                                max_no_drone_per_subtype=args.max_no_drone_per_subtype,
                                exclude_subtypes=exclude_list,
                                dataset_root=args.dataset_root)

    if args.min_quality is not None:
        print(f"[INFO] min_quality={args.min_quality}: dropped low-quality drone clips from train")
    if exclude_list:
        print(f"[INFO] exclude_subtypes={exclude_list}: removed from training and noise pool")
    if args.max_no_drone_per_subtype is not None:
        print(f"[INFO] max_no_drone_per_subtype={args.max_no_drone_per_subtype}: "
              f"train set has {len(train_ds)} samples")

    val_ds = AudioDataset(args.val_csv, task="stage1", cfg=cfg, dataset_root=args.dataset_root)

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

    # SpecAugment
    spec_augment = []
    if args.freq_mask > 0:
        spec_augment.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=args.freq_mask))
    if args.time_mask > 0:
        spec_augment.append(torchaudio.transforms.TimeMasking(time_mask_param=args.time_mask))
    if spec_augment:
        print(f"[INFO] SpecAugment: freq_mask={args.freq_mask}, time_mask={args.time_mask}")
    else:
        spec_augment = None

    model = ModelClass(num_classes=2, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n_params:,}")

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    best_score = -1.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device,
                                   label_smoothing=args.label_smoothing,
                                   spec_augment=spec_augment)
        va_loss, va_y, va_p, va_meta = run_epoch(model, val_loader, device=device,
                                                  collect_meta=True)
        if scheduler is not None:
            scheduler.step()

        cm = confusion_matrix(2, va_y, va_p)
        mf1       = macro_f1(cm)
        f1_drone  = f1_from_cm(cm, class_index=1)
        f1_nodrone = f1_from_cm(cm, class_index=0)
        lr_now = scheduler.get_last_lr()[0] if scheduler is not None else args.lr

        if args.use_macro_f1:
            score = mf1
            print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
                  f"| macroF1 {mf1:.4f} | F1(drone) {f1_drone:.4f} | F1(no_drone) {f1_nodrone:.4f} | lr {lr_now:.2e}")
        else:
            score = f1_drone
            print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
                  f"| F1(drone) {f1_drone:.4f} | lr {lr_now:.2e}")
        print("  CM:", cm.tolist())
        print_error_breakdown(va_y, va_p, va_meta)

        if score > best_score:
            best_score = score
            ckpt = {
                "model_state": model.state_dict(),
                "cfg": cfg.__dict__,
                "epoch": epoch,
                "noise_mix_prob": args.noise_mix_prob,
                "snr_range": [args.snr_low, args.snr_high],
                "gain_jitter_db": args.gain_jitter_db,
            }
            if args.use_macro_f1:
                ckpt["best_macro_f1"] = best_score
            else:
                ckpt["best_f1_drone"] = best_score
            torch.save(ckpt, out_path)
            print(f"  ** Saved best -> {out_path} ({'macroF1' if args.use_macro_f1 else 'F1_drone'}={best_score:.4f})")

    metric_name = "macroF1" if args.use_macro_f1 else "F1(drone)"
    print(f"[DONE] Best {metric_name} = {best_score:.4f}")


if __name__ == "__main__":
    main()
