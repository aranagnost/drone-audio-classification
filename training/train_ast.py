# training/train_ast.py
"""
AST (Audio Spectrogram Transformer) fine-tuning script.
Handles both stage1 (binary drone detection) and stage2 (motor count).

Usage:
    python training/train_ast.py --task stage1
    python training/train_ast.py --task stage2 --use_weighted_sampler
    python training/train_ast.py --task stage1 --freeze_encoder   # head-only, fast test
    python training/train_ast.py --task stage1 --eval             # evaluate saved checkpoint
    python training/train_ast.py --task stage1 --dataset_root /kaggle/input/.../audio

NOTE: AST checkpoints are not compatible with eval.py (no AudioConfig saved).
      Use --eval in this script instead.
"""
from __future__ import annotations

import argparse
import csv as csv_mod
import importlib.util
import json
import os
import warnings
from pathlib import Path

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ASTFeatureExtractor

from data.ast_dataset import ASTAudioDataset
from data.train_utils import confusion_matrix, f1_from_cm, macro_f1, set_seed

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

STAGE1_LABELS = ["no_drone", "drone"]
STAGE2_LABELS = ["2_motors", "4_motors", "6_motors", "8_motors"]


def load_model_class(model_name: str):
    model_py = MODELS_DIR / model_name / "model.py"
    spec = importlib.util.spec_from_file_location(f"{model_name}_model", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_train_config(model_name: str, stage: str) -> dict:
    cfg_path = MODELS_DIR / model_name / "train_config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        full = json.load(f)
    return full.get(stage, {})


def make_stage1_sampler(csv_path: str):
    labels = []
    counts = {"no_drone": 0, "drone": 0}
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            lbl = row["binary_label"]
            counts[lbl] += 1
            labels.append(0 if lbl == "no_drone" else 1)
    class_counts = [counts["no_drone"], counts["drone"]]
    weights = [1.0 / class_counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), class_counts


def make_stage2_sampler(csv_path: str, min_quality: int = 1):
    motor_map = {"2_motors": 0, "4_motors": 1, "6_motors": 2, "8_motors": 3}
    labels = []
    counts = [0, 0, 0, 0]
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            if row["binary_label"] != "drone":
                continue
            try:
                if int(row.get("quality", "0")) < min_quality:
                    continue
            except ValueError:
                continue
            y = motor_map[row["motor_label"]]
            counts[y] += 1
            labels.append(y)
    weights = [1.0 / counts[y] for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), counts


def run_epoch(model, loader, optimizer=None, device="cpu",
              label_smoothing=0.0, grad_clip=1.0):
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    ys, ps = [], []

    for x, y, _ in loader:
        x = x.to(device)
        y = torch.as_tensor(y, device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        ps.append(logits.argmax(1).detach().cpu())
        ys.append(y.detach().cpu())

    return total_loss / len(loader.dataset), torch.cat(ys), torch.cat(ps)


def main():
    # Pass 1: parse --model and --task to load correct config section
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="ast_v1")
    pre.add_argument("--task", default="stage1", choices=["stage1", "stage2"])
    pre_args, _ = pre.parse_known_args()
    model_name = pre_args.model
    task = pre_args.task

    cfg_dict = load_train_config(model_name, task)

    # Pass 2: full parser with config-driven defaults
    ap = argparse.ArgumentParser(description="AST fine-tuning for drone audio classification")
    ap.add_argument("--model",  default="ast_v1")
    ap.add_argument("--task",   default="stage1", choices=["stage1", "stage2"])
    ap.add_argument("--train_csv", default="data/splits/train.csv" if task == "stage1"
                                   else "data/splits_stage2/train.csv")
    ap.add_argument("--val_csv",   default="data/splits/val.csv" if task == "stage1"
                                   else "data/splits_stage2/val.csv")
    ap.add_argument("--test_csv",  default="data/splits/test.csv" if task == "stage1"
                                   else "data/splits_stage2/test.csv")
    ap.add_argument("--epochs",       type=int,   default=10)
    ap.add_argument("--batch",        type=int,   default=8)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--dropout",      type=float, default=0.0)
    ap.add_argument("--grad_clip",    type=float, default=1.0)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out",          default=None)
    ap.add_argument("--num_workers",  type=int,   default=2)
    ap.add_argument("--min_quality",  type=int,   default=None)
    ap.add_argument("--max_no_drone_per_subtype", type=int, default=None)
    ap.add_argument("--use_weighted_sampler", action="store_true")
    ap.add_argument("--use_macro_f1",         action="store_true")
    ap.add_argument("--freeze_encoder",       action="store_true",
                    help="Freeze AST encoder, train classifier head only")
    ap.add_argument("--unfreeze_top_n", type=int, default=0,
                    help="With --freeze_encoder: also unfreeze the last N transformer blocks + layernorm")
    ap.add_argument("--mlp_head", action="store_true",
                    help="Replace the linear classifier with a 2-layer MLP head (768->256->num_classes)")
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--cache_dir", default=None,
                    help="Cache pre-computed AST features to disk (e.g. /kaggle/working/ast_cache)")
    ap.add_argument("--pretrained_model",
                    default="MIT/ast-finetuned-audioset-10-10-0.4593")
    ap.add_argument("--eval", action="store_true",
                    help="Evaluate checkpoint on --test_csv (no training)")
    ap.add_argument("--patience", type=int, default=0,
                    help="Early stopping: stop if score does not improve for N epochs (0 = disabled)")

    ap.set_defaults(**cfg_dict)
    args = ap.parse_args()

    if args.out is None:
        args.out = f"artifacts/checkpoints/{task}_{model_name}.pt"

    model_mod = load_model_class(model_name)
    model_class_name = cfg_dict.get("model_class", "ASTClassifier")
    ModelClass = getattr(model_mod, model_class_name)
    num_classes = 2 if task == "stage1" else 4
    labels_list = STAGE1_LABELS if task == "stage1" else STAGE2_LABELS

    extractor = ASTFeatureExtractor.from_pretrained(args.pretrained_model)

    if args.eval:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(args.out, map_location=device, weights_only=False)
        model = ModelClass(num_classes=num_classes, dropout=0.0,
                           pretrained_model=args.pretrained_model,
                           mlp_head=args.mlp_head).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        ds = ASTAudioDataset(args.test_csv, task=task, extractor=extractor,
                             dataset_root=args.dataset_root, min_quality=args.min_quality,
                             cache_dir=args.cache_dir)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)
        _, y_t, p_t = run_epoch(model, loader, device=device)
        cm = confusion_matrix(num_classes, y_t, p_t)
        sep = "═" * 60
        print(f"\n{sep}")
        print(f"  Evaluation  : {args.test_csv}")
        print(f"  Checkpoint  : {args.out}")
        print(f"{sep}")
        print(f"  macroF1     : {macro_f1(cm):.4f}")
        for i, lbl in enumerate(labels_list):
            print(f"  F1({lbl}): {f1_from_cm(cm, i):.4f}")
        print(f"  CM: {cm.tolist()}")
        return

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model={model_name}  task={task}")
    print(f"[INFO] pretrained={args.pretrained_model}")

    if args.cache_dir:
        print(f"[INFO] Feature cache: {args.cache_dir}")

    train_ds = ASTAudioDataset(
        args.train_csv, task=task, extractor=extractor,
        dataset_root=args.dataset_root, min_quality=args.min_quality,
        max_no_drone_per_subtype=args.max_no_drone_per_subtype,
        cache_dir=args.cache_dir,
        augment=True,
    )
    val_ds = ASTAudioDataset(
        args.val_csv, task=task, extractor=extractor,
        dataset_root=args.dataset_root, min_quality=args.min_quality,
        cache_dir=args.cache_dir,
    )
    print(f"[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}")

    if args.use_weighted_sampler:
        if task == "stage1":
            sampler, counts = make_stage1_sampler(args.train_csv)
            print(f"[INFO] Class counts: no_drone={counts[0]} drone={counts[1]}")
        else:
            sampler, counts = make_stage2_sampler(args.train_csv,
                                                   min_quality=args.min_quality or 1)
            print(f"[INFO] Class counts: {counts}")
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers)

    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

    model = ModelClass(num_classes=num_classes, dropout=args.dropout,
                       pretrained_model=args.pretrained_model,
                       mlp_head=args.mlp_head).to(device)

    if args.freeze_encoder:
        # Freeze everything first
        for p in model.parameters():
            p.requires_grad_(False)
        # Always unfreeze the classifier head
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad_(True)
        if args.unfreeze_top_n > 0:
            # AST has 12 transformer blocks: encoder.layer.0 … encoder.layer.11
            n_layers = 12
            unfreeze_from = n_layers - args.unfreeze_top_n
            for name, p in model.named_parameters():
                # Unfreeze last N transformer blocks
                for i in range(unfreeze_from, n_layers):
                    if f"encoder.layer.{i}." in name:
                        p.requires_grad_(True)
                        break
                # Unfreeze layernorm after the transformer stack
                if "audio_spectrogram_transformer.layernorm" in name:
                    p.requires_grad_(True)
            print(f"[INFO] Encoder partially frozen — training last {args.unfreeze_top_n} "
                  f"transformer blocks + layernorm + classifier head")
        else:
            print("[INFO] Encoder frozen — training classifier head only")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Differential LR: encoder at lr, head at lr * 10
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "classifier" not in n]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "classifier" in n]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.lr},
        {"params": head_params,    "lr": args.lr * 10},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-7)

    best_score = -1.0
    no_improve = 0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer=optimizer, device=device,
                                  label_smoothing=args.label_smoothing,
                                  grad_clip=args.grad_clip)
        va_loss, va_y, va_p = run_epoch(model, val_loader, device=device)
        scheduler.step()

        cm = confusion_matrix(num_classes, va_y, va_p)
        mf1 = macro_f1(cm)
        lr_now = scheduler.get_last_lr()[0]

        if task == "stage1" and not args.use_macro_f1:
            score = f1_from_cm(cm, class_index=1)
            print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
                  f"| F1(drone) {score:.4f} | macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        else:
            score = mf1
            print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
                  f"| macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        print(f"  CM: {cm.tolist()}")

        if score > best_score:
            best_score = score
            no_improve = 0
            ckpt = {
                "model_state": model.state_dict(),
                "model_class": model_class_name,
                "pretrained_model": args.pretrained_model,
                "task": task,
                "num_classes": num_classes,
                "epoch": epoch,
                "best_score": best_score,
                "min_quality": args.min_quality,
            }
            torch.save(ckpt, out_path)
            print(f"  ** Saved best -> {out_path} (score={best_score:.4f})")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  [Early stop] No improvement for {args.patience} epochs.")
                break

    print(f"[DONE] Best score = {best_score:.4f}")


if __name__ == "__main__":
    main()
