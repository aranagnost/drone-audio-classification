"""PaSST fine-tuning for stage2 (drone motor count).

Mirrors training/train_ast.py, but:
  - Dataset: PaSSTAudioDataset (waveforms at 32 kHz)
  - Model:   PaSSTClassifier (loads via hear21passt)
  - No HuggingFace ASTFeatureExtractor needed.

Usage (Kaggle):
    python training/train_passt.py \\
        --task stage2 --context_seconds 10 \\
        --dataset_root /kaggle/input/.../audio \\
        --cache_dir /kaggle/working/ast_cache_10s \\
        --out artifacts/checkpoints/stage2_passt_v1.pt \\
        --freeze_encoder --unfreeze_top_n 3 \\
        --lr 5e-6 --dropout 0.3 --mlp_head --epochs 5
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import warnings
from pathlib import Path

import csv as csv_mod
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import ASTFeatureExtractor  # used only as a 16 kHz sample-rate carrier

from data.passt_dataset import PaSSTAudioDataset
from data.train_utils import confusion_matrix, f1_from_cm, macro_f1, set_seed

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

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
        return json.load(f).get(stage, {})


def make_stage1_sampler_from_rows(rows):
    """Stage 1 weighted sampler aligned to dataset rows (post --max_no_drone_per_subtype)."""
    labels = []
    counts = {"no_drone": 0, "drone": 0}
    for r in rows:
        lbl = r["binary_label"]
        counts[lbl] = counts.get(lbl, 0) + 1
        labels.append(0 if lbl == "no_drone" else 1)
    class_counts = [counts.get("no_drone", 0), counts.get("drone", 0)]
    weights = [1.0 / max(class_counts[y], 1) for y in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True), class_counts


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
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", default="passt_v1")
    pre.add_argument("--task", default="stage2", choices=["stage1", "stage2"])
    pre_args, _ = pre.parse_known_args()
    cfg = load_train_config(pre_args.model, pre_args.task)
    task = pre_args.task

    ap = argparse.ArgumentParser(description="PaSST fine-tuning")
    ap.add_argument("--model", default="passt_v1")
    ap.add_argument("--task",  default="stage2", choices=["stage1", "stage2"])
    ap.add_argument("--train_csv", default="data/splits/train.csv" if task == "stage1"
                                  else "data/splits_stage2/train.csv")
    ap.add_argument("--val_csv",   default="data/splits/val.csv"   if task == "stage1"
                                  else "data/splits_stage2/val.csv")
    ap.add_argument("--epochs",    type=int, default=cfg.get("epochs", 5))
    ap.add_argument("--batch",     type=int, default=cfg.get("batch", 8))
    ap.add_argument("--lr",        type=float, default=cfg.get("lr", 5e-6))
    ap.add_argument("--weight_decay", type=float, default=cfg.get("weight_decay", 1e-2))
    ap.add_argument("--label_smoothing", type=float, default=cfg.get("label_smoothing", 0.1))
    ap.add_argument("--dropout",   type=float, default=cfg.get("dropout", 0.3))
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--num_workers", type=int, default=cfg.get("num_workers", 2))
    ap.add_argument("--min_quality", type=int, default=None)
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--unfreeze_top_n", type=int, default=0,
                    help="Unfreeze last N transformer blocks of the PaSST backbone.")
    ap.add_argument("--mlp_head", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="For stage1: oversample the minority drone class.")
    ap.add_argument("--max_no_drone_per_subtype", type=int, default=None,
                    help="For stage1: cap no_drone clips per subtype during training.")
    ap.add_argument("--dataset_root", default=None)
    ap.add_argument("--cache_dir",    default=None)
    ap.add_argument("--context_seconds", type=float, default=10.0)
    ap.add_argument("--patience", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}  model={args.model}  task={args.task}")

    # Load model class (PaSSTClassifier)
    model_mod = load_model_class(args.model)
    model_class_name = cfg.get("model_class", "PaSSTClassifier")
    ModelClass = getattr(model_mod, model_class_name)
    num_classes = 2 if args.task == "stage1" else 4

    # 16 kHz extractor — only used to seed the parent dataset's internal sample_rate.
    extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    train_ds = PaSSTAudioDataset(
        args.train_csv, task=args.task, extractor=extractor,
        dataset_root=args.dataset_root, min_quality=args.min_quality,
        max_no_drone_per_subtype=args.max_no_drone_per_subtype,
        cache_dir=args.cache_dir, augment=False,  # SpecAugment is for mel; PaSST does its own internally
        context_seconds=args.context_seconds,
    )
    val_ds = PaSSTAudioDataset(
        args.val_csv, task=args.task, extractor=extractor,
        dataset_root=args.dataset_root, min_quality=args.min_quality,
        cache_dir=args.cache_dir, context_seconds=args.context_seconds,
    )
    print(f"[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}")

    if train_ds._stitch_enabled:
        print("[INFO] Prestitching train groups...", flush=True)
        train_ds.prestitch_groups(verbose=True)
    if val_ds._stitch_enabled:
        print("[INFO] Prestitching val groups...", flush=True)
        val_ds.prestitch_groups(verbose=True)

    if args.use_weighted_sampler and args.task == "stage1":
        sampler, counts = make_stage1_sampler_from_rows(train_ds.rows)
        print(f"[INFO] Class counts (post-filter): no_drone={counts[0]} drone={counts[1]}  "
              f"(sampler aligned to dataset of {len(train_ds)} rows)")
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                                  num_workers=args.num_workers)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                                  num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

    model = ModelClass(num_classes=num_classes, dropout=args.dropout,
                       mlp_head=args.mlp_head).to(device)

    if args.freeze_encoder:
        for p in model.parameters():
            p.requires_grad_(False)
        for name, p in model.named_parameters():
            if "classifier" in name:
                p.requires_grad_(True)
        if args.unfreeze_top_n > 0:
            n_layers = 12  # PaSST-S has 12 transformer blocks
            unfreeze_from = n_layers - args.unfreeze_top_n
            for name, p in model.named_parameters():
                # Unfreeze last N transformer blocks: matches names like
                # "backbone.net.blocks.{i}." — defensively check substring.
                for i in range(unfreeze_from, n_layers):
                    if f"blocks.{i}." in name:
                        p.requires_grad_(True)
                        break
                # Final norm after the transformer stack
                if name.endswith(".norm.weight") or name.endswith(".norm.bias"):
                    p.requires_grad_(True)
            print(f"[INFO] PaSST backbone partially frozen — training last "
                  f"{args.unfreeze_top_n} blocks + norm + classifier head")
        else:
            print("[INFO] PaSST backbone frozen — training classifier head only")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {n_params:,} total, {n_trainable:,} trainable")

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
        tr_loss, _, _ = run_epoch(model, train_loader, optimizer=optimizer,
                                  device=device,
                                  label_smoothing=args.label_smoothing,
                                  grad_clip=args.grad_clip)
        va_loss, va_y, va_p = run_epoch(model, val_loader, device=device)
        scheduler.step()

        cm = confusion_matrix(num_classes, va_y, va_p)
        mf1 = macro_f1(cm)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
              f"| macroF1 {mf1:.4f} | lr {lr_now:.2e}")
        print(f"  CM: {cm.tolist()}")

        if mf1 > best_score:
            best_score = mf1
            no_improve = 0
            ckpt = {
                "model_state": model.state_dict(),
                "model_class": model_class_name,
                "task": args.task,
                "num_classes": num_classes,
                "epoch": epoch,
                "best_score": best_score,
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
