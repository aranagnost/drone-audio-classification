# training/eval.py
from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.audio_dataset import AudioDataset, AudioConfig
from data.train_utils import confusion_matrix, macro_f1, f1_from_cm

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.io")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_model_from_checkpoint(path: str, model_dir: Path, num_classes: int, dropout: float = 0.0):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = AudioConfig(**ckpt["cfg"])
    model_py = model_dir / "model.py"
    spec = importlib.util.spec_from_file_location("model_mod", str(model_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Pick first class in the module that has num_classes in its __init__ signature
    # (SmallCNN, SmallCNNv2, etc.) — user can override via --model_class if needed
    model_class_name = ckpt.get("model_class", None)
    if model_class_name:
        ModelClass = getattr(mod, model_class_name)
    else:
        # Guess: pick the first nn.Module subclass defined in the file
        import torch.nn as nn
        candidates = [v for k, v in vars(mod).items()
                      if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module]
        if not candidates:
            raise ValueError(f"No nn.Module subclass found in {model_py}")
        ModelClass = candidates[0]
    model = ModelClass(num_classes=num_classes, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def eval_model(model, ds, batch=32, num_workers=4):
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=num_workers)
    ys, ps = [], []
    for x, y, meta in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        ys.append(torch.as_tensor(y))
        ps.append(pred.cpu())
    ys = torch.cat(ys)
    ps = torch.cat(ps)
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Model variant name (subdirectory of models/)")
    ap.add_argument("--stage1_ckpt", default=None,
                    help="Stage1 checkpoint path (default: artifacts/checkpoints/stage1_<model>.pt)")
    ap.add_argument("--stage2_ckpt", default=None,
                    help="Stage2 checkpoint path (default: artifacts/checkpoints/stage2_<model>.pt)")
    ap.add_argument("--test_csv",    default="data/splits/test.csv")
    ap.add_argument("--batch",       type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_quality_stage2", type=int, default=1)
    ap.add_argument("--dataset_root", default=None,
                    help="Override absolute filepaths in CSVs using relpath column "
                         "(e.g. /content/drive/MyDrive/drone-audio/datasets/Drone_Audio_Dataset/audio)")
    args = ap.parse_args()

    model_dir = MODELS_DIR / args.model
    if args.stage1_ckpt is None:
        args.stage1_ckpt = f"artifacts/checkpoints/stage1_{args.model}.pt"
    if args.stage2_ckpt is None:
        args.stage2_ckpt = f"artifacts/checkpoints/stage2_{args.model}.pt"

    # Stage 1
    m1, cfg1 = load_model_from_checkpoint(args.stage1_ckpt, model_dir, num_classes=2)
    ds1 = AudioDataset(args.test_csv, task="stage1", cfg=cfg1, dataset_root=args.dataset_root)
    y1, p1 = eval_model(m1, ds1, batch=args.batch, num_workers=args.num_workers)
    cm1 = confusion_matrix(2, y1, p1)
    f1_drone = f1_from_cm(cm1, 1)
    acc1 = (y1 == p1).float().mean().item()

    print("\n=== Stage 1 (drone vs no_drone) TEST ===")
    print("Confusion (rows=true, cols=pred):\n", cm1)
    print(f"Accuracy: {acc1:.4f}")
    print(f"F1(drone): {f1_drone:.4f}")

    # Stage 2
    m2, cfg2 = load_model_from_checkpoint(args.stage2_ckpt, model_dir, num_classes=4)
    ds2 = AudioDataset(args.test_csv, task="stage2", cfg=cfg2, min_quality=args.min_quality_stage2, dataset_root=args.dataset_root)
    y2, p2 = eval_model(m2, ds2, batch=args.batch, num_workers=args.num_workers)
    cm2 = confusion_matrix(4, y2, p2)
    mf1 = macro_f1(cm2)
    acc2 = (y2 == p2).float().mean().item()

    print("\n=== Stage 2 (2/4/6/8 motors) TEST (drone only) ===")
    print("Confusion (rows=true, cols=pred):\n", cm2)
    print(f"Accuracy: {acc2:.4f}")
    print(f"Macro-F1: {mf1:.4f}")


if __name__ == "__main__":
    main()
