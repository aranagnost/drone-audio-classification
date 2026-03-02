# ml/train/eval.py
from __future__ import annotations

import argparse
import warnings
import torch
from torch.utils.data import DataLoader

from ml.utils.audio_dataset import AudioDataset, AudioConfig
from ml.models.small_cnn import SmallCNN
from ml.utils.train_utils import confusion_matrix, macro_f1, f1_from_cm

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.io")


def load_model(path: str, num_classes: int):
    ckpt = torch.load(path, map_location="cpu")
    cfg = AudioConfig(**ckpt["cfg"])
    model = SmallCNN(num_classes=num_classes)
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
    ap.add_argument("--stage1_ckpt", default="ml/models/stage1_smallcnn.pt")
    ap.add_argument("--stage2_ckpt", default="ml/models/stage2_smallcnn.pt")
    ap.add_argument("--test_csv", default="ml/splits/test.csv")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--min_quality_stage2", type=int, default=1)
    args = ap.parse_args()

    # Stage 1
    m1, cfg1 = load_model(args.stage1_ckpt, num_classes=2)
    ds1 = AudioDataset(args.test_csv, task="stage1", cfg=cfg1)
    y1, p1 = eval_model(m1, ds1, batch=args.batch, num_workers=args.num_workers)
    cm1 = confusion_matrix(2, y1, p1)
    f1_drone = f1_from_cm(cm1, 1)
    acc1 = (y1 == p1).float().mean().item()

    print("\n=== Stage 1 (drone vs no_drone) TEST ===")
    print("Confusion (rows=true, cols=pred):\n", cm1)
    print(f"Accuracy: {acc1:.4f}")
    print(f"F1(drone): {f1_drone:.4f}")

    # Stage 2
    m2, cfg2 = load_model(args.stage2_ckpt, num_classes=4)
    ds2 = AudioDataset(args.test_csv, task="stage2", cfg=cfg2, min_quality=args.min_quality_stage2)
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
