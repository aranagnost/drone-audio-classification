# ml/train/eval_stage1_only.py
import argparse
import torch
from torch.utils.data import DataLoader

from ml.utils.audio_dataset import AudioDataset, AudioConfig
from ml.models.small_cnn import SmallCNN
from ml.utils.train_utils import confusion_matrix, f1_from_cm

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ml/models/stage1_smallcnn.pt")
    ap.add_argument("--test_csv", default="ml/splits/test.csv")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = AudioConfig(**ckpt["cfg"])

    model = SmallCNN(num_classes=2)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = AudioDataset(args.test_csv, task="stage1", cfg=cfg)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4)

    ys, ps = [], []
    for x, y, meta in loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        ys.append(torch.as_tensor(y))
        ps.append(pred.cpu())
    y = torch.cat(ys)
    p = torch.cat(ps)

    cm = confusion_matrix(2, y, p)
    acc = (y == p).float().mean().item()
    f1d = f1_from_cm(cm, 1)

    print("=== Stage 1 TEST ===")
    print("Confusion (rows=true, cols=pred):\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1(drone): {f1d:.4f}")

if __name__ == "__main__":
    main()
