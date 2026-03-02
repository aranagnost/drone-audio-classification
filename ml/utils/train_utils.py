# ml/utils/train_utils.py
from __future__ import annotations
import torch


@torch.no_grad()
def confusion_matrix(num_classes: int, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        cm[int(t), int(p)] += 1
    return cm


def f1_from_cm(cm: torch.Tensor, class_index: int) -> float:
    # binary-style F1 for a given class (one-vs-all)
    tp = cm[class_index, class_index].item()
    fp = cm[:, class_index].sum().item() - tp
    fn = cm[class_index, :].sum().item() - tp
    denom = (2 * tp + fp + fn)
    return (2 * tp / denom) if denom > 0 else 0.0


def macro_f1(cm: torch.Tensor) -> float:
    f1s = [f1_from_cm(cm, i) for i in range(cm.shape[0])]
    return float(sum(f1s) / len(f1s))


def set_seed(seed: int = 42):
    import random
    import os
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
