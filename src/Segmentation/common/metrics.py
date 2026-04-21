"""mIoU and pixel accuracy."""
from __future__ import annotations

import torch


@torch.no_grad()
def confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
) -> torch.Tensor:
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    k = (targets * num_classes + preds).long()
    bincount = torch.bincount(k, minlength=num_classes**2)
    return bincount.view(num_classes, num_classes).float()


def miou_from_confusion(conf: torch.Tensor, eps: float = 1e-7) -> tuple[float, torch.Tensor]:
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter + eps
    iou = inter / union
    miou = iou.mean().item()
    return miou, iou


def pixel_accuracy(conf: torch.Tensor) -> float:
    total = conf.sum()
    correct = conf.diag().sum()
    return (correct / (total + 1e-7)).item()
