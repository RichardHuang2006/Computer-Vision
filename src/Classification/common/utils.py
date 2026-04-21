"""Meters, accuracy, checkpoints, time formatting for classification."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0


def format_hms(seconds: float) -> str:
    s = int(round(seconds))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    if h:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m:
        return f"{m:d}m{s:02d}s"
    return f"{s}s"


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)
) -> List[float]:
    """Compute top-k accuracy for batch (float, not tensor)."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res: List[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((correct_k / batch_size).item())
    return res


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: Optional[Dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    dev = device or torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location=dev, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=dev)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


class ETATracker:
    def __init__(self, window: int = 3) -> None:
        from collections import deque

        self.times: "deque[float]" = deque(maxlen=window)

    def update(self, seconds: float) -> None:
        self.times.append(seconds)

    def eta(self, epochs_remaining: int) -> float:
        if not self.times:
            return 0.0
        avg = sum(self.times) / len(self.times)
        return avg * epochs_remaining
