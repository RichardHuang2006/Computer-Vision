"""Generic detection train helper (optimizer loop skeleton)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.Classification.common.utils import format_hms, save_checkpoint


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    scaler: GradScaler,
    max_norm: float = 0.1,
) -> float:
    model.train()
    total = 0.0
    n = 0
    for images, targets in loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss_dict = loss_fn(model, images, targets)
            if isinstance(loss_dict, dict):
                losses = sum(loss_dict.values())
            else:
                losses = loss_dict
        scaler.scale(losses).backward()
        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        total += float(losses.detach())
        n += 1
    return total / max(n, 1)


def fit_detection(
    model: nn.Module,
    train_loader,
    val_loader,
    loss_fn: Callable,
    out_dir: Path,
    epochs: int = 24,
    lr: float = 1e-4,
    device: Optional[torch.device] = None,
) -> None:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 1e9
    for ep in range(epochs):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        print(f"epoch {ep+1}/{epochs} train_loss={tr:.4f} time={format_hms(time.time()-t0)}", flush=True)
        save_checkpoint(
            out_dir / "last.pt",
            model,
            optimizer,
            ep,
            tr,
            extra={},
        )
        if tr < best:
            best = tr
            save_checkpoint(out_dir / "best.pt", model, optimizer, ep, tr, extra={})
