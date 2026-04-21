"""Segmentation training with poly LR."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.Classification.common.utils import format_hms, save_checkpoint
from src.Segmentation.common.data import build_voc_loaders
from src.Segmentation.common.metrics import confusion_matrix, miou_from_confusion


@dataclass
class SegTrainDefaults:
    model_name: str = "fcn"
    batch_size: int = 8
    epochs: int = 50
    lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    num_workers: int = 4
    crop_size: int = 480


def poly_lr(base: float, iter_curr: int, max_iter: int, power: float = 0.9) -> float:
    return base * (1.0 - iter_curr / max(1, max_iter)) ** power


def run(model: nn.Module, defaults: SegTrainDefaults, args: argparse.Namespace | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/voc2012"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args() if args is None else args
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, num_classes = build_voc_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=defaults.num_workers,
        crop_size=defaults.crop_size,
    )

    opt = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=defaults.momentum,
        weight_decay=defaults.weight_decay,
    )
    crit = nn.CrossEntropyLoss(ignore_index=255)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    max_iter = args.epochs * len(train_loader)
    iter_curr = 0
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    best_miou = 0.0

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            lr = poly_lr(args.lr, iter_curr, max_iter)
            for g in opt.param_groups:
                g["lr"] = lr
            iter_curr += 1
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = crit(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # val
        model.eval()
        conf = torch.zeros(num_classes, num_classes, device=device)
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                conf += confusion_matrix(preds, masks, num_classes)
        miou, _ = miou_from_confusion(conf.cpu())
        print(
            f"epoch {epoch+1}/{args.epochs} mIoU={miou:.4f} time={format_hms(time.time()-t0)}",
            flush=True,
        )
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(
                out / "best.pt",
                model,
                opt,
                epoch,
                best_miou,
                extra={"num_classes": num_classes},
            )
        save_checkpoint(
            out / "last.pt",
            model,
            opt,
            epoch,
            best_miou,
            extra={"num_classes": num_classes},
        )
