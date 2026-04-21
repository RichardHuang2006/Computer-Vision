"""Classification training loop: AMP, schedulers, checkpointing."""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .data import build_imagenet100_loaders
from .utils import ETATracker, AverageMeter, accuracy, format_hms, save_checkpoint, load_checkpoint

ModelFactory = Callable[..., nn.Module]


@dataclass
class TrainDefaults:
    model_name: str
    arch: str = "resnet50"
    batch_size: int = 128
    epochs: int = 90
    lr: float = 0.1
    min_lr: float = 0.0
    weight_decay: float = 1e-4
    momentum: float = 0.9
    optimizer: str = "sgd"  # "sgd" | "adamw"
    scheduler: str = "cosine"  # "cosine" | "step" | "none"
    warmup_epochs: int = 0
    step_size: int = 30
    gamma: float = 0.1
    label_smoothing: float = 0.0
    num_workers: int = 8
    image_size: int = 224
    grad_clip: float = 0.0
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    strong_aug: bool = False


def _mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample().item()
    lam = float(lam)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def _mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def _build_parser(defaults: TrainDefaults) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Train {defaults.model_name}")
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--arch", type=str, default=defaults.arch)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--lr", type=float, default=defaults.lr)
    p.add_argument("--min-lr", type=float, default=defaults.min_lr)
    p.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    p.add_argument("--momentum", type=float, default=defaults.momentum)
    p.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default=defaults.optimizer,
    )
    p.add_argument(
        "--scheduler",
        choices=("cosine", "step", "none"),
        default=defaults.scheduler,
    )
    p.add_argument("--warmup-epochs", type=int, default=defaults.warmup_epochs)
    p.add_argument("--step-size", type=int, default=defaults.step_size)
    p.add_argument("--gamma", type=float, default=defaults.gamma)
    p.add_argument("--label-smoothing", type=float, default=defaults.label_smoothing)
    p.add_argument("--num-workers", type=int, default=defaults.num_workers)
    p.add_argument("--image-size", type=int, default=defaults.image_size)
    p.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    p.add_argument("--mixup-alpha", type=float, default=defaults.mixup_alpha)
    p.add_argument("--strong-aug", action="store_true", default=defaults.strong_aug)
    p.add_argument(
        "--amp",
        choices=("on", "off"),
        default="on",
        help="fp16 autocast + GradScaler; disable for networks with LRN (e.g. AlexNet)",
    )
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    return p


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_at_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
    scheduler: str,
    step_size: int,
    gamma: float,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / max(warmup_epochs, 1)
    e = epoch - warmup_epochs
    te = max(total_epochs - warmup_epochs, 1)
    if scheduler == "cosine":
        t = e / te
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    if scheduler == "step":
        return base_lr * (gamma ** (e // step_size))
    return base_lr


def run(
    model_factory: ModelFactory,
    defaults: TrainDefaults,
    args: Optional[argparse.Namespace] = None,
) -> None:
    if args is None:
        args = _build_parser(defaults).parse_args()
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, num_classes = build_imagenet100_loaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_strong_aug=args.strong_aug,
    )

    model = model_factory(args.arch, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    use_amp = args.amp == "on" and device.type == "cuda"
    scaler = GradScaler(device.type, enabled=use_amp)

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_acc1 = float(ckpt.get("best_metric", 0.0))

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
    except ImportError:
        pass

    eta_tr = ETATracker()
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        lr = _lr_at_epoch(
            epoch,
            total_epochs,
            args.lr,
            args.min_lr,
            args.warmup_epochs,
            args.scheduler,
            args.step_size,
            args.gamma,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        loss_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        t0 = time.time()

        for step, (images, targets) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if args.mixup_alpha > 0:
                images, ya, yb, lam = _mixup_data(
                    images, targets, args.mixup_alpha, device
                )
            else:
                ya, yb, lam = targets, targets, 1.0

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(images)
                if isinstance(out, tuple):
                    logits, *aux = out
                    if args.mixup_alpha > 0:
                        loss_main = _mixup_criterion(criterion, logits, ya, yb, lam)
                        loss_aux = sum(
                            _mixup_criterion(criterion, a, ya, yb, lam) for a in aux
                        )
                    else:
                        loss_main = criterion(logits, targets)
                        loss_aux = sum(criterion(a, targets) for a in aux)
                    loss = loss_main + 0.3 * loss_aux
                    logits = logits
                elif args.mixup_alpha > 0:
                    loss = _mixup_criterion(criterion, out, ya, yb, lam)
                    logits = out
                else:
                    loss = criterion(out, targets)
                    logits = out

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_m.update(loss.item(), images.size(0))
            if args.mixup_alpha <= 0:
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                top1_m.update(acc1, images.size(0))
                top5_m.update(acc5, images.size(0))

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(
                    f"  epoch {epoch + 1}/{total_epochs} step {step + 1}/{len(train_loader)} "
                    f"loss={loss_m.avg:.4f} lr={lr:.2e}",
                    flush=True,
                )

        train_time = time.time() - t0
        eta_tr.update(train_time)
        # val
        model.eval()
        v_loss = AverageMeter()
        v1 = AverageMeter()
        v5 = AverageMeter()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = criterion(logits, targets)
                acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
                v_loss.update(loss.item(), images.size(0))
                v1.update(acc1, images.size(0))
                v5.update(acc5, images.size(0))

        print(
            f"Epoch {epoch + 1}/{total_epochs}  train_loss={loss_m.avg:.4f}  "
            f"val_loss={v_loss.avg:.4f}  val_top1={v1.avg * 100:.2f}  val_top5={v5.avg * 100:.2f}  "
            f"lr={lr:.2e}  train_t={format_hms(train_time)}  ETA~{format_hms(eta_tr.eta(total_epochs - epoch - 1))}",
            flush=True,
        )
        if writer:
            writer.add_scalar("train/loss", loss_m.avg, epoch)
            writer.add_scalar("val/loss", v_loss.avg, epoch)
            writer.add_scalar("val/top1", v1.avg, epoch)
            writer.add_scalar("val/lr", lr, epoch)

        acc1_score = v1.avg
        is_best = acc1_score > best_acc1
        best_acc1 = max(best_acc1, acc1_score)
        ckpt_path = out_dir / "last.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            best_acc1,
            extra={"args": vars(args), "num_classes": num_classes},
        )
        if is_best:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_acc1,
                extra={"args": vars(args), "num_classes": num_classes},
            )

    if writer:
        writer.close()


def main_stub() -> None:
    raise RuntimeError("Use per-architecture train.py")
