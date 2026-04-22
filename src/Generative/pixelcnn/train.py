"""Train PixelCNN on CIFAR-10 (negative log-likelihood / bits per dimension)."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.Classification.common.utils import AverageMeter, ETATracker, format_hms, load_checkpoint, save_checkpoint
from src.Generative.pixelcnn.data import build_cifar10_loaders
from src.Generative.pixelcnn.model import PixelCNN


def lr_at_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    t = min(max(t, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def bits_per_dim(loss_nats: float) -> float:
    """CE with default reduction='mean' is nats per subpixel."""
    return float(loss_nats) / math.log(2.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--n-filters", type=int, default=126, help="Bottleneck F; 2F must be divisible by 3")
    p.add_argument("--n-res", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log-interval", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_cifar10_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )

    h = 2 * args.n_filters
    if h % 3 != 0:
        raise SystemExit(f"--n-filters={args.n_filters} gives 2F={h}; choose n_filters so 2F % 3 == 0")

    model = PixelCNN(in_channels=3, image_size=32, n_filters=args.n_filters, n_res=args.n_res).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=use_amp)

    start_epoch = 0
    best_val_bits = float("inf")
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, opt, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_bits = float(ckpt.get("best_metric", float("inf")))

    eta_tr = ETATracker()
    total_epochs = args.epochs

    for epoch in range(start_epoch, total_epochs):
        lr = lr_at_epoch(epoch, total_epochs, args.lr, args.min_lr, args.warmup_epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for step, (x01, target) in enumerate(train_loader):
            x = (x01 * 2.0 - 1.0).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, target)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_m.update(loss.item(), x.size(0))

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                bpd = bits_per_dim(loss_m.avg)
                print(
                    f"  epoch {epoch + 1}/{total_epochs} step {step + 1}/{len(train_loader)} "
                    f"nll={loss_m.avg:.4f} bpd={bpd:.4f} lr={lr:.2e}",
                    flush=True,
                )

        train_time = time.time() - t0
        eta_tr.update(train_time)

        model.eval()
        v_m = AverageMeter()
        with torch.no_grad():
            for x01, target in val_loader:
                x = (x01 * 2.0 - 1.0).to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, target)
                v_m.update(loss.item(), x.size(0))

        val_bpd = bits_per_dim(v_m.avg)
        is_best = val_bpd < best_val_bits
        best_val_bits = min(best_val_bits, val_bpd)

        print(
            f"Epoch {epoch + 1}/{total_epochs}  train_nll={loss_m.avg:.4f}  val_nll={v_m.avg:.4f}  "
            f"val_bpd={val_bpd:.4f}  best_val_bpd={best_val_bits:.4f}  lr={lr:.2e}  "
            f"train_t={format_hms(train_time)}  ETA~{format_hms(eta_tr.eta(total_epochs - epoch - 1))}",
            flush=True,
        )

        extra = {
            "args": vars(args),
            "arch": "pixelcnn_cifar",
            "n_filters": args.n_filters,
            "n_res": args.n_res,
            "image_size": 32,
        }
        save_checkpoint(
            out_dir / "last.pt",
            model,
            opt,
            epoch,
            best_val_bits,
            extra=extra,
        )
        if is_best:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                opt,
                epoch,
                best_val_bits,
                extra=extra,
            )


if __name__ == "__main__":
    main()
