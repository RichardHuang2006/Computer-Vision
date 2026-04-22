"""MAE self-supervised pre-training on ImageFolder (e.g. ImageNet-100 train split)."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD
from src.Classification.common.utils import AverageMeter, format_hms
from src.SelfSupervised.mae.model import MAE


def build_train_loader(
    data_dir: Path, batch_size: int, num_workers: int
) -> DataLoader:
    root = Path(data_dir) / "train"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected ImageFolder train root at {root}")
    tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds = datasets.ImageFolder(str(root), transform=tfm)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


def lr_at_step(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    t = min(max(t, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak LR after warmup; default 1.5e-4 * batch_size/256",
    )
    p.add_argument("--min-lr", type=float, default=1.0e-6)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument(
        "--no-norm-pix-loss",
        action="store_true",
        help="Disable per-patch mean/var normalization in reconstruction loss",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_lr = args.lr if args.lr is not None else 1.5e-4 * (args.batch_size / 256.0)
    loader = build_train_loader(args.data_dir, args.batch_size, args.num_workers)
    steps_per_epoch = len(loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    model = MAE(norm_pix_loss=not args.no_norm_pix_loss).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            lr = lr_at_step(global_step, warmup_steps, total_steps, base_lr, args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _, _ = model(images, mask_ratio=args.mask_ratio)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_m.update(loss.item(), images.size(0))
            global_step += 1

        print(
            f"epoch {epoch + 1}/{args.epochs} loss={loss_m.avg:.4f} "
            f"time={format_hms(time.time() - t0)}",
            flush=True,
        )
        torch.save(
            {
                "encoder": model.encoder.state_dict(),
                "decoder": model.decoder.state_dict(),
                "args": vars(args),
            },
            out_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
        )

    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "decoder": model.decoder.state_dict(),
            "args": vars(args),
        },
        out_dir / "mae_pretrained.pt",
    )


if __name__ == "__main__":
    main()
