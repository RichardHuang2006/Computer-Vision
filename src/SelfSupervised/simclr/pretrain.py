"""SimCLR self-supervised pre-training (ImageFolder train split)."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Tuple

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD
from src.Classification.common.utils import AverageMeter, format_hms
from src.SelfSupervised.simclr.model import SimCLR, nt_xent_loss


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


class TwoViewDataset(Dataset):
    """Returns two augmented views per image."""

    def __init__(self, root: Path) -> None:
        self.ds = datasets.ImageFolder(str(root))
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        blur = transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
        base = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([blur], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        self.tfm = transforms.Compose(base)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        path, y = self.ds.samples[idx]
        img = datasets.folder.default_loader(path)
        v1 = self.tfm(img)
        v2 = self.tfm(img)
        return v1, v2, y


def build_loader(data_dir: Path, batch_size: int, num_workers: int) -> DataLoader:
    root = Path(data_dir) / "train"
    if not root.is_dir():
        raise FileNotFoundError(f"Expected ImageFolder train root at {root}")
    ds = TwoViewDataset(root)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )


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
        help="Peak LR after warmup; default 0.3 * batch_size/256",
    )
    p.add_argument("--min-lr", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--tau", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_lr = args.lr if args.lr is not None else 0.3 * (args.batch_size / 256.0)
    loader = build_loader(args.data_dir, args.batch_size, args.num_workers)
    steps_per_epoch = len(loader)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)

    model = SimCLR().to(device)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for v1, v2, _ in loader:
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            lr = lr_at_step(global_step, warmup_steps, total_steps, base_lr, args.min_lr)
            for pg in opt.param_groups:
                pg["lr"] = lr

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                z1 = model(v1)
                z2 = model(v2)
                loss = nt_xent_loss(z1, z2, tau=args.tau)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_m.update(loss.item(), v1.size(0))
            global_step += 1

        print(
            f"epoch {epoch + 1}/{args.epochs} loss={loss_m.avg:.4f} time={format_hms(time.time() - t0)}",
            flush=True,
        )
        torch.save(
            {
                "encoder": model.encoder.resnet.state_dict(),
                "proj_head": model.proj_head.state_dict(),
                "args": vars(args),
            },
            out_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
        )

    torch.save(
        {
            "encoder": model.encoder.resnet.state_dict(),
            "proj_head": model.proj_head.state_dict(),
            "args": vars(args),
        },
        out_dir / "simclr_pretrained.pt",
    )


if __name__ == "__main__":
    main()
