"""Linear probe on frozen ResNet-50 after MoCo v1 pre-training."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.Classification.common.data import build_imagenet100_loaders
from src.Classification.common.utils import AverageMeter, accuracy, save_checkpoint
from src.SelfSupervised.simclr.model import ResNetEncoder


class ProbeClassifier(nn.Module):
    def __init__(self, encoder: ResNetEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(encoder.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.fc(z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument(
        "--pretrained",
        type=Path,
        required=True,
        help="moco_pretrained.pt from pretrain.py",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    enc = ResNetEncoder().to(device)
    enc.resnet.load_state_dict(ckpt["encoder"], strict=True)

    train_loader, val_loader, num_classes = build_imagenet100_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    model = ProbeClassifier(enc, num_classes).to(device)
    opt = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = crit(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_m.update(loss.item(), images.size(0))

        model.eval()
        v1 = AverageMeter()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                a1, _ = accuracy(logits, targets, topk=(1, 5))
                v1.update(a1, images.size(0))
        print(
            f"epoch {epoch + 1} val_top1={v1.avg * 100:.2f} loss={loss_m.avg:.4f}",
            flush=True,
        )
        if v1.avg > best:
            best = v1.avg
            save_checkpoint(
                out_dir / "probe_best.pt",
                model,
                opt,
                epoch,
                best,
                extra={"num_classes": num_classes},
            )


if __name__ == "__main__":
    main()
