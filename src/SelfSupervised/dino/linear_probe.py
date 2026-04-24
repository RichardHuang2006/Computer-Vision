"""Linear probe on frozen DINO ViT encoder (ImageNet-100 classification)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.Classification.common.data import build_imagenet100_loaders, imagenet100_class_names
from src.Classification.common.utils import AverageMeter, accuracy, format_hms, save_checkpoint
from src.SelfSupervised.dino.model import ViTEncoder


class ProbeClassifier(nn.Module):
    def __init__(self, encoder: ViTEncoder, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(encoder.embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.fc(z)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument("--pretrained", type=Path, required=True, help="dino_pretrained.pt")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print train progress (img/s, ETA) every N batches; 0 disables.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    enc = ViTEncoder(img_size=224, patch_size=16, embed_dim=384).to(device)
    enc.load_state_dict(ckpt["student_enc"], strict=True)

    train_loader, val_loader, num_classes = build_imagenet100_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    class_names = imagenet100_class_names(args.data_dir)
    model = ProbeClassifier(enc, num_classes).to(device)
    opt = torch.optim.SGD(model.fc.parameters(), lr=args.lr, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    steps_per_epoch = len(train_loader)
    epoch_wall_times: list[float] = []
    print(
        f"device={device} steps/epoch(train)={steps_per_epoch} val_batches={len(val_loader)} "
        f"epochs={args.epochs}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        t_epoch = time.time()
        t_log = t_epoch
        imgs_since_log = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
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
            imgs_since_log += images.size(0)

            if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
                now = time.time()
                done = batch_idx + 1
                dt = max(now - t_log, 1e-6)
                ips = imgs_since_log / dt
                eta_epoch_s = (steps_per_epoch - done) * (now - t_epoch) / max(done, 1)
                frac = max(done / steps_per_epoch, 1e-6)
                est_epoch_total = (now - t_epoch) / frac
                rem_full_epochs = args.epochs - epoch - 1
                eta_run_s = eta_epoch_s + rem_full_epochs * est_epoch_total
                print(
                    f"  ep {epoch + 1} train {done}/{steps_per_epoch} "
                    f"loss={loss_m.avg:.4f} img/s={ips:.1f} "
                    f"eta_epoch={format_hms(eta_epoch_s)} eta_run={format_hms(eta_run_s)}",
                    flush=True,
                )
                t_log = now
                imgs_since_log = 0

        model.eval()
        v1 = AverageMeter()
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                a1, _ = accuracy(logits, targets, topk=(1, 5))
                v1.update(a1, images.size(0))

        epoch_wall = time.time() - t_epoch
        epoch_wall_times.append(epoch_wall)
        avg_epoch = sum(epoch_wall_times) / len(epoch_wall_times)
        rem_epochs = args.epochs - (epoch + 1)
        eta_run_rem = avg_epoch * rem_epochs
        print(
            f"epoch {epoch + 1}/{args.epochs} val_top1={v1.avg * 100:.2f} loss={loss_m.avg:.4f} "
            f"wall={format_hms(epoch_wall)} eta_run_rem={format_hms(eta_run_rem)}",
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
                extra={"num_classes": num_classes, "class_names": class_names},
            )


if __name__ == "__main__":
    main()
