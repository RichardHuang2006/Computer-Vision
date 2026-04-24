"""YOLOv1 training on COCO-mini."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.Classification.common.utils import format_hms, save_checkpoint
from src.Detection.common.data import CocoMini, collate_fn
from src.Detection.common.targets import yolo_assign
from src.Detection.yolov1.model import YOLOv1


def pad_batch(images) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Pad variable-size CHW image tensors to a common batch size."""
    sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
    max_h = max(h for h, _ in sizes)
    max_w = max(w for _, w in sizes)
    padded = [
        F.pad(img, (0, max_w - w, 0, max_h - h), value=0.0)
        for img, (h, w) in zip(images, sizes)
    ]
    return torch.stack(padded, dim=0), sizes


def yolo_loss(pred: torch.Tensor, target: torch.Tensor, C: int) -> torch.Tensor:
    """Simplified YOLO loss: coord + obj + class (B=1)."""
    pxy = pred[..., :5]
    pcls = pred[..., 5 : 5 + C]
    txy = target[..., :5]
    tcls = target[..., 5 : 5 + C]
    obj_mask = (txy[..., 4] > 0.5).float()
    noobj_mask = 1.0 - obj_mask
    coord = ((pxy[..., :4] - txy[..., :4]) ** 2).sum(dim=-1) * obj_mask
    conf = (pxy[..., 4] - txy[..., 4]) ** 2
    loss_conf = (conf * obj_mask).mean() + 0.5 * (conf * noobj_mask).mean()
    loss_cls = F.binary_cross_entropy_with_logits(pcls, tcls, reduction="none")
    loss_cls = (loss_cls * obj_mask.unsqueeze(-1)).mean()
    return 5.0 * coord.mean() + loss_conf + loss_cls


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/coco-mini"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print batch progress (img/s, ETA) every N steps; 0 disables.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoMini(args.data_dir, max_size=args.image_size)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = YOLOv1(num_classes=ds.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    S = 7
    steps_per_epoch = len(loader)
    epoch_wall_times: list[float] = []
    print(
        f"device={device} steps/epoch={steps_per_epoch} batch_size={args.batch_size} epochs={args.epochs}",
        flush=True,
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        t_epoch = time.time()
        t_log = t_epoch
        imgs_since_log = 0
        for batch_idx, (images, targets) in enumerate(loader):
            images, sizes = pad_batch(images)
            images = images.to(device)
            bs = images.size(0)
            targ = torch.zeros(bs, S, S, 5 + ds.num_classes, device=device)
            for i in range(bs):
                t = targets[i]
                if t["boxes"].numel() == 0:
                    continue
                h, w = sizes[i]
                targ[i] = yolo_assign(
                    t["boxes"].to(device),
                    t["labels"].to(device),
                    S,
                    h,
                    w,
                    ds.num_classes,
                )
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                pred = model(images)
                loss = yolo_loss(pred, targ, ds.num_classes)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            imgs_since_log += bs

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
                    f"  ep {epoch + 1} step {done}/{steps_per_epoch} "
                    f"loss_batch={float(loss.detach()):.4f} img/s={ips:.1f} "
                    f"eta_epoch={format_hms(eta_epoch_s)} eta_run={format_hms(eta_run_s)}",
                    flush=True,
                )
                t_log = now
                imgs_since_log = 0

        epoch_wall = time.time() - t_epoch
        epoch_wall_times.append(epoch_wall)
        avg_epoch = sum(epoch_wall_times) / len(epoch_wall_times)
        rem_epochs = args.epochs - (epoch + 1)
        eta_run_rem = avg_epoch * rem_epochs
        print(
            f"epoch {epoch + 1}/{args.epochs} loss={total / max(len(loader), 1):.4f} "
            f"wall={format_hms(epoch_wall)} eta_run_rem={format_hms(eta_run_rem)}",
            flush=True,
        )
        save_checkpoint(
            Path(args.out_dir) / "last.pt",
            model,
            opt,
            epoch,
            total,
            extra={"num_classes": ds.num_classes},
        )


if __name__ == "__main__":
    main()
