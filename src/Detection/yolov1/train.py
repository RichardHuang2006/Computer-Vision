"""YOLOv1 training on COCO-mini."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.Classification.common.utils import save_checkpoint
from src.Detection.common.data import CocoMini, collate_fn
from src.Detection.common.targets import yolo_assign
from src.Detection.yolov1.model import YOLOv1


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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoMini(args.data_dir, max_size=args.image_size)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    model = YOLOv1(num_classes=ds.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    S = 7

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = torch.stack([img for img in images], dim=0).to(device)
            bs = images.size(0)
            targ = torch.zeros(bs, S, S, 5 + ds.num_classes, device=device)
            for i in range(bs):
                t = targets[i]
                if t["boxes"].numel() == 0:
                    continue
                _, h, w = images[i].shape
                targ[i] = yolo_assign(
                    t["boxes"], t["labels"], S, h, w, ds.num_classes
                )
            opt.zero_grad(set_to_none=True)
            with autocast(device.type == "cuda"):
                pred = model(images)
                loss = yolo_loss(pred, targ, ds.num_classes)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
        print(f"epoch {epoch+1} loss={total/max(len(loader),1):.4f}", flush=True)
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
