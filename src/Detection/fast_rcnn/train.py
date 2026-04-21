"""Same training loop as Faster R-CNN; uses Fast R-CNN (RoI) head + RPN."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.Classification.common.utils import save_checkpoint
from src.Detection.common.data import CocoMini, collate_fn
from src.Detection.fast_rcnn.model import build_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/coco-mini"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoMini(args.data_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    model = build_model(ds.num_classes).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(opt)
            scaler.update()
            total += float(losses.detach())
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
