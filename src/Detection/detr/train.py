"""DETR training with Hungarian matching on IoU cost."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader

from src.Classification.common.utils import save_checkpoint
from src.Detection.common.boxes import generalized_box_iou
from src.Detection.common.data import CocoMini, collate_fn
from src.Detection.detr.model import DETR


def box_cxcywh_to_xyxy(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    cx, cy, bw, bh = x.unbind(-1)
    return torch.stack(
        [
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ],
        dim=-1,
    )


def train_step(
    model: DETR,
    images: torch.Tensor,
    targets: list,
    criterion_ce: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    logits, pred_boxes = model(images)
    b, q, nc = logits.shape
    total = logits.new_tensor(0.0)
    for i in range(b):
        tgt = targets[i]
        gt_boxes = tgt["boxes"].to(device)
        gt_labels = tgt["labels"].to(device)
        h, w = int(tgt["size"][0].item()), int(tgt["size"][1].item())
        pb = box_cxcywh_to_xyxy(pred_boxes[i], (h, w))
        if gt_boxes.numel() == 0:
            tgt_cls = torch.full((q,), nc - 1, dtype=torch.long, device=device)
            total = total + criterion_ce(logits[i], tgt_cls)
            continue
        with torch.no_grad():
            giou = generalized_box_iou(pb, gt_boxes)
            cost = (-giou).detach().cpu().numpy()
            r, c = linear_sum_assignment(cost)
        tgt_cls = torch.full((q,), nc - 1, dtype=torch.long, device=device)
        for ri, ci in zip(r, c):
            tgt_cls[ri] = gt_labels[ci]
        loss_cls = criterion_ce(logits[i], tgt_cls)
        loss_bbox = logits.new_tensor(0.0)
        for ri, ci in zip(r, c):
            loss_bbox = loss_bbox + F.l1_loss(pb[ri], gt_boxes[ci])
        n_m = max(len(r), 1)
        total = total + loss_cls + loss_bbox / n_m
    return total / b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/coco-mini"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CocoMini(args.data_dir)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    model = DETR(ds.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        tot = 0.0
        for imgs, targets in loader:
            imgs_b = torch.stack(imgs, dim=0).to(device)
            opt.zero_grad(set_to_none=True)
            loss = train_step(model, imgs_b, list(targets), ce, device)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
        print(f"epoch {epoch+1} loss={tot/max(len(loader),1):.4f}", flush=True)
        save_checkpoint(
            Path(args.out_dir) / "last.pt",
            model,
            opt,
            epoch,
            tot,
            extra={"num_classes": ds.num_classes},
        )


if __name__ == "__main__":
    main()
