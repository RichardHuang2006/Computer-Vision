"""DETR inference — top queries by objectness (non-background)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD
from src.Detection.common.infer import draw_boxes
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score-thresh", type=float, default=0.3)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    extra = ckpt.get("extra") or {}
    nc = int(extra.get("num_classes", 40))
    m = DETR(nc).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    img = Image.open(args.image).convert("RGB")
    w0, h0 = img.size
    sc = 800 / max(w0, h0)
    rw, rh = int(w0 * sc), int(h0 * sc)
    img_r = img.resize((rw, rh), Image.BILINEAR)
    t = TF.to_tensor(img_r)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, boxes = m(t)
    prob = logits.softmax(-1)[0]
    scores, labels = prob[:, :-1].max(-1)
    boxes_xy = box_cxcywh_to_xyxy(boxes[0], (rh, rw))
    keep = scores > args.score_thresh
    sx, sy = w0 / rw, h0 / rh
    bx = boxes_xy[keep]
    bx[:, 0] *= sx
    bx[:, 2] *= sx
    bx[:, 1] *= sy
    bx[:, 3] *= sy
    draw_boxes(
        args.image,
        bx.cpu(),
        scores[keep].cpu(),
        labels[keep].cpu().long(),
        args.out,
        score_thresh=0.0,
    )


if __name__ == "__main__":
    main()
