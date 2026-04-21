"""Faster R-CNN inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD
from src.Detection.common.infer import draw_boxes
from src.Detection.faster_rcnn.model import build_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--score-thresh", type=float, default=0.5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ncls = int(ckpt.get("extra", {}).get("num_classes", 40)) if isinstance(
        ckpt.get("extra"), dict
    ) else int(ckpt.get("num_classes", 40))
    model = build_model(ncls).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    from PIL import Image

    img = Image.open(args.image).convert("RGB")
    w0, h0 = img.size
    m = 800 / max(w0, h0)
    rw, rh = int(w0 * m), int(h0 * m)
    img = img.resize((rw, rh), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).to(device)
    with torch.no_grad():
        out = model([t])[0]
    boxes = out["boxes"].cpu().clone()
    scores = out["scores"].cpu()
    labels = out["labels"].cpu()
    sx, sy = w0 / rw, h0 / rh
    boxes[:, 0] *= sx
    boxes[:, 2] *= sx
    boxes[:, 1] *= sy
    boxes[:, 3] *= sy
    draw_boxes(args.image, boxes, scores, labels, args.out, score_thresh=args.score_thresh)


if __name__ == "__main__":
    main()
