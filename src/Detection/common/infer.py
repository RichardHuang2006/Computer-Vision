"""Draw boxes on image for detection checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD


def draw_boxes(
    img_path: Path,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    out_path: Path,
    score_thresh: float = 0.5,
) -> None:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for b, s, lab in zip(boxes, scores, labels):
        if float(s) < score_thresh:
            continue
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{int(lab)}:{float(s):.2f}", fill="yellow")
    img.save(out_path)


def load_image_tensor(path: Path, device: torch.device, max_size: int = 800) -> tuple[torch.Tensor, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    w0, h0 = img.size
    scale = max_size / max(w0, h0)
    w, h = int(w0 * scale), int(h0 * scale)
    img = img.resize((w, h), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.to(device), (h0, w0)
