"""Segmentation inference — save overlay PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD

# VOC12 palette (class 0 = bg)
VOC_PALETTE = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(VOC_PALETTE)):
        color[mask == i] = VOC_PALETTE[i]
    return color


def run_infer(
    model: torch.nn.Module,
    image_path: Path,
    out_path: Path,
    device: torch.device,
) -> None:
    model.eval()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    t = TF.to_tensor(img)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t)
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    # upsample pred to original size
    pred_img = Image.fromarray(pred, mode="L")
    pred_img = pred_img.resize((w, h), Image.NEAREST)
    pred = np.array(pred_img)
    color = colorize_mask(pred)
    Image.fromarray(color).save(out_path)


def cli_load_and_run(model_builder, default_ckpt_help: str = "") -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_builder()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    run_infer(model, args.image, args.out, device)
