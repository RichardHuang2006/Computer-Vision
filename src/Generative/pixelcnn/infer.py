"""Compute bits per dimension for an image under a PixelCNN checkpoint."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from src.Generative.pixelcnn.model import PixelCNN


def load_pixelcnn(ckpt_path: Path, device: torch.device) -> PixelCNN:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ex = ckpt.get("extra") or {}
    args = ex.get("args") or {}
    n_filters = int(ex.get("n_filters", args.get("n_filters", 126)))
    n_res = int(ex.get("n_res", args.get("n_res", 15)))
    image_size = int(ex.get("image_size", 32))
    model = PixelCNN(in_channels=3, image_size=image_size, n_filters=n_filters, n_res=n_res).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pixelcnn(args.ckpt.resolve(), device)
    h = w = model.image_size

    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    x01 = tfm(img).unsqueeze(0).to(device)
    target = (x01 * 255.0).round().clamp(0, 255).long()
    x = x01 * 2.0 - 1.0

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(x)
        loss = criterion(logits, target).item()
    bpd = loss / math.log(2.0)
    print(f"nll(nats_per_subpixel)={loss:.6f}  bits_per_dim={bpd:.6f}")


if __name__ == "__main__":
    main()
