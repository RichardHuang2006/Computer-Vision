"""Autoregressive sampling from a trained PixelCNN checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

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


def sample_batch(model: PixelCNN, n: int, device: torch.device) -> torch.Tensor:
    """Return images in [0, 1], shape (n, 3, H, W)."""
    h = w = model.image_size
    x = torch.zeros(n, 3, h, w, device=device)
    with torch.no_grad():
        for yi in range(h):
            for xi in range(w):
                for c in range(3):
                    x_in = x * 2.0 - 1.0
                    logits = model(x_in)
                    probs = F.softmax(logits[:, :, c, yi, xi], dim=1)
                    idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    v = idx.float() / 255.0
                    x[:, c, yi, xi] = v
    return x.clamp(0.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output PNG path (grid of samples)")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pixelcnn(args.ckpt.resolve(), device)
    n = args.num_samples
    xs = sample_batch(model, n, device)
    nrow = int(round(n**0.5))
    nrow = max(1, nrow)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(xs, args.out, nrow=nrow, padding=2)


if __name__ == "__main__":
    main()
