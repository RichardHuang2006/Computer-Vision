"""ViT inference CLI."""
from __future__ import annotations

from pathlib import Path

import torch

from src.Classification.common.infer import run_image, load_model_from_ckpt

from .model import build_model


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--arch", type=str, default="vit_s16")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(args.ckpt, build_model, args.arch, device)
    pairs = run_image(model, args.image, device, topk=args.topk, image_size=args.image_size)
    for c, s in pairs:
        print(f"class_{c}: {s:.4f}")


if __name__ == "__main__":
    main()
