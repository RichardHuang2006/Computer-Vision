"""Generate a sample grid from a DCGAN checkpoint (same as ``sample.py``)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.Generative.dcgan.sample import load_generator, sample_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Generate images from a trained DCGAN")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("dcgan_samples.png"))
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt.resolve()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ex = ckpt.get("extra") or {}
    args_ck = ex.get("args") or {}
    nz = int(ex.get("nz", args_ck.get("nz", 100)))

    net_g = load_generator(ckpt_path, device)
    xs = sample_grid(net_g, args.num_samples, nz, device)
    nrow = max(1, int(round(args.num_samples**0.5)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(xs, args.out, nrow=nrow, padding=2)
    print(f"Saved {args.num_samples} samples to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
