"""Generate a sample grid from a DiT checkpoint (same as ``sample.py``)."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.Generative.dit.diffusion import build_diffusion
from src.Generative.dit.sample import build_labels, load_dit_for_sampling, sample_grid


def main() -> None:
    p = argparse.ArgumentParser(description="Generate images from a trained DiT (DDIM + CFG)")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("dit_samples.png"))
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--classes", type=str, default="random")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt.resolve()

    model, meta = load_dit_for_sampling(ckpt_path, device)
    diffusion = build_diffusion(
        num_timesteps=meta["T"],
        beta_start=meta["beta_start"],
        beta_end=meta["beta_end"],
    ).to(device)

    y = build_labels(args.classes, args.num_samples, meta["num_classes"], device)
    xs = sample_grid(
        model,
        diffusion,
        y,
        meta["num_classes"],
        meta["image_size"],
        args.cfg_scale,
        args.steps,
        device,
    ).cpu()

    nrow = max(1, int(round(args.num_samples**0.5)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(xs, args.out, nrow=nrow, padding=2)
    print(f"Saved {args.num_samples} samples to {args.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
