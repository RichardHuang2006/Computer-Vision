"""Sample from a trained VAE checkpoint (prior z ~ N(0, I))."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.Generative.vae.model import VAE


def vae_from_checkpoint_dict(ckpt: dict, device: torch.device) -> VAE:
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        ex = ckpt.get("extra") or {}
    else:
        sd = ckpt["model"]
        ex = {}
    args = ex.get("args") or ckpt.get("args") or {}
    z_dim = int(ex.get("z_dim", args.get("z_dim", 128)))
    image_size = int(ex.get("image_size", args.get("image_size", 32)))
    base = int(ex.get("base", args.get("base", 32)))
    in_channels = int(ex.get("in_channels", args.get("in_channels", 3)))

    model = VAE(in_channels=in_channels, image_size=image_size, base=base, z_dim=z_dim).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def load_vae(ckpt_path: Path, device: torch.device) -> VAE:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    return vae_from_checkpoint_dict(ckpt, device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output PNG path (grid of samples)")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vae(args.ckpt.resolve(), device)
    n = args.num_samples
    xs = model.sample(n, device=device).cpu()
    nrow = int(round(n**0.5))
    nrow = max(1, nrow)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(xs, args.out, nrow=nrow, padding=2)


if __name__ == "__main__":
    main()
