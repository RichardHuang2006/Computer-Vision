"""Sample images from a trained DCGAN generator checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.Generative.dcgan.model import build_generator


def load_generator(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ex = ckpt.get("extra") or {}
    run_args = ex.get("args") or {}
    nz = int(ex.get("nz", run_args.get("nz", 100)))
    ngf = int(ex.get("ngf", run_args.get("ngf", 64)))
    nc = int(ex.get("nc", 3))
    g = build_generator(nz=nz, ngf=ngf, nc=nc).to(device)
    g.load_state_dict(ckpt["generator"], strict=True)
    g.eval()
    return g


@torch.no_grad()
def sample_grid(
    net_g: torch.nn.Module,
    num: int,
    nz: int,
    device: torch.device,
) -> torch.Tensor:
    """Return ``(num, 3, 32, 32)`` in ``[0, 1]``."""
    z = torch.randn(num, nz, 1, 1, device=device)
    x = net_g(z)
    return (x + 1.0) * 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output PNG path (grid of samples)")
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


if __name__ == "__main__":
    main()
