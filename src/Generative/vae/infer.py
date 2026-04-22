"""Run a single image through a trained VAE: reconstruction and ELBO-based metrics.

``bits_per_dim`` uses the negative ELBO (reconstruction + beta * KL) with
``reduction='sum'``, divided by ``C * H * W * log(2)``. This is an upper bound on
true bits per dimension under the model, not the exact marginal NLL.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from src.Generative.vae.model import vae_loss
from src.Generative.vae.sample import vae_from_checkpoint_dict


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional PNG path: side-by-side original | reconstruction",
    )
    p.add_argument(
        "--kl-weight",
        type=float,
        default=None,
        help="Beta in beta-VAE loss; default reads from checkpoint args or 1.0",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt.resolve()
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ex = ckpt.get("extra") or {}
    train_args = ex.get("args") or ckpt.get("args") or {}
    kl_weight = float(args.kl_weight if args.kl_weight is not None else train_args.get("kl_weight", 1.0))

    model = vae_from_checkpoint_dict(ckpt, device)
    h = w = model.image_size
    c = model.in_channels

    tfm = transforms.Compose(
        [
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ]
    )
    img = Image.open(args.image).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model(x)
        loss, recon, kl = vae_loss(out, x, kl_weight=kl_weight, reduction="sum")

    neg_elbo = loss.item()
    recon_nats = recon.item()
    kl_nats = kl.item()
    denom = float(c * h * w * math.log(2.0))
    bpd = neg_elbo / denom

    print(
        f"recon_nats={recon_nats:.4f}  kl_nats={kl_nats:.4f}  "
        f"neg_elbo_nats={neg_elbo:.4f}  bits_per_dim_elbo_upper_bound={bpd:.6f}"
    )

    if args.out is not None:
        x_hat = torch.sigmoid(out.x_logits).cpu()
        x_cpu = x.cpu()
        grid = torch.cat([x_cpu, x_hat], dim=0)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        nrow = 2 if x.size(0) == 1 else x.size(0)
        save_image(grid, args.out, nrow=nrow, padding=2)


if __name__ == "__main__":
    main()
