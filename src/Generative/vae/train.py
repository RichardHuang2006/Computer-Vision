"""Train the VAE on CIFAR-10.

Run from the ``Computer Vision`` directory with ``PYTHONPATH=.``:

    python -m src.Generative.vae.train --data-dir data/cifar10 --out-dir runs/vae

CIFAR-10 is downloaded automatically via torchvision on first run.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from src.Generative.vae.data import build_cifar10_loaders
from src.Generative.vae.model import VAE, vae_loss


def run_epoch(
    model: VAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    kl_weight: float,
) -> tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total = recon_sum = kl_sum = 0.0
    n = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            loss, recon, kl = vae_loss(out, x, kl_weight=kl_weight)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            total += loss.item() * bs
            recon_sum += recon.item() * bs
            kl_sum += kl.item() * bs
            n += bs
    return total / n, recon_sum / n, kl_sum / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/vae"))
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--z-dim", type=int, default=128)
    p.add_argument("--kl-weight", type=float, default=1.0, help="beta in beta-VAE; 1.0 = standard ELBO")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_loader, val_loader = build_cifar10_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers, download=True
    )
    model = VAE(in_channels=3, image_size=32, z_dim=args.z_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Fixed validation batch for reconstruction grids (first batch, up to 8 images).
    x_fixed, _ = next(iter(val_loader))
    x_fixed = x_fixed[:8].to(device, non_blocking=True)

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, args.kl_weight)
        va = run_epoch(model, val_loader, None, device, args.kl_weight)
        print(
            f"epoch {epoch:3d} | "
            f"train loss {tr[0]:7.2f} recon {tr[1]:7.2f} kl {tr[2]:6.2f} | "
            f"val loss {va[0]:7.2f} recon {va[1]:7.2f} kl {va[2]:6.2f}"
        )

        samples = model.sample(64, device=device).cpu()
        vutils.save_image(samples, args.out_dir / f"samples_epoch{epoch:03d}.png", nrow=8)

        model.eval()
        with torch.no_grad():
            recon = torch.sigmoid(model(x_fixed).x_logits).cpu()
        model.train()
        vutils.save_image(
            torch.cat([x_fixed.cpu(), recon], dim=0),
            args.out_dir / f"recon_epoch{epoch:03d}.png",
            nrow=x_fixed.size(0),
        )

        extra = {
            "args": vars(args),
            "z_dim": args.z_dim,
            "image_size": 32,
            "base": 32,
            "in_channels": 3,
        }
        ckpt = {"state_dict": model.state_dict(), "extra": extra}
        if va[0] < best:
            best = va[0]
            torch.save(ckpt, args.out_dir / "best.pt")

    last_extra = {
        "args": vars(args),
        "z_dim": args.z_dim,
        "image_size": 32,
        "base": 32,
        "in_channels": 3,
    }
    torch.save({"state_dict": model.state_dict(), "extra": last_extra}, args.out_dir / "last.pt")


if __name__ == "__main__":
    main()
