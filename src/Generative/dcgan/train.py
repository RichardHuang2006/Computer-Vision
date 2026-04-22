"""Train DCGAN on CIFAR-10 (Radford et al., ICLR workshop 2016).

Images are scaled to ``[-1, 1]`` to match the generator ``Tanh`` output.
Optimizer: Adam with lr=2e-4, betas=(0.5, 0.999) as in the reference implementation.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

from src.Classification.common.utils import AverageMeter, format_hms
from src.Generative.dcgan.model import build_discriminator, build_generator


def get_loaders(data_dir: Path, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Expected CIFAR-10 root at {data_dir} (run scripts/prepare_cifar10.py)")
    tfm = transforms.ToTensor()
    train_ds = datasets.CIFAR10(str(data_dir), train=True, download=False, transform=tfm)
    val_ds = datasets.CIFAR10(str(data_dir), train=False, download=False, transform=tfm)
    train = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train, val


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--nz", type=int, default=100, help="Latent dimension")
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--n-disc", type=int, default=1, help="D steps per G step")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, _ = get_loaders(args.data_dir, args.batch_size, args.num_workers)

    net_g = build_generator(nz=args.nz, ngf=args.ngf, nc=3).to(device)
    net_d = build_discriminator(nc=3, ndf=args.ndf).to(device)

    opt_g = torch.optim.Adam(net_g.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    opt_d = torch.optim.Adam(net_d.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    bce = nn.BCEWithLogitsLoss()

    use_amp = device.type == "cuda"
    scaler_g = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=use_amp)
    scaler_d = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=use_amp)

    real_label = 1.0
    fake_label = 0.0

    ckpt: dict | None = None
    for epoch in range(args.epochs):
        net_g.train()
        net_d.train()
        loss_d_m = AverageMeter()
        loss_g_m = AverageMeter()
        t0 = time.time()
        for step, (real_cpu, _) in enumerate(train_loader):
            bsz = real_cpu.size(0)
            real = (real_cpu * 2.0 - 1.0).to(device, non_blocking=True)
            noise = torch.randn(bsz, args.nz, 1, 1, device=device)

            # --- Update D ---
            for _ in range(args.n_disc):
                net_d.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=use_amp):
                    out_real = net_d(real)
                    loss_real = bce(out_real, torch.full_like(out_real, real_label))
                    fake = net_g(noise).detach()
                    out_fake = net_d(fake)
                    loss_fake = bce(out_fake, torch.full_like(out_fake, fake_label))
                    loss_d = (loss_real + loss_fake) * 0.5
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()
                loss_d_m.update(loss_d.item(), bsz)

            # --- Update G ---
            net_g.zero_grad(set_to_none=True)
            noise = torch.randn(bsz, args.nz, 1, 1, device=device)
            with autocast(device_type=device.type, enabled=use_amp):
                fake = net_g(noise)
                out_g = net_d(fake)
                loss_g = bce(out_g, torch.full_like(out_g, real_label))
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            loss_g_m.update(loss_g.item(), bsz)

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(
                    f"  epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(train_loader)} "
                    f"loss_d={loss_d_m.avg:.4f} loss_g={loss_g_m.avg:.4f}",
                    flush=True,
                )

        dt = time.time() - t0
        print(
            f"epoch {epoch + 1}/{args.epochs} loss_d={loss_d_m.avg:.4f} loss_g={loss_g_m.avg:.4f} "
            f"time={format_hms(dt)}",
            flush=True,
        )

        with torch.no_grad():
            net_g.eval()
            grid = net_g(torch.randn(64, args.nz, 1, 1, device=device))
            vutils.save_image(
                (grid + 1.0) * 0.5,
                out_dir / f"fakes_epoch_{epoch + 1:04d}.png",
                nrow=8,
                padding=2,
            )

        ckpt = {
            "generator": net_g.state_dict(),
            "discriminator": net_d.state_dict(),
            "epoch": epoch,
            "extra": {
                "args": vars(args),
                "nz": args.nz,
                "ngf": args.ngf,
                "ndf": args.ndf,
                "nc": 3,
                "image_size": 32,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")

    if ckpt is not None:
        torch.save(ckpt, out_dir / "best.pt")
    print(f"Checkpoints written to {out_dir} (last.pt each epoch, best.pt final)", flush=True)


if __name__ == "__main__":
    main()
