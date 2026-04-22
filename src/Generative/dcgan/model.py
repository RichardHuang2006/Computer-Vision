"""DCGAN (Radford et al., ICLR workshop 2016) — CIFAR-10 (3×32×32).

Generator maps ``(B, nz, 1, 1)`` noise to ``(B, 3, 32, 32)`` in ``[-1, 1]`` (``Tanh``).
Discriminator maps RGB to a single logit (``BCEWithLogitsLoss`` in training).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: Normal(0, 0.02) on Conv / ConvTranspose weights; BN as in PyTorch reference."""
    cname = m.__class__.__name__
    if "Conv" in cname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in cname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    """``nz`` -> 4×4×(8·ngf) -> … -> ``nc``×32×32."""

    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.nz = nz
        # 1x1 nz -> 4x4 -> 8 -> 16 -> 32 (CIFAR-10); final 3x3 conv keeps spatial size
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)


class Discriminator(nn.Module):
    """``nc``×32×32 -> one logit per image (no Sigmoid; use ``BCEWithLogitsLoss``)."""

    def __init__(self, nc: int = 3, ndf: int = 64) -> None:
        super().__init__()
        # 32 -> 16 -> 8 -> 4 -> 1 (matches 32x32 CIFAR after three stride-2 blocks)
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x).view(-1, 1).squeeze(1)


def build_generator(nz: int = 100, ngf: int = 64, nc: int = 3) -> Generator:
    g = Generator(nz=nz, ngf=ngf, nc=nc)
    g.apply(weights_init)
    return g


def build_discriminator(nc: int = 3, ndf: int = 64) -> Discriminator:
    d = Discriminator(nc=nc, ndf=ndf)
    d.apply(weights_init)
    return d
