"""Variational Autoencoder (Kingma & Welling, ICLR 2014) — ``Auto-Encoding Variational Bayes``.

Convolutional encoder/decoder for 32 x 32 RGB images. Posterior q(z|x) is a
diagonal Gaussian N(mu, sigma^2); prior p(z) is N(0, I); decoder outputs
per-pixel Bernoulli logits (images scaled to [0, 1]).

ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
     = -BCE(x_hat_logits, x) - 0.5 * sum(1 + log_var - mu^2 - exp(log_var))

``reparameterize`` implements the pathwise gradient trick: z = mu + sigma * eps.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    x_logits: torch.Tensor  # (N, C, H, W) Bernoulli logits
    mu: torch.Tensor        # (N, z_dim)
    log_var: torch.Tensor   # (N, z_dim)
    z: torch.Tensor         # (N, z_dim) sampled latent


class Encoder(nn.Module):
    """32 x 32 -> 4 x 4 conv tower, then FC heads for mu and log_var."""

    def __init__(self, in_channels: int = 3, base: int = 32, z_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),      # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),          # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),      # 8 -> 4
            nn.ReLU(inplace=True),
        )
        self.flat_dim = base * 4 * 4 * 4
        self.fc_mu = nn.Linear(self.flat_dim, z_dim)
        self.fc_log_var = nn.Linear(self.flat_dim, z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_log_var(h)


class Decoder(nn.Module):
    """z -> 4 x 4 feature map -> transposed conv tower -> 32 x 32 logits."""

    def __init__(self, out_channels: int = 3, base: int = 32, z_dim: int = 128) -> None:
        super().__init__()
        self.base = base
        self.fc = nn.Linear(z_dim, base * 4 * 4 * 4)
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),       # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, out_channels, 4, 2, 1),   # 16 -> 32
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.base * 4, 4, 4)
        return self.net(h)


class VAE(nn.Module):
    """Convolutional VAE with diagonal-Gaussian posterior and Bernoulli decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        base: int = 32,
        z_dim: int = 128,
    ) -> None:
        super().__init__()
        if image_size != 32:
            raise ValueError(f"This VAE expects 32 x 32 inputs, got image_size={image_size}")
        self.in_channels = in_channels
        self.image_size = image_size
        self.z_dim = z_dim
        self.encoder = Encoder(in_channels, base, z_dim)
        self.decoder = Decoder(in_channels, base, z_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps, eps ~ N(0, I). Differentiable wrt mu, log_var."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x: torch.Tensor) -> VAEOutput:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_logits = self.decoder(z)
        return VAEOutput(x_logits=x_logits, mu=mu, log_var=log_var, z=z)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device | str = "cpu") -> torch.Tensor:
        """Draw z ~ N(0, I) and return decoded images in [0, 1]."""
        z = torch.randn(n, self.z_dim, device=device)
        return torch.sigmoid(self.decoder(z))


def vae_loss(
    out: VAEOutput,
    x: torch.Tensor,
    kl_weight: float = 1.0,
    reduction: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Negative ELBO = BCE(reconstruction) + beta * KL.

    Returns ``(loss, recon, kl)`` each reduced over the batch per ``reduction``.
    Input ``x`` must be in [0, 1]; ``out.x_logits`` is the Bernoulli logit map.
    """
    recon_per = F.binary_cross_entropy_with_logits(out.x_logits, x, reduction="none")
    recon = recon_per.flatten(1).sum(dim=1)  # (N,) nats per image
    kl = -0.5 * (1.0 + out.log_var - out.mu.pow(2) - out.log_var.exp()).sum(dim=1)
    loss = recon + kl_weight * kl
    if reduction == "mean":
        return loss.mean(), recon.mean(), kl.mean()
    if reduction == "sum":
        return loss.sum(), recon.sum(), kl.sum()
    return loss, recon, kl


def build_model(arch: str, in_channels: int = 3, image_size: int = 32) -> nn.Module:
    if arch != "vae_cifar":
        raise ValueError(f"Unknown arch {arch!r}; use vae_cifar")
    return VAE(in_channels=in_channels, image_size=image_size)
