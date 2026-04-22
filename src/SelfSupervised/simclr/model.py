"""SimCLR (Chen et al., ICML 2020): ResNet-50 + MLP projection + NT-Xent."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Classification.resnet.model import build_model


class ResNetEncoder(nn.Module):
    """ResNet-50 backbone; ``fc`` is identity; features are 2048-d."""

    feat_dim = 2048

    def __init__(self) -> None:
        super().__init__()
        net = build_model("resnet50", num_classes=1)
        net.fc = nn.Identity()
        self.resnet = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResNetEncoder()
        self.proj_head = ProjectionHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.proj_head(h)
        return F.normalize(z, dim=-1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """NT-Xent over two L2-normalized views ``z1``, ``z2`` each (B, D)."""
    b = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=-1)
    sim = torch.mm(z, z.t()) / tau
    mask = torch.eye(2 * b, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat(
        [
            torch.arange(b, 2 * b, device=z.device),
            torch.arange(0, b, device=z.device),
        ],
        dim=0,
    )
    return F.cross_entropy(sim, labels)
