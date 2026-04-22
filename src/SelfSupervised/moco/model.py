"""MoCo v1 (He et al., CVPR 2020): momentum key encoder + queue + InfoNCE.

Shuffle-BN from the paper is omitted (single-GPU training).
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Classification.resnet.model import build_model


class MoCoResNetEncoder(nn.Module):
    """ResNet-50 + linear projection to ``proj_dim`` (MoCo v1)."""

    def __init__(self, proj_dim: int = 128) -> None:
        super().__init__()
        net = build_model("resnet50", num_classes=1)
        net.fc = nn.Identity()
        self.resnet = net
        self.proj = nn.Linear(2048, proj_dim)
        self.feat_dim = 2048
        self.proj_dim = proj_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.resnet(x)
        return self.proj(h)


class MoCo(nn.Module):
    def __init__(self, queue_size: int = 16384, proj_dim: int = 128, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau
        self.queue_size = queue_size
        self.proj_dim = proj_dim

        self.encoder_q = MoCoResNetEncoder(proj_dim=proj_dim)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False

        _q = torch.randn(queue_size, proj_dim)
        self.register_buffer("queue", F.normalize(_q, dim=1))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_key_encoder(self, m: float) -> None:
        for pq, pk in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            pk.data.mul_(m).add_(pq.data, alpha=1.0 - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """keys: (B, proj_dim) L2-normalized."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, "queue_size must be divisible by batch_size"

        self.queue[ptr : ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(self, im_q: torch.Tensor, im_k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``logits`` (B, 1+K), ``labels`` (B,), ``k`` detached normalized keys."""
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=-1)

        with torch.no_grad():
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=-1)

        l_pos = (q * k).sum(dim=-1, keepdim=True)
        l_neg = torch.mm(q, self.queue.clone().detach().t())
        logits = torch.cat([l_pos, l_neg], dim=1) / self.tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        with torch.no_grad():
            self._dequeue_and_enqueue(k)

        return logits, labels, k
