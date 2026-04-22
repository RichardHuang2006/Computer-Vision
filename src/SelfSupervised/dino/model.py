"""DINO self-supervised ViT-S/16 (Caron et al., ICCV 2021).

Student path::

    Input crops (global 224 or local 96)                           (N, 3, H, W)
    ViT backbone (patch + blocks + norm)                          (N, D)
    DINO projection MLP (3 layers) + L2 norm                      (N, K)

Teacher path is the **EMA** of the student (same architecture), applied to global crops only.

High-level (conceptual) rows::

    Patch + pos + blocks (same as ViT)                            tokens -> CLS (D)
    Head: Linear(D->H)+BN+GELU -> Linear(H->H)+BN+GELU -> Linear(H->K)   (K prototype dim)

``DINO.forward_student(x)`` and ``DINO.forward_teacher(x)`` return L2-normalized K-dim descriptors.

Linear probe uses frozen backbone + ``nn.Linear(D, C)``.
"""
from __future__ import annotations

import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Classification.common.layers import PatchEmbed, TransformerBlock, init_vit


class DINOHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 2048, bottleneck_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return F.normalize(x, dim=-1, p=2)


class ViTEncoder(nn.Module):
    """ViT trunk ending at CLS token (before classifier)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    dp,
                )
                for dp in dpr
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(init_vit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]


class DINO(nn.Module):
    def __init__(
        self,
        out_dim: int = 65536,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.student_enc = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )
        self.student_head = DINOHead(embed_dim, out_dim)
        self.teacher_enc = copy.deepcopy(self.student_enc)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher_enc.parameters():
            p.requires_grad = False
        for p in self.teacher_head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self, m: float) -> None:
        for ps, pt in zip(self.student_enc.parameters(), self.teacher_enc.parameters()):
            pt.data.mul_(m).add_((1.0 - m) * ps.data)
        for ps, pt in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            pt.data.mul_(m).add_((1.0 - m) * ps.data)

    def forward_student(self, x: torch.Tensor) -> torch.Tensor:
        z = self.student_enc(x)
        return self.student_head(z)

    @torch.no_grad()
    def forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.teacher_enc(x)
            return self.teacher_head(z)


def build_backbone(arch: str, num_classes: int) -> nn.Module:
    """For linear probe: encoder + linear classifier."""
    del arch, num_classes  # use fixed vit
    raise RuntimeError("Use LinearClassifier in linear_probe.py")


def build_dino_head(out_dim: int = 65536) -> DINO:
    return DINO(out_dim=out_dim)
