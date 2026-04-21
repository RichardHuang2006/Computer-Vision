"""Vision Transformer ViT-S/16 (Dosovitskiy et al., ICLR 2021, small variant).

Diagram for ``img_size=224``, ``patch_size=16``, ``embed=384``, ``depth=12``, ``heads=6``::

    Input image                                              (3, 224, 224)
    PatchEmbed  Conv16 s16                                     (14*14, 384) tokens
    + CLS + pos emb                                            (197, 384)
    L=12 x [ LayerNorm -> MSA -> +residual
             LayerNorm -> MLP  -> +residual ]                (197, 384)
    Take CLS index                                             (384,)
    LayerNorm + Linear(C)                                      (C,)

``forward(x) -> logits`` for ``x`` ``(N,3,224,224)``.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.Classification.common.layers import PatchEmbed, TransformerBlock, init_vit


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    drop_path,
                )
                for drop_path in dpr
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

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
        x = x[:, 0]
        return self.head(x)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch != "vit_s16":
        raise ValueError(f"Unknown arch {arch!r}; use vit_s16")
    return VisionTransformer(num_classes=num_classes)
