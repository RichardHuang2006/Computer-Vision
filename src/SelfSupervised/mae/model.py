"""Masked Autoencoder ViT-S/16 (He et al., CVPR 2022, lite decoder variant).

Encoder weights match ``src.SelfSupervised.dino.model.ViTEncoder`` for linear probe
``strict=True`` loading. Encoder forward follows the official MAE layout (patch pos before
mask, then CLS + visible patches through ViT blocks).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from src.Classification.common.layers import TransformerBlock, init_vit
from src.SelfSupervised.dino.model import ViTEncoder


def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """imgs: (N, 3, H, W) -> (N, L, patch_size**2 * 3)."""
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    return x


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """x: (N, L, patch_size**2 * 3) -> (N, 3, H, W)."""
    p = patch_size
    h = w = int(math.sqrt(x.shape[1]))
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def random_masking(
    x: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample shuffle masking. x: (N, L, D) patch tokens with pos already added.

    Returns ``x_masked`` (N, len_keep, D), ``mask`` (N, L) 0=keep 1=remove in **original**
    patch order, ``ids_restore`` (N, L) to unshuffle patch tokens.
    """
    n, ell, d = x.shape
    len_keep = int(ell * (1.0 - mask_ratio))
    noise = torch.rand(n, ell, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, d))
    mask = torch.ones([n, ell], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


class MAEDecoder(nn.Module):
    def __init__(
        self,
        num_patches: int,
        encoder_dim: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, decoder_depth)]
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    decoder_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    drop_rate,
                    attn_drop_rate,
                    dp,
                )
                for dp in dpr
            ]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * 3, bias=True)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.apply(init_vit)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """x: (N, 1+len_keep, encoder_dim) encoder output. Returns pred patches (N, L, P*P*3)."""
        x = self.decoder_embed(x)
        n, _, d_dec = x.shape
        ell = ids_restore.shape[1]
        len_keep = x.shape[1] - 1
        n_masked = ell - len_keep
        mask_tokens = self.mask_token.expand(n, n_masked, -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, d_dec)
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]


class MAE(nn.Module):
    """ViT-S/16 encoder + lightweight decoder; encoder is a ``ViTEncoder`` instance."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        encoder_dim: int = 384,
        encoder_depth: int = 12,
        encoder_num_heads: int = 6,
        decoder_dim: int = 192,
        decoder_depth: int = 4,
        decoder_num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_pix_loss: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )
        num_patches = self.encoder.patch_embed.num_patches
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward_encoder(
        self, imgs: torch.Tensor, mask_ratio: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = self.encoder
        x = enc.patch_embed(imgs)
        x = x + enc.pos_embed[:, 1:, :]
        x, mask, ids_restore = random_masking(x, mask_ratio)
        cls_tok = enc.cls_token + enc.pos_embed[:, :1, :]
        cls_tok = cls_tok.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tok, x), dim=1)
        x = enc.pos_drop(x)
        for blk in enc.blocks:
            x = blk(x)
        x = enc.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        return self.decoder(x, ids_restore)

    def forward_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        target = patchify(imgs, self.patch_size)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def forward(
        self, imgs: torch.Tensor, mask_ratio: float = 0.75
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
