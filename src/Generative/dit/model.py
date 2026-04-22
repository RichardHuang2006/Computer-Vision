"""Diffusion Transformer (DiT, Peebles & Xie, ICLR 2023) — CIFAR-10 (3×32×32).

Class-conditional noise prediction with adaLN-Zero, sinusoidal timestep embedding,
learned class embeddings (including a null token for classifier-free guidance).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x: (B, N, D); shift, scale: (B, D) -> broadcast to (B, 1, D)."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """2D sin-cos position embedding (fixed), shape (1 +) grid_size**2, embed_dim."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # noqa: B008
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class TimestepEmbedder(nn.Module):
    """Sinusoidal t -> MLP -> hidden."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class LabelEmbedder(nn.Module):
    """Embedding for class labels + null index ``num_classes`` for CFG."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor | None) -> torch.Tensor:
        if force_drop_ids is None:
            drop = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            out = torch.where(drop, self.num_classes, labels)
            return out
        return torch.where(force_drop_ids, self.num_classes, labels)

    def forward(
        self,
        labels: torch.Tensor,
        train: bool,
        force_drop_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (train and self.dropout_prob > 0) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding(labels)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)


def mlp_gelu(dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim, hidden_dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(hidden_dim, dim, bias=True),
    )


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero (shift, scale, gate for attn and MLP)."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = mlp_gelu(hidden_size, mlp_hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """adaLN + linear to patch channels."""

    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Noise predictor epsilon(x, t, y) for images in [-1, 1]."""

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        learn_sigma: bool = False,
        class_dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (input_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size), requires_grad=False
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(N, T, patch**2 * C) -> (N, C, H, W)."""
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(shape=(x.shape[0], c, h * p, h * p))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, H, W), t: (N,) int/long, y: (N,) class indices.
        Returns epsilon (N, C, H, W) or (N, 2C, H, W) if learn_sigma.
        """
        x_tokens = self.x_embedder(x).flatten(2).transpose(1, 2)
        x_tokens = x_tokens + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training)
        c = t_emb + y_emb
        for blk in self.blocks:
            x_tokens = blk(x_tokens, c)
        x_tokens = self.final_layer(x_tokens, c)
        x_tokens = self.unpatchify(x_tokens)
        if self.learn_sigma:
            return torch.split(x_tokens, self.in_channels, dim=1)[0]
        return x_tokens


DIT_CONFIGS: dict[str, dict[str, Any]] = {
    "DiT-Ti/2": {"depth": 12, "hidden_size": 192, "patch_size": 2, "num_heads": 3},
    "DiT-S/2": {"depth": 12, "hidden_size": 384, "patch_size": 2, "num_heads": 6},
    "DiT-B/2": {"depth": 12, "hidden_size": 768, "patch_size": 2, "num_heads": 12},
}


def build_dit(
    arch: str = "DiT-S/2",
    input_size: int = 32,
    in_channels: int = 3,
    num_classes: int = 10,
    class_dropout_prob: float = 0.1,
    learn_sigma: bool = False,
) -> DiT:
    if arch not in DIT_CONFIGS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {list(DIT_CONFIGS)}")
    cfg = DIT_CONFIGS[arch]
    return DiT(
        input_size=input_size,
        patch_size=int(cfg["patch_size"]),
        in_channels=in_channels,
        hidden_size=int(cfg["hidden_size"]),
        depth=int(cfg["depth"]),
        num_heads=int(cfg["num_heads"]),
        num_classes=num_classes,
        learn_sigma=learn_sigma,
        class_dropout_prob=class_dropout_prob,
    )
