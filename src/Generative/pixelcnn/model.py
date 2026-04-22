"""Original PixelCNN (van den Oord et al., ICML 2016) — masked conv + residual blocks.

RGB autoregressive order: raster scan over pixels; within each pixel, R then G then B.
``2 * n_filters`` must be divisible by 3 so channels partition into R/G/B groups.
Default ``n_filters=126`` gives 252 channels (84 per group).
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _in_group_index(in_channels: int, ic: int) -> int:
    if in_channels == 3:
        return ic
    return ic // (in_channels // 3)


def _out_group_index(out_channels: int, oc: int) -> int:
    return oc // (out_channels // 3)


def build_spatial_channel_mask(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    mask_type: Literal["A", "B"],
) -> torch.Tensor:
    """Binary mask (out_c, in_c, ky, kx) for masked convolution."""
    if in_channels % 3 != 0 or out_channels % 3 != 0:
        raise ValueError(
            f"in_channels={in_channels} and out_channels={out_channels} must be multiples of 3"
        )
    k = kernel_size
    mid = k // 2
    mask = torch.zeros(out_channels, in_channels, k, k)
    for ky in range(k):
        for kx in range(k):
            if ky < mid or (ky == mid and kx < mid):
                mask[:, :, ky, kx] = 1.0
            elif ky == mid and kx == mid:
                for oc in range(out_channels):
                    og = _out_group_index(out_channels, oc)
                    for ic in range(in_channels):
                        ig = _in_group_index(in_channels, ic)
                        if mask_type == "A":
                            ok = ig < og
                        else:
                            ok = ig <= og
                        if ok:
                            mask[oc, ic, ky, kx] = 1.0
    return mask


class MaskedConv2d(nn.Conv2d):
    """2D conv with autoregressive mask (spatial + RGB channel ordering)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_type: Literal["A", "B"],
        **kwargs,
    ) -> None:
        if "padding" not in kwargs:
            kwargs["padding"] = kernel_size // 2
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer("mask", build_spatial_channel_mask(in_channels, out_channels, kernel_size, mask_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ResBlock(nn.Module):
    """Bottleneck residual: 2F -> F -> F -> 2F with type-B masked convolutions."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if channels % 6 != 0:
            raise ValueError(f"ResBlock channels={channels} must be divisible by 6 (2F with F%3==0)")
        f = channels // 2
        self.conv1 = MaskedConv2d(channels, f, 1, "B")
        self.conv2 = MaskedConv2d(f, f, kernel_size, "B")
        self.conv3 = MaskedConv2d(f, channels, 1, "B")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x), inplace=True)
        h = F.relu(self.conv2(h), inplace=True)
        h = self.conv3(h)
        return x + h


class PixelCNN(nn.Module):
    """PixelCNN for 3 x H x W images; outputs logits (N, 256, 3, H, W)."""

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        n_filters: int = 126,
        n_res: int = 15,
        num_levels: int = 256,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_filters = n_filters
        self.n_res = n_res
        self.n_classes = num_levels
        h = 2 * n_filters
        if h % 3 != 0:
            raise ValueError(f"2 * n_filters = {h} must be divisible by 3 for RGB masking")
        layers: list[nn.Module] = [
            MaskedConv2d(in_channels, h, 7, "A"),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_res):
            layers.append(ResBlock(h, 3))
            layers.append(nn.ReLU(inplace=True))
        layers.extend(
            [
                MaskedConv2d(h, h, 1, "B"),
                nn.ReLU(inplace=True),
                MaskedConv2d(h, h, 1, "B"),
                nn.ReLU(inplace=True),
                MaskedConv2d(h, 3 * num_levels, 1, "B"),
            ]
        )
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.uniform_(m.weight, -math.sqrt(6.0 / n), math.sqrt(6.0 / n))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 3, H, W) in roughly [-1, 1]. Returns logits (N, 256, 3, H, W)."""
        out = self.net(x)
        n, _, h, w = out.shape
        return out.view(n, 3, self.n_classes, h, w).permute(0, 2, 1, 3, 4).contiguous()


def build_model(arch: str, in_channels: int = 3, image_size: int = 32) -> nn.Module:
    if arch != "pixelcnn_cifar":
        raise ValueError(f"Unknown arch {arch!r}; use pixelcnn_cifar")
    _ = in_channels  # CIFAR-10 RGB
    return PixelCNN(in_channels=3, image_size=image_size)
