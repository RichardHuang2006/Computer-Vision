"""Anchor grid generators for RPN / multiscale."""
from __future__ import annotations

import torch
from typing import List, Tuple


def generate_grid_anchors(
    sizes: Tuple[int, int],
    stride: int,
    scales: Tuple[float, ...] = (8.0, 16.0, 32.0),
    ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return anchors in xyxy for feature map of ``sizes`` (fh, fw), in image pixel coords."""
    fh, fw = sizes
    anchors: List[torch.Tensor] = []
    for y in range(fh):
        for x in range(fw):
            cx = (x + 0.5) * stride
            cy = (y + 0.5) * stride
            for s in scales:
                for r in ratios:
                    w = s * (r**0.5)
                    h = s / (r**0.5)
                    xa = cx - w / 2
                    ya = cy - h / 2
                    xb = cx + w / 2
                    yb = cy + h / 2
                    anchors.append(
                        torch.tensor([xa, ya, xb, yb], device=device, dtype=torch.float32)
                    )
    if not anchors:
        return torch.empty(0, 4, device=device)
    return torch.stack(anchors, dim=0)
