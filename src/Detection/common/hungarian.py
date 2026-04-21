"""Bipartite matching for DETR-style set prediction."""
from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment


def hungarian_match(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``cost`` ``(N, M)`` -> row indices, col indices (CPU numpy)."""
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)
