"""YOLO cell assignment helpers."""
from __future__ import annotations

import torch


def yolo_assign(
    boxes_xyxy: torch.Tensor,
    labels: torch.Tensor,
    S: int,
    img_h: int,
    img_w: int,
    num_classes: int,
) -> torch.Tensor:
    """Build ``(S, S, 5 + num_classes)`` target for one image (center grid cell)."""
    t = torch.zeros(S, S, 5 + num_classes, device=boxes_xyxy.device, dtype=torch.float32)
    if boxes_xyxy.numel() == 0:
        return t
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5 / img_w
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) * 0.5 / img_h
    gw = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) / img_w
    gh = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]) / img_h
    gx = (cx * S).long().clamp(0, S - 1)
    gy = (cy * S).long().clamp(0, S - 1)
    for i in range(boxes_xyxy.size(0)):
        j, k = int(gx[i]), int(gy[i])
        lab = int(labels[i])
        if lab < 0 or lab >= num_classes:
            continue
        t[k, j, 0] = cx[i]
        t[k, j, 1] = cy[i]
        t[k, j, 2] = gw[i]
        t[k, j, 3] = gh[i]
        t[k, j, 4] = 1.0
        t[k, j, 5 + lab] = 1.0
    return t
