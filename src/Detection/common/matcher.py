"""Assign ground truth to anchors by IoU."""
from __future__ import annotations

import torch

from .boxes import box_iou


def match_anchors_to_gt(
    anchors_xyxy: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pos_iou_thr: float = 0.7,
    neg_iou_thr: float = 0.3,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``anchor_labels`` (K,) {-1 ignore, 0 bg, >0 class}, ``anchor_target_boxes`` (K,4)."""
    device = device or anchors_xyxy.device
    num_a = anchors_xyxy.size(0)
    if gt_boxes.numel() == 0:
        return (
            torch.zeros(num_a, dtype=torch.long, device=device),
            torch.zeros(num_a, 4, device=device),
        )
    ious = box_iou(anchors_xyxy, gt_boxes)
    max_iou, max_idx = ious.max(dim=1)
    labels = torch.zeros(num_a, dtype=torch.long, device=device)
    labels[max_iou < neg_iou_thr] = 0  # bg
    labels[(max_iou >= neg_iou_thr) & (max_iou < pos_iou_thr)] = -1  # ignore
    pos_mask = max_iou >= pos_iou_thr
    labels[pos_mask] = gt_labels[max_idx[pos_mask]]
    target_boxes = gt_boxes[max_idx]
    return labels, target_boxes
