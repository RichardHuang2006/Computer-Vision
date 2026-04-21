"""Box IoU, NMS, coordinate transforms."""
from __future__ import annotations

import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """``boxes`` ``(N,4)`` xyxy."""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU, ``boxes*`` ``(N,4)`` ``(M,4)`` -> ``(N,M)``."""
    area1 = box_area(boxes1).unsqueeze(1)
    area2 = box_area(boxes2).unsqueeze(0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1 + area2 - inter + 1e-7
    return inter / union


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Return indices to keep. ``boxes`` ``(N,4)`` xyxy, ``scores`` ``(N,)``."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    idx = scores.argsort(descending=True)
    keep: list[int] = []
    while idx.numel() > 0:
        i = idx[0].item()
        keep.append(i)
        if idx.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[idx[1:]])[0]
        idx = idx[1:][ious <= iou_threshold]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


def clip_boxes_to_image(boxes: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(0, w)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(0, h)
    return boxes


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise GIoU (N,4) vs (M,4) -> (N,M)."""
    from torchvision.ops import generalized_box_iou as giou_tv

    return giou_tv(boxes1, boxes2)
