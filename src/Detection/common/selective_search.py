"""Selective Search–style region proposals (fallback: multi-scale sliding windows).

Caches ``proposals.pt``: dict mapping image_id (int) -> FloatTensor ``(N,4)`` xyxy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image


def _grid_proposals(w: int, h: int, max_boxes: int = 200) -> np.ndarray:
    boxes: list[list[float]] = []
    for s in np.linspace(0.1, 1.0, 8):
        ww = w * s * 0.5
        hh = h * s * 0.5
        for cx in np.linspace(ww / 2, w - ww / 2, 6):
            for cy in np.linspace(hh / 2, h - hh / 2, 6):
                x1 = max(0, cx - ww / 2)
                y1 = max(0, cy - hh / 2)
                x2 = min(w, cx + ww / 2)
                y2 = min(h, cy + hh / 2)
                if x2 - x1 > 4 and y2 - y1 > 4:
                    boxes.append([x1, y1, x2, y2])
                if len(boxes) >= max_boxes:
                    return np.array(boxes, dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def build_proposals_cache(
    image_dir: Path,
    out_path: Path,
    annotations_json: Path,
    max_boxes: int = 200,
) -> Dict[int, torch.Tensor]:
    import json

    ann = json.loads(annotations_json.read_text(encoding="utf-8"))
    cache: Dict[int, torch.Tensor] = {}
    for im in ann["images"]:
        iid = im["id"]
        path = Path(image_dir) / im["file_name"]
        img = Image.open(path).convert("RGB")
        w, h = img.size
        arr = _grid_proposals(w, h, max_boxes=max_boxes)
        cache[iid] = torch.from_numpy(arr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    return cache


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("proposals.pt"))
    args = p.parse_args()
    build_proposals_cache(args.images, args.out, args.annotations)
