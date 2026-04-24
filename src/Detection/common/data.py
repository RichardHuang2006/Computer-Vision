"""COCO-mini detection dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD


class CocoMini(Dataset):
    def __init__(self, root: Path, split: str = "train", max_size: int = 800) -> None:
        root = Path(root)
        ann_path = root / "annotations.json"
        if not ann_path.is_file():
            print(
                f"ERROR: COCO-mini layout not found at {root.resolve()}\n\n"
                "Expected:\n"
                f"  {root / 'annotations.json'}\n"
                f"  {root / 'images'}/<image files>\n\n"
                "Build the subset from full COCO 2017 train (images + instances JSON):\n"
                "  python scripts/prepare_coco_mini.py ^\n"
                "    --images-dir path/to/train2017 ^\n"
                "    --annotations path/to/instances_train2017.json ^\n"
                "    --out-dir data/coco-mini\n",
                file=sys.stderr,
            )
            raise FileNotFoundError(ann_path)
        self.images_root = root / "images"
        if not self.images_root.is_dir():
            print(
                f"ERROR: missing images directory:\n  {self.images_root}\n",
                file=sys.stderr,
            )
            raise FileNotFoundError(self.images_root)
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        self.id_to_info = {im["id"]: im for im in ann["images"]}
        self.cat_ids = sorted([c["id"] for c in ann["categories"]])
        self.cat_to_idx = {c: i for i, c in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)
        by_img: Dict[int, List[Dict[str, Any]]] = {}
        for a in ann["annotations"]:
            by_img.setdefault(a["image_id"], []).append(a)
        self.image_ids = [im["id"] for im in ann["images"] if im["id"] in by_img]
        self.by_img = by_img
        self.max_size = max_size

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        iid = self.image_ids[idx]
        info = self.id_to_info[iid]
        path = self.images_root / info["file_name"]
        img = Image.open(path).convert("RGB")
        w0, h0 = img.size
        scale = self.max_size / max(w0, h0)
        w, h = int(w0 * scale), int(h0 * scale)
        img = img.resize((w, h), Image.BILINEAR)
        im_t = TF.to_tensor(img)
        im_t = TF.normalize(im_t, IMAGENET_MEAN, IMAGENET_STD)

        anns = self.by_img[iid]
        boxes: List[List[float]] = []
        labels: List[int] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            x1 = x * scale
            y1 = y * scale
            x2 = (x + bw) * scale
            y2 = (y + bh) * scale
            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_to_idx[ann["category_id"]])
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_id": torch.tensor([iid], dtype=torch.long),
            "orig_size": torch.tensor([h0, w0], dtype=torch.long),
            "size": torch.tensor([h, w], dtype=torch.long),
        }
        return im_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def build_coco_loaders(
    root: Path,
    batch_size: int = 4,
    num_workers: int = 4,
) -> Tuple[DataLoader, CocoMini]:
    ds = CocoMini(root)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader, ds
