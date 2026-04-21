"""VOC 2012 segmentation loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD


class VOCSegmentation(Dataset):
    def __init__(
        self,
        root: Path,
        split: str = "train",
        crop_size: int = 480,
        is_train: bool = True,
        val_max_size: int = 512,
    ) -> None:
        root = Path(root)
        self.voc_root = root / "VOCdevkit" / "VOC2012"
        ids_file = self.voc_root / "ImageSets" / "Segmentation" / f"{split}.txt"
        with open(ids_file, encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.img_dir = self.voc_root / "JPEGImages"
        self.mask_dir = self.voc_root / "SegmentationClass"
        self.crop_size = crop_size
        self.is_train = is_train
        self.val_max_size = val_max_size

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        img = Image.open(self.img_dir / f"{sid}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{sid}.png")

        if self.is_train:
            # random scale + crop + flip
            w, h = img.size
            scale = np.random.uniform(0.5, 2.0)
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)
            i = np.random.randint(0, max(nh - self.crop_size, 1) + 1)
            j = np.random.randint(0, max(nw - self.crop_size, 1) + 1)
            i2 = min(i + self.crop_size, nh)
            j2 = min(j + self.crop_size, nw)
            img = img.crop((j, i, j2, i2))
            mask = mask.crop((j, i, j2, i2))
            if img.size[0] != self.crop_size or img.size[1] != self.crop_size:
                img = img.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                mask = mask.resize((self.crop_size, self.crop_size), Image.NEAREST)
            if np.random.rand() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
        else:
            w, h = img.size
            s = self.val_max_size / max(w, h)
            nw, nh = int(w * s), int(h * s)
            img = img.resize((nw, nh), Image.BILINEAR)
            mask = mask.resize((nw, nh), Image.NEAREST)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(np.array(mask)).long()
        mask_t[mask_t == 255] = 255  # ignore
        return img_t, mask_t


def build_voc_loaders(
    root: Path,
    batch_size: int = 8,
    num_workers: int = 4,
    crop_size: int = 480,
    val_max_size: int = 512,
) -> Tuple[DataLoader, DataLoader, int]:
    train_set = VOCSegmentation(root, "train", crop_size=crop_size, is_train=True)
    val_set = VOCSegmentation(
        root, "val", crop_size=crop_size, is_train=False, val_max_size=val_max_size
    )
    num_classes = 21
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, num_classes
