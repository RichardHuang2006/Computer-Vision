"""Multi-view augmentation for DINO (simplified: all views 224x224)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD


def _make_globals() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.25, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _make_locals_weak() -> transforms.Compose:
    """Smaller random crop / stronger jitter (still output 224 for shared ViT)."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.05, 0.25), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiCropImageNet(Dataset):
    """Returns tensor ``views`` of shape ``(num_views, 3, 224, 224)`` and label."""

    def __init__(self, root: Path, split: str = "train", num_global: int = 2, num_local: int = 6) -> None:
        root = Path(root) / split
        self.ds = datasets.ImageFolder(str(root))
        self.num_views = num_global + num_local
        self.t_global = _make_globals()
        self.t_local = _make_locals_weak()

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        path, label = self.ds.samples[idx]
        img = datasets.folder.default_loader(path)
        views: List[torch.Tensor] = []
        for _ in range(2):
            views.append(self.t_global(img))
        for _ in range(self.num_views - 2):
            views.append(self.t_local(img))
        return torch.stack(views, dim=0), label


def build_multicrop_loader(
    root: Path,
    batch_size: int,
    num_workers: int = 8,
) -> Tuple[DataLoader, int]:
    """Train loader only (no val for SSL). Returns loader, dummy num_classes for labels."""
    ds = MultiCropImageNet(root, "train")
    num_classes = len(ds.ds.classes)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    return loader, num_classes
