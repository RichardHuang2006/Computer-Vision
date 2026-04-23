"""Multi-view augmentation for DINO (simplified: all views 224x224).

Expects a flat dataset layout with no class subfolders and no labels.json::

    data/imagenet100/
        train/
            img_0000001.jpg
            img_0000002.jpg
            ...
        val/
            img_0000001.jpg
            ...

All images directly under ``<root>/<split>/`` are used (recursively). Labels are
not needed for self-supervised pretraining, so a dummy ``0`` is returned.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _list_images(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected image directory at {root}. Put your images directly under "
            f"this folder (no class subfolders, no labels.json)."
        )
    paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
    if not paths:
        raise RuntimeError(f"No images found under {root} (extensions: {IMG_EXTENSIONS}).")
    paths.sort()
    return paths


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
    """Flat-folder multi-crop dataset.

    Returns ``(views, 0)`` where ``views`` has shape ``(num_views, 3, 224, 224)``.
    The label is a dummy ``0`` since DINO pretraining ignores labels.
    """

    def __init__(self, root: Path, split: str = "train", num_global: int = 2, num_local: int = 6) -> None:
        root = Path(root) / split
        self.samples = _list_images(root)
        self.num_views = num_global + num_local
        self.t_global = _make_globals()
        self.t_local = _make_locals_weak()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        img = datasets.folder.default_loader(str(path))
        views: List[torch.Tensor] = []
        for _ in range(2):
            views.append(self.t_global(img))
        for _ in range(self.num_views - 2):
            views.append(self.t_local(img))
        return torch.stack(views, dim=0), 0


def build_multicrop_loader(
    root: Path,
    batch_size: int,
    num_workers: int = 8,
    num_global: int = 2,
    num_local: int = 6,
) -> Tuple[DataLoader, int]:
    """Train loader only (no val for SSL). Returns ``(loader, 0)``.

    The second return value used to be ``num_classes`` for labeled ImageFolder
    mode; it is always ``0`` here because the flat layout has no class labels.
    """
    ds = MultiCropImageNet(root, "train", num_global=num_global, num_local=num_local)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return loader, 0
