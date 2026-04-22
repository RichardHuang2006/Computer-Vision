"""CIFAR-10 loaders for PixelCNN (RGB 8-bit targets)."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_loaders(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders. ``root`` should contain torchvision CIFAR-10 files (see ``prepare_cifar10.py``).

    Each batch is ``(x_01, target)`` where ``x_01`` is float ``(N, 3, 32, 32)`` in ``[0, 1]`` and
    ``target`` is ``long`` ``(N, 3, 32, 32)`` with values in ``[0, 255]`` (class indices per subpixel).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Expected CIFAR-10 root at {root} (run scripts/prepare_cifar10.py)")

    train_tf = transforms.ToTensor()
    val_tf = transforms.ToTensor()

    train_set = datasets.CIFAR10(str(root), train=True, download=False, transform=train_tf)
    val_set = datasets.CIFAR10(str(root), train=False, download=False, transform=val_tf)

    def collate(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([b[0] for b in batch], dim=0)
        # uint8 targets in [0, 255] for CrossEntropyLoss
        targets = (xs * 255.0).round().clamp(0, 255).long()
        return xs, targets

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )
    return train_loader, val_loader
