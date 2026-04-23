"""CIFAR-10 loaders for DiT: RGB float in ``[-1, 1]`` with class labels."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_loaders(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders. Each batch is ``(x, y)`` with ``x`` ``(N, 3, 32, 32)`` in ``[-1, 1]``, ``y`` ``(N,)`` long.

    ``root`` must contain torchvision CIFAR-10 files unless ``download=True``.
    """
    root = Path(root)
    if not download and not root.is_dir():
        raise FileNotFoundError(f"Expected CIFAR-10 root at {root} (run scripts/prepare_cifar10.py)")

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t * 2.0 - 1.0),
        ]
    )
    train_set = datasets.CIFAR10(str(root), train=True, download=download, transform=tfm)
    val_set = datasets.CIFAR10(str(root), train=False, download=download, transform=tfm)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
