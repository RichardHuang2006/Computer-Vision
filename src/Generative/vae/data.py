"""CIFAR-10 loaders for VAE (float RGB in [0, 1])."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_cifar10_loaders(
    root: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Train/val loaders. Each batch is ``(x, _)`` with ``x`` float ``(N, 3, 32, 32)`` in ``[0, 1]``.

    ``root`` is passed to ``torchvision.datasets.CIFAR10``. When ``download=True``,
    CIFAR-10 is fetched automatically on first run.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tfm = transforms.ToTensor()
    train_set = datasets.CIFAR10(str(root), train=True, download=download, transform=tfm)
    val_set = datasets.CIFAR10(str(root), train=False, download=download, transform=tfm)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader
