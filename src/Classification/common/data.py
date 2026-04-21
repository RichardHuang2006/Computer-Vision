"""ImageNet-100 folder layout loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_imagenet100_loaders(
    root: Path,
    image_size: int = 224,
    batch_size: int = 256,
    num_workers: int = 8,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
    use_strong_aug: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Return train_loader, val_loader, num_classes.

    ``root`` should contain ``train/`` and ``val/`` class subfolders.
    """
    root = Path(root)
    train_root = root / train_dir_name
    val_root = root / val_dir_name
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"Missing {val_root}")

    train_t: list = [
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if use_strong_aug:
        train_t.extend(
            [
                transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            ]
        )
    else:
        train_t.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
    train_t.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(train_t)

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_set = datasets.ImageFolder(str(train_root), transform=train_transform)
    val_set = datasets.ImageFolder(str(val_root), transform=val_transform)
    num_classes = len(train_set.classes)

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
    )
    return train_loader, val_loader, num_classes


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    mean_t = tensor.new_tensor(mean).view(3, 1, 1)
    std_t = tensor.new_tensor(std).view(3, 1, 1)
    return tensor * std_t + mean_t
