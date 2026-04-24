"""ImageNet-100 folder layout loaders."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, find_classes


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _has_imagefolder_layout(split_root: Path) -> bool:
    """True if ``split_root`` has at least one subdirectory that directly holds an image."""
    for child in split_root.iterdir():
        if not child.is_dir():
            continue
        for p in child.iterdir():
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
                return True
    return False


def _list_flat_images(split_root: Path) -> List[Path]:
    paths = [
        p
        for p in split_root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    ]
    if not paths:
        raise RuntimeError(
            f"No images under {split_root} (extensions: {IMG_EXTENSIONS}). "
            "For SSL-style flat data, filenames must look like ``<class>__<name>.ext`` "
            "(see scripts/prepare_imagenet100.py)."
        )
    paths.sort()
    return paths


def _class_from_flat_filename(path: Path) -> str:
    """Label prefix from ``prepare_imagenet100.py`` output: ``<wnid>__<stem>.JPEG``."""
    stem = path.stem
    if "__" not in stem:
        raise ValueError(
            f"Flat layout expects ``class__filename.ext``; got {path.name!r} under {path.parent}"
        )
    return stem.split("__", 1)[0]


def _build_flat_class_to_idx(train_root: Path, val_root: Path) -> Dict[str, int]:
    labels = set()
    for p in _list_flat_images(train_root):
        labels.add(_class_from_flat_filename(p))
    for p in _list_flat_images(val_root):
        labels.add(_class_from_flat_filename(p))
    classes = sorted(labels)
    return {c: i for i, c in enumerate(classes)}


def imagenet100_class_names(
    root: Path,
    train_dir_name: str = "train",
    val_dir_name: str = "val",
) -> List[str]:
    """Names for class indices ``0 .. num_classes-1`` (same order as training).

    * **ImageFolder** layout: subfolder names (usually WordNet ids), sorted as in torchvision.
    * **Flat** layout (``<wnid>__file.ext``): WordNet id strings parsed from filenames.
    """
    root = Path(root)
    train_root = root / train_dir_name
    val_root = root / val_dir_name
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"Missing {val_root}")
    if _has_imagefolder_layout(train_root):
        classes, _ = find_classes(str(train_root))
        return list(classes)
    class_to_idx = _build_flat_class_to_idx(train_root, val_root)
    return sorted(class_to_idx.keys(), key=lambda c: class_to_idx[c])


class FlatLabeledImageFolder(Dataset):
    """Images under one directory; class parsed from ``<class>__`` filename prefix."""

    def __init__(self, split_root: Path, transform, class_to_idx: Dict[str, int]) -> None:
        self.paths = _list_flat_images(split_root)
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.targets: List[int] = []
        for p in self.paths:
            c = _class_from_flat_filename(p)
            if c not in class_to_idx:
                raise KeyError(f"Unknown class {c!r} from {p} (not in train/val union)")
            self.targets.append(class_to_idx[c])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = default_loader(str(path))
        return self.transform(img), self.targets[idx]


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

    Supports two layouts under ``root/train`` and ``root/val``:

    1. **ImageFolder** — one subfolder per class, images inside class folders.
    2. **Flat (Kaggle prepare script)** — all images directly under ``train/`` / ``val/``,
       filenames ``<class>__<original_name>.ext`` (see ``scripts/prepare_imagenet100.py``).
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

    if _has_imagefolder_layout(train_root):
        train_set = datasets.ImageFolder(str(train_root), transform=train_transform)
        val_set = datasets.ImageFolder(str(val_root), transform=val_transform)
        num_classes = len(train_set.classes)
    else:
        class_to_idx = _build_flat_class_to_idx(train_root, val_root)
        train_set = FlatLabeledImageFolder(train_root, train_transform, class_to_idx)
        val_set = FlatLabeledImageFolder(val_root, val_transform, class_to_idx)
        num_classes = len(class_to_idx)

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
