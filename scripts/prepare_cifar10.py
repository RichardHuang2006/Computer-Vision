"""Download CIFAR-10 via torchvision into ``data/cifar10/`` (torchvision on-disk layout)."""
from __future__ import annotations

import argparse
from pathlib import Path

from torchvision import datasets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=Path("data/cifar10"),
        help="Directory where torchvision will store ``cifar-10-batches-py/``",
    )
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    datasets.CIFAR10(str(root), train=True, download=True)
    datasets.CIFAR10(str(root), train=False, download=True)
    print(f"CIFAR-10 ready under {root}")


if __name__ == "__main__":
    main()
