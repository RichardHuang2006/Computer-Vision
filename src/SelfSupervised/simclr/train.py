"""Alias: SimCLR uses ``pretrain.py`` then ``linear_probe.py``."""

from __future__ import annotations


def main() -> None:
    print(
        "SimCLR training uses pretrain.py for self-supervised phase, "
        "then linear_probe.py for evaluation.\n"
        "  python -m src.SelfSupervised.simclr.pretrain --data-dir data/imagenet100 --out-dir runs/simclr\n"
        "  python -m src.SelfSupervised.simclr.linear_probe --pretrained runs/simclr/simclr_pretrained.pt "
        "--out-dir runs/simclr_probe --data-dir data/imagenet100"
    )


if __name__ == "__main__":
    main()
