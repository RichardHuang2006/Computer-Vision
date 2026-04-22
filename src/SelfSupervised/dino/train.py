"""Alias: DINO uses ``pretrain.py`` for SSL; this prints guidance."""

from __future__ import annotations


def main() -> None:
    print(
        "DINO training uses pretrain.py for self-supervised phase, "
        "then linear_probe.py for evaluation.\n"
        "  python -m src.SelfSupervised.dino.pretrain --data-dir data/imagenet100 --out-dir runs/dino\n"
        "  python -m src.SelfSupervised.dino.linear_probe --pretrained runs/dino/dino_pretrained.pt "
        "--out-dir runs/dino_probe --data-dir data/imagenet100"
    )


if __name__ == "__main__":
    main()
