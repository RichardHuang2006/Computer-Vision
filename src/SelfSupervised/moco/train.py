"""Alias: MoCo v1 uses ``pretrain.py`` then ``linear_probe.py``."""

from __future__ import annotations


def main() -> None:
    print(
        "MoCo v1 training uses pretrain.py for self-supervised phase, "
        "then linear_probe.py for evaluation.\n"
        "  python -m src.SelfSupervised.moco.pretrain --data-dir data/imagenet100 --out-dir runs/moco\n"
        "  python -m src.SelfSupervised.moco.linear_probe --pretrained runs/moco/moco_pretrained.pt "
        "--out-dir runs/moco_probe --data-dir data/imagenet100\n"
        "Note: queue size K must divide batch size (default K=16384, batch 256). "
        "Shuffle-BN from the paper is omitted (single-GPU)."
    )


if __name__ == "__main__":
    main()
