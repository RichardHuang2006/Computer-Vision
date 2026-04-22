"""Alias: MAE uses ``pretrain.py`` for SSL, then ``linear_probe.py`` for evaluation."""

from __future__ import annotations


def main() -> None:
    print(
        "MAE training uses pretrain.py for self-supervised phase, "
        "then linear_probe.py for evaluation.\n"
        "  python -m src.SelfSupervised.mae.pretrain --data-dir data/imagenet100 --out-dir runs/mae\n"
        "  python -m src.SelfSupervised.mae.linear_probe --pretrained runs/mae/mae_pretrained.pt "
        "--out-dir runs/mae_probe --data-dir data/imagenet100"
    )


if __name__ == "__main__":
    main()
