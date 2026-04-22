"""Alias: CPC (simplified) uses ``pretrain.py`` then ``linear_probe.py``."""

from __future__ import annotations


def main() -> None:
    print(
        "CPC (simplified) training uses pretrain.py for self-supervised phase, "
        "then linear_probe.py for evaluation (256px inputs).\n"
        "  python -m src.SelfSupervised.cpc.pretrain --data-dir data/imagenet100 --out-dir runs/cpc\n"
        "  python -m src.SelfSupervised.cpc.linear_probe --pretrained runs/cpc/cpc_pretrained.pt "
        "--out-dir runs/cpc_probe --data-dir data/imagenet100"
    )


if __name__ == "__main__":
    main()
