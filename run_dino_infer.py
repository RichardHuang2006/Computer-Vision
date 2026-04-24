#!/usr/bin/env python3
"""Launch ``src.SelfSupervised.dino.infer`` from any cwd (fixes ``No module named 'src'``).

Example from ``Projects``::

    python Computer Vision/run_dino_infer.py ^
      --ckpt runs/dino_probe/probe_best.pt ^
      --pretrained-encoder runs/dino/dino_pretrained.pt ^
      --image data/imagenet100/train/n01440764__n01440764_10026.JPEG
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from src.SelfSupervised.dino.infer import main as infer_main

    infer_main()


if __name__ == "__main__":
    main()
