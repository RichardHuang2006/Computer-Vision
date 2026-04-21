"""FCN-8s training on VOC 2012."""
from __future__ import annotations

import argparse

from src.Segmentation.common.train import SegTrainDefaults, run
from src.Segmentation.fcn.model import build_model


def main() -> None:
    run(
        build_model(num_classes=21),
        SegTrainDefaults(model_name="fcn8s", batch_size=8, epochs=50, lr=1e-3),
    )


if __name__ == "__main__":
    main()
