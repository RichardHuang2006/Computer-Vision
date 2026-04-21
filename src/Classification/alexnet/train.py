"""AlexNet training entry (ImageNet-100)."""
from __future__ import annotations

from src.Classification.common.train import TrainDefaults, run

from .model import build_model


def main() -> None:
    run(
        model_factory=build_model,
        defaults=TrainDefaults(
            model_name="alexnet",
            arch="alexnet",
            batch_size=128,
            epochs=30,
            lr=0.01,
            weight_decay=5e-4,
            optimizer="sgd",
            scheduler="step",
            step_size=10,
            gamma=0.1,
            label_smoothing=0.0,
            warmup_epochs=0,
            num_workers=8,
            image_size=224,
        ),
    )


if __name__ == "__main__":
    main()
