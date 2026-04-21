"""GoogLeNet training."""
from __future__ import annotations

from src.Classification.common.train import TrainDefaults, run

from .model import build_model


def main() -> None:
    run(
        model_factory=build_model,
        defaults=TrainDefaults(
            model_name="googlenet",
            arch="googlenet",
            batch_size=64,
            epochs=60,
            lr=0.1,
            weight_decay=1e-4,
            optimizer="sgd",
            scheduler="cosine",
            min_lr=1e-6,
            warmup_epochs=5,
            label_smoothing=0.1,
            num_workers=8,
        ),
    )


if __name__ == "__main__":
    main()
