"""ViT-S/16 training with stronger augmentation defaults."""
from __future__ import annotations

from src.Classification.common.train import TrainDefaults, run

from .model import build_model


def main() -> None:
    run(
        model_factory=build_model,
        defaults=TrainDefaults(
            model_name="vit_s16",
            arch="vit_s16",
            batch_size=128,
            epochs=100,
            lr=3e-3,
            weight_decay=0.05,
            optimizer="adamw",
            scheduler="cosine",
            min_lr=1e-5,
            warmup_epochs=5,
            label_smoothing=0.1,
            mixup_alpha=0.2,
            strong_aug=True,
            num_workers=8,
        ),
    )


if __name__ == "__main__":
    main()
