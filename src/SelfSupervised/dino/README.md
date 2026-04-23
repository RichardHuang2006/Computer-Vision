# DINO

Caron et al., Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021).

## Dataset layout

DINO pretraining is self-supervised — no labels and no `labels.json` are needed.
Drop your images into a flat `train/` (and optional `val/`) directory:

    data/imagenet100/
        train/
            img_0000001.jpg
            img_0000002.jpg
            ...
        val/
            img_0000001.jpg
            ...

Any of `.jpg .jpeg .png .bmp .webp .tif .tiff` are picked up recursively.

## Self-supervised pretrain

    python -m src.SelfSupervised.dino.pretrain --data-dir data/imagenet100 --out-dir runs/dino

## Linear probe on frozen encoder

The linear probe needs labels, so it expects the standard `ImageFolder` layout
(`train/<class>/*.jpg`, `val/<class>/*.jpg`) — not the flat layout above.

    python -m src.SelfSupervised.dino.linear_probe --pretrained runs/dino/dino_pretrained.pt --out-dir runs/dino_probe --data-dir data/imagenet100
