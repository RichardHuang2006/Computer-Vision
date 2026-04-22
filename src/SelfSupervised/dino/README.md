# DINO

Caron et al., Emerging Properties in Self-Supervised Vision Transformers (ICCV 2021).

Self-supervised pretrain::

    python -m src.SelfSupervised.dino.pretrain --data-dir data/imagenet100 --out-dir runs/dino

Linear probe on frozen encoder::

    python -m src.SelfSupervised.dino.linear_probe --pretrained runs/dino/dino_pretrained.pt --out-dir runs/dino_probe --data-dir data/imagenet100
