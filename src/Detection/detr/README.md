# DETR

Carion et al., End-to-End Object Detection with Transformers (ECCV 2020).

Minimal ResNet50 + PyTorch ``TransformerEncoder/Decoder`` + learnable queries.

``python -m src.Detection.detr.train --data-dir data/coco-mini --out-dir runs/detr``
