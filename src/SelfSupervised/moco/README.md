# MoCo v1

He et al., *Momentum Contrast for Unsupervised Visual Representation Learning* (CVPR 2020).

Self-supervised pretrain (ResNet-50 + linear projection + momentum key encoder + queue + InfoNCE):

```bash
python -m src.SelfSupervised.moco.pretrain --data-dir data/imagenet100 --out-dir runs/moco
```

Linear probe on frozen ResNet-50 trunk:

```bash
python -m src.SelfSupervised.moco.linear_probe --pretrained runs/moco/moco_pretrained.pt --out-dir runs/moco_probe --data-dir data/imagenet100
```

**Notes:** `--K` (queue length, default 16384) must be divisible by `--batch-size`. Shuffle-BN from the paper is omitted (single-GPU training).
