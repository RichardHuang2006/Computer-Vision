# SimCLR

Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations* (ICML 2020).

Self-supervised pretrain (ResNet-50 + MLP projection + NT-Xent on ImageNet-100 `train/`):

```bash
python -m src.SelfSupervised.simclr.pretrain --data-dir data/imagenet100 --out-dir runs/simclr
```

Linear probe on frozen backbone:

```bash
python -m src.SelfSupervised.simclr.linear_probe --pretrained runs/simclr/simclr_pretrained.pt --out-dir runs/simclr_probe --data-dir data/imagenet100
```

Default peak LR scales with batch size (`0.3 * batch_size/256`). Use `--tau` for NT-Xent temperature (default 0.5).
