# MAE

He et al., *Masked Autoencoders Are Scalable Vision Learners* (CVPR 2022).

Self-supervised pretrain (ViT-S/16 encoder + lightweight decoder on ImageNet-100 `train/`):

```bash
python -m src.SelfSupervised.mae.pretrain --data-dir data/imagenet100 --out-dir runs/mae
```

Linear probe on frozen encoder:

```bash
python -m src.SelfSupervised.mae.linear_probe --pretrained runs/mae/mae_pretrained.pt --out-dir runs/mae_probe --data-dir data/imagenet100
```

Use `--no-norm-pix-loss` to disable per-patch normalization in the reconstruction target.
