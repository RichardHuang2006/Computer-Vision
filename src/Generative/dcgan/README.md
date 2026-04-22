# DCGAN

Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks* (ICLR workshop 2016).

Train on CIFAR-10 (expects `data/cifar10/` from `scripts/prepare_cifar10.py`):

```bash
python -m src.Generative.dcgan.train --data-dir data/cifar10 --out-dir runs/dcgan
```

Sample a grid from a checkpoint:

```bash
python -m src.Generative.dcgan.sample --ckpt runs/dcgan/last.pt --out runs/dcgan/grid.png
python -m src.Generative.dcgan.infer --ckpt runs/dcgan/last.pt --out runs/dcgan/infer.png
```

Hyperparameters: `--nz` (latent dim, default 100), `--ngf` / `--ndf` (channel width, default 64), Adam `lr=2e-4`, `beta1=0.5`.
