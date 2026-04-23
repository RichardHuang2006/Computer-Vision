# DiT (Diffusion Transformer)

Peebles & Xie, *Scalable Diffusion Models with Transformers* (ICLR 2023).

Class-conditional DiT on CIFAR-10 (32×32, RGB in `[-1, 1]`): adaLN-Zero blocks, sinusoidal timestep embedding, learned class embeddings with label dropout for classifier-free guidance (CFG). Training uses epsilon prediction and MSE (`L_simple`); sampling uses DDIM (default) or full DDPM in code.

## Train

From the `Computer Vision` directory with `PYTHONPATH` set to the project root (or `cd` into `Computer Vision` and `set PYTHONPATH=%CD%` on Windows):

```bash
python -m src.Generative.dit.train --data-dir data/cifar10 --out-dir runs/dit
```

If CIFAR-10 is not prepared yet, either run `python scripts/prepare_cifar10.py` or pass `--download`.

## Sample

```bash
python -m src.Generative.dit.sample --ckpt runs/dit/best.pt --out runs/dit/grid.png --cfg-scale 1.5 --steps 50
python -m src.Generative.dit.infer --ckpt runs/dit/best.pt --out runs/dit/infer.png
```

Use `--classes random` (default) or e.g. `--classes 0,1,2` (cycled to `--num-samples`). Checkpoints prefer `ema_state_dict` when present.

## Architectures

`--arch`: `DiT-Ti/2`, `DiT-S/2` (default), `DiT-B/2`. AdamW `lr=1e-4`, linear warmup + cosine decay, EMA decay `0.9999`, DDPM linear betas `1e-4` … `2e-2` over `T=1000`.
