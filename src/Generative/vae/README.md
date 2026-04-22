# Variational Autoencoder (VAE)

Kingma & Welling, *Auto-Encoding Variational Bayes* (ICLR 2014). Convolutional encoder/decoder for 32×32 RGB, diagonal Gaussian posterior ``q(z|x)``, standard normal prior, Bernoulli decoder (inputs scaled to ``[0, 1]``). Training objective is the ELBO; use ``--kl-weight`` for a beta-VAE.

CIFAR-10 is downloaded automatically on first train run (torchvision) into ``data/cifar10/``. Optionally you can pre-populate data with::

    python scripts/prepare_cifar10.py

Train::

    python -m src.Generative.vae.train --data-dir data/cifar10 --out-dir runs/vae

Sample from the prior (fast)::

    python -m src.Generative.vae.sample --ckpt runs/vae/best.pt --out runs/vae/samples.png

Reconstruction and ELBO-based **bits per dimension** on an image (resized to 32×32). The printed ``bits_per_dim_elbo_upper_bound`` uses the negative ELBO summed over the image, divided by ``C * H * W * log(2)``. It is an upper bound on the true marginal under the model, not exact NLL::

    python -m src.Generative.vae.infer --ckpt runs/vae/best.pt --image path/to.jpg --out runs/vae/recon.png

Checkpoints use ``{"state_dict": ..., "extra": {...}}``. Older runs saved ``{"model": ..., "args": ...}``; ``sample`` / ``infer`` still load those.
