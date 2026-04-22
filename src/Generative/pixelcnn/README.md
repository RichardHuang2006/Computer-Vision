# PixelCNN (original)

van den Oord et al., *Pixel Recurrent Neural Networks* (ICML 2016), CNN variant with masked convolutions (type A/B) and RGB channel masking. Trained on CIFAR-10 with 256-way softmax per R/G/B subpixel.

Channel width: use ``n_filters`` such that ``2 * n_filters`` is divisible by 3 (default ``126`` → 252 feature maps).

Prepare data (torchvision download into ``data/cifar10/``)::

    python scripts/prepare_cifar10.py

Train::

    python -m src.Generative.pixelcnn.train --data-dir data/cifar10 --out-dir runs/pixelcnn

Sample (slow: full forward per subpixel)::

    python -m src.Generative.pixelcnn.sample --ckpt runs/pixelcnn/best.pt --out runs/pixelcnn/samples.png

Bits per dimension on an image (resized to 32×32)::

    python -m src.Generative.pixelcnn.infer --ckpt runs/pixelcnn/best.pt --image path/to.jpg
