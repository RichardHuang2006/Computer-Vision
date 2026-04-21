# AlexNet

Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks (NIPS 2012).

Train (from `Computer Vision`, `PYTHONPATH=.`)::

    python -m src.Classification.alexnet.train --data-dir data/imagenet100 --out-dir runs/alexnet_imagenet100

Infer::

    python -m src.Classification.alexnet.infer --ckpt runs/alexnet_imagenet100/best.pt --image path/to.jpg
