# Computer Vision

Paper-faithful implementations with shared `common/` training and inference per task.

## Layout

- [`data/`](data/) — cached datasets (gitignored except `.gitkeep`)
- [`runs/`](runs/) — checkpoints, logs, TensorBoard
- [`src/Classification/`](src/Classification/) — AlexNet, VGG16-BN, GoogLeNet, ResNet, ViT-S/16, DINO
- [`src/Segmentation/`](src/Segmentation/) — FCN-8s (VOC 2012)
- [`src/Detection/`](src/Detection/) — Fast R-CNN, Faster R-CNN, YOLOv1, DETR (COCO-mini)

## Setup

```bash
cd "Computer Vision"
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

Run modules from this directory so `src` is importable:

```bash
set PYTHONPATH=%CD%
python -m src.Classification.alexnet.train --data-dir data/imagenet100 --out-dir runs/alexnet_test
```

On Linux/macOS: `export PYTHONPATH=$PWD`

## Data

1. **ImageNet-100** — `python scripts/prepare_imagenet100.py` (needs Kaggle API / `kaggle.json` in user home or project root)
2. **VOC 2012** — `python scripts/prepare_voc2012.py` (downloads from official host)
3. **COCO-mini** — place COCO 2017 `train2017` + `annotations` then `python scripts/prepare_coco_mini.py`

See each script’s docstring for paths.

## Training / inference

Each architecture folder has `train.py` and `infer.py`:

```bash
python -m src.Classification.resnet.train --data-dir data/imagenet100 --out-dir runs/resnet50 --arch resnet50
python -m src.Classification.resnet.infer --ckpt runs/resnet50/best.pt --image path/to.jpg
python -m src.Segmentation.fcn.train --data-dir data/voc2012 --out-dir runs/fcn
python -m src.Detection.faster_rcnn.train --data-dir data/coco-mini --out-dir runs/frcnn
```

DINO uses `pretrain.py` and `linear_probe.py` instead of standard `train.py` for the SSL phase.
