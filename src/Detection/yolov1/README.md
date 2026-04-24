# YOLOv1

Redmon et al., You Only Look Once: Unified, Real-Time Object Detection.

Build ``data/coco-mini`` — **download** (official COCO URLs; only the annotations zip + JPEGs needed for the subset, not the full 18GB train zip)::

    python scripts/prepare_coco_mini.py --download --out-dir data/coco-mini

If HTTPS fails (SSL / proxy), the script retries **HTTP** automatically; as a last resort use ``--insecure-ssl``.

Or **local** (you already unzipped train2017 + annotations)::

    python scripts/prepare_coco_mini.py --images-dir path/to/train2017 --annotations path/to/instances_train2017.json --out-dir data/coco-mini

Train::

    python -m src.Detection.yolov1.train --data-dir data/coco-mini --out-dir runs/yolov1

Progress / ETA: ``--log-every 50`` (default) prints ``eta_epoch`` and ``eta_run`` mid-epoch; each epoch line includes ``wall`` and ``eta_run_rem``. Use ``--log-every 0`` to disable mid-epoch lines.

Detection ``eval`` uses ``pycocotools``; install with ``pip install pycocotools`` (listed in ``requirements.txt``).
