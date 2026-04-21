# Fast R-CNN (RoI head)

Uses the same Torchvision ``fasterrcnn_resnet50_fpn`` module as Faster R-CNN (RoI classification + box regression). Historically Fast R-CNN used **Selective Search** proposals; here the RPN provides region proposals. See `../common/selective_search.py` for offline proposal caching.

``python -m src.Detection.fast_rcnn.train --data-dir data/coco-mini --out-dir runs/fastrcnn``
