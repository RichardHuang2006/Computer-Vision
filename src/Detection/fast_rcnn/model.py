"""Fast R-CNN (Girshick, ICCV 2015) detection head with ResNet50-FPN.

In this educational repo we reuse the **same** Torchvision ``fasterrcnn_resnet50_fpn``
implementation as Faster R-CNN: the RoI head and bbox regression path are the *Fast R-CNN*
stage; the difference to the historical Fast R-CNN paper is that proposals come from the
**learned RPN** instead of Selective Search. Use ``common/selective_search.py`` + frozen-RPN
experiments if you need SS proposals for ablations.

Diagram::

    FPN features -> (RPN proposals) -> ROI Align -> FC heads -> cls + box deltas

``build_model`` matches ``faster_rcnn.model.build_model``.
"""
from __future__ import annotations

from src.Detection.faster_rcnn.model import build_model

__all__ = ["build_model"]
