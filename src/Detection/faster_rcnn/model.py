"""Faster R-CNN with ResNet50-FPN (Ren et al., NeurIPS 2015 / He et al., 2017 FPN).

High-level Torchvision module::

    Backbone ResNet50 + FPN                                    P2..P6 feature pyramids
    Region Proposal Network                                    objectness + box2d per anchor
    ROI Align (multiscale) + head                              FC -> cls + reg

``forward(images, targets?)`` — training returns loss dict; inference returns detections.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn


def build_model(num_classes: int) -> nn.Module:
    # num_classes includes background for torchvision (+1 handled internally via +1 here)
    return fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
        num_classes=num_classes + 1,
    )
