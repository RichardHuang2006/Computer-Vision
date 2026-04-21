"""FCN-8s with ResNet-50 backbone (Long et al., CVPR 2015; ResNet variant common in practice).

High-level data path::

    Input RGB                                                    (3, H, W)
    ResNet50 stem + layer1                                       (256, H/4, W/4)
    Layer2                                                       (512, H/8, H/8)
    Layer3                                                       (1024, H/16, W/16)
    Layer4                                                       (2048, H/32, W/32)
    1x1 score on layer4                                          (C, H/32, W/32)
    Bilinear x2 + 1x1 score on layer3 fusion                     (C, H/16, W/16)
    Bilinear x2 + 1x1 score on layer2 fusion                     (C, H/8, W/8)
    Bilinear x8 to input resolution                              (C, H, W)

``forward(x)`` returns per-pixel logits ``(N, C, H, W)`` matching input size.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


class FCN8s(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained_backbone: bool = True) -> None:
        super().__init__()
        w = ResNet50_Weights.DEFAULT if pretrained_backbone else None
        rn = resnet50(weights=w)
        self.conv1 = rn.conv1
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.maxpool = rn.maxpool
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.layer4 = rn.layer4

        self.score_l4 = nn.Conv2d(2048, num_classes, kernel_size=1)
        self.score_l3 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.score_l2 = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        s4 = self.score_l4(l4)
        s3 = self.score_l3(l3)
        s2 = self.score_l2(l2)

        o = F.interpolate(s4, size=l3.shape[2:], mode="bilinear", align_corners=False) + s3
        o = F.interpolate(o, size=l2.shape[2:], mode="bilinear", align_corners=False) + s2
        o = F.interpolate(o, size=inp.shape[2:], mode="bilinear", align_corners=False)
        return o


def build_model(num_classes: int = 21) -> FCN8s:
    return FCN8s(num_classes=num_classes, pretrained_backbone=True)
