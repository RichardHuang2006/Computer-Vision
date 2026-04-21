"""VGG-16 with Batch Normalization (Simonyan & Zisserman, ICLR 2015).

High-level block-wise flow (``224 x 224`` input)::

    Stem    Conv3x3x64  x2  + MaxPool                     (64, 112, 112) -> (64, 56, 56)
    Block1  Conv3x3x128 x2  + MaxPool                     (128, 28, 28)
    Block2  Conv3x3x256 x3  + MaxPool                     (256, 14, 14)
    Block3  Conv3x3x512 x3  + MaxPool                     (512, 7, 7)
    Block4  Conv3x3x512 x3  + MaxPool                     (512, 7, 7)   (optional pool to 7)
    Head    Flatten -> FC4096+ReLU+Drop -> FC4096+ReLU+Drop -> FC(C)

``forward(x) -> logits`` with ``x`` ``(N,3,224,224)``.
"""
from __future__ import annotations

from typing import List, Union

import torch
import torch.nn as nn


def _make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_ch = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)


cfgs = {
    "vgg16_bn": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, features: nn.Sequential, num_classes: int = 1000, init_weights: bool = True) -> None:
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch != "vgg16_bn":
        raise ValueError(f"Unknown arch {arch!r}; use vgg16_bn")
    features = _make_layers(cfgs["vgg16_bn"], batch_norm=True)
    return VGG(features, num_classes=num_classes)
