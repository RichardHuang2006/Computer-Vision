"""ResNet (He et al., CVPR 2016).

ResNet-50 layer flow (``224`` input, bottleneck blocks)::

    Stem   Conv7x7/64 s2 + BN + ReLU + MaxPool              (64, 56, 56)
    Stage1 3x Bottleneck(64->256)        stride 1 on first   (256, 56, 56)
    Stage2 4x Bottleneck stride 2 then x3                 (512, 28, 28)
    Stage3 6x Bottleneck stride 2 then x5                   (1024, 14, 14)
    Stage4 3x Bottleneck stride 2 then x2                   (2048, 7, 7)
    Head   GAP + FC(C)

ResNet-18/34 use BasicBlock (2 convs) with widths [64,128,256,512].

``forward(x) -> logits`` for ``x`` ``(N,3,224,224)``.
``build_model(arch, num_classes)`` with ``arch`` in ``resnet18|resnet34|resnet50``.
"""
from __future__ import annotations

from typing import List, Optional, Type

import torch
import torch.nn as nn

from src.Classification.common.layers import BasicBlock, Bottleneck


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int = 1000,
        in_ch: int = 3,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[nn.Module],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        expansion = block.expansion  # type: ignore
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * expansion),
            )

        layers_list: List[nn.Module] = []
        layers_list.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * expansion
        for _ in range(1, blocks):
            layers_list.append(block(self.inplanes, planes))
        return nn.Sequential(*layers_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def build_model(arch: str, num_classes: int) -> nn.Module:
    arch = arch.lower()
    if arch == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if arch == "resnet34":
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    if arch == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    raise ValueError(f"Unknown arch {arch!r}; use resnet18|resnet34|resnet50")
