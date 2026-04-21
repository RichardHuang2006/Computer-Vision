"""GoogLeNet / Inception v1 (Szegedy et al., CVPR 2015).

Block diagram (single forward path; two auxiliary towers during training)::

    Stem        Conv7x7/64 s2 + LRN + MaxPool                         (64, 56, 56)
                Conv1x1/64 + Conv3x3/192 + LRN + MaxPool               (192, 28, 28)
    Inception3a 1x1/64 1x1/96->3x3/128 1x16->5x5/32 pool->1x1/32     (256, 28, 28)
    Inception3b ...                                                (480, 28, 28)
    MaxPool                                                             (480, 14, 14)
    Inception4a ...                                                (512, 14, 14)
    Inception4b ...                                                (512, 14, 14)
    Inception4c ...                                                (512, 14, 14)
    Inception4d ...                                                (528, 14, 14)
    Inception4e ...                                                (832, 14, 14)
    MaxPool                                                             (832, 7, 7)
    Inception5a ...                                                (832, 7, 7)
    Inception5b ...                                                (1024, 7, 7)
    GAP 7x7 -> 1024                                                  (1024,)
    FC(num_classes)

 Auxiliary A (after 4a): AvgPool5x5 -> 512 -> FC -> C   (training only, weight 0.3)
 Auxiliary B (after 4d): AvgPool5x5 -> 528 -> FC -> C   (training only, weight 0.3)

``forward``: training returns ``(logits, aux1, aux2)``; eval returns ``logits`` only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(
        self,
        in_ch: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        super().__init__()
        self.branch1 = BasicConv2d(in_ch, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_ch, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BasicConv2d(in_ch, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class GoogLeNet(nn.Module):
    def __init__(self, num_classes: int = 1000, aux_logits: bool = True) -> None:
        super().__init__()
        self.aux_logits = aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if aux_logits:
            self.aux1 = nn.Sequential(
                nn.AvgPool2d(5, stride=3),
                BasicConv2d(512, 128, kernel_size=1),
                nn.Flatten(),
                nn.Linear(4 * 4 * 128, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.7),
                nn.Linear(1024, num_classes),
            )
            self.aux2 = nn.Sequential(
                nn.AvgPool2d(5, stride=3),
                BasicConv2d(528, 128, kernel_size=1),
                nn.Flatten(),
                nn.Linear(4 * 4 * 128, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.7),
                nn.Linear(1024, num_classes),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # type: ignore
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        else:
            aux1 = None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        else:
            aux2 = None
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        if self.training and self.aux_logits and aux1 is not None and aux2 is not None:
            return logits, aux1, aux2
        return logits


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch != "googlenet":
        raise ValueError(f"Unknown arch {arch!r}")
    return GoogLeNet(num_classes=num_classes, aux_logits=True)
