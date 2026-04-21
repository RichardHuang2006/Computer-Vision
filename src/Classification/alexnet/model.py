"""AlexNet (Krizhevsky et al., NIPS 2012).

Standard single-stream variant (torchvision-style widths + LRN). Shapes for ``224 x 224``::

    Input                                                      (3, 224, 224)
    Conv(64, 11x11, s=4, pad=2) + ReLU + LRN + MaxPool(3,s=2) (64, 27, 27)
    Conv(192, 5x5, pad=2)        + ReLU + LRN + MaxPool(3,s=2) (192, 13, 13)
    Conv(384, 3x3, pad=1)        + ReLU                         (384, 13, 13)
    Conv(256, 3x3, pad=1)        + ReLU                         (256, 13, 13)
    Conv(256, 3x3, pad=1)        + ReLU + MaxPool(3,s=2)        (256, 6, 6)
    AdaptiveAvgPool -> 6x6                                     (256, 6, 6)
    Flatten                                                    (9216,)
    Linear(9216->4096) + ReLU + Dropout(0.5)                   (4096,)
    Linear(4096->4096) + ReLU + Dropout(0.5)                   (4096,)
    Linear(4096->num_classes)                                  (C,)

Forward: ``logits = model(x)`` with ``x`` shape ``(N, 3, 224, 224)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch != "alexnet":
        raise ValueError(f"Unknown arch {arch!r}")
    return AlexNet(num_classes=num_classes)
