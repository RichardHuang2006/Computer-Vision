"""YOLOv1 grid detector (Redmon et al., CVPR 2016) — ResNet backbone + detection head.

Conceptual stack (``448`` input, ``S=7``, ``B=2``)::

    ResNet50 layers 1–4 (stride 32)                               (2048, 14, 14) @448? 
    Actually 448/32 = 14 — for paper 7x7, use AdaptiveAvgPool or extra conv to 7x7
    Flatten+FC layers reproducing 7x7x(5B+C) tensor               reshape to (S,S,5*B+C)

Forward::

    ``logits`` ``(N, S, S, 5*B + C)`` with ``x,y,w,h,obj`` per box (sigmoid in loss).

"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


class YOLOv1(nn.Module):
    def __init__(self, num_classes: int, S: int = 7, B: int = 1) -> None:
        super().__init__()
        self.S = S
        self.B = B  # code assumes B=1 for head linear size
        self.num_classes = num_classes
        rn = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        modules = list(rn.children())[:-2]
        self.backbone = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((S, S))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048 * S * S, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (5 + num_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        f = self.backbone(x)
        f = self.pool(f)
        o = self.fc(f)
        return o.view(n, self.S, self.S, 5 + self.num_classes)


def build_model(num_classes: int) -> YOLOv1:
    return YOLOv1(num_classes=num_classes)
