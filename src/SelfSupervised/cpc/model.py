"""Simplified CPC on images (Oord et al., 2018): patch grid + GRU + InfoNCE."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.Classification.common.layers import BasicBlock

IMG_SIZE = 256
PATCH_SIZE = 64
STRIDE = 32
GRID = 7
NUM_PATCHES = GRID * GRID


def extract_patches(x: torch.Tensor, patch_size: int = PATCH_SIZE, stride: int = STRIDE) -> torch.Tensor:
    """(B, 3, H, W) -> (B, NUM_PATCHES, 3, patch_size, patch_size)."""
    patches = x.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    b, c, nh, nw, p1, p2 = patches.shape
    return patches.permute(0, 2, 3, 1, 4, 5).reshape(b, nh * nw, c, patch_size, patch_size)


class PatchEncoder(nn.Module):
    """Small ResNet-style encoder for 64x64 patches -> 512-d."""

    feat_dim = 512

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.inplanes = 64
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        in_ch = self.inplanes
        downsample = None
        if stride != 1 or in_ch != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_ch,
                    planes * BasicBlock.expansion,
                    1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers: List[nn.Module] = [BasicBlock(in_ch, planes, stride, downsample)]
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class CPC(nn.Module):
    """Row-mean GRU context; predict mean patch embedding of future rows (InfoNCE)."""

    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau
        self.patch_enc = PatchEncoder()
        self.gru = nn.GRU(
            input_size=PatchEncoder.feat_dim,
            hidden_size=256,
            batch_first=True,
        )
        self.predictors = nn.ModuleList([nn.Linear(256, PatchEncoder.feat_dim) for _ in range(4)])

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        b = imgs.shape[0]
        patches = extract_patches(imgs)
        flat = patches.reshape(b * NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE)
        emb = self.patch_enc(flat).reshape(b, GRID, GRID, PatchEncoder.feat_dim)
        row = emb.mean(dim=2)
        out, _ = self.gru(row)
        loss_acc = torch.zeros((), device=imgs.device, dtype=torch.float32)
        n_terms = 0
        for t in range(GRID):
            for k in range(1, 5):
                if t + k >= GRID:
                    continue
                pred = self.predictors[k - 1](out[:, t, :])
                pos = emb[:, t + k].mean(dim=1)
                pred_n = F.normalize(pred, dim=-1)
                pos_n = F.normalize(pos, dim=-1)
                logits = torch.mm(pred_n, pos_n.t()) / self.tau
                labels = torch.arange(b, device=imgs.device)
                loss_acc = loss_acc + F.cross_entropy(logits, labels).float()
                n_terms += 1
        return loss_acc / max(1, n_terms)
