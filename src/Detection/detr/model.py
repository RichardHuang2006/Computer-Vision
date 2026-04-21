"""DETR (Carion et al., ECCV 2020) — minimal ResNet50 + Transformer.

Diagram::

    ResNet50 stride-32 feature map                     (B, 2048, H', W')
    1x1 reduce to hidden_dim                           (B, d, H', W')
    Flatten spatial + sinusoidal PE -> encoder         (B, H'W', d)
    Transformer encoder x6                             same
    Learned queries + decoder x6                       (B, Q, d)
    Class head (C+1 incl no-object) + MLP bbox head   (B, Q, C+1), (B, Q, 4) cxcywh [0,1]

``forward(images)`` returns ``pred_logits``, ``pred_boxes``.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torchvision.models as tvm


class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int, out: int, num_layers: int) -> None:
        super().__init__()
        layers = []
        c = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(c, hid), nn.ReLU(inplace=True)])
            c = hid
        layers.append(nn.Linear(c, out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DETR(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        rn = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(rn.children())[:-2])
        self.input_proj = nn.Conv2d(2048, hidden_dim, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            hidden_dim, nheads, dim_feedforward=2048, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)
        dec_layer = nn.TransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward=2048, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(images)
        x = self.input_proj(feats)
        b, c, h, w = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        x = self.encoder(x)
        q = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        hs = self.decoder(q, x)
        logits = self.class_embed(hs)
        boxes = self.bbox_embed(hs).sigmoid()
        return logits, boxes


def build_model(num_classes: int) -> DETR:
    return DETR(num_classes=num_classes)
