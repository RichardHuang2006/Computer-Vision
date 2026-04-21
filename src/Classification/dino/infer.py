"""Run linear probe checkpoint on an image."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.Classification.common.infer import run_image
from src.Classification.dino.linear_probe import ProbeClassifier
from src.Classification.dino.model import ViTEncoder


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pretrained-encoder", type=Path, default=None)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.ckpt, map_location=device)
    ex = ckpt.get("extra") or {}
    num_classes = int(ex.get("num_classes", ckpt.get("num_classes", 100)))
    enc = ViTEncoder(img_size=224, patch_size=16, embed_dim=384).to(device)
    if args.pretrained_encoder:
        pre = torch.load(args.pretrained_encoder, map_location=device, weights_only=False)
        enc.load_state_dict(pre["student_enc"], strict=True)
    model = ProbeClassifier(enc, num_classes).to(device)
    sd = ckpt["state_dict"]
    model.load_state_dict(sd, strict=True)
    model.eval()
    pairs = run_image(model, args.image, device, topk=args.topk)
    for c, s in pairs:
        print(f"class_{c}: {s:.4f}")


if __name__ == "__main__":
    main()
