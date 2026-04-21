"""Single-image / directory inference for classifiers."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD

ModelFactory = Callable[..., nn.Module]


def build_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


@torch.no_grad()
def run_image(
    model: nn.Module,
    image_path: Path,
    device: torch.device,
    topk: int = 5,
    image_size: int = 224,
) -> List[Tuple[int, float]]:
    model.eval()
    tfm = build_val_transform(image_size)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = logits.softmax(dim=1)[0]
    scores, idx = probs.topk(min(topk, probs.numel()))
    return [(int(i), float(s)) for i, s in zip(idx.tolist(), scores.tolist())]


def load_model_from_ckpt(
    ckpt_path: Path,
    model_factory: ModelFactory,
    arch: str,
    device: torch.device,
) -> nn.Module:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = int(ckpt.get("num_classes", 100))
    ex = ckpt.get("extra")
    if isinstance(ex, dict) and "num_classes" in ex:
        num_classes = int(ex["num_classes"])
    model = model_factory(arch, num_classes).to(device)
    sd = ckpt.get("state_dict", ckpt)
    if not isinstance(sd, dict):
        sd = ckpt["state_dict"]
    if sd and any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def cli_main(model_factory: ModelFactory, default_arch: str) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--arch", type=str, default=default_arch)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_from_ckpt(args.ckpt, model_factory, args.arch, device)
    pairs = run_image(model, args.image, device, topk=args.topk, image_size=args.image_size)
    for c, s in pairs:
        print(f"class_{c}: {s:.4f}")
