"""Evaluate every trained classification checkpoint.

Two modes:
  - ``--image PATH``   : print top-k predictions for a single image, for every
                        architecture that has a checkpoint under ``--runs-dir``.
  - ``--data-dir PATH``: run full val-set evaluation and print val_top1 / val_top5
                        for each architecture.

Assumes the layout produced by ``src/Classification/<arch>/train.py``::

    runs/
      alexnet/best.pt
      vgg/best.pt
      googlenet/best.pt
      resnet/best.pt
      vit/best.pt
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets

from src.Classification.common.data import build_imagenet100_loaders
from src.Classification.common.infer import (
    build_val_transform,
    load_model_from_ckpt,
    run_image,
)
from src.Classification.common.utils import AverageMeter, accuracy


ARCHS = [
    ("alexnet", "alexnet"),
    ("vgg", "vgg16_bn"),
    ("googlenet", "googlenet"),
    ("resnet", "resnet50"),
    ("vit", "vit_s16"),
]


def _get_factory(arch_pkg: str):
    mod = importlib.import_module(f"src.Classification.{arch_pkg}.model")
    return mod.build_model


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    t0 = time.time()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        if isinstance(logits, tuple):
            logits = logits[0]
        a1, a5 = accuracy(logits, targets, topk=(1, 5))
        top1.update(a1, images.size(0))
        top5.update(a5, images.size(0))
    return top1.avg * 100, top5.avg * 100, time.time() - t0


def _idx_to_synset(val_root: Path) -> list[str]:
    """ImageFolder sorts class names alphabetically; use the same order."""
    return sorted(d.name for d in val_root.iterdir() if d.is_dir())


def main() -> None:
    p = argparse.ArgumentParser(description="Batch-evaluate all trained classifiers")
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--ckpt-name", default="best.pt", help="best.pt or last.pt")
    p.add_argument("--arch", choices=[a for a, _ in ARCHS], default=None,
                   help="Evaluate only this architecture. Omit to do all.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="If given, run full val-set evaluation (needs val/ folder).")
    p.add_argument("--image", type=Path, default=None,
                   help="If given, run single-image top-k across all archs.")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--labels-json", type=Path, default=None,
                   help="Optional Labels.json (synset -> readable) for nicer output.")
    args = p.parse_args()

    if args.image is None and args.data_dir is None:
        p.error("Provide either --image or --data-dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    label_map: Optional[dict[str, str]] = None
    if args.labels_json and args.labels_json.is_file():
        with open(args.labels_json) as f:
            label_map = json.load(f)

    results: list[tuple[str, float, float, float]] = []

    archs = [(p, a) for p, a in ARCHS if args.arch is None or p == args.arch]
    for pkg, arch in archs:
        ckpt = args.runs_dir / pkg / args.ckpt_name
        if not ckpt.is_file():
            print(f"[skip] {pkg}: {ckpt} not found")
            continue

        print(f"\n=== {pkg} ({arch}) — {ckpt} ===")
        try:
            factory = _get_factory(pkg)
            model = load_model_from_ckpt(ckpt, factory, arch, device)
        except Exception as e:
            print(f"  failed to load: {e}")
            continue

        if args.image is not None:
            pairs = run_image(model, args.image, device, topk=args.topk,
                              image_size=args.image_size)
            synsets = None
            if args.data_dir is not None and (args.data_dir / "val").is_dir():
                synsets = _idx_to_synset(args.data_dir / "val")
            for c, s in pairs:
                label = f"class_{c}"
                if synsets is not None and c < len(synsets):
                    syn = synsets[c]
                    readable = label_map.get(syn, syn) if label_map else syn
                    label = f"{syn}  {readable}"
                print(f"  {label}: {s:.4f}")

        if args.data_dir is not None:
            train_loader, val_loader, _ = build_imagenet100_loaders(
                args.data_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            del train_loader
            top1, top5, dt = _evaluate(model, val_loader, device)
            print(f"  val_top1={top1:.2f}  val_top5={top5:.2f}  ({dt:.1f}s)")
            results.append((pkg, top1, top5, dt))

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results:
        print("\n================ SUMMARY ================")
        print(f"{'arch':<12} {'top1':>7} {'top5':>7} {'time(s)':>9}")
        for name, t1, t5, dt in sorted(results, key=lambda r: -r[1]):
            print(f"{name:<12} {t1:>7.2f} {t5:>7.2f} {dt:>9.1f}")


if __name__ == "__main__":
    main()
