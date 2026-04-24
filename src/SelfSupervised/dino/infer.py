"""Run linear probe checkpoint on an image."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ``Computer Vision`` (parent of ``src/``) on path so ``python -m src...`` works
# even when the shell cwd is not the repo root (e.g. ``Projects``).
_CV_ROOT = Path(__file__).resolve().parents[3]
if str(_CV_ROOT) not in sys.path:
    sys.path.insert(0, str(_CV_ROOT))

import torch
import torch.nn as nn

from src.Classification.common.data import imagenet100_class_names
from src.Classification.common.infer import run_image
from src.SelfSupervised.dino.linear_probe import ProbeClassifier
from src.SelfSupervised.dino.model import ViTEncoder


def _resolve_repo_path(p: Path) -> Path:
    """Resolve file or directory path, trying repo root if missing from cwd."""
    p = p.expanduser()
    if p.exists():
        return p.resolve()
    if p.is_absolute():
        return p.resolve()
    cand = (_CV_ROOT / p).resolve()
    if cand.exists():
        return cand
    return p.resolve()


def _ground_truth_label(image_path: Path) -> str | None:
    """Label if inferable from path: flat ``<wnid>__...`` or ImageFolder ``.../train/<wnid>/file``."""
    stem = image_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    parent = image_path.parent.name
    gparent = image_path.parent.parent.name.lower() if len(image_path.parents) > 1 else ""
    if gparent in ("train", "val") and parent not in ("train", "val", ""):
        return parent
    return None


def _resolve_data_path(p: Path) -> Path:
    """Resolve ``p``; if missing, try under repo root (so cwd need not be ``Computer Vision``)."""
    p = p.expanduser()
    if p.is_file():
        return p.resolve()
    if p.is_absolute():
        return p.resolve()
    under = (_CV_ROOT / p).resolve()
    if under.is_file():
        return under
    # Prefer reporting the repo-root path for relative args (clearer when cwd is wrong).
    if under.parent.is_dir():
        return under
    return p.resolve()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Classify one image with a trained DINO linear probe (probe_best.pt)."
    )
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--pretrained-encoder", type=Path, default=None)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="ImageNet-100 root (train/val). Used to print class names if the checkpoint "
        "has no ``class_names`` (older probes).",
    )
    args = p.parse_args()

    ckpt_path = _resolve_data_path(args.ckpt)
    img_path = _resolve_data_path(args.image)
    if not ckpt_path.is_file():
        print(
            f"ERROR: probe checkpoint not found:\n  {ckpt_path}\n\n"
            "Create it from the Computer Vision repo (any cwd is fine now):\n"
            "  python -m src.SelfSupervised.dino.linear_probe "
            "--data-dir data/imagenet100 "
            "--pretrained runs/dino/dino_pretrained.pt "
            "--out-dir runs/dino_probe\n\n"
            "That saves runs/dino_probe/probe_best.pt when validation improves.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not img_path.is_file():
        print(f"ERROR: image not found:\n  {img_path}", file=sys.stderr)
        sys.exit(1)
    enc_path: Path | None = None
    if args.pretrained_encoder is not None:
        enc_path = _resolve_data_path(args.pretrained_encoder)
        if not enc_path.is_file():
            print(f"ERROR: --pretrained-encoder not found:\n  {enc_path}", file=sys.stderr)
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    ex = ckpt.get("extra") or {}
    num_classes = int(ex.get("num_classes", ckpt.get("num_classes", 100)))
    enc = ViTEncoder(img_size=224, patch_size=16, embed_dim=384).to(device)
    if enc_path is not None:
        pre = torch.load(enc_path, map_location=device, weights_only=False)
        enc.load_state_dict(pre["student_enc"], strict=True)
    model = ProbeClassifier(enc, num_classes).to(device)
    sd = ckpt["state_dict"]
    model.load_state_dict(sd, strict=True)
    names: list[str] | None = None
    raw = ckpt.get("class_names")
    if isinstance(raw, list) and raw and all(isinstance(x, str) for x in raw):
        names = list(raw)
    elif args.data_dir is not None:
        root = _resolve_repo_path(args.data_dir)
        if root.is_dir():
            names = imagenet100_class_names(root)

    actual = _ground_truth_label(img_path)
    if actual is not None:
        extra = ""
        if names is not None and actual in names:
            extra = f" (class_index={names.index(actual)})"
        print(f"actual_class: {actual}{extra}", flush=True)
    else:
        print(
            "actual_class: unknown (expected flat ``<wnid>__<stem>.ext`` "
            "or ImageFolder ``.../train/<wnid>/file.ext``)",
            flush=True,
        )

    model.eval()
    pairs = run_image(model, img_path, device, topk=args.topk)

    print("top_5:", flush=True)
    for rank, (c, s) in enumerate(pairs, start=1):
        if names is not None and 0 <= c < len(names):
            print(f"  {rank}. class_{c} ({names[c]}): {s:.4f}", flush=True)
        else:
            print(f"  {rank}. class_{c}: {s:.4f}", flush=True)
    if names is None:
        print(
            "\nTip: pass --data-dir data/imagenet100 (or re-train probe) to show WordNet ids on top_5 lines.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
