"""Download and unpack Kaggle ImageNet-100 (ambityga/imagenet100).

Requires Kaggle credentials at ``~/.kaggle/kaggle.json`` or env vars
``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.

Layout produced::

    data/imagenet100/train/<class_name>/*.JPEG
    data/imagenet100/val/<class_name>/*.JPEG

Run from ``Computer Vision``::

    python scripts/prepare_imagenet100.py --out-dir data/imagenet100
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def _flatten_train(out: Path) -> None:
    """Kaggle bundle ships ``train.X1..X4`` plus ``val.X``. Merge into train/ and val/."""
    # Create canonical split folders
    train_root = out / "train"
    val_root = out / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    for sub in list(out.iterdir()):
        if not sub.is_dir():
            continue
        name = sub.name
        if name.startswith("train"):
            for cls_dir in sub.iterdir():
                if cls_dir.is_dir():
                    dest = train_root / cls_dir.name
                    if dest.exists():
                        # move contents
                        for f in cls_dir.iterdir():
                            shutil.move(str(f), str(dest / f.name))
                        cls_dir.rmdir()
                    else:
                        shutil.move(str(cls_dir), str(dest))
            sub.rmdir()
        elif name.startswith("val"):
            for cls_dir in sub.iterdir():
                if cls_dir.is_dir():
                    dest = val_root / cls_dir.name
                    if dest.exists():
                        for f in cls_dir.iterdir():
                            shutil.move(str(f), str(dest / f.name))
                        cls_dir.rmdir()
                    else:
                        shutil.move(str(cls_dir), str(dest))
            sub.rmdir()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/imagenet100"),
        help="Destination root for train/val split folders.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        default="ambityga/imagenet100",
        help="Kaggle dataset slug.",
    )
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("Install the kaggle package: pip install kaggle", file=sys.stderr)
        raise SystemExit(1)

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        print(
            "Kaggle auth failed. Put kaggle.json in ~/.kaggle/ or set "
            "KAGGLE_USERNAME/KAGGLE_KEY env vars.",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    print(f"Downloading {args.dataset} to {out} ...")
    api.dataset_download_files(args.dataset, path=str(out), quiet=False, unzip=False)

    # Find the downloaded zip
    zips = sorted(out.glob("*.zip"))
    if not zips:
        print("No zip found after download.", file=sys.stderr)
        raise SystemExit(1)
    zip_path = zips[0]

    print(f"Extracting {zip_path} ...")
    with tempfile.TemporaryDirectory(dir=str(out)) as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp_path)
        # Move top-level children into out/
        children = [p for p in tmp_path.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            inner = children[0]
            children = list(inner.iterdir())
        for sub in children:
            dest = out / sub.name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.move(str(sub), str(dest))

    zip_path.unlink(missing_ok=True)

    _flatten_train(out)

    n_train = len(list((out / "train").iterdir())) if (out / "train").exists() else 0
    n_val = len(list((out / "val").iterdir())) if (out / "val").exists() else 0
    print(f"Done. train classes = {n_train}, val classes = {n_val}, root = {out}")


if __name__ == "__main__":
    main()
