"""Download and unpack Kaggle ImageNet-100 (ambityga/imagenet100).

Requires Kaggle credentials at ``~/.kaggle/kaggle.json`` or env vars
``KAGGLE_USERNAME`` / ``KAGGLE_KEY``.

Produces a flat, label-free layout suitable for self-supervised pretraining
(DINO, MAE, etc.)::

    data/imagenet100/
        train/
            <class>__<original_filename>.JPEG
            ...
        val/
            <class>__<original_filename>.JPEG
            ...

Class information from the Kaggle archive (``train.X1..X4`` / ``val.X`` with
class subfolders) is folded into the filename as ``<class>__<stem>`` so files
stay unique and you can still recover the label if you ever need to.

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


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def _flatten_split(extracted_root: Path, split_dest: Path, split_prefixes: tuple[str, ...]) -> int:
    """Walk every ``train*``/``val*`` bundle under ``extracted_root`` and move
    all image files into ``split_dest`` as ``<class>__<name>`` (flat)."""
    split_dest.mkdir(parents=True, exist_ok=True)
    moved = 0
    for sub in list(extracted_root.iterdir()):
        if not sub.is_dir():
            continue
        if not sub.name.lower().startswith(split_prefixes):
            continue
        for path in sub.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTENSIONS:
                continue
            cls_name = path.parent.name
            dest_name = f"{cls_name}__{path.name}"
            dest = split_dest / dest_name
            if dest.exists():
                dest = split_dest / f"{cls_name}__{path.stem}_{moved}{path.suffix}"
            shutil.move(str(path), str(dest))
            moved += 1
        shutil.rmtree(sub, ignore_errors=True)
    return moved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/imagenet100"),
        help="Destination root. Will contain train/ and val/ with flat image files.",
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

        children = [p for p in tmp_path.iterdir()]
        if len(children) == 1 and children[0].is_dir():
            extracted_root = children[0]
        else:
            extracted_root = tmp_path

        train_dest = out / "train"
        val_dest = out / "val"
        # Clear any previous run
        for d in (train_dest, val_dest):
            if d.exists():
                shutil.rmtree(d)

        n_train = _flatten_split(extracted_root, train_dest, ("train",))
        n_val = _flatten_split(extracted_root, val_dest, ("val",))

    zip_path.unlink(missing_ok=True)

    print(
        f"Done. {n_train} train images, {n_val} val images at {out}\n"
        f"  {train_dest}\n  {val_dest}"
    )


if __name__ == "__main__":
    main()
