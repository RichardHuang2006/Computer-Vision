"""Download Pascal VOC 2012 train/val (segmentation + detection).

Uses the official PASCAL host. Produces::

    data/voc2012/VOCdevkit/VOC2012/
        JPEGImages/
        SegmentationClass/
        Annotations/
        ImageSets/

Run::

    python scripts/prepare_voc2012.py --out-dir data/voc2012
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

VOC2012_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/voc2012"))
    args = p.parse_args()
    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "VOCtrainval_11-May-2012.tar"
    if not tar_path.exists():
        print(f"Downloading {VOC2012_URL} ...")
        urllib.request.urlretrieve(VOC2012_URL, tar_path)
    print(f"Extracting {tar_path} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(root)
    print(f"VOC 2012 extracted under {root / 'VOCdevkit'}")


if __name__ == "__main__":
    main()
