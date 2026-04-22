"""Download Pascal VOC 2012 train/val (segmentation + detection).

Uses the official PASCAL host. Produces::

    data/voc2012/VOCdevkit/VOC2012/
        JPEGImages/
        SegmentationClass/
        Annotations/
        ImageSets/

Run::

    python scripts/prepare_voc2012.py --out-dir data/voc2012

The download and extraction are both streamed in fixed-size chunks so
peak memory usage stays around a few MB even though the tar is ~2 GB.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm

VOC2012_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)

DOWNLOAD_CHUNK = 1 << 20  # 1 MiB


def stream_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` in chunks with a resumable .part file."""
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "voc2012-prepare/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", "0")) or None
        with open(part, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
        ) as bar:
            while True:
                chunk = resp.read(DOWNLOAD_CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    part.replace(dest)


def stream_extract(tar_path: Path, out_dir: Path) -> None:
    """Extract ``tar_path`` member-by-member in streaming mode.

    ``tarfile.open(mode="r|")`` disables random access, which means the
    library does not build an in-memory index of every member. That keeps
    peak memory bounded regardless of archive size.
    """
    with open(tar_path, "rb") as raw, tarfile.open(fileobj=raw, mode="r|") as tf:
        for member in tqdm(tf, desc=f"extracting {tar_path.name}", unit="file"):
            tf.extract(member, out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=Path("data/voc2012"))
    p.add_argument(
        "--keep-tar",
        action="store_true",
        help="keep the downloaded .tar after extraction",
    )
    args = p.parse_args()

    root = args.out_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / "VOCtrainval_11-May-2012.tar"

    if not tar_path.exists():
        print(f"Downloading {VOC2012_URL} -> {tar_path}")
        try:
            stream_download(VOC2012_URL, tar_path)
        except BaseException:
            part = tar_path.with_suffix(tar_path.suffix + ".part")
            if part.exists():
                part.unlink(missing_ok=True)
            raise
    else:
        print(f"Using cached archive {tar_path}")

    print(f"Extracting {tar_path} -> {root}")
    stream_extract(tar_path, root)

    if not args.keep_tar:
        tar_path.unlink(missing_ok=True)

    voc_root = root / "VOCdevkit"
    if not voc_root.exists():
        raise RuntimeError(
            f"Extraction finished but {voc_root} is missing; archive may be corrupt"
        )
    print(f"VOC 2012 extracted under {voc_root}")
    print(f"Free space on target drive: {shutil.disk_usage(root).free / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
