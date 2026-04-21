"""Download and unpack Kaggle ImageNet-100 (ambityga/imagenet100).

Requires ``kaggle`` CLI or API credentials (``~/.kaggle/kaggle.json``).

Layout produced::

    data/imagenet100/train/<class_name>/*.JPEG
    data/imagenet100/val/<class_name>/*.JPEG

Run from ``Computer Vision``::

    python scripts/prepare_imagenet100.py --out-dir data/imagenet100
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/imagenet100"),
        help="Destination root for train/val split folders.",
    )
    args = p.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "download",
                "-d",
                "ambityga/imagenet100",
                "-p",
                str(out),
                "--force",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print("kaggle download failed; ensure kaggle.json is configured.", file=sys.stderr)
        raise SystemExit(e.returncode) from e

    zip_path = out / "imagenet100.zip"
    if not zip_path.exists():
        print(f"Expected {zip_path} after download.", file=sys.stderr)
        raise SystemExit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        shutil.unpack_archive(zip_path, tmp_path)
        # Kaggle bundle: often train.X/ val.X or single folder — move children
        children = list(tmp_path.iterdir())
        if len(children) == 1 and children[0].is_dir():
            inner = children[0]
            for sub in inner.iterdir():
                dest = out / sub.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(sub), str(dest))
        else:
            for sub in children:
                dest = out / sub.name
                if sub.is_dir() and dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(sub), str(dest))

    zip_path.unlink(missing_ok=True)
    print(f"ImageNet-100 ready under {out}")


if __name__ == "__main__":
    main()
