"""Build a COCO-mini subset (~4k images, ~40 categories) for detection training.

**Local mode** — you already have COCO 2017 train images + JSON::

    python scripts/prepare_coco_mini.py --images-dir path/to/train2017 \\
        --annotations path/to/instances_train2017.json --out-dir data/coco-mini

**Download mode** — fetches official files from cocodataset.org (no full 18GB train zip;
only the annotations zip + the JPEGs needed for the mini subset)::

    python scripts/prepare_coco_mini.py --download --out-dir data/coco-mini

Output::

    data/coco-mini/images/<*.jpg>
    data/coco-mini/annotations.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import ssl
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set
from urllib.error import URLError
from urllib.request import Request, urlopen

COCO_ANNOTATIONS_ZIP_URL = (
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ZIP_JSON_MEMBER = "annotations/instances_train2017.json"
TRAIN2017_IMAGE_URL = "https://images.cocodataset.org/train2017/{}"


def _candidate_urls(url: str) -> List[str]:
    """Try HTTPS first, then plain HTTP (avoids some SSL / corporate-proxy issues)."""
    urls = [url]
    if url.startswith("https://"):
        urls.append("http://" + url[len("https://") :])
    return urls


def _download_file(
    url: str,
    dest: Path,
    label: str,
    *,
    timeout: int = 600,
    insecure_ssl: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[skip] {label} already at {dest}", flush=True)
        return
    ctx: ssl.SSLContext | None = None
    if insecure_ssl:
        ctx = ssl._create_unverified_context()

    tmp = dest.with_suffix(dest.suffix + ".partial")
    last_err: BaseException | None = None
    for u in _candidate_urls(url):
        print(f"[get] {label}\n  {u}\n  -> {dest}", flush=True)
        req = Request(u, headers={"User-Agent": "prepare_coco_mini/1.2"})
        try:
            open_kw: dict = {"timeout": timeout}
            if ctx is not None:
                open_kw["context"] = ctx
            with urlopen(req, **open_kw) as resp:
                total = resp.headers.get("Content-Length")
                total_i = int(total) if total and total.isdigit() else None
                n = 0
                chunk = 1024 * 1024
                with open(tmp, "wb") as f:
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
                        n += len(buf)
                        if total_i and n % (10 * chunk) < chunk:
                            print(
                                f"  ... {n / (1024 * 1024):.1f} / {total_i / (1024 * 1024):.1f} MiB",
                                flush=True,
                            )
                        elif not total_i and n % (20 * chunk) < chunk:
                            print(f"  ... {n / (1024 * 1024):.1f} MiB", flush=True)
            tmp.replace(dest)
            return
        except (URLError, OSError) as e:
            last_err = e
            print(f"[warn] download failed ({e!r}); trying next URL if any…", flush=True)
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
        except BaseException:
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
            raise
    assert last_err is not None
    raise last_err


def _ensure_instances_json(cache_dir: Path, *, insecure_ssl: bool) -> Path:
    """Download annotations zip if needed; return path to instances_train2017.json."""
    cache_dir = cache_dir.resolve()
    ann_json = cache_dir / "instances_train2017.json"
    if ann_json.is_file():
        return ann_json
    zip_path = cache_dir / "annotations_trainval2017.zip"
    _download_file(
        COCO_ANNOTATIONS_ZIP_URL,
        zip_path,
        "annotations_trainval2017.zip",
        timeout=900,
        insecure_ssl=insecure_ssl,
    )
    print(f"[extract] {ZIP_JSON_MEMBER} from zip", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if ZIP_JSON_MEMBER not in names:
            raise RuntimeError(
                f"Member {ZIP_JSON_MEMBER!r} not in zip (has {len(names)} entries). "
                "COCO may have changed the archive layout."
            )
        ann_json.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(ZIP_JSON_MEMBER) as zsrc, open(ann_json, "wb") as out:
            shutil.copyfileobj(zsrc, out)
    return ann_json


def _copy_or_download_image(
    fname: str,
    dst: Path,
    images_dir: Path | None,
    use_download: bool,
    *,
    insecure_ssl: bool,
) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    if use_download:
        url = TRAIN2017_IMAGE_URL.format(fname)
        _download_file(url, dst, fname, timeout=120, insecure_ssl=insecure_ssl)
        return
    assert images_dir is not None
    src = images_dir / fname
    if not src.is_file():
        raise FileNotFoundError(f"Missing image (local mode): {src}")
    shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build COCO-mini: subset of COCO 2017 train for detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Download annotations zip + only the train2017 JPEGs needed for the subset "
        "(no local COCO install required).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/.coco_cache"),
        help="Where to store downloaded annotations zip / extracted JSON (download mode).",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Folder of train2017 JPEGs (required in local mode; ignored with --download).",
    )
    p.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Path to instances_train2017.json (required in local mode; ignored with --download).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("data/coco-mini"))
    p.add_argument("--num-images", type=int, default=4000)
    p.add_argument("--num-categories", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--insecure-ssl",
        action="store_true",
        help="Disable TLS certificate verification (use if you see SSL: CERTIFICATE_VERIFY_FAILED "
        "or hostname mismatch behind a proxy). Less safe; prefer fixing trust store when possible.",
    )
    args = p.parse_args()
    random.seed(args.seed)

    use_dl = args.download
    if use_dl:
        ann_path = _ensure_instances_json(args.cache_dir, insecure_ssl=args.insecure_ssl)
        images_dir = None
    else:
        if args.images_dir is None or args.annotations is None:
            p.error("Local mode requires both --images-dir and --annotations (or use --download).")
        ann_path = args.annotations.expanduser().resolve()
        images_dir = args.images_dir.expanduser().resolve()
        if not ann_path.is_file():
            p.error(
                f"annotations file not found:\n  {ann_path}\n\n"
                "Use --download to fetch from cocodataset.org, or place instances_train2017.json locally."
            )
        if not images_dir.is_dir():
            p.error(
                f"images directory not found:\n  {images_dir}\n\n"
                "Use --download to fetch images from cocodataset.org, or point to your train2017 folder."
            )

    with open(ann_path, encoding="utf-8") as f:
        coco: Dict[str, Any] = json.load(f)

    cat_freq: Counter[int] = Counter()
    for ann in coco.get("annotations", []):
        cat_freq[ann["category_id"]] += 1

    top_cats = [c for c, _ in cat_freq.most_common(args.num_categories)]
    old_to_new: Dict[int, int] = {c: i + 1 for i, c in enumerate(top_cats)}

    img_ids_with_top: Set[int] = set()
    for ann in coco["annotations"]:
        if ann["category_id"] in old_to_new:
            img_ids_with_top.add(ann["image_id"])

    eligible = [img["id"] for img in coco["images"] if img["id"] in img_ids_with_top]
    random.shuffle(eligible)
    chosen_ids = set(eligible[: args.num_images])

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    out_root = args.out_dir.resolve()
    img_out = out_root / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    new_images: List[Dict[str, Any]] = []
    chosen_ordered = sorted(chosen_ids)
    n_need = len(chosen_ordered)
    for j, iid in enumerate(chosen_ordered):
        fn = id_to_file[iid]
        dst = img_out / fn
        if use_dl and (j + 1) % 100 == 0:
            print(f"[images] {j + 1}/{n_need}", flush=True)
        _copy_or_download_image(
            fn, dst, images_dir, use_dl, insecure_ssl=args.insecure_ssl
        )
        new_images.append(next(im for im in coco["images"] if im["id"] == iid))

    new_anns: List[Dict[str, Any]] = []
    new_id = 1
    for ann in coco["annotations"]:
        if ann["image_id"] not in chosen_ids:
            continue
        cid = ann["category_id"]
        if cid not in old_to_new:
            continue
        a = dict(ann)
        a["id"] = new_id
        new_id += 1
        a["category_id"] = old_to_new[cid]
        new_anns.append(a)

    new_cats = []
    for c in coco["categories"]:
        if c["id"] in old_to_new:
            nc = dict(c)
            nc["id"] = old_to_new[c["id"]]
            new_cats.append(nc)

    out_json = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_anns,
        "categories": sorted(new_cats, key=lambda x: x["id"]),
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f)

    mode = "download" if use_dl else "local"
    print(
        f"Done ({mode}): {len(new_images)} images, {len(new_anns)} annotations -> {out_root}",
        flush=True,
    )
    if use_dl:
        print(
            f"Cache (annotations zip / json): {args.cache_dir.resolve()}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
