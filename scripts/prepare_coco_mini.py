"""Build a 4k-image COCO-mini subset with ~40 frequent categories.

Place full COCO 2017 train images and instances JSON, then run::

    python scripts/prepare_coco_mini.py ^
        --images-dir path/to/train2017 ^
        --annotations path/to/instances_train2017.json ^
        --out-dir data/coco-mini ^
        --num-images 4000 ^
        --num-categories 40

Output::

    data/coco-mini/images/
    data/coco-mini/annotations.json

Images are copied (or symlink on Unix) into ``images/``; annotation ids are remapped.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True, help="COCO train2017 images folder.")
    p.add_argument("--annotations", type=Path, required=True, help="instances_train2017.json")
    p.add_argument("--out-dir", type=Path, default=Path("data/coco-mini"))
    p.add_argument("--num-images", type=int, default=4000)
    p.add_argument("--num-categories", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    with open(args.annotations, encoding="utf-8") as f:
        coco: Dict[str, Any] = json.load(f)

    cat_freq: Counter[int] = Counter()
    for ann in coco.get("annotations", []):
        cat_freq[ann["category_id"]] += 1

    top_cats = [c for c, _ in cat_freq.most_common(args.num_categories)]
    old_to_new: Dict[int, int] = {c: i + 1 for i, c in enumerate(top_cats)}  # 0 reserved bg in DETR

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
    for iid in chosen_ids:
        fn = id_to_file[iid]
        src = args.images_dir / fn
        dst = img_out / fn
        if not dst.exists():
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, dst)
        new_images.append(
            next(im for im in coco["images"] if im["id"] == iid)
        )

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

    print(f"Wrote {len(new_images)} images, {len(new_anns)} annotations to {out_root}")


if __name__ == "__main__":
    main()
