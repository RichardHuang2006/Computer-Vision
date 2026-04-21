"""COCO mAP via pycocotools."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_evaluate(
    coco_gt_json: str,
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """``predictions``: list of dict with keys image_id, category_id, bbox xywh, score."""
    coco_gt = COCO(coco_gt_json)
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP_50": float(coco_eval.stats[1]),
    }
