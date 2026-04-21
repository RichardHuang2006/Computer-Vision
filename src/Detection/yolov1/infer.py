"""YOLOv1 inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from src.Classification.common.data import IMAGENET_MEAN, IMAGENET_STD
from src.Detection.common.infer import draw_boxes
from src.Detection.yolov1.model import YOLOv1


def decode(
    pred: torch.Tensor, S: int, num_classes: int, w_pix: float, h_pix: float, conf_th: float = 0.25
):
    """pred (S,S,5+C) -> boxes xyxy in **original full-image pixel** space if w_pix,h_pix set."""
    boxes = []
    scores = []
    labels = []
    for i in range(S):
        for j in range(S):
            o = pred[i, j]
            obj = torch.sigmoid(o[4])
            if obj < conf_th:
                continue
            cx, cy, bw, bh = o[0:4]
            cls_prob = torch.sigmoid(o[5:])
            lab = int(cls_prob.argmax().item())
            sc = float(obj * cls_prob[lab])
            x1 = (cx.item() - bw.item() / 2) * w_pix
            y1 = (cy.item() - bh.item() / 2) * h_pix
            x2 = (cx.item() + bw.item() / 2) * w_pix
            y2 = (cy.item() + bh.item() / 2) * h_pix
            boxes.append([x1, y1, x2, y2])
            scores.append(sc)
            labels.append(lab)
    if not boxes:
        return (
            torch.empty(0, 4),
            torch.empty(0),
            torch.empty(0, dtype=torch.long),
        )
    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores),
        torch.tensor(labels, dtype=torch.long),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    nc = int((ckpt.get("extra") or {}).get("num_classes", 40))
    model = YOLOv1(num_classes=nc).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    img = Image.open(args.image).convert("RGB")
    w0, h0 = img.size
    sc = 448 / max(w0, h0)
    rw, rh = int(w0 * sc), int(h0 * sc)
    img_r = img.resize((rw, rh), Image.BILINEAR)
    t = TF.to_tensor(img_r)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        p = model(t)[0]
    bx, scs, labs = decode(p, 7, nc, float(rw), float(rh))
    sx, sy = w0 / rw, h0 / rh
    bx[:, 0] *= sx
    bx[:, 2] *= sx
    bx[:, 1] *= sy
    bx[:, 3] *= sy
    draw_boxes(args.image, bx, scs, labs, args.out)


if __name__ == "__main__":
    main()
