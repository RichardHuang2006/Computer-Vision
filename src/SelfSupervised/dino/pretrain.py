"""DINO self-supervised pre-training entrypoint."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.Classification.common.utils import AverageMeter, format_hms, save_checkpoint
from src.SelfSupervised.dino.multicrop_data import build_multicrop_loader
from src.SelfSupervised.dino.model import DINO


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_out: torch.Tensor, beta: float) -> None:
    batch_mean = teacher_out.mean(dim=0)
    center.mul_(beta).add_((1.0 - beta) * batch_mean)


def dino_loss(
    student_out: torch.Tensor,
    teacher_out: torch.Tensor,
    center: torch.Tensor,
    tau_s: float,
    tau_t: float,
) -> torch.Tensor:
    """Cross-entropy between teacher prob and student log-prob (K dim)."""
    teacher_out = teacher_out.detach()
    t = F.softmax((teacher_out - center) / tau_t, dim=-1)
    s = F.log_softmax(student_out / tau_s, dim=-1)
    return -(t * s).sum(dim=-1).mean()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--momentum-teacher", type=float, default=0.996)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--tau-s", type=float, default=0.1)
    p.add_argument("--tau-t", type=float, default=0.04)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, _ = build_multicrop_loader(args.data_dir, args.batch_size, args.num_workers)
    model = DINO(out_dim=args.out_dim, img_size=224, patch_size=16, embed_dim=384).to(device)

    center = torch.zeros(args.out_dim, device=device)
    opt = torch.optim.AdamW(
        list(model.student_enc.parameters()) + list(model.student_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch

    step = 0
    for epoch in range(args.epochs):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for views, _ in loader:
            views = views.to(device, non_blocking=True)  # (B, V, 3, 224, 224)
            b, v, c, h, w = views.shape
            # teacher: first two globals averaged
            with torch.no_grad():
                t1 = model.forward_teacher(views[:, 0])
                t2 = model.forward_teacher(views[:, 1])
                t = (t1 + t2) / 2.0
                update_center(center, t, args.center_momentum)

            loss_total = 0.0
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                for i in range(v):
                    s_out = model.forward_student(views[:, i])
                    loss_total = loss_total + dino_loss(s_out, t, center, args.tau_s, args.tau_t)
                loss_total = loss_total / v

            scaler.scale(loss_total).backward()
            scaler.step(opt)
            scaler.update()

            m = args.momentum_teacher
            model.update_teacher(m)

            loss_m.update(loss_total.item(), b)
            step += 1

        print(
            f"epoch {epoch+1}/{args.epochs} loss={loss_m.avg:.4f} time={format_hms(time.time()-t0)}",
            flush=True,
        )
        torch.save(
            {
                "student_enc": model.student_enc.state_dict(),
                "student_head": model.student_head.state_dict(),
                "center": center.cpu(),
                "args": vars(args),
            },
            out_dir / f"checkpoint_epoch_{epoch+1:04d}.pt",
        )
    torch.save(
        {
            "student_enc": model.student_enc.state_dict(),
            "student_head": model.student_head.state_dict(),
            "center": center.cpu(),
            "args": vars(args),
        },
        out_dir / "dino_pretrained.pt",
    )


if __name__ == "__main__":
    main()
