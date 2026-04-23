"""DINO self-supervised pre-training entrypoint."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast

from src.Classification.common.utils import AverageMeter, format_hms
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


def _latest_resume_ckpt(out_dir: Path) -> Optional[Path]:
    """Return the most recent ``checkpoint_epoch_*.pt`` in ``out_dir``, if any."""
    candidates = sorted(out_dir.glob("checkpoint_epoch_*.pt"))
    return candidates[-1] if candidates else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/imagenet100"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-global", type=int, default=2)
    p.add_argument("--num-local", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--momentum-teacher", type=float, default=0.996)
    p.add_argument("--center-momentum", type=float, default=0.9)
    p.add_argument("--tau-s", type=float, default=0.1)
    p.add_argument("--tau-t", type=float, default=0.04)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a progress line every N optimizer steps (set 0 to disable).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="auto",
        help=(
            "'auto' (default) loads the latest checkpoint in --out-dir if present; "
            "'none' forces a fresh start; any other value is treated as an explicit "
            "path to a checkpoint .pt file."
        ),
    )
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, _ = build_multicrop_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        num_global=args.num_global,
        num_local=args.num_local,
    )
    model = DINO(out_dim=args.out_dim, img_size=224, patch_size=16, embed_dim=384).to(device)

    center = torch.zeros(args.out_dim, device=device)
    opt = torch.optim.AdamW(
        list(model.student_enc.parameters()) + list(model.student_head.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Resume logic
    start_epoch = 0
    resume_path: Optional[Path] = None
    if args.resume == "auto":
        resume_path = _latest_resume_ckpt(out_dir)
    elif args.resume and args.resume.lower() != "none":
        resume_path = Path(args.resume)

    if resume_path is not None and resume_path.is_file():
        print(f"[resume] loading {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.student_enc.load_state_dict(ckpt["student_enc"], strict=True)
        model.student_head.load_state_dict(ckpt["student_head"], strict=True)
        if "teacher_enc" in ckpt and "teacher_head" in ckpt:
            model.teacher_enc.load_state_dict(ckpt["teacher_enc"], strict=True)
            model.teacher_head.load_state_dict(ckpt["teacher_head"], strict=True)
        else:
            print("[resume] no teacher state in checkpoint, copying from student", flush=True)
            model.teacher_enc.load_state_dict(model.student_enc.state_dict())
            model.teacher_head.load_state_dict(model.student_head.state_dict())
        center = ckpt["center"].to(device)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0))
        print(f"[resume] starting from epoch {start_epoch + 1}/{args.epochs}", flush=True)
    else:
        if args.resume not in ("auto", "none"):
            print(f"[resume] {resume_path} not found; starting fresh", flush=True)

    if start_epoch >= args.epochs:
        print(f"Already trained for {start_epoch} epochs (>= --epochs {args.epochs}); nothing to do.")
        return

    steps_per_epoch = len(loader)
    step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        t_last = t0
        imgs_since_last = 0
        for batch_idx, (views, _) in enumerate(loader):
            views = views.to(device, non_blocking=True)  # (B, V, 3, 224, 224)
            b, v = views.shape[0], views.shape[1]

            # Teacher target from the two global views (no-grad, no activation cost).
            with torch.no_grad():
                with autocast(device_type=device.type, enabled=use_amp):
                    t1 = model.forward_teacher(views[:, 0])
                    t2 = model.forward_teacher(views[:, 1])
                t_target = ((t1 + t2) / 2.0).float()
                update_center(center, t_target, args.center_momentum)

            # Accumulate gradients view-by-view so we never hold more than a
            # single view's worth of ViT activations in memory at once.
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for i in range(v):
                with autocast(device_type=device.type, enabled=use_amp):
                    s_out = model.forward_student(views[:, i])
                    loss_i = dino_loss(s_out, t_target, center, args.tau_s, args.tau_t) / v
                scaler.scale(loss_i).backward()
                loss_sum += loss_i.detach().item()

            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(
                    list(model.student_enc.parameters()) + list(model.student_head.parameters()),
                    args.grad_clip,
                )
            scaler.step(opt)
            scaler.update()

            model.update_teacher(args.momentum_teacher)

            loss_m.update(loss_sum, b)
            step += 1
            imgs_since_last += b

            if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
                now = time.time()
                dt = max(now - t_last, 1e-6)
                ips = imgs_since_last / dt
                steps_done = batch_idx + 1
                steps_left = steps_per_epoch - steps_done
                eta_s = steps_left * (now - t0) / max(steps_done, 1)
                print(
                    f"  ep {epoch+1} step {steps_done}/{steps_per_epoch} "
                    f"loss={loss_m.avg:.4f} img/s={ips:.1f} "
                    f"eta={format_hms(eta_s)}",
                    flush=True,
                )
                t_last = now
                imgs_since_last = 0

        print(
            f"epoch {epoch+1}/{args.epochs} loss={loss_m.avg:.4f} time={format_hms(time.time()-t0)}",
            flush=True,
        )
        _save_full_ckpt(
            out_dir / f"checkpoint_epoch_{epoch+1:04d}.pt",
            model, opt, scaler, center, args, epoch + 1,
        )

    _save_full_ckpt(
        out_dir / "dino_pretrained.pt",
        model, opt, scaler, center, args, args.epochs,
    )


def _save_full_ckpt(
    path: Path,
    model: DINO,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    center: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
) -> None:
    torch.save(
        {
            "student_enc": model.student_enc.state_dict(),
            "student_head": model.student_head.state_dict(),
            "teacher_enc": model.teacher_enc.state_dict(),
            "teacher_head": model.teacher_head.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "center": center.cpu(),
            "epoch": epoch,
            "args": vars(args),
        },
        path,
    )


if __name__ == "__main__":
    main()
