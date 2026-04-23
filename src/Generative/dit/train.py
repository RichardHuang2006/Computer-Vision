"""Train class-conditional DiT on CIFAR-10 (epsilon prediction, DDPM noise schedule)."""
from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import utils as vutils

from src.Classification.common.utils import AverageMeter, ETATracker, format_hms, load_checkpoint, save_checkpoint
from src.Generative.dit.data import build_cifar10_loaders
from src.Generative.dit.diffusion import build_diffusion
from src.Generative.dit.model import DIT_CONFIGS, build_dit


def lr_at_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    min_lr: float,
    warmup_epochs: int,
) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    t = min(max(t, 0.0), 1.0)
    return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def update_ema(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    ema_sd = ema_model.state_dict()
    m_sd = model.state_dict()
    for k in ema_sd:
        v_ema = ema_sd[k]
        v_m = m_sd[k]
        if v_ema.dtype.is_floating_point:
            v_ema.mul_(decay).add_(v_m, alpha=1.0 - decay)
        else:
            v_ema.copy_(v_m)


def run_epoch_val_loss(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    m = AverageMeter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device_type=device.type, enabled=use_amp):
                loss = diffusion.training_losses(model, x, y)
            m.update(loss.item(), x.size(0))
    return m.avg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--arch", type=str, default="DiT-S/2", choices=list(DIT_CONFIGS.keys()))
    p.add_argument("--class-dropout", type=float, default=0.1)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--T", type=int, default=1000, help="Diffusion timesteps")
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--sample-cfg", type=float, default=1.5, help="CFG scale for epoch sample grids")
    p.add_argument("--sample-steps", type=int, default=50, help="DDIM steps for epoch sample grids")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 10
    train_loader, val_loader = build_cifar10_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
    )

    model = build_dit(
        arch=args.arch,
        input_size=32,
        in_channels=3,
        num_classes=num_classes,
        class_dropout_prob=args.class_dropout,
        learn_sigma=False,
    ).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()

    diffusion = build_diffusion(num_timesteps=args.T, beta_start=args.beta_start, beta_end=args.beta_end).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu", enabled=use_amp)

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, opt, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_metric", float("inf")))
        esd = ckpt.get("ema_state_dict")
        if esd is not None:
            ema.load_state_dict(esd, strict=True)

    eta_tr = ETATracker()
    total_epochs = args.epochs
    image_size = 32

    for epoch in range(start_epoch, total_epochs):
        lr = lr_at_epoch(epoch, total_epochs, args.lr, args.min_lr, args.warmup_epochs)
        for pg in opt.param_groups:
            pg["lr"] = lr

        model.train()
        loss_m = AverageMeter()
        t0 = time.time()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=use_amp):
                loss = diffusion.training_losses(model, x, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model, args.ema_decay)

            loss_m.update(loss.item(), x.size(0))

            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                print(
                    f"  epoch {epoch + 1}/{total_epochs} step {step + 1}/{len(train_loader)} "
                    f"loss={loss_m.avg:.4f} lr={lr:.2e}",
                    flush=True,
                )

        train_time = time.time() - t0
        eta_tr.update(train_time)

        val_loss = run_epoch_val_loss(model, diffusion, val_loader, device, use_amp)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)

        print(
            f"Epoch {epoch + 1}/{total_epochs}  train_mse={loss_m.avg:.4f}  val_mse={val_loss:.4f}  "
            f"best_val_mse={best_val:.4f}  lr={lr:.2e}  train_t={format_hms(train_time)}  "
            f"ETA~{format_hms(eta_tr.eta(total_epochs - epoch - 1))}",
            flush=True,
        )

        ema.eval()
        with torch.no_grad():
            y_grid = torch.randint(0, num_classes, (64,), device=device)
            samples = diffusion.ddim_sample_loop(
                ema,
                shape=(64, 3, image_size, image_size),
                y=y_grid,
                num_classes=num_classes,
                cfg_scale=args.sample_cfg,
                timesteps=args.sample_steps,
                eta=0.0,
                clip_denoised=True,
            )
        vutils.save_image(
            (samples + 1.0) * 0.5,
            out_dir / f"samples_epoch_{epoch + 1:04d}.png",
            nrow=8,
            padding=2,
        )

        flat_extra = {
            "args": vars(args),
            "arch": args.arch,
            "num_classes": num_classes,
            "image_size": image_size,
            "class_dropout": args.class_dropout,
            "T": args.T,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "dit_config": DIT_CONFIGS[args.arch],
            "ema_state_dict": ema.state_dict(),
        }
        save_checkpoint(
            out_dir / "last.pt",
            model,
            opt,
            epoch,
            best_val,
            extra=flat_extra,
        )
        if is_best:
            save_checkpoint(
                out_dir / "best.pt",
                model,
                opt,
                epoch,
                best_val,
                extra=flat_extra,
            )

    print(f"Checkpoints written to {out_dir} (last.pt each epoch, best.pt when val improves)", flush=True)


if __name__ == "__main__":
    main()
