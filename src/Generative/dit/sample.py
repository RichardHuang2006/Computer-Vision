"""Sample from a trained DiT checkpoint (DDIM + CFG); prefers EMA weights if present."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from src.Generative.dit.diffusion import build_diffusion
from src.Generative.dit.model import build_dit


def load_dit_for_sampling(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    run_args = ckpt.get("args") or {}
    arch = str(ckpt.get("arch", run_args.get("arch", "DiT-S/2")))
    num_classes = int(ckpt.get("num_classes", run_args.get("num_classes", 10)))
    image_size = int(ckpt.get("image_size", run_args.get("image_size", 32)))
    class_dropout = float(ckpt.get("class_dropout", run_args.get("class_dropout", 0.1)))
    T = int(ckpt.get("T", run_args.get("T", 1000)))
    beta_start = float(ckpt.get("beta_start", run_args.get("beta_start", 1e-4)))
    beta_end = float(ckpt.get("beta_end", run_args.get("beta_end", 2e-2)))

    model = build_dit(
        arch=arch,
        input_size=image_size,
        in_channels=3,
        num_classes=num_classes,
        class_dropout_prob=class_dropout,
        learn_sigma=False,
    ).to(device)

    ema_sd = ckpt.get("ema_state_dict")
    if ema_sd is not None:
        model.load_state_dict(ema_sd, strict=True)
    else:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    meta = {
        "num_classes": num_classes,
        "image_size": image_size,
        "T": T,
        "beta_start": beta_start,
        "beta_end": beta_end,
    }
    return model, meta


def build_labels(
    classes_spec: str,
    num_samples: int,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    spec = classes_spec.strip().lower()
    if spec == "random":
        return torch.randint(0, num_classes, (num_samples,), device=device, dtype=torch.long)
    parts = [int(x.strip()) for x in classes_spec.split(",") if x.strip()]
    if not parts:
        raise ValueError("Empty --classes; use 'random' or comma-separated class ids (0-9 for CIFAR-10)")
    for c in parts:
        if c < 0 or c >= num_classes:
            raise ValueError(f"Class {c} out of range [0, {num_classes - 1}]")
    out: list[int] = []
    for i in range(num_samples):
        out.append(parts[i % len(parts)])
    return torch.tensor(out, device=device, dtype=torch.long)


@torch.no_grad()
def sample_grid(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    y: torch.Tensor,
    num_classes: int,
    image_size: int,
    cfg_scale: float,
    ddim_steps: int,
    device: torch.device,
) -> torch.Tensor:
    n = y.shape[0]
    shape = (n, 3, image_size, image_size)
    x = diffusion.ddim_sample_loop(
        model,
        shape=shape,
        y=y,
        num_classes=num_classes,
        cfg_scale=cfg_scale,
        timesteps=ddim_steps,
        eta=0.0,
        clip_denoised=True,
    )
    return (x + 1.0) * 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="Output PNG path (grid of samples)")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--steps", type=int, default=50, help="DDIM steps")
    p.add_argument(
        "--classes",
        type=str,
        default="random",
        help="'random' or comma-separated class ids (e.g. '0,1,2' cycled to fill --num-samples)",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = args.ckpt.resolve()

    model, meta = load_dit_for_sampling(ckpt_path, device)
    diffusion = build_diffusion(
        num_timesteps=meta["T"],
        beta_start=meta["beta_start"],
        beta_end=meta["beta_end"],
    ).to(device)

    y = build_labels(args.classes, args.num_samples, meta["num_classes"], device)
    xs = sample_grid(
        model,
        diffusion,
        y,
        meta["num_classes"],
        meta["image_size"],
        args.cfg_scale,
        args.steps,
        device,
    ).cpu()

    nrow = max(1, int(round(args.num_samples**0.5)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(xs, args.out, nrow=nrow, padding=2)


if __name__ == "__main__":
    main()
