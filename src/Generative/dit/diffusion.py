"""Gaussian diffusion (DDPM) with epsilon prediction; DDPM + DDIM sampling and CFG."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
    """Gather diffusion constants for batch timesteps ``t``; broadcast to ``x_shape``."""
    b = t.shape[0]
    out = a.gather(0, t).float()
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule_linear(num_timesteps: int, beta_start: float, beta_end: float) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


class GaussianDiffusion(nn.Module):
    """Linear beta schedule, fixed-variance reverse process (DDPM); sampling via DDPM or DDIM."""

    def __init__(
        self,
        betas: torch.Tensor,
        model_mean_type: str = "epsilon",
    ) -> None:
        super().__init__()
        betas = betas.float()
        if (betas <= 0).any() or (betas >= 1).any():
            raise ValueError("betas must be in (0, 1)")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]])

        self.num_timesteps = int(betas.shape[0])
        self.model_mean_type = model_mean_type

        register = self.register_buffer
        register("betas", betas)
        register("alphas_cumprod", alphas_cumprod)
        register("alphas_cumprod_prev", alphas_cumprod_prev)
        register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register("posterior_variance", posterior_variance)
        register(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        register(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        register(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def apply_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        num_classes: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        if cfg_scale <= 1.0:
            return model(x, t, y)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_null = torch.full_like(y, num_classes)
        y_in = torch.cat([y_null, y], dim=0)
        eps = model(x_in, t_in, y_in)
        eps_u, eps_c = eps.chunk(2, dim=0)
        return eps_u + cfg_scale * (eps_c - eps_u)

    def training_losses(
        self,
        model: nn.Module,
        x_start: torch.Tensor,
        y: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)
        eps_pred = model(x_t, t, y)
        return torch.nn.functional.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample_ddpm_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: int,
        y: torch.Tensor,
        num_classes: int,
        cfg_scale: float,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """One ancestral DDPM step from timestep ``t`` to ``t-1``."""
        b = x.shape[0]
        device = x.device
        t_b = torch.full((b,), t, device=device, dtype=torch.long)
        eps = self.apply_model(model, x, t_b, y, num_classes, cfg_scale)
        x_recon = self.predict_start_from_noise(x, t_b, eps)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

        posterior_mean = (
            _extract(self.posterior_mean_coef1, t_b, x.shape) * x_recon
            + _extract(self.posterior_mean_coef2, t_b, x.shape) * x
        )
        posterior_log_var = _extract(self.posterior_log_variance_clipped, t_b, x.shape)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return posterior_mean + torch.exp(0.5 * posterior_log_var) * noise

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        y: torch.Tensor,
        num_classes: int,
        cfg_scale: float,
        clip_denoised: bool = True,
        progress: Callable[[int], None] | None = None,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        img = torch.randn(*shape, device=device)
        for t_idx in reversed(range(self.num_timesteps)):
            img = self.p_sample_ddpm_step(model, img, t_idx, y, num_classes, cfg_scale, clip_denoised)
            if progress is not None:
                progress(t_idx)
        return img

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        y: torch.Tensor,
        num_classes: int,
        cfg_scale: float,
        timesteps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """DDIM with ``timesteps`` evenly spaced indices in [0, T-1] (inclusive), descending."""
        device = next(model.parameters()).device
        T = self.num_timesteps
        if timesteps > T:
            raise ValueError(f"timesteps={timesteps} must be <= T={T}")
        c = T // timesteps
        seq = list(range(0, T, c))
        if seq[-1] != T - 1:
            seq.append(T - 1)
        seq = sorted(set(seq))
        # Descending noise level: high t -> low t (each step predicts toward cleaner image).
        seq_desc = seq[::-1]
        img = torch.randn(*shape, device=device)
        for t_cur, t_next in zip(seq_desc[:-1], seq_desc[1:]):
            t_b = torch.full((shape[0],), t_cur, device=device, dtype=torch.long)
            eps = self.apply_model(model, img, t_b, y, num_classes, cfg_scale)
            alpha_bar = float(self.alphas_cumprod[t_cur])
            alpha_bar_next = float(self.alphas_cumprod[t_next])
            sqrt_ab = math.sqrt(alpha_bar)
            sqrt_omab = math.sqrt(1.0 - alpha_bar)
            pred_x0 = (img - sqrt_omab * eps) / sqrt_ab
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            sqrt_ab_next = math.sqrt(alpha_bar_next)
            sigma = (
                eta
                * math.sqrt((1.0 - alpha_bar_next) / max(1e-20, 1.0 - alpha_bar))
                * math.sqrt(max(0.0, 1.0 - alpha_bar / max(alpha_bar_next, 1e-20)))
            )
            dir_xt = math.sqrt(max(1e-20, 1.0 - alpha_bar_next - sigma**2)) * eps
            rand = torch.randn_like(img) if eta > 0 else torch.zeros_like(img)
            img = sqrt_ab_next * pred_x0 + dir_xt + sigma * rand
        return img


def build_diffusion(
    num_timesteps: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> GaussianDiffusion:
    betas = make_beta_schedule_linear(num_timesteps, beta_start, beta_end)
    return GaussianDiffusion(betas=betas)
