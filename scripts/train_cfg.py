"""
CFG-Primary Training — train the conditioned path through the CFG combination.

Standard diffusion training gives eps_cond and eps_null separate losses. The CFG
combination eps_null + s*(eps_cond - eps_null) for s>1 produces worse denoising
than either alone. CFG-primary training fixes this by training the conditioned
path exclusively through the CFG combination:

    eps_cfg = (1 - s) * eps_null + s * eps_cond
    L = masked_mse(eps_cfg, noise_target)

No detach on eps_null. No separate L_null. Varying s regularizes both modes.

Six configs:
  A: Baseline — standard training + 50% conditioning dropout (single UNet fwd)
  B: CFG-primary, fixed s
  C: CFG-primary, random s ~ U(s_min, s_max)
  D: CFG-primary, random s + warmup (s_max ramps up)
  E: CFG-primary, learned s(t) via GuidanceSchedule MLP
  F: CFG-primary, learned s(t) + warmup clamp
"""
import csv
import gc
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset
from src.utils.mask_utils import (
    create_latent_band_mask,
    dilate_mask_batch,
    downsample_mask_maxpool,
    unet_roundtrip_masks,
)
from src.utils.optim_utils import (
    L2SPRegularizer,
    build_norm_param_id_set,
    flatten_modules,
    split_decay_no_decay,
)
from src.utils.crop_utils import clip_crop_multi
from src.inference.generate import generate_anomagic_single

_SEED = 42


def _worker_init_fn(worker_id):
    """Seed Python random + numpy in DataLoader workers (required on Windows/spawn)."""
    worker_seed = _SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ──────────────────────────────────────────────────────────────────────
# GuidanceSchedule — learned s(t) for configs E and F
# ──────────────────────────────────────────────────────────────────────

class GuidanceSchedule(nn.Module):
    """Learned guidance scale as a function of normalised timestep.

    Maps t_normalized in [0, 1] (0=clean, 1=full noise) to s >= 1.0
    via ``s = 1.0 + softplus(MLP(t))``.

    Initialised so output ~ 0 => softplus(0) ~ 0.69 => s ~ 1.69.
    ~17K params total.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
        # Small init so output starts near 0
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, t_normalized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_normalized: [B, 1] in [0, 1], where 0=clean, 1=full noise.

        Returns:
            s: [B, 1] guidance scale >= 1.0.
        """
        return 1.0 + F.softplus(self.net(t_normalized))


# ──────────────────────────────────────────────────────────────────────
# Guidance scale sampling
# ──────────────────────────────────────────────────────────────────────

def sample_guidance_scale(
    cfg_config: str,
    batch_size: int,
    device: torch.device,
    timesteps: torch.Tensor = None,
    guidance_schedule: GuidanceSchedule = None,
    progress: float = 0.0,
    cfg_scale: float = 3.0,
    cfg_s_min: float = 1.5,
    cfg_s_max: float = 4.0,
) -> torch.Tensor:
    """Return guidance scale s as [B, 1, 1, 1] for broadcasting.

    Args:
        cfg_config: One of A, B, C, D, E, F.
        batch_size: Number of samples.
        device: Target device.
        timesteps: [B] diffusion timesteps (needed for E, F).
        guidance_schedule: GuidanceSchedule module (needed for E, F).
        progress: Training progress in [0, 1] (needed for D, F).
        cfg_scale: Fixed scale for config B.
        cfg_s_min: Min scale for random sampling (C, D).
        cfg_s_max: Max scale for random sampling (C, D).

    Returns:
        s: [B, 1, 1, 1] guidance scale tensor.
    """
    if cfg_config == "A":
        # Not used for loss computation in config A
        return torch.ones(batch_size, 1, 1, 1, device=device)

    if cfg_config == "B":
        return torch.full((batch_size, 1, 1, 1), cfg_scale, device=device)

    if cfg_config == "C":
        # Random s ~ U(s_min, s_max) per sample
        s = cfg_s_min + torch.rand(batch_size, 1, 1, 1, device=device) * (cfg_s_max - cfg_s_min)
        return s

    if cfg_config == "D":
        # Random s with warmup — s_max ramps up over training
        if progress < 0.2:
            s_min_t, s_max_t = 1.0, 1.5
        elif progress < 0.5:
            ramp = (progress - 0.2) / 0.3
            s_min_t = 1.0 + 0.5 * ramp       # 1.0 -> 1.5
            s_max_t = 1.5 + 2.5 * ramp        # 1.5 -> 4.0
        else:
            s_min_t = cfg_s_min
            s_max_t = cfg_s_max
        s = s_min_t + torch.rand(batch_size, 1, 1, 1, device=device) * (s_max_t - s_min_t)
        return s

    if cfg_config in ("E", "F"):
        assert guidance_schedule is not None, "GuidanceSchedule required for config E/F"
        assert timesteps is not None, "timesteps required for config E/F"
        t_norm = (timesteps.float() / 1000.0).unsqueeze(-1)  # [B, 1]
        s_raw = guidance_schedule(t_norm)  # [B, 1]

        if cfg_config == "F":
            # Clamp during warmup
            if progress < 0.2:
                s_raw = s_raw.clamp(1.0, 1.5)
            elif progress < 0.5:
                ramp = (progress - 0.2) / 0.3
                s_raw = s_raw.clamp(1.0, 1.5 + 2.5 * ramp)
            # else: unclamped

        return s_raw.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

    raise ValueError(f"Unknown cfg_config: {cfg_config}")


# ──────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────

def compute_cfg_loss(
    pipeline, ip_adapter, images, masks, references,
    batch_types, captions,
    cfg_config: str,
    # Scale params
    cfg_scale: float = 3.0,
    cfg_s_min: float = 1.5,
    cfg_s_max: float = 4.0,
    dropout_prob: float = 0.5,
    progress: float = 0.0,
    guidance_schedule: GuidanceSchedule = None,
    # Standard params
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    binary_cross_attn_mask: bool = False,
    t2i_adapter=None,
    clip_masks=None,
    clip_core_masks=None,
    references_2=None,
    clip_masks_2=None,
    clip_core_masks_2=None,
    group_valid=None,
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    mask_visual: bool = True,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
):
    """Compute CFG-primary loss for configs A-F.

    For configs B-F: packs cond + null into a 2B UNet forward.
    For config A: single UNet forward with dropout.

    Returns:
        loss: Scalar loss tensor with gradient.
        extras: Dict of diagnostic values.
    """
    batch_size = images.shape[0]
    device = images.device
    images = images.float()
    masks_fp32 = masks.float()

    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        # --- CLIP mask ---
        if clip_masks is None:
            clip_dilated = dilate_mask_batch(masks_fp32, clip_dilation_min_r, clip_dilation_max_r)
        else:
            clip_dilated = None

        # Encode image to latent space
        latents = pipeline.encode_image(images)

        # Timestep sampling
        if timestep_sampling == "logit_normal":
            u = torch.sigmoid(logit_normal_mean + logit_normal_std * torch.randn(batch_size, device=device))
            timesteps = (u * 1000).long().clamp(0, 999)
        else:
            timesteps = torch.randint(0, 1000, (batch_size,), device=device, dtype=torch.long)

        noise = torch.randn_like(latents)
        if noise_offset > 0:
            noise += noise_offset * torch.randn(
                latents.shape[0], latents.shape[1], 1, 1, device=device, dtype=latents.dtype
            )
        noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

        # --- UNet masks: maxpool to 64x64, then latent-space band dilation ---
        latent_h = latents.shape[-1]
        kernel = masks_fp32.shape[-1] // latent_h
        core_mask_64 = F.max_pool2d(masks_fp32, kernel_size=kernel)
        core_mask_64 = (core_mask_64 > 0.5).float()

        dilated_binary_64, alpha_map_64, weight_map_64, band_mask_64 = create_latent_band_mask(
            core_mask_64, band_mode, core_ratio=loss_core_ratio,
        )

        # --- Pathway 1: Text conditioning ---
        prompts = []
        for i, atype in enumerate(batch_types):
            cap = captions[i] if i < len(captions) else ""
            if cap:
                prompts.append(cap)
            else:
                type_word = atype.replace("_", " ")
                prompts.append(f"a photo of a {type_word} defect")
        text_emb = pipeline.encode_text(prompts, enable_grad=False)

        # --- Pathway 2: Visual conditioning via IP-Adapter ---
        ref_01 = (references + 1.0) / 2.0
        if clip_masks is not None:
            clip_mask = clip_masks
            clip_core = clip_core_masks
        else:
            clip_mask = clip_dilated if True else None
            clip_core = None
        ip_cond = ip_adapter.encode_image(ref_01, mask=clip_mask, core_mask=clip_core)

        # --- Multi-crop: encode second crop and concatenate ---
        null_token_mask = None
        if references_2 is not None:
            ref_2_01 = (references_2 + 1.0) / 2.0
            clip_mask_2 = clip_masks_2 if clip_masks_2 is not None else None
            clip_core_2 = clip_core_masks_2 if clip_core_masks_2 is not None else None
            ip_cond_2 = ip_adapter.encode_image(ref_2_01, mask=clip_mask_2, core_mask=clip_core_2)

            K = ip_cond.shape[1]
            ip_cond = torch.cat([ip_cond, ip_cond_2], dim=1)  # [B, 2K, dim]

            if group_valid is not None:
                mask_1 = group_valid[:, 0:1].expand(-1, K)
                mask_2 = group_valid[:, 1:2].expand(-1, K)
                null_token_mask = torch.cat([mask_1, mask_2], dim=1)

        # --- Inpainting inputs ---
        mask_latents = dilated_binary_64
        unet_mask_512 = F.interpolate(dilated_binary_64, size=images.shape[-2:], mode='nearest')
        masked_image = images * (1 - unet_mask_512)
        masked_image_latents = pipeline.encode_image(masked_image)
        model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

        # --- T2I-Adapter spatial features ---
        t2i_features = None
        t2i_kwargs = {}
        if t2i_adapter is not None:
            t2i_input = torch.cat([core_mask_64, band_mask_64], dim=1)
            t2i_features = t2i_adapter(t2i_input, mask=dilated_binary_64)
            t2i_kwargs = t2i_adapter.prepare_unet_kwargs(t2i_features)

        # --- Cross-attention mask ---
        cross_attn_mask = dilated_binary_64 if binary_cross_attn_mask else alpha_map_64

        if cfg_config == "A":
            # ============================================================
            # Config A: Standard training + conditioning dropout
            # Single UNet forward, randomly use cond or null IP embeddings
            # No extra diagnostic forwards — CFG metrics are N/A for baseline
            # ============================================================
            ip_null = torch.zeros_like(ip_cond)
            null_mask = torch.rand(batch_size, device=device) < dropout_prob

            ip_embeds = torch.where(
                null_mask[:, None, None].expand_as(ip_cond),
                ip_null, ip_cond,
            )

            cross_attn_kwargs = {"ip_adapter_image_embeds": ip_embeds}
            cross_attn_kwargs["ip_adapter_mask"] = cross_attn_mask
            if null_token_mask is not None:
                cross_attn_kwargs["null_token_mask"] = null_token_mask

            noise_pred = pipeline.unet(
                model_input, timesteps,
                encoder_hidden_states=text_emb,
                cross_attention_kwargs=cross_attn_kwargs,
                **t2i_kwargs,
            ).sample.float()
            torch.cuda.synchronize()

            # Config A: no separate eps_cond/eps_null available without extra forwards
            eps_cond_diag = None
            eps_null_diag = None

        else:
            # ============================================================
            # Configs B-F: 2B packed UNet forward
            # ============================================================
            ip_null = torch.zeros_like(ip_cond)

            model_input_2b = torch.cat([model_input, model_input], dim=0)
            text_emb_2b = torch.cat([text_emb, text_emb], dim=0)
            ip_embeds_2b = torch.cat([ip_cond, ip_null], dim=0)
            alpha_2b = torch.cat([cross_attn_mask, cross_attn_mask], dim=0)
            timesteps_2b = torch.cat([timesteps, timesteps], dim=0)

            cross_attn_kwargs_2b = {"ip_adapter_image_embeds": ip_embeds_2b}
            cross_attn_kwargs_2b["ip_adapter_mask"] = alpha_2b

            if null_token_mask is not None:
                null_token_mask_2b = torch.cat([null_token_mask, null_token_mask], dim=0)
                cross_attn_kwargs_2b["null_token_mask"] = null_token_mask_2b

            # T2I features doubled
            t2i_kwargs_2b = {}
            if t2i_features is not None:
                t2i_features_2b = [torch.cat([f, f], dim=0) for f in t2i_features]
                t2i_kwargs_2b = t2i_adapter.prepare_unet_kwargs(t2i_features_2b)

            eps_all = pipeline.unet(
                model_input_2b, timesteps_2b,
                encoder_hidden_states=text_emb_2b,
                cross_attention_kwargs=cross_attn_kwargs_2b,
                **t2i_kwargs_2b,
            ).sample.float()
            torch.cuda.synchronize()

            eps_cond, eps_null = eps_all.chunk(2, dim=0)

            # For diagnostics (already have both)
            eps_cond_diag = eps_cond.detach()
            eps_null_diag = eps_null.detach()

            # We set noise_pred to None — loss is computed via CFG combination
            noise_pred = None

    # ── Loss computed OUTSIDE autocast (fp32) ──
    weight_expanded = weight_map_64.expand(-1, 4, -1, -1)  # [B, 4, 64, 64]

    if cfg_config == "A":
        # Standard MSE loss
        diff = (noise_pred - noise.float()) ** 2
    else:
        # CFG combination
        s = sample_guidance_scale(
            cfg_config, batch_size, device,
            timesteps=timesteps,
            guidance_schedule=guidance_schedule,
            progress=progress,
            cfg_scale=cfg_scale,
            cfg_s_min=cfg_s_min,
            cfg_s_max=cfg_s_max,
        )
        eps_cfg = (1 - s) * eps_null + s * eps_cond
        diff = (eps_cfg - noise.float()) ** 2

    weighted_diff = diff * weight_expanded
    weight_sum = weight_expanded.sum()
    if weight_sum < 1e-8:
        return torch.tensor(0.0, device=device, requires_grad=True), {}
    loss = weighted_diff.sum() / weight_sum

    # ── Diagnostics (all no grad) ──
    extras = {}
    with torch.no_grad():
        # Core vs band decomposition
        core_expanded = core_mask_64.expand(-1, 4, -1, -1)
        band_expanded = band_mask_64.expand(-1, 4, -1, -1)
        core_n = core_expanded.sum().clamp(min=1)
        band_n = band_expanded.sum().clamp(min=1)
        extras["core_loss"] = (diff * core_expanded).sum().item() / core_n.item()
        extras["band_loss"] = (diff * band_expanded).sum().item() / band_n.item()

        # s stats
        if cfg_config != "A":
            extras["s_mean"] = s.mean().item()
            extras["s_std"] = s.std().item() if batch_size > 1 else 0.0
            extras["s_min"] = s.min().item()
            extras["s_max"] = s.max().item()
        else:
            extras["s_mean"] = extras["s_std"] = extras["s_min"] = extras["s_max"] = 0.0

        # CFG diagnostics: only available when we have both eps_cond and eps_null
        if eps_cond_diag is not None and eps_null_diag is not None:
            eps_null_d = eps_null_diag.float()
            eps_cond_d = eps_cond_diag.float()
            noise_f = noise.float()
            null_diff = (eps_null_d - noise_f) ** 2
            cond_diff = (eps_cond_d - noise_f) ** 2
            extras["L_null"] = (null_diff * weight_expanded).sum().item() / weight_sum.item()
            extras["L_cond"] = (cond_diff * weight_expanded).sum().item() / weight_sum.item()
            extras["delta_norm"] = (eps_cond_d - eps_null_d).pow(2).mean().sqrt().item()

            for s_eval in [1.0, 1.5, 2.0, 3.0, 5.0]:
                eps_cfg_eval = (1 - s_eval) * eps_null_d + s_eval * eps_cond_d
                cfg_diff = (eps_cfg_eval - noise_f) ** 2
                extras[f"L_cfg_s{s_eval}"] = (cfg_diff * weight_expanded).sum().item() / weight_sum.item()
        else:
            # Config A: no CFG diagnostics (would require 2 extra UNet forwards)
            extras["L_null"] = extras["L_cond"] = extras["delta_norm"] = 0.0
            for s_eval in [1.0, 1.5, 2.0, 3.0, 5.0]:
                extras[f"L_cfg_s{s_eval}"] = 0.0

    return loss, extras


# ──────────────────────────────────────────────────────────────────────
# Loss plot (static, saved to file)
# ──────────────────────────────────────────────────────────────────────

def _ema_smooth(values, smoothing: float = 0.99):
    """Exponential moving average."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def save_loss_plot(losses: list, stats_file: Path, save_path: Path):
    """Render 2x3 CFG training diagnostics plot."""
    import matplotlib.pyplot as plt

    if len(losses) < 2:
        return

    # Parse stats CSV
    stats = _parse_stats_csv(stats_file)
    if not stats:
        return

    SM = 0.99
    SF = 0.6
    steps = np.arange(len(losses))
    arr = np.array(losses)
    ema_fast = _ema_smooth(arr, SF)
    ema_slow = _ema_smooth(arr, SM)

    s_steps = stats["step"]
    has_data = len(s_steps) > 0

    fig, axes = plt.subplots(2, 3, figsize=(21, 10))

    # [0,0] Loss trend (EMA)
    ax = axes[0, 0]
    ax.plot(steps, ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
    ax.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"Smooth Trend — step {len(losses)}, trend: {ema_slow[-1]:.4f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # [0,1] L_cfg vs s (THE diagnostic)
    ax = axes[0, 1]
    if has_data:
        cfg_colors = {
            "1.0": "#2196F3", "1.5": "#4CAF50", "2.0": "#FF9800",
            "3.0": "#F44336", "5.0": "#9C27B0",
        }
        for s_val in ["1.0", "1.5", "2.0", "3.0", "5.0"]:
            key = f"L_cfg_s{s_val}"
            if key in stats and len(stats[key]) > 1:
                ema_val = _ema_smooth(stats[key], SM)
                ax.plot(s_steps, ema_val, color=cfg_colors[s_val], linewidth=2,
                        label=f"s={s_val} ({ema_val[-1]:.4f})")
        # L_null and L_cond as dashed
        if "L_null" in stats and len(stats["L_null"]) > 1:
            ema_null = _ema_smooth(stats["L_null"], SM)
            ax.plot(s_steps, ema_null, color="black", linewidth=1.5, linestyle="--",
                    label=f"L_null ({ema_null[-1]:.4f})")
        if "L_cond" in stats and len(stats["L_cond"]) > 1:
            ema_cond = _ema_smooth(stats["L_cond"], SM)
            ax.plot(s_steps, ema_cond, color="gray", linewidth=1.5, linestyle="--",
                    label=f"L_cond ({ema_cond[-1]:.4f})")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("L_cfg vs s — want curves to flatten")
    else:
        ax.set_title("L_cfg vs s (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # [0,2] Grad Norm
    ax = axes[0, 2]
    if has_data and "grad_norm" in stats and len(stats["grad_norm"]) > 1:
        gn = np.array(stats["grad_norm"])
        gn_ema_fast = _ema_smooth(gn, SF)
        gn_ema_slow = _ema_smooth(gn, SM)
        ax.plot(s_steps, gn_ema_fast, color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
        ax.plot(s_steps, gn_ema_slow, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax.axhline(y=1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="clip=1.0")
        ax.set_title(f"Grad Norm — trend: {gn_ema_slow[-1]:.4f}")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_title("Grad Norm (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.grid(True, alpha=0.3)

    # [1,0] Delta norm
    ax = axes[1, 0]
    if has_data and "delta_norm" in stats and len(stats["delta_norm"]) > 1:
        dn = np.array(stats["delta_norm"])
        ema_dn = _ema_smooth(dn, SM)
        ax.plot(s_steps, _ema_smooth(dn, SF), color="red", linewidth=0.8, alpha=0.3, label=f"EMA ({SF})")
        ax.plot(s_steps, ema_dn, color="darkblue", linewidth=2.5, label=f"EMA ({SM})")
        ax.set_title(f"Delta Norm — trend: {ema_dn[-1]:.4f}")
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.set_title("Delta Norm (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("||eps_cond - eps_null||")
    ax.grid(True, alpha=0.3)

    # [1,1] Core/Band decomposition
    ax = axes[1, 1]
    if has_data and "core_loss" in stats and len(stats["core_loss"]) > 1:
        s_core = np.array(stats["core_loss"])
        s_band = np.array(stats["band_loss"])
        if s_core.any():
            w_core_s = 0.8 * s_core
            w_band_s = 0.2 * s_band
            w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
            core_share = w_core_s / w_total_s
            core_share_ps = np.interp(steps, s_steps, core_share)
            core_ps = arr * core_share_ps / 0.8
            band_ps = arr * (1.0 - core_share_ps) / 0.2
            ce = _ema_smooth(core_ps, SM)
            be = _ema_smooth(band_ps, SM)
            ax.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
            ax.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
            ax.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
            ax.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
            ax.plot(steps, ema_slow, color="black", linewidth=2, linestyle="--",
                    label=f"0.8\u00d7core+0.2\u00d7band = {ema_slow[-1]:.4f}")
            ax.annotate(f"{ce[-1]:.3f}", xy=(steps[-1], ce[-1] / 2),
                        fontsize=9, fontweight="bold", color="#1565C0", ha="right")
            ax.annotate(f"{be[-1]:.3f}", xy=(steps[-1], ce[-1] + be[-1] / 2),
                        fontsize=9, fontweight="bold", color="#E65100", ha="right")
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(f"Core vs Band, EMA({SM}) — weighted: {ema_slow[-1]:.4f}")
        else:
            ax.set_title("Core vs Band (no data)")
    else:
        ax.set_title("Core vs Band (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE")
    ax.grid(True, alpha=0.3)

    # [1,2] Gates or s stats
    ax = axes[1, 2]
    # Auto-detect: if s_mean varies, show s stats; else show gates
    if has_data and "s_mean" in stats and len(stats["s_mean"]) > 1:
        s_mean = np.array(stats["s_mean"])
        if s_mean.max() > 0:
            s_min_arr = np.array(stats["s_min"])
            s_max_arr = np.array(stats["s_max"])
            ax.plot(s_steps, s_mean, color="blue", linewidth=2, label=f"s_mean ({s_mean[-1]:.2f})")
            ax.fill_between(s_steps, s_min_arr, s_max_arr, color="blue", alpha=0.15, label="s range")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_title(f"Guidance Scale — mean={s_mean[-1]:.2f}")
            ax.set_xlabel("Step"); ax.set_ylabel("s")
            ax.grid(True, alpha=0.3)
        else:
            _plot_gates(ax, stats, s_steps)
    elif has_data:
        _plot_gates(ax, stats, s_steps)
    else:
        ax.set_title("Gates / s stats (no data yet)")
        ax.grid(True, alpha=0.3)

    # Run title from config
    config_path = stats_file.parent / "run_config.json"
    try:
        with open(config_path) as cf:
            cfg = json.load(cf)
        parts = [
            f"CFG-{cfg.get('cfg_config', '?')}",
            f"BS={cfg.get('batch_size', '?')}",
            f"lr={cfg.get('lr_pretrained', '?')}",
        ]
        cc = cfg.get("cfg_config", "")
        if cc == "B":
            parts.append(f"s={cfg.get('cfg_scale', '?')}")
        elif cc in ("C", "D"):
            parts.append(f"s=[{cfg.get('cfg_s_min', '?')},{cfg.get('cfg_s_max', '?')}]")
        elif cc in ("E", "F"):
            parts.append("learned_s(t)")
        if cc == "A":
            parts.append(f"dropout={cfg.get('dropout_prob', 0.5)}")
        title = " | ".join(str(p) for p in parts)
        fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_gates(ax, stats, s_steps):
    """Plot gate values on the given axis."""
    if "attn_gate" in stats and len(stats["attn_gate"]) > 1:
        ax.plot(s_steps, np.array(stats["attn_gate"]), color="blue", linewidth=1.5, label="Attn gate")
        ax.plot(s_steps, np.array(stats["ff_gate"]), color="green", linewidth=1.5, label="FF gate")
        ax.set_title(f"Gates — attn={stats['attn_gate'][-1]:.4f}, ff={stats['ff_gate'][-1]:.4f}")
        ax.legend(loc="upper left", fontsize=8)
    else:
        ax.set_title("Gates (no data)")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate value")
    ax.grid(True, alpha=0.3)


def _parse_stats_csv(stats_file: Path) -> dict:
    """Parse CFG stats CSV into dict of lists."""
    result = {}
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            cols = header.split(",")
            for col in cols:
                result[col] = []
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != len(cols):
                    continue
                for i, col in enumerate(cols):
                    try:
                        result[col].append(float(parts[i]))
                    except ValueError:
                        result[col].append(0.0)
        # Convert to numpy
        for col in result:
            result[col] = np.array(result[col])
    except FileNotFoundError:
        pass
    return result


# ──────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────

def save_cfg_checkpoint(
    ip_adapter, optimizer, save_dir,
    band_mode: int = 2, t2i_adapter=None,
    guidance_schedule: GuidanceSchedule = None,
    step: int = 0, losses: list = None,
):
    """Save IP-Adapter + T2I-Adapter + GuidanceSchedule checkpoint."""
    save_dir = Path(save_dir)
    ip_adapter.save_ip_adapter(save_dir)

    state = {
        "optimizer": optimizer.state_dict(),
        "band_mode": band_mode,
        "step": step,
        "losses": losses or [],
    }
    if t2i_adapter is not None:
        state["t2i_adapter"] = t2i_adapter.state_dict()
    if guidance_schedule is not None:
        state["guidance_schedule"] = guidance_schedule.state_dict()
    torch.save(state, save_dir / "training_state.pt")


# ──────────────────────────────────────────────────────────────────────
# Sample generation
# ──────────────────────────────────────────────────────────────────────

def generate_cfg_samples(
    pipeline, ip_adapter, dataset,
    save_path, device,
    band_mode: int = 2,
    t2i_adapter=None,
    guidance_schedule: GuidanceSchedule = None,
    cfg_config: str = "C",
):
    """Generate 7-row sample grid for CFG training.

    Rows:
        0: Real image
        1: Mask
        2: Reference CLIP crop
        3: Gen s=0.0 (pure null — unconditional baseline)
        4: Gen s=1.0 (no guidance amplification)
        5: Gen s=3.0 (fixed high guidance)
        6: Gen s=learned_mean (E/F) or s=2.0 (A-D)
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)

    # --- Sample selection (same fixed samples as train_anomagic.py) ---
    sample_rng = random.Random(42)
    _fixed_samples = [
        ("cut_lead", "AnomVerse_data_filtered/mvtec/mvtec/transistor/test/cut_lead/005.png"),
        ("broken_large", "AnomVerse_data_filtered/mvtec/mvtec/bottle/test/broken_large/011.png"),
        ("fold", "AnomVerse_data_filtered/mvtec/mvtec/leather/test/fold/016.png"),
        ("cut", "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/cut/011.png"),
        ("thread", "AnomVerse_data_filtered/mvtec/mvtec/carpet/test/thread/011.png"),
        ("faulty_imprint", "AnomVerse_data_filtered/mvtec/mvtec/pill/test/faulty_imprint/011.png"),
        ("contamination", "realiad_1024/mint/NG/ZW/S0070/mint_0070_NG_ZW_C5_20230910095530.jpg"),
    ]
    _path_to_idx = {}
    for i, s in enumerate(dataset.samples):
        p = s["image_path"].replace("\\", "/")
        _path_to_idx[p] = i
        for prefix in ("AnomVerse_data_filtered/", "realiad_1024/"):
            idx = p.find(prefix)
            if idx >= 0:
                _path_to_idx[p[idx:]] = i
                break
    types_to_show = []
    fixed_indices = {}
    for atype, img_path in _fixed_samples:
        if atype in dataset.type_to_samples and img_path in _path_to_idx:
            types_to_show.append(atype)
            fixed_indices[atype] = _path_to_idx[img_path]
    if len(types_to_show) < 6:
        remaining = [t for t in dataset.anomaly_types if t not in types_to_show]
        for t in sample_rng.sample(remaining, min(6 - len(types_to_show), len(remaining))):
            types_to_show.append(t)

    n_cols = len(types_to_show)

    # --- Collect sample data ---
    all_samples = []
    for i, atype in enumerate(types_to_show):
        if atype in fixed_indices:
            idx_val = fixed_indices[atype]
        else:
            indices = dataset.type_to_samples[atype]
            idx_val = sample_rng.choice(indices)
        sample = dataset[idx_val]

        image = sample["image"].unsqueeze(0).to(device)
        mask_t = sample["mask"].unsqueeze(0).to(device)
        reference = sample["reference"].unsqueeze(0).to(device)

        img_path = sample.get("image_path", "unknown")
        path_parts = Path(img_path).parts
        short_path = "/".join(path_parts[-3:]) if len(path_parts) >= 3 else img_path

        img_np = ((image[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        mask_np = mask_t[0, 0].cpu().numpy()
        ref_np = ((reference[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

        all_samples.append({
            "atype": atype, "image": image, "mask": mask_t, "reference": reference,
            "caption": sample.get("caption", ""),
            "img_np": img_np, "mask_np": mask_np, "ref_np": ref_np,
            "short_path": short_path,
        })

    # --- Determine s for row 6 ---
    if cfg_config in ("E", "F") and guidance_schedule is not None:
        # Use mean of learned schedule evaluated at t=0.5
        with torch.no_grad():
            t_probe = torch.tensor([[0.5]], device=device)
            s_learned = guidance_schedule(t_probe).item()
        row6_scale = s_learned
        row6_label = f"Gen (s=learned {row6_scale:.1f})"
    else:
        row6_scale = 2.0
        row6_label = "Gen (s=2.0)"

    # --- Generate ---
    gen_kwargs = dict(
        num_steps=30, noise_strength=1.0,
        reference_mode=dataset.reference_mode,
        band_mode=band_mode, t2i_adapter=t2i_adapter,
    )

    for i, s in enumerate(all_samples):
        with torch.no_grad():
            # s=0 (pure null): guidance_scale=0 doesn't work with standard generate.
            # Use guidance_scale=1.0 but zero out IP embeds via cfg_mode trick.
            # Actually, generate with guidance_scale=1 and cfg_mode to get s=1 behavior.
            # For "pure null" we pass zeros for reference. Create a zero ref.
            s["gen_s0"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=1.0, cfg_mode="visual", seed=42 + i,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

            s["gen_s1"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=1.0, cfg_mode="both", seed=42 + i,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

            s["gen_s3"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=3.0, cfg_mode="visual", seed=42 + i,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

            s["gen_s_row6"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=row6_scale, cfg_mode="visual", seed=42 + i,
                **gen_kwargs,
            )
            torch.cuda.synchronize()

    def _to_np(t):
        return ((t[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

    # --- Plot 7-row grid ---
    fig, axes = plt.subplots(7, n_cols, figsize=(3.5 * n_cols, 22), squeeze=False)
    ref_label = "clip_crop" if dataset.augment else dataset.reference_mode

    for i, s in enumerate(all_samples):
        axes[0, i].imshow(s["img_np"])
        axes[0, i].set_title(f"Real ({s['atype']})\n{s['short_path']}", fontsize=6)
        axes[0, i].axis("off")

        axes[1, i].imshow(s["mask_np"], cmap="gray")
        axes[1, i].set_title(f"Mask ({s['mask_np'].mean() * 100:.1f}%)", fontsize=7)
        axes[1, i].axis("off")

        axes[2, i].imshow(s["ref_np"])
        axes[2, i].set_title(f"Reference ({ref_label})", fontsize=7)
        axes[2, i].axis("off")

        axes[3, i].imshow(_to_np(s["gen_s0"]))
        axes[3, i].set_title("Gen (CFG=1, visual uncond)", fontsize=7)
        axes[3, i].axis("off")

        axes[4, i].imshow(_to_np(s["gen_s1"]))
        axes[4, i].set_title("Gen (s=1.0, no guidance)", fontsize=7)
        axes[4, i].axis("off")

        axes[5, i].imshow(_to_np(s["gen_s3"]))
        axes[5, i].set_title("Gen (s=3.0)", fontsize=7)
        axes[5, i].axis("off")

        axes[6, i].imshow(_to_np(s["gen_s_row6"]))
        axes[6, i].set_title(row6_label, fontsize=7)
        axes[6, i].axis("off")

    plt.suptitle("CFG-Primary Training Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Live loss viewer launcher
# ──────────────────────────────────────────────────────────────────────

def _launch_live_loss(loss_file: Path):
    """Launch live loss viewer as a detached subprocess."""
    import subprocess
    script = Path(__file__).parent / "live_loss_cfg.py"
    loss_file.write_text("")
    try:
        subprocess.Popen(
            [sys.executable, str(script), str(loss_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  Live loss viewer launched (reading {loss_file})")
    except Exception as e:
        print(f"  Warning: could not launch live loss viewer: {e}")


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ──────────────────────────────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────────────────────────────

def train_cfg(
    splits_dir: Path,
    save_dir: Path,
    captions_file: Path,
    cfg_config: str = "C",
    cfg_scale: float = 3.0,
    cfg_s_min: float = 1.5,
    cfg_s_max: float = 4.0,
    dropout_prob: float = 0.5,
    n_steps: int = 50000,
    batch_size: int = 6,
    lr: float = 1e-4,
    lr_pretrained: float = 1e-4,
    lambda_sp: float = 0.0,
    device: str = "cuda",
    save_every: int = 2000,
    anomaly_types: list = None,
    exclude_sources: list = None,
    data_root: Path = None,
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 16,
    ip_adapter_scale: float = 1.0,
    mask_visual: bool = True,
    clip_dilation_min_r: int = 2,
    clip_dilation_max_r: int = 10,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    binary_cross_attn_mask: bool = False,
    t2i_adapter_mode: str = "cascade",
    augment: bool = True,
    resume_dir: Path = None,
    visual_mode: int = 3,
    learnable_gates: bool = True,
    force_gates: bool = False,
    sa_num_layers: int = 3,
    sa_num_heads: int = 12,
    multi_crop: bool = True,
    no_live_viewer: bool = False,
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    clip_align: bool = True,
):
    """Train IP-Adapter with CFG-primary objective."""
    # Seed everything
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Effective UNet batch size for configs B-F
    unet_bs = batch_size * 2 if cfg_config != "A" else batch_size

    print("=" * 70)
    print("CFG-PRIMARY TRAINING")
    print("=" * 70)
    print(f"Config: {cfg_config}")
    if cfg_config == "A":
        print(f"  Baseline: standard training + {dropout_prob:.0%} dropout")
    elif cfg_config == "B":
        print(f"  Fixed s = {cfg_scale}")
    elif cfg_config in ("C", "D"):
        print(f"  Random s ~ U({cfg_s_min}, {cfg_s_max})")
        if cfg_config == "D":
            print(f"  Warmup: s_max ramps up over first 50% of training")
    elif cfg_config in ("E", "F"):
        print(f"  Learned s(t) via GuidanceSchedule MLP")
        if cfg_config == "F":
            print(f"  Warmup: s clamped during early training")
    print(f"Batch size: {batch_size} (UNet sees {unet_bs})")
    print(f"Learning rate (scratch): {lr}")
    print(f"Learning rate (pretrained): {lr_pretrained}")
    print(f"L2-SP lambda: {lambda_sp}")
    print(f"Training steps: {n_steps}")
    print(f"IP-Adapter: {ip_adapter_type} (K={ip_adapter_k})")
    print(f"Band mode: {band_mode}")
    print(f"Loss ratio: {loss_core_ratio:.0%} core / {1-loss_core_ratio:.0%} band")
    print(f"Augmentation: {augment}")
    print(f"Visual mode: {visual_mode}")
    print(f"Gates: learnable={learnable_gates}, force={force_gates}")
    print(f"SA layers: {sa_num_layers}, heads: {sa_num_heads}")
    print(f"Multi-crop: {multi_crop}")
    print(f"CLIP-UNet alignment: {clip_align}")
    print(f"Noise offset: {noise_offset}")
    print(f"Timestep sampling: {timestep_sampling}")
    if resume_dir:
        print(f"Resume from: {resume_dir}")
    print()

    # =========================================
    # Load Data
    # =========================================
    dataset = AnomalyDataset(
        splits_dir,
        anomaly_types=anomaly_types,
        exclude_sources=exclude_sources,
        data_root=data_root,
        captions_file=captions_file,
        return_reference=True,
        reference_mode="full",
        reference_crop_size=224,
        augment=augment,
        multi_crop=multi_crop,
        band_mode=band_mode,
        clip_align=clip_align,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    n_captions = sum(1 for s in dataset.samples if s.get("caption"))
    print(f"Samples with captions: {n_captions}/{len(dataset.samples)}")

    # =========================================
    # Initialize Pipeline
    # =========================================
    print("\nInitializing models...")

    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    # =========================================
    # Initialize IP-Adapter
    # =========================================
    print(f"\nLoading IP-Adapter ({ip_adapter_type}, K={ip_adapter_k}, visual_mode={visual_mode})...")
    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=ip_adapter_type,
        num_tokens=ip_adapter_k,
        scale=ip_adapter_scale,
        load_pretrained=True,
        mask_visual=mask_visual,
        visual_mode=visual_mode,
        learnable_gates=learnable_gates,
        force_gates=force_gates,
        sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )
    ip_adapter.freeze_image_encoder()

    # =========================================
    # Initialize T2I-Adapter
    # =========================================
    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(in_channels=2, injection_mode=t2i_adapter_mode).to(device)
        n_t2i = sum(p.numel() for p in t2i_adapter.parameters())
        print(f"\nT2I-Adapter ({t2i_adapter_mode}): {n_t2i:,} params")

    # =========================================
    # Initialize GuidanceSchedule (configs E, F)
    # =========================================
    guidance_schedule = None
    if cfg_config in ("E", "F"):
        guidance_schedule = GuidanceSchedule().to(device)
        n_gs = sum(p.numel() for p in guidance_schedule.parameters())
        print(f"\nGuidanceSchedule: {n_gs:,} params")

    # Ensure all trainable modules are fp32
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # =========================================
    # Build Optimizer Parameter Groups
    # =========================================
    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)
    if guidance_schedule is not None:
        group_b_modules.append(guidance_schedule)

    norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)

    # Group A: pretrained IP-Adapter
    a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

    # Group C: gate/scale scalars
    group_c_params = []
    if hasattr(ip_adapter, 'masked_self_attn') and ip_adapter.masked_self_attn.learnable_gates:
        for layer in ip_adapter.masked_self_attn.layers:
            group_c_params.append(layer["attn_gate"].gate)
            if "ff_gate" in layer:
                group_c_params.append(layer["ff_gate"].gate)
    if t2i_adapter is not None:
        for s in t2i_adapter.scales:
            group_c_params.append(s)
    group_c_ids = {id(p) for p in group_c_params}

    # Group B: from-scratch modules (minus Group C params)
    b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
    b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
    b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

    param_groups = [
        {"params": a_decay,     "lr": lr_pretrained, "weight_decay": 0.0, "label": "A_decay"},
        {"params": a_no_decay,  "lr": lr_pretrained, "weight_decay": 0.0, "label": "A_no_decay"},
        {"params": b_decay,     "lr": lr,            "weight_decay": 1e-4, "label": "B_decay"},
        {"params": b_no_decay,  "lr": lr,            "weight_decay": 0.0, "label": "B_no_decay"},
        {"params": group_c_params, "lr": lr,         "weight_decay": 0.0, "label": "C_gates"},
    ]

    # Verify no duplicates
    all_group_ids = []
    for pg in param_groups:
        for p in pg["params"]:
            pid = id(p)
            assert pid not in all_group_ids, f"Duplicate param in group {pg['label']}"
            all_group_ids.append(pid)

    # Verify total param count
    group_numel = sum(p.numel() for pg in param_groups for p in pg["params"])
    expected_params = ip_adapter.get_trainable_parameters()
    if t2i_adapter is not None:
        expected_params.extend(list(t2i_adapter.parameters()))
    if guidance_schedule is not None:
        expected_params.extend(list(guidance_schedule.parameters()))
    expected_numel = sum(p.numel() for p in expected_params)
    assert group_numel == expected_numel, (
        f"Param count mismatch: groups={group_numel:,} vs expected={expected_numel:,}"
    )

    trainable_params = [p for pg in param_groups for p in pg["params"]]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  Trainable parameters: {n_trainable:,}")
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"    {pg['label']:12s}: {n:>10,} params  (lr={pg['lr']:.1e}, wd={pg['weight_decay']:.1e})")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # L2-SP regularizer
    l2sp_reg = None
    if lambda_sp > 0 and len(a_decay) > 0:
        l2sp_reg = L2SPRegularizer(a_decay, lambda_sp=lambda_sp)
        print(f"\n  L2-SP: {l2sp_reg.total_elements:,} elements, lambda={lambda_sp:.1e}")

    # AMP
    use_amp = True
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # =========================================
    # Resume from checkpoint
    # =========================================
    start_step = 0
    resumed_losses = []

    if resume_dir is not None:
        resume_dir = Path(resume_dir)
        print(f"\nResuming from checkpoint: {resume_dir}")

        ip_ckpt = resume_dir / "ip_adapter.pt"
        if ip_ckpt.exists():
            ip_adapter.load_finetuned(resume_dir)
            ip_adapter.masked_self_attn.float()
            ip_adapter.image_projection.float()
            for proc in ip_adapter.attn_processors.values():
                proc.float()
            print(f"  Loaded IP-Adapter weights from {ip_ckpt}")
        else:
            print(f"  WARNING: {ip_ckpt} not found, starting with pretrained weights")

        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)

            try:
                optimizer.load_state_dict(state["optimizer"])
                print(f"  Loaded optimizer state")
            except Exception as e:
                print(f"  WARNING: Could not load optimizer state ({e}), using fresh optimizer")

            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")

            resumed_losses = state.get("losses", [])
            if resumed_losses:
                print(f"  Loaded {len(resumed_losses)} previous loss values")

            if t2i_adapter is not None and "t2i_adapter" in state:
                t2i_adapter.load_state_dict(state["t2i_adapter"])
                print(f"  Loaded T2I-Adapter weights")

            if guidance_schedule is not None and "guidance_schedule" in state:
                guidance_schedule.load_state_dict(state["guidance_schedule"])
                print(f"  Loaded GuidanceSchedule weights")

            del state
            torch.cuda.empty_cache()
        else:
            print(f"  WARNING: {state_path} not found, starting fresh")

    remaining_steps = n_steps - start_step
    if remaining_steps <= 0:
        print(f"\nAlready completed {start_step}/{n_steps} steps. Nothing to do.")
        return

    # =========================================
    # Step-0 baseline samples
    # =========================================
    if start_step == 0:
        print("\nGenerating step-0 baseline samples...")
        torch.cuda.empty_cache()
        try:
            generate_cfg_samples(
                pipeline, ip_adapter, dataset,
                save_dir / "samples_0_pretrained.png", device,
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                guidance_schedule=guidance_schedule,
                cfg_config=cfg_config,
            )
            print(f"  Saved to {save_dir / 'samples_0_pretrained.png'}")
        except RuntimeError as e:
            print(f"  Warning: baseline sample generation failed ({e})")
        torch.cuda.empty_cache()

    # =========================================
    # Training Loop
    # =========================================
    print(f"\nStarting training for {remaining_steps} remaining steps ({start_step}/{n_steps} done)...")

    # Save run config
    config_file = save_dir / "run_config.json"
    run_config = {
        "cfg_config": cfg_config,
        "cfg_scale": cfg_scale,
        "cfg_s_min": cfg_s_min,
        "cfg_s_max": cfg_s_max,
        "dropout_prob": dropout_prob,
        "visual_mode": visual_mode,
        "batch_size": batch_size,
        "unet_batch_size": unet_bs,
        "lr_pretrained": lr_pretrained,
        "lr": lr,
        "band_mode": band_mode,
        "noise_offset": noise_offset,
        "timestep_sampling": timestep_sampling,
        "sa_num_layers": sa_num_layers,
        "sa_num_heads": sa_num_heads,
        "learnable_gates": learnable_gates,
        "force_gates": force_gates,
        "multi_crop": multi_crop,
        "ip_adapter_type": ip_adapter_type,
        "ip_adapter_k": ip_adapter_k,
        "clip_align": clip_align,
        "lambda_sp": lambda_sp,
    }
    with open(config_file, "w") as cf:
        json.dump(run_config, cf, indent=2)

    # Launch live loss viewer
    loss_file = save_dir / "losses.txt"
    if resumed_losses:
        with open(loss_file, "w") as lf:
            lf.writelines(f"{v}\n" for v in resumed_losses)
    if not no_live_viewer:
        _launch_live_loss(loss_file)

    losses = resumed_losses
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=False, persistent_workers=True,
        drop_last=True, worker_init_fn=_worker_init_fn,
        generator=g,
    )
    data_iter = iter(infinite_loader(loader))

    loss_fh = open(loss_file, "a")

    # Stats CSV: 23 columns
    stats_file = save_dir / "stats.csv"
    stats_fh = open(stats_file, "a")
    if stats_fh.tell() == 0:
        stats_fh.write(
            "step,loss,L_null,L_cond,delta_norm,"
            "L_cfg_s1.0,L_cfg_s1.5,L_cfg_s2.0,L_cfg_s3.0,L_cfg_s5.0,"
            "s_mean,s_std,s_min,s_max,"
            "core_loss,band_loss,grad_norm,"
            "attn_gate,ff_gate,lr_pretrained,lr_scratch,l2sp,progress\n"
        )

    pbar = tqdm(range(start_step, n_steps), desc="Training", initial=start_step, total=n_steps)
    skipped_nan = 0
    prev_ckpt_dir = None

    for step in pbar:
        _t0 = time.time()
        batch = next(data_iter)
        _t_data = time.time()

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        references = batch["reference"].to(device)
        batch_types = batch["anomaly_type"]
        batch_captions = batch.get("caption", [""] * batch_size)
        clip_masks = batch["clip_mask"].to(device) if "clip_mask" in batch else None
        clip_core_masks = batch["clip_core_mask"].to(device) if "clip_core_mask" in batch else None
        references_2 = batch["reference_2"].to(device) if "reference_2" in batch else None
        clip_masks_2 = batch["clip_mask_2"].to(device) if "clip_mask_2" in batch else None
        clip_core_masks_2 = batch["clip_core_mask_2"].to(device) if "clip_core_mask_2" in batch else None
        group_valid = batch["group_valid"].to(device) if "group_valid" in batch else None
        _t_transfer = time.time()

        optimizer.zero_grad()

        progress = step / max(n_steps - 1, 1)

        loss_diff, loss_extras = compute_cfg_loss(
            pipeline, ip_adapter, images, masks, references,
            batch_types, batch_captions,
            cfg_config=cfg_config,
            cfg_scale=cfg_scale,
            cfg_s_min=cfg_s_min,
            cfg_s_max=cfg_s_max,
            dropout_prob=dropout_prob,
            progress=progress,
            guidance_schedule=guidance_schedule,
            clip_dilation_min_r=clip_dilation_min_r,
            clip_dilation_max_r=clip_dilation_max_r,
            band_mode=band_mode,
            loss_core_ratio=loss_core_ratio,
            binary_cross_attn_mask=binary_cross_attn_mask,
            t2i_adapter=t2i_adapter,
            clip_masks=clip_masks,
            clip_core_masks=clip_core_masks,
            references_2=references_2,
            clip_masks_2=clip_masks_2,
            clip_core_masks_2=clip_core_masks_2,
            group_valid=group_valid,
            noise_offset=noise_offset,
            timestep_sampling=timestep_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            mask_visual=mask_visual,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
        )
        _t_fwd = time.time()

        # L2-SP regularization
        l2sp_val = 0.0
        if l2sp_reg is not None:
            l2sp_loss = l2sp_reg.compute()
            l2sp_val = l2sp_loss.item()
            loss = loss_diff + l2sp_loss
        else:
            loss = loss_diff

        if torch.isnan(loss) or torch.isinf(loss):
            skipped_nan += 1
            del loss_diff, loss, loss_extras
            gc.collect()
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss).backward()
        torch.cuda.synchronize()
        _t_bwd = time.time()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        scaler.step(optimizer)
        scaler.update()
        _t_opt = time.time()

        loss_val = loss_diff.item()
        losses.append(loss_val)

        loss_fh.write(f"{loss_val}\n")
        loss_fh.flush()

        # Log stats
        attn_gate = ff_gate = 0.0
        if hasattr(ip_adapter, 'masked_self_attn'):
            sa = ip_adapter.masked_self_attn
            if sa.learnable_gates:
                for layer in sa.layers:
                    attn_gate = layer["attn_gate"].gate.item()
                    if "ff_gate" in layer:
                        ff_gate = layer["ff_gate"].gate.item()
                    break
            else:
                attn_gate = ff_gate = 1.0
        lr_pre = optimizer.param_groups[0]["lr"]
        lr_scr = optimizer.param_groups[2]["lr"]

        stats_fh.write(
            f"{step},{loss_val},"
            f"{loss_extras.get('L_null', 0.0)},{loss_extras.get('L_cond', 0.0)},"
            f"{loss_extras.get('delta_norm', 0.0)},"
            f"{loss_extras.get('L_cfg_s1.0', 0.0)},{loss_extras.get('L_cfg_s1.5', 0.0)},"
            f"{loss_extras.get('L_cfg_s2.0', 0.0)},{loss_extras.get('L_cfg_s3.0', 0.0)},"
            f"{loss_extras.get('L_cfg_s5.0', 0.0)},"
            f"{loss_extras.get('s_mean', 0.0)},{loss_extras.get('s_std', 0.0)},"
            f"{loss_extras.get('s_min', 0.0)},{loss_extras.get('s_max', 0.0)},"
            f"{loss_extras.get('core_loss', 0.0)},{loss_extras.get('band_loss', 0.0)},"
            f"{grad_norm},"
            f"{attn_gate},{ff_gate},{lr_pre},{lr_scr},{l2sp_val},{progress}\n"
        )
        stats_fh.flush()

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(loss=f"{avg:.4f}")

        # Timing diagnostics (first 20 steps)
        if step - start_step < 20:
            _t_end = time.time()
            print(f"  [TIMING step {step}] data={(_t_data-_t0)*1000:.0f}ms "
                  f"transfer={(_t_transfer-_t_data)*1000:.0f}ms "
                  f"fwd={(_t_fwd-_t_transfer)*1000:.0f}ms "
                  f"bwd={(_t_bwd-_t_fwd)*1000:.0f}ms "
                  f"opt={(_t_opt-_t_bwd)*1000:.0f}ms "
                  f"rest={(_t_end-_t_opt)*1000:.0f}ms "
                  f"TOTAL={(_t_end-_t0)*1000:.0f}ms", flush=True)

        # Free CUDA tensors
        del images, masks, references, clip_masks, clip_core_masks
        del references_2, clip_masks_2, clip_core_masks_2, group_valid
        del loss, loss_diff, loss_extras

        # Save learned s(t) curve periodically (E, F only)
        if cfg_config in ("E", "F") and guidance_schedule is not None and (step + 1) % 100 == 0:
            _save_learned_s_curve(guidance_schedule, save_dir, step + 1, device)

        # Early sample snapshots
        if (step + 1) in (500, 1000, 2000) and (step + 1) % save_every != 0:
            torch.cuda.empty_cache()
            try:
                generate_cfg_samples(
                    pipeline, ip_adapter, dataset,
                    save_dir / f"samples_{step + 1}.png", device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
                    guidance_schedule=guidance_schedule,
                    cfg_config=cfg_config,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            torch.cuda.empty_cache()
            save_loss_plot(losses, stats_file, save_dir / f"loss_{step + 1}.png")

        # Checkpoints + samples
        if (step + 1) % save_every == 0:
            print(f"\n  Checkpoint at step {step + 1}...", flush=True)
            ckpt_dir = save_dir / f"checkpoint_{step + 1}"
            save_cfg_checkpoint(
                ip_adapter, optimizer, ckpt_dir,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                guidance_schedule=guidance_schedule,
                step=step + 1,
                losses=losses,
            )
            # Delete previous checkpoint (keep only latest + final)
            if prev_ckpt_dir is not None and prev_ckpt_dir.exists():
                import shutil
                shutil.rmtree(prev_ckpt_dir)
            prev_ckpt_dir = ckpt_dir
            torch.cuda.empty_cache()
            try:
                generate_cfg_samples(
                    pipeline, ip_adapter, dataset,
                    save_dir / f"samples_{step + 1}.png", device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
                    guidance_schedule=guidance_schedule,
                    cfg_config=cfg_config,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            torch.cuda.empty_cache()
            save_loss_plot(losses, stats_file, save_dir / f"loss_{step + 1}.png")

    loss_fh.close()
    stats_fh.close()

    # =========================================
    # Final Outputs
    # =========================================
    print("\nTraining complete!")
    print(f"  Total steps: {n_steps} (resumed from {start_step})" if start_step > 0 else f"  Total steps: {n_steps}")
    print(f"  Skipped (NaN/Inf): {skipped_nan}")
    print(f"  Valid losses recorded: {len(losses)}")

    save_loss_plot(losses, stats_file, save_dir / "training_loss.png")

    save_cfg_checkpoint(
        ip_adapter, optimizer,
        save_dir / "checkpoint_final",
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
        guidance_schedule=guidance_schedule,
        step=n_steps,
        losses=losses,
    )

    torch.cuda.empty_cache()
    try:
        generate_cfg_samples(
            pipeline, ip_adapter, dataset,
            save_dir / "final_samples.png", device,
            band_mode=band_mode, t2i_adapter=t2i_adapter,
            guidance_schedule=guidance_schedule,
            cfg_config=cfg_config,
        )
    except RuntimeError as e:
        print(f"  Warning: final samples failed ({e})")

    # Final learned s(t) curve
    if cfg_config in ("E", "F") and guidance_schedule is not None:
        _save_learned_s_curve(guidance_schedule, save_dir, n_steps, device)

    print(f"\nResults saved to: {save_dir}")


def _save_learned_s_curve(guidance_schedule: GuidanceSchedule, save_dir: Path, step: int, device: str):
    """Save learned s(t) curve plot."""
    import matplotlib.pyplot as plt

    with torch.no_grad():
        t_probe = torch.linspace(0, 1, 50, device=device).unsqueeze(-1)  # [50, 1]
        s_probe = guidance_schedule(t_probe).squeeze(-1).cpu().numpy()  # [50]
        t_vals = t_probe.squeeze(-1).cpu().numpy()  # [50]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(t_vals, s_probe, color="blue", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="s=1 (no guidance)")
    ax.set_xlabel("t (normalized timestep, 0=clean, 1=noise)")
    ax.set_ylabel("s (guidance scale)")
    ax.set_title(f"Learned s(t) — step {step}, mean={s_probe.mean():.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    curve_dir = save_dir / "s_curves"
    curve_dir.mkdir(exist_ok=True)
    fig.savefig(curve_dir / f"learned_s_vs_t_{step}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CFG-Primary training: IP-Adapter + captions")

    # CFG-specific
    parser.add_argument("--cfg-config", type=str, required=True,
                        choices=["A", "B", "C", "D", "E", "F"],
                        help="CFG training config (A=baseline, B=fixed s, C=random s, "
                             "D=random s+warmup, E=learned s(t), F=learned s(t)+warmup)")
    parser.add_argument("--cfg-scale", type=float, default=3.0,
                        help="Fixed guidance scale for config B (default 3.0)")
    parser.add_argument("--cfg-s-min", type=float, default=1.5,
                        help="Min guidance scale for random sampling (configs C, D; default 1.5)")
    parser.add_argument("--cfg-s-max", type=float, default=4.0,
                        help="Max guidance scale for random sampling (configs C, D; default 4.0)")
    parser.add_argument("--dropout-prob", type=float, default=0.5,
                        help="Conditioning dropout probability for config A (default 0.5)")

    # Standard training args
    parser.add_argument("--splits-dir", type=str, default="data/concepts",
                        help="Directory with anomaly type JSONs")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions.json")
    parser.add_argument("--save-dir", type=str, default="results/cfg_training",
                        help="Where to save results")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint directory to resume from")
    parser.add_argument("--steps", type=int, default=50000,
                        help="Training steps")
    parser.add_argument("--save-every", type=int, default=2000,
                        help="Save checkpoint + samples every N steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for from-scratch params")
    parser.add_argument("--lr-pretrained", type=float, default=1e-4,
                        help="Learning rate for pretrained IP-Adapter params")
    parser.add_argument("--lambda-sp", type=float, default=0.0,
                        help="L2-SP regularization strength (default 0=disabled)")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Batch size (UNet sees 2x for configs B-F)")
    parser.add_argument("--ip-adapter-type", type=str, default="plus",
                        choices=["standard", "plus"])
    parser.add_argument("--ip-adapter-k", type=int, default=16, choices=[1, 4, 16])
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--no-mask-visual", action="store_true",
                        help="Disable masked cross-attention for visual pathway")
    parser.add_argument("--visual-mode", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--no-learnable-gates", action="store_true")
    parser.add_argument("--force-gates", action="store_true")
    parser.add_argument("--sa-num-layers", type=int, default=3)
    parser.add_argument("--sa-num-heads", type=int, default=12)
    parser.add_argument("--no-multi-crop", action="store_true")
    parser.add_argument("--no-clip-align", action="store_true")
    parser.add_argument("--band-mode", type=int, default=2, choices=[1, 2])
    parser.add_argument("--loss-core-ratio", type=float, default=0.8)
    parser.add_argument("--binary-cross-attn-mask", action="store_true")
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"])
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--types", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-sources", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-live-viewer", action="store_true")
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--timestep-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--logit-normal-mean", type=float, default=0.0)
    parser.add_argument("--logit-normal-std", type=float, default=1.0)

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    captions_file = Path(args.captions_file)
    if not captions_file.is_absolute():
        captions_file = project_root / captions_file

    resume_dir = Path(args.resume) if args.resume else None
    if resume_dir and not resume_dir.is_absolute():
        resume_dir = project_root / resume_dir

    train_cfg(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        captions_file=captions_file,
        cfg_config=args.cfg_config,
        cfg_scale=args.cfg_scale,
        cfg_s_min=args.cfg_s_min,
        cfg_s_max=args.cfg_s_max,
        dropout_prob=args.dropout_prob,
        n_steps=args.steps,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_pretrained=args.lr_pretrained,
        lambda_sp=args.lambda_sp,
        anomaly_types=args.types,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        device=args.device,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        mask_visual=not args.no_mask_visual,
        band_mode=args.band_mode,
        loss_core_ratio=args.loss_core_ratio,
        binary_cross_attn_mask=args.binary_cross_attn_mask,
        t2i_adapter_mode=args.t2i_adapter_mode,
        augment=args.augment,
        resume_dir=resume_dir,
        visual_mode=args.visual_mode,
        learnable_gates=not args.no_learnable_gates,
        force_gates=args.force_gates,
        sa_num_layers=args.sa_num_layers,
        sa_num_heads=args.sa_num_heads,
        multi_crop=not args.no_multi_crop,
        no_live_viewer=args.no_live_viewer,
        noise_offset=args.noise_offset,
        timestep_sampling=args.timestep_sampling.replace("-", "_"),
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        clip_align=not args.no_clip_align,
    )
