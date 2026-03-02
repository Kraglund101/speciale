"""
CLIP Consistency Training — standard diffusion + CLIP consistency loss.

The IP-Adapter's visual pathway may not strongly attract predictions toward
containing the reference anomaly. This script adds a CLIP consistency loss:
decode predicted x̂₀ → apply same augmentations as CLIP reference → CLIP encode
→ masked cosine distance. Forces predictions to be semantically consistent with
the reference in the anomaly region.

Loss structure (single-B UNet forward, NOT 2B packed):
  Per conditioned sample:  L = L_eps + gamma * L_clip
  Per unconditional sample: L = L_eps only

Extensive diagnostics (A-H) instrument the full conditioning pipeline.
"""
import csv
import gc
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image
from tqdm import tqdm
from torch.utils.checkpoint import checkpoint as grad_checkpoint

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
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def _worker_init_fn(worker_id):
    """Seed Python random + numpy in DataLoader workers (required on Windows/spawn)."""
    worker_seed = _SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ──────────────────────────────────────────────────────────────────────
# CLIP Consistency helpers
# ──────────────────────────────────────────────────────────────────────

def apply_spatial_transforms(
    images: torch.Tensor,
    flip_h: torch.Tensor,
    flip_v: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """Apply per-sample spatial transforms (flip, rotation) to decoded images.

    Args:
        images: [B, 3, H, W] in [0, 1]
        flip_h: [B] float, 1.0=flip
        flip_v: [B] float, 1.0=flip
        rotation: [B] float, 0/90/180/270

    Returns:
        Transformed [B, 3, H, W]
    """
    B = images.shape[0]
    result = images.clone()
    for i in range(B):
        if flip_h[i] > 0.5:
            result[i] = result[i].flip(-1)  # horizontal flip
        if flip_v[i] > 0.5:
            result[i] = result[i].flip(-2)  # vertical flip
        rot = int(rotation[i].item())
        if rot == 90:
            result[i] = result[i].rot90(1, [-2, -1])
        elif rot == 180:
            result[i] = result[i].rot90(2, [-2, -1])
        elif rot == 270:
            result[i] = result[i].rot90(3, [-2, -1])
    return result


def extract_crops(
    images: torch.Tensor,
    crop_top: torch.Tensor,
    crop_left: torch.Tensor,
    crop_h: torch.Tensor,
    crop_w: torch.Tensor,
    resized: torch.Tensor,
    orig_h: torch.Tensor,
    orig_w: torch.Tensor,
    out_size: int = 224,
) -> torch.Tensor:
    """Extract per-sample crops from decoded images at scaled coordinates.

    Args:
        images: [B, 3, H_dec, W_dec] (decoded, e.g. 512x512)
        crop_top/left/h/w: [B] float, coordinates in original-resolution space
        resized: [B] float, 1.0 if crop was resized (bbox > 224)
        orig_h/w: [B] float, original image dimensions (before decode scaling)
        out_size: output crop size (224)

    Returns:
        [B, 3, out_size, out_size]
    """
    B, C, H_dec, W_dec = images.shape
    crops = torch.zeros(B, C, out_size, out_size, device=images.device, dtype=images.dtype)

    for i in range(B):
        oh, ow = orig_h[i].item(), orig_w[i].item()
        if oh < 1 or ow < 1:
            continue
        # Scale coordinates from original resolution to decoded resolution
        sy = H_dec / oh
        sx = W_dec / ow
        t = int(crop_top[i].item() * sy)
        l = int(crop_left[i].item() * sx)
        h = max(1, int(crop_h[i].item() * sy))
        w = max(1, int(crop_w[i].item() * sx))
        # Clamp
        t = max(0, min(t, H_dec - 1))
        l = max(0, min(l, W_dec - 1))
        h = min(h, H_dec - t)
        w = min(w, W_dec - l)
        if h < 1 or w < 1:
            continue
        crop = images[i:i+1, :, t:t+h, l:l+w]
        crops[i] = F.interpolate(crop, size=(out_size, out_size),
                                 mode='bilinear', align_corners=False).squeeze(0)
    return crops


def clip_normalize(images: torch.Tensor) -> torch.Tensor:
    """Apply CLIP normalization to [B, 3, H, W] images in [0, 1]."""
    mean = torch.tensor(_CLIP_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def make_patch_mask(crop_mask: torch.Tensor, grid_size: int = 16) -> torch.Tensor:
    """Downsample crop mask to patch grid via maxpool → flatten.

    Args:
        crop_mask: [B, 1, H, W] in {0, 1}
        grid_size: patch grid size (16 for ViT-H)

    Returns:
        [B, grid_size*grid_size] binary mask
    """
    kernel = crop_mask.shape[-1] // grid_size
    mask_grid = F.max_pool2d(crop_mask.float(), kernel_size=kernel)
    return (mask_grid > 0.5).float().view(crop_mask.shape[0], -1)


# ──────────────────────────────────────────────────────────────────────
# Diagnostic B: Feature Drift
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_feature_drift(
    pipeline, ip_adapter, model_input: torch.Tensor,
    timesteps: torch.Tensor, text_emb: torch.Tensor,
    ip_cond: torch.Tensor, alpha_map: torch.Tensor,
    t2i_kwargs: dict, null_token_mask: Optional[torch.Tensor],
) -> float:
    """Two no_grad UNet forwards: real IP vs zero IP. Return ||delta|| inside mask.

    Args:
        model_input: [B, 9, 64, 64] (noisy + mask + masked_image)
        ip_cond: [B, K, dim] real IP embeddings
        alpha_map: [B, 1, 64, 64] spatial mask

    Returns:
        Mean L2 norm of (eps_cond - eps_null) within masked region.
    """
    cross_kwargs_real = {
        "ip_adapter_image_embeds": ip_cond,
        "ip_adapter_mask": alpha_map,
    }
    if null_token_mask is not None:
        cross_kwargs_real["null_token_mask"] = null_token_mask

    eps_real = pipeline.unet(
        model_input, timesteps,
        encoder_hidden_states=text_emb,
        cross_attention_kwargs=cross_kwargs_real,
        **t2i_kwargs,
    ).sample.float()

    cross_kwargs_null = {
        "ip_adapter_image_embeds": torch.zeros_like(ip_cond),
        "ip_adapter_mask": alpha_map,
    }
    if null_token_mask is not None:
        cross_kwargs_null["null_token_mask"] = null_token_mask

    eps_null = pipeline.unet(
        model_input, timesteps,
        encoder_hidden_states=text_emb,
        cross_attention_kwargs=cross_kwargs_null,
        **t2i_kwargs,
    ).sample.float()

    # Delta inside mask
    weight = alpha_map.expand(-1, 4, -1, -1)
    delta = (eps_real - eps_null) * weight
    return delta.norm().item() / max(weight.sum().item(), 1.0)


# ──────────────────────────────────────────────────────────────────────
# Loss computation
# ──────────────────────────────────────────────────────────────────────

def compute_clip_consistency_loss(
    pipeline, ip_adapter,
    noisy_latents: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    eps_pred: torch.Tensor,
    is_cond: torch.Tensor,
    batch_meta: dict,
    multi_crop: bool,
    clip_grad_checkpoint: bool = True,
    clip_alpha_cutoff: float = 0.5,
) -> Tuple[float, Optional[torch.Tensor], dict]:
    """Compute CLIP consistency loss — per-sample to avoid VRAM blowup.

    Processes each conditioned sample individually: VAE decode → CLIP encode →
    cosine loss → backward → accumulate x0 grad → free graph. This keeps peak
    Phase 2 memory to ~1 sample's worth of VAE+CLIP activations.

    Returns:
        (L_clip_value: float, clip_grad_eps: [B,4,64,64] or None, extras dict)
    """
    device = eps_pred.device
    B = eps_pred.shape[0]
    extras = {"cos_sim_anomaly": float('nan'), "cos_sim_all": float('nan')}

    cond_mask = is_cond  # [B] bool
    cond_cpu = cond_mask.cpu()
    n_cond = cond_mask.sum().item()
    if n_cond == 0:
        return float('nan'), None, extras

    # 1. Predict x̂₀ — DETACHED from UNet graph
    scheduler = pipeline.scheduler
    alpha_t = scheduler.alphas_cumprod.to(device)[timesteps]  # [B]
    sqrt_alpha = alpha_t.sqrt().view(B, 1, 1, 1)
    sqrt_1_minus_alpha = (1 - alpha_t).sqrt().view(B, 1, 1, 1)

    with torch.no_grad():
        x0_pred = (noisy_latents - sqrt_1_minus_alpha * eps_pred) / sqrt_alpha.clamp(min=1e-6)

    x0_cond_all = x0_pred[cond_mask].detach()  # [n_cond, 4, 64, 64]
    scaling_factor = pipeline.vae.config.scaling_factor

    # Gather per-sample metadata
    flip_h = batch_meta["clip_flip_h"][cond_cpu]
    flip_v = batch_meta["clip_flip_v"][cond_cpu]
    rotation = batch_meta["clip_rotation"][cond_cpu]
    orig_h_all = batch_meta["orig_h"][cond_cpu]
    orig_w_all = batch_meta["orig_w"][cond_cpu]
    crop_top_all = batch_meta["clip_crop_top"][cond_cpu]
    crop_left_all = batch_meta["clip_crop_left"][cond_cpu]
    crop_h_all = batch_meta["clip_crop_h"][cond_cpu]
    crop_w_all = batch_meta["clip_crop_w"][cond_cpu]
    resized_all = batch_meta["clip_resized"][cond_cpu]
    ref_01_1_all = batch_meta["ref_01_1"][cond_cpu]  # [n_cond, 3, 224, 224]
    patch_mask_1_all = batch_meta["patch_mask_1"][cond_cpu]  # [n_cond, 256]

    # Multi-crop metadata
    has_crop2 = multi_crop and "clip_crop_top_2" in batch_meta
    group_valid_gpu = None
    if has_crop2:
        gv = batch_meta.get("group_valid")
        if gv is not None:
            group_valid_gpu = gv[cond_mask]  # [n_cond, 2]

    # 2. Batch-encode reference tokens (no grad, cheap)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        ref_1_norm = clip_normalize(ref_01_1_all.to(device))
        ref_tokens_1 = ip_adapter.image_encoder(
            ref_1_norm, output_hidden_states=True,
        ).hidden_states[-2][:, 1:, :]  # [n_cond, 256, 1280]

        ref_tokens_2_all = None
        if has_crop2 and "ref_01_2" in batch_meta:
            v2_mask = group_valid_gpu[:, 1] > 0.5 if group_valid_gpu is not None else None
            if v2_mask is not None and v2_mask.any():
                v2_cpu = v2_mask.cpu()
                ref_2_norm = clip_normalize(batch_meta["ref_01_2"][cond_cpu][v2_cpu].to(device))
                ref_tokens_2_all = ip_adapter.image_encoder(
                    ref_2_norm, output_hidden_states=True,
                ).hidden_states[-2][:, 1:, :]
    del ref_1_norm
    torch.cuda.empty_cache()

    # 3. Per-sample forward+backward through VAE+CLIP (memory-safe)
    #    Skip samples where noise power > 0.5 (alpha_bar < 0.5) — x0 prediction
    #    is unreliable so CLIP gradient would not be semantically meaningful.
    grad_x0_accum = torch.zeros_like(x0_cond_all)  # [n_cond, 4, 64, 64]
    alpha_cond = alpha_t[cond_mask]  # [n_cond]
    total_weighted_cos_dist = 0.0  # accumulate for L_clip value
    total_anomaly_tokens = 0.0
    total_weighted_cos_sim = 0.0  # accumulate for cos_sim_anomaly (same weighting as L_clip)
    cos_sim_all_sum = 0.0
    n_clip_computed = 0

    for i in range(n_cond):
        # Skip high-noise samples — x0 prediction is unreliable
        if clip_alpha_cutoff > 0 and alpha_cond[i].item() < clip_alpha_cutoff:
            continue
        n_clip_computed += 1

        x0_i = x0_cond_all[i:i+1].requires_grad_(True)  # [1, 4, 64, 64]

        # VAE decode
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            dec_i = pipeline.vae.decode(x0_i / scaling_factor).sample
        dec_i = dec_i.float().clamp(-1, 1)
        dec_01 = (dec_i + 1) / 2  # [1, 3, 512, 512]

        # Spatial transforms
        aligned = apply_spatial_transforms(
            dec_01,
            flip_h[i:i+1].to(device), flip_v[i:i+1].to(device),
            rotation[i:i+1].to(device),
        )

        # Extract crop 1
        crop_1 = extract_crops(
            aligned,
            crop_top_all[i:i+1].to(device), crop_left_all[i:i+1].to(device),
            crop_h_all[i:i+1].to(device), crop_w_all[i:i+1].to(device),
            resized_all[i:i+1].to(device),
            orig_h_all[i:i+1].to(device), orig_w_all[i:i+1].to(device),
            out_size=224,
        )

        # CLIP encode with gradient
        crop_1_norm = clip_normalize(crop_1)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred_tok_1 = ip_adapter.image_encoder(
                crop_1_norm, output_hidden_states=True,
            ).hidden_states[-2][:, 1:, :]  # [1, 256, 1280]

        # Cosine distance (crop 1)
        ref_tok_1 = ref_tokens_1[i:i+1].detach()
        cos_dist_1 = 1 - F.cosine_similarity(pred_tok_1, ref_tok_1, dim=-1)  # [1, 256]
        pm_1 = patch_mask_1_all[i:i+1].to(device)  # [1, 256]
        n1_i = pm_1.sum()
        weighted_1 = (cos_dist_1 * pm_1).sum()

        # Multi-crop: crop 2
        weighted_2 = torch.tensor(0.0, device=device)
        n2_i = torch.tensor(0.0, device=device)
        if has_crop2 and group_valid_gpu is not None and group_valid_gpu[i, 1] > 0.5:
            v2_cpu_mask = (group_valid_gpu[:, 1] > 0.5).cpu()
            # Find index into ref_tokens_2_all for this sample
            v2_idx = v2_cpu_mask[:i+1].sum().item() - 1  # 0-based index among valid_2 samples

            crop_top_2 = batch_meta["clip_crop_top_2"][cond_cpu][i:i+1].to(device)
            crop_left_2 = batch_meta["clip_crop_left_2"][cond_cpu][i:i+1].to(device)
            crop_h_2 = batch_meta["clip_crop_h_2"][cond_cpu][i:i+1].to(device)
            crop_w_2 = batch_meta["clip_crop_w_2"][cond_cpu][i:i+1].to(device)
            resized_2 = batch_meta["clip_resized_2"][cond_cpu][i:i+1].to(device)

            crop_2 = extract_crops(
                aligned, crop_top_2, crop_left_2, crop_h_2, crop_w_2,
                resized_2, orig_h_all[i:i+1].to(device), orig_w_all[i:i+1].to(device),
                out_size=224,
            )
            crop_2_norm = clip_normalize(crop_2)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred_tok_2 = ip_adapter.image_encoder(
                    crop_2_norm, output_hidden_states=True,
                ).hidden_states[-2][:, 1:, :]

            ref_tok_2 = ref_tokens_2_all[v2_idx:v2_idx+1].detach()
            cos_dist_2 = 1 - F.cosine_similarity(pred_tok_2, ref_tok_2, dim=-1)
            pm_2 = batch_meta["patch_mask_2"][cond_cpu][i:i+1].to(device)
            n2_i = pm_2.sum()
            weighted_2 = (cos_dist_2 * pm_2).sum()

        # Accumulate loss numerator/denominator
        total_weighted_cos_dist += (weighted_1 + weighted_2).item()
        total_anomaly_tokens += (n1_i + n2_i).item()

        # Diagnostics: token-weighted cos_sim across both crops (matches L_clip weighting)
        with torch.no_grad():
            cos_sim_all_sum += F.cosine_similarity(pred_tok_1.detach(), ref_tok_1, dim=-1).mean().item()
            # Anomaly cos_sim: raw weighted sums (divided by total_anomaly_tokens later)
            sim_1 = (F.cosine_similarity(pred_tok_1.detach(), ref_tok_1, dim=-1) * pm_1).sum()
            total_weighted_cos_sim += sim_1.item()
            if n2_i > 0:
                sim_2 = (F.cosine_similarity(pred_tok_2.detach(), ref_tok_2, dim=-1) * pm_2).sum()
                total_weighted_cos_sim += sim_2.item()

        # Per-sample loss for backward
        loss_i = (weighted_1 + weighted_2) / (n1_i + n2_i).clamp(min=1)
        loss_i.backward()

        # Save gradient and free graph
        grad_x0_accum[i] = x0_i.grad.detach()
        del x0_i, dec_i, dec_01, aligned, crop_1, crop_1_norm, pred_tok_1
        del cos_dist_1, pm_1, loss_i, weighted_1, weighted_2
        torch.cuda.empty_cache()

    # 4. Aggregate (only over samples that passed the noise power threshold)
    #    cos_sim_anomaly uses same token-weighted average as L_clip → L_clip = 1 - cos_sim exactly
    L_clip_val = total_weighted_cos_dist / max(total_anomaly_tokens, 1.0)
    extras["cos_sim_anomaly"] = total_weighted_cos_sim / max(total_anomaly_tokens, 1.0)
    extras["cos_sim_all"] = cos_sim_all_sum / max(n_clip_computed, 1)
    extras["n_clip_computed"] = n_clip_computed

    # If no samples passed the noise threshold, return NaN (no deflation of trend)
    if n_clip_computed == 0:
        return float('nan'), None, extras

    # 5. Confidence-weighted gradient transfer: x0-space grad → eps-space
    #    weight = sqrt(alpha_bar_t): ~1.0 at t=0 (reliable x0), ~0.006 at t=999 (noise).
    #    Samples with alpha < 0.5 already have zero grad (skipped above).
    #    Negative sign: increasing eps worsens x0.
    clip_grad_eps = torch.zeros_like(eps_pred)  # [B, 4, 64, 64]
    sqrt_alpha_cond = sqrt_alpha[cond_mask]
    confidence_scale = -sqrt_alpha_cond  # [n_cond, 1, 1, 1]
    clip_grad_eps[cond_mask] = grad_x0_accum * confidence_scale

    return L_clip_val, clip_grad_eps.detach(), extras


# ──────────────────────────────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────────────────────────────

def save_checkpoint(ip_adapter, optimizer, save_dir, band_mode, t2i_adapter, step, losses,
                    unet=None, qo_params_named=None):
    """Save training checkpoint."""
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
    if unet is not None:
        from peft import get_peft_model_state_dict
        state["lora_weights"] = get_peft_model_state_dict(unet)
    if qo_params_named:
        state["qo_weights"] = {name: param.data for name, param in qo_params_named}
    torch.save(state, save_dir / "training_state.pt")


# ──────────────────────────────────────────────────────────────────────
# Sample grid generation
# ──────────────────────────────────────────────────────────────────────

def generate_samples(
    pipeline, ip_adapter, dataset, save_path, device,
    band_mode: int = 2, t2i_adapter=None,
):
    """Generate sample grid: 7 rows (real, mask, ref, CFG=1, CFG=3, heatmap, trajectory).

    Row 0: Real image
    Row 1: Mask
    Row 2: Reference CLIP crop
    Row 3: Generated (CFG=1, no amplification)
    Row 4: Generated (CFG=3, visual mode)
    Row 5: CLIP similarity heatmap (16x16 cos_sim, upscaled)
    Row 6: Denoising trajectory strip (x0 at t=900,500,300,100,50,10)
    """
    import matplotlib.pyplot as plt

    save_path = Path(save_path)

    # --- Sample selection (same fixed samples as other scripts) ---
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

    # --- Generate at two CFG scales ---
    gen_kwargs = dict(
        num_steps=30, noise_strength=1.0,
        reference_mode=dataset.reference_mode,
        band_mode=band_mode, t2i_adapter=t2i_adapter,
    )

    for i, s in enumerate(all_samples):
        with torch.no_grad():
            s["gen_cfg1"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=1.0, cfg_mode="visual", seed=42 + i,
                **gen_kwargs,
            )
            torch.cuda.empty_cache()
            s["gen_cfg3"] = generate_anomagic_single(
                pipeline, ip_adapter, s["image"], s["mask"], s["reference"],
                s["atype"], s["caption"],
                guidance_scale=3.0, cfg_mode="visual", seed=42 + i,
                **gen_kwargs,
            )
            # Move results to CPU immediately to free GPU memory
            s["gen_cfg1"] = s["gen_cfg1"].cpu()
            s["gen_cfg3"] = s["gen_cfg3"].cpu()
            # Free per-sample GPU tensors
            del s["image"], s["mask"], s["reference"]
            torch.cuda.empty_cache()

    def _to_np(t):
        return ((t[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

    # --- Plot grid: 5 rows x n_cols ---
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.2 * n_rows), squeeze=False)
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

        axes[3, i].imshow(_to_np(s["gen_cfg1"]))
        axes[3, i].set_title("Gen (CFG=1)", fontsize=7)
        axes[3, i].axis("off")

        axes[4, i].imshow(_to_np(s["gen_cfg3"]))
        axes[4, i].set_title("Gen (CFG=3, visual)", fontsize=7)
        axes[4, i].axis("off")

    plt.suptitle("CLIP Consistency Training Samples", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Loss plot (static)
# ──────────────────────────────────────────────────────────────────────

def save_loss_plot(losses, stats_file, save_path):
    """Save static 3x3 loss plot — matches live_loss_clip.py quality."""
    import matplotlib.pyplot as plt

    SM = 0.99

    def ema(arr, alpha=0.99):
        """NaN-aware EMA — carries forward on NaN."""
        out = np.empty_like(arr)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                out[i] = out[i - 1]
            elif np.isnan(out[i - 1]):
                out[i] = arr[i]
            else:
                out[i] = alpha * out[i-1] + (1 - alpha) * arr[i]
        return out

    def parse_stats(fp):
        result = {}
        try:
            with open(fp, "r") as f:
                header = f.readline().strip().split(",")
                for col in header:
                    result[col] = []
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != len(header):
                        continue
                    for i, col in enumerate(header):
                        try:
                            result[col].append(float(parts[i]))
                        except ValueError:
                            result[col].append(0.0)
            for col in result:
                result[col] = np.array(result[col])
        except FileNotFoundError:
            pass
        return result

    arr = np.array(losses)
    if len(arr) < 2:
        return
    steps = np.arange(len(arr))
    stats = parse_stats(str(stats_file))
    s_steps = stats.get("step", np.array([]))
    has_stats = len(s_steps) > 0

    fig, axes = plt.subplots(3, 3, figsize=(21, 15))

    # [0,0] Loss trend
    ax = axes[0, 0]
    ema_slow = ema(arr, SM)
    ax.plot(steps, ema_slow, color="darkblue", lw=2.5, label=f"EMA({SM})")
    ax.plot(steps, ema(arr, 0.6), color="red", lw=0.8, alpha=0.3, label="EMA(0.6)")
    ax.set_title(f"Loss Trend \u2014 step {len(arr)}, trend: {ema_slow[-1]:.4f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # [0,1] L_diff vs L_clip (NaN-aware EMA)
    ax = axes[0, 1]
    if has_stats and "L_diffusion" in stats and len(stats["L_diffusion"]) > 1:
        ema_diff = ema(stats["L_diffusion"], SM)
        ax.plot(s_steps, ema_diff, color="blue", lw=2, label=f"L_diff ({ema_diff[-1]:.4f})")
        if "L_clip_scaled" in stats and len(stats["L_clip_scaled"]) > 1:
            ema_clip = ema(stats["L_clip_scaled"], SM)
            ax.plot(s_steps, ema_clip, color="red", lw=2, label=f"L_clip*\u03b3 ({ema_clip[-1]:.4f})")
        if "L_clip" in stats and len(stats["L_clip"]) > 1:
            ema_clip_raw = ema(stats["L_clip"], SM)
            ax.plot(s_steps, ema_clip_raw, color="red", lw=1, ls="--", alpha=0.5,
                    label=f"L_clip raw ({ema_clip_raw[-1]:.4f})")
        ax.legend(fontsize=7)
        ax.set_title("L_diff vs L_clip \u2014 want both to decrease")
    else:
        ax.set_title("L_diff vs L_clip (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)

    # [0,2] Cosine Similarity (NaN-aware EMA)
    ax = axes[0, 2]
    if has_stats and "cos_sim_anomaly" in stats and len(stats["cos_sim_anomaly"]) > 1:
        ema_anom = ema(stats["cos_sim_anomaly"], SM)
        ema_all = ema(stats.get("cos_sim_all", np.zeros_like(stats["cos_sim_anomaly"])), SM)
        ax.plot(s_steps, ema_anom, color="green", lw=2.5, label=f"anomaly ({ema_anom[-1]:.3f})")
        ax.plot(s_steps, ema_all, color="gray", lw=1.5, ls="--", label=f"all ({ema_all[-1]:.3f})")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"Cosine Similarity \u2014 anomaly: {ema_anom[-1]:.3f}")
    else:
        ax.set_title("Cosine Similarity (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("cos_sim"); ax.grid(True, alpha=0.3)

    # [1,0] Core vs Band (weighted, stacked)
    ax = axes[1, 0]
    if has_stats and "core_loss" in stats and len(stats["core_loss"]) > 1 and stats["core_loss"].any():
        s_core = stats["core_loss"]
        s_band = stats["band_loss"]
        w_core_s = 0.8 * s_core
        w_band_s = 0.2 * s_band
        w_total_s = np.maximum(w_core_s + w_band_s, 1e-10)
        core_share = w_core_s / w_total_s

        L_diff = stats.get("L_diffusion", np.zeros_like(s_core))
        L_diff_ema = ema(L_diff, SM)
        core_share_ema = ema(core_share, SM)
        ce = L_diff_ema * core_share_ema / 0.8
        be = L_diff_ema * (1.0 - core_share_ema) / 0.2

        ax.fill_between(s_steps, 0, ce, color="#2196F3", alpha=0.4)
        ax.fill_between(s_steps, ce, ce + be, color="#FF9800", alpha=0.4)
        ax.plot(s_steps, ce, color="#1565C0", lw=1.5, label="Core")
        ax.plot(s_steps, ce + be, color="#E65100", lw=1.5, label="Band (stacked)")
        ax.plot(s_steps, L_diff_ema, color="black", lw=2, ls="--",
                label=f"0.8\u00d7core+0.2\u00d7band = {L_diff_ema[-1]:.4f}")
        ax.annotate(f"{ce[-1]:.3f}", xy=(s_steps[-1], ce[-1] / 2),
                    fontsize=9, fontweight="bold", color="#1565C0", ha="right")
        ax.annotate(f"{be[-1]:.3f}", xy=(s_steps[-1], ce[-1] + be[-1] / 2),
                    fontsize=9, fontweight="bold", color="#E65100", ha="right")
        ax.legend(fontsize=8)
        ax.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {L_diff_ema[-1]:.4f}")
    else:
        ax.set_title("Core vs Band (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE"); ax.grid(True, alpha=0.3)

    # [1,1] IP Ratio (only nonzero points from diag_interval)
    ax = axes[1, 1]
    if has_stats and "ip_attn_norm" in stats and stats["ip_attn_norm"].any():
        ip_norm = stats["ip_attn_norm"]
        h_pre = stats.get("h_pre_norm", np.ones_like(ip_norm))
        nz = ip_norm > 0
        if nz.any():
            ratio_nz = np.where(h_pre[nz] > 1e-8, ip_norm[nz] / h_pre[nz], 0.0)
            ema_ratio = ema(ratio_nz, SM)
            ax.plot(s_steps[nz], ema_ratio, color="purple", lw=2, marker=".", ms=3,
                    label=f"ratio ({ema_ratio[-1]:.4f})")
            ax.legend(fontsize=8)
            ax.set_title(f"IP Ratio (||ip_out|| / ||h_pre||) \u2014 {ema_ratio[-1]:.4f}")
        else:
            ax.set_title("IP Ratio (no nonzero data)")
    else:
        ax.set_title("IP Ratio (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Ratio"); ax.grid(True, alpha=0.3)

    # [1,2] Cross-Attn Entropy (only nonzero points)
    ax = axes[1, 2]
    if has_stats and "ip_entropy" in stats and stats["ip_entropy"].any():
        ent = stats["ip_entropy"]
        nz = ent > 0
        if nz.any():
            ema_ent = ema(ent[nz], SM)
            ax.plot(s_steps[nz], ema_ent, color="teal", lw=2, marker=".", ms=3,
                    label=f"entropy ({ema_ent[-1]:.2f})")
            ax.legend(fontsize=8)
            ax.set_title(f"IP Attention Entropy \u2014 {ema_ent[-1]:.2f}")
        else:
            ax.set_title("IP Attention Entropy (no nonzero data)")
    else:
        ax.set_title("IP Attention Entropy (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Entropy"); ax.grid(True, alpha=0.3)

    # [2,0] Feature Drift (only nonzero points, no EMA)
    ax = axes[2, 0]
    if has_stats and "feature_drift" in stats and stats["feature_drift"].any():
        mask = stats["feature_drift"] > 0
        if mask.any():
            ax.plot(s_steps[mask], stats["feature_drift"][mask], color="darkorange",
                    lw=1.5, marker=".", ms=3, label=f"drift ({stats['feature_drift'][mask][-1]:.4f})")
            ax.legend(fontsize=8)
            ax.set_title("Feature Drift (B) \u2014 want > 0 (IP has effect)")
        else:
            ax.set_title("Feature Drift (B) (no nonzero data)")
    else:
        ax.set_title("Feature Drift (B) (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("||delta|| / mask_area"); ax.grid(True, alpha=0.3)

    # [2,1] Grad Norm Split (total=EMA, g_diff/g_clip=nonzero only)
    ax = axes[2, 1]
    if has_stats and "grad_norm" in stats and len(stats["grad_norm"]) > 1:
        gn_ema = ema(stats["grad_norm"], SM)
        ax.plot(s_steps, gn_ema, color="darkblue", lw=2, label=f"total ({gn_ema[-1]:.2f})")
        if "g_diff_norm" in stats and stats["g_diff_norm"].any():
            mask = stats["g_diff_norm"] > 0
            if mask.any():
                ax.plot(s_steps[mask], stats["g_diff_norm"][mask], color="blue",
                        lw=1, marker=".", ms=2, label=f"g_diff ({stats['g_diff_norm'][mask][-1]:.2f})")
                ax.plot(s_steps[mask], stats["g_clip_norm"][mask], color="red",
                        lw=1, marker=".", ms=2, label=f"g_clip ({stats['g_clip_norm'][mask][-1]:.2f})")
        ax.axhline(1.0, color="black", ls="--", alpha=0.5, label="clip=1.0")
        ax.legend(fontsize=7)
        ax.set_title(f"Grad Norm \u2014 {gn_ema[-1]:.2f}")
    else:
        ax.set_title("Grad Norm (no data yet)")
    ax.set_xlabel("Step"); ax.set_ylabel("Norm"); ax.grid(True, alpha=0.3)

    # [2,2] Gates
    ax = axes[2, 2]
    if has_stats and "attn_gate" in stats and len(stats["attn_gate"]) > 1:
        ax.plot(s_steps, stats["attn_gate"], color="blue", lw=1.5, label="Attn gate")
        ax.plot(s_steps, stats["ff_gate"], color="green", lw=1.5, label="FF gate")
        ax.set_title(f"Gates \u2014 attn={stats['attn_gate'][-1]:.4f}, ff={stats['ff_gate'][-1]:.4f}")
        ax.legend(fontsize=8)
    else:
        ax.set_title("Gates (no data)")
    ax.set_xlabel("Step"); ax.set_ylabel("Gate value"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Live loss viewer launcher
# ──────────────────────────────────────────────────────────────────────

def _launch_live_loss(loss_file: Path):
    """Launch live loss viewer as a detached subprocess."""
    import subprocess
    script = Path(__file__).parent / "live_loss_clip.py"
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

def train_clip_consistency(
    splits_dir: Path,
    save_dir: Path,
    captions_file: Path,
    # CLIP consistency
    clip_gamma: float = 1.0,
    clip_grad_checkpoint: bool = True,
    clip_alpha_cutoff: float = 0.5,
    # Diagnostics
    diag_interval: int = 50,
    drift_interval: int = 500,
    grad_split_interval: int = 100,
    # Standard training
    n_steps: int = 50000,
    batch_size: int = 10,
    save_every: int = 2000,
    lr: float = 1e-4,
    lr_pretrained: float = 1e-4,
    lambda_sp: float = 0.0,
    # Conditioning dropout
    drop_image_prob: float = 0.10,
    drop_text_prob: float = 0.10,
    drop_both_prob: float = 0.05,
    # IP-Adapter
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 16,
    ip_adapter_scale: float = 1.0,
    mask_visual: bool = True,
    visual_mode: int = 3,
    learnable_gates: bool = True,
    force_gates: bool = False,
    sa_num_layers: int = 3,
    sa_num_heads: int = 12,
    # LoRA
    lora_rank: int = 0,
    lora_alpha: int = 16,
    lora_lr: float = 5e-5,
    lora_mode: str = "all",
    # UNet unfreezing
    unfreeze_qo: str = "",
    unfreeze_qo_lr: float = 1e-5,
    # Data
    data_root: Optional[Path] = None,
    anomaly_types: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
    augment: bool = True,
    multi_crop: bool = True,
    clip_align: bool = True,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    binary_cross_attn_mask: bool = False,
    t2i_adapter_mode: str = "cascade",
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    # Resume
    resume_dir: Optional[str] = None,
    # Viewer
    no_live_viewer: bool = False,
    # Pilot subset
    pilot_subset_path: Optional[str] = None,
    pilot_subset_size: int = 0,
    device: str = "cuda",
    seed: int = _SEED,
):
    """Train IP-Adapter + T2I-Adapter with CLIP consistency loss."""

    # =========================================
    # Seed everything
    # =========================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CLIP Consistency Training")
    print("=" * 70)
    print(f"  clip_gamma:       {clip_gamma}")
    print(f"  clip_alpha_cutoff:{clip_alpha_cutoff}")
    print(f"  ip_adapter_scale: {ip_adapter_scale}")
    print(f"  batch_size:       {batch_size}")
    print(f"  steps:            {n_steps}")
    print(f"  lr_pretrained:    {lr_pretrained}")
    print(f"  lr_scratch:       {lr}")
    print(f"  IP-Adapter:       {ip_adapter_type}, K={ip_adapter_k}")
    print(f"  visual_mode:      {visual_mode}")
    print(f"  multi_crop:       {multi_crop}")
    print(f"  augment:          {augment}")
    print(f"  band_mode:        {band_mode}")
    print(f"  conditioning dropout: img={drop_image_prob}, txt={drop_text_prob}, both={drop_both_prob}")
    print(f"  diag_interval:    {diag_interval}")
    print(f"  drift_interval:   {drift_interval}")
    print(f"  grad_split_interval: {grad_split_interval}")
    print(f"  clip_grad_checkpoint: {clip_grad_checkpoint}")
    print(f"  save_dir:         {save_dir}")
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
        return_clip_meta=True,
    )

    if len(dataset) == 0:
        print("ERROR: No data loaded!")
        return

    # =========================================
    # Pilot Subset (optional)
    # =========================================
    from src.utils.pilot_subset import (
        generate_pilot_subset, apply_pilot_subset,
        load_pilot_subset, save_pilot_subset,
    )
    if pilot_subset_path and Path(pilot_subset_path).exists():
        pilot_indices = load_pilot_subset(pilot_subset_path)
        apply_pilot_subset(dataset, pilot_indices)
    elif pilot_subset_size > 0 and pilot_subset_size < len(dataset):
        pilot_indices = generate_pilot_subset(
            dataset.samples, dataset.type_to_samples,
            target_size=pilot_subset_size,
        )
        save_pilot_subset(pilot_indices, save_dir / "pilot_indices.json")
        apply_pilot_subset(dataset, pilot_indices)
    full_dataset = dataset

    # =========================================
    # Initialize Pipeline
    # =========================================
    print("\nInitializing models...")

    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    # No UNet gradient checkpointing needed — using detached-gradient approach
    # for CLIP loss (VAE+CLIP graph is separate from UNet graph)

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

    # Ensure all trainable modules are fp32
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # =========================================
    # LoRA (optional)
    # =========================================
    lora_params = []
    if lora_rank > 0:
        from peft import LoraConfig
        if lora_mode == "cross":
            target_modules = [r"attn2\.(to_k|to_q|to_v|to_out\.0)"]
        elif lora_mode == "mid_up":
            # Only mid_block + up_blocks, cross-attention (attn2) only
            target_modules = [r"(mid_block|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn2\.(to_k|to_q|to_v|to_out\.0)"]
        else:
            target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipeline.unet.add_adapter(lora_config)
        for p in pipeline.unet.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        ip_proc_ids = {id(p) for proc in ip_adapter.attn_processors.values() for p in proc.parameters()}
        lora_params = [p for p in pipeline.unet.parameters() if p.requires_grad and id(p) not in ip_proc_ids]
        n_lora = sum(p.numel() for p in lora_params)
        print(f"\nLoRA (rank={lora_rank}, alpha={lora_alpha}, mode={lora_mode}, lr={lora_lr}): {n_lora:,} params")

    # =========================================
    # Unfreeze W_Q/W_O (optional)
    # =========================================
    qo_params = []
    qo_params_named = []
    if unfreeze_qo:
        for name, param in pipeline.unet.named_parameters():
            if unfreeze_qo == "mid_up":
                if not (name.startswith("mid_block") or name.startswith("up_blocks")):
                    continue
            if unfreeze_qo in ("mid_up", "all_cross"):
                if "attn2" not in name:
                    continue
            else:
                if "attn1" not in name and "attn2" not in name:
                    continue
            if not (name.endswith("to_q.weight") or name.endswith("to_q.bias")
                    or "to_out.0.weight" in name or "to_out.0.bias" in name):
                continue
            param.requires_grad_(True)
            param.data = param.data.float()
            qo_params.append(param)
            qo_params_named.append((name, param))
        n_qo = sum(p.numel() for p in qo_params)
        mode_desc = {"mid_up": "mid+up cross-attn", "all_cross": "all cross-attn",
                     "all": "all cross+self attn"}[unfreeze_qo]
        print(f"\nUnfrozen W_Q + W_O ({mode_desc}, lr={unfreeze_qo_lr}): {n_qo:,} params")

    # =========================================
    # Build Optimizer
    # =========================================
    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)

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

    # Optional Group D: LoRA
    if lora_params:
        qo_ids = {id(p) for p in qo_params}
        lora_params_clean = [p for p in lora_params if id(p) not in qo_ids]
        param_groups.append(
            {"params": lora_params_clean, "lr": lora_lr, "weight_decay": 1e-3, "label": "D_lora"}
        )

    # Optional Group E: Unfrozen W_Q + W_O
    if qo_params:
        param_groups.append(
            {"params": qo_params, "lr": unfreeze_qo_lr, "weight_decay": 0.0, "label": "E_qo"}
        )

    # Verify no duplicates
    all_group_ids = []
    for pg in param_groups:
        for p in pg["params"]:
            pid = id(p)
            assert pid not in all_group_ids, f"Duplicate param in group {pg['label']}"
            all_group_ids.append(pid)

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

    # AMP — bf16 has same exponent range as fp32, no GradScaler needed
    use_amp = True
    amp_dtype = torch.bfloat16

    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # =========================================
    # Resume from checkpoint
    # =========================================
    start_step = 0
    resumed_losses = []

    if resume_dir is not None:
        resume_dir_path = Path(resume_dir)
        print(f"\nResuming from checkpoint: {resume_dir_path}")

        ip_ckpt = resume_dir_path / "ip_adapter.pt"
        if ip_ckpt.exists():
            ip_adapter.load_finetuned(resume_dir_path)
            ip_adapter.masked_self_attn.float()
            ip_adapter.image_projection.float()
            for proc in ip_adapter.attn_processors.values():
                proc.float()
            print(f"  Loaded IP-Adapter weights from {ip_ckpt}")
        else:
            print(f"  WARNING: {ip_ckpt} not found, starting with pretrained weights")

        state_path = resume_dir_path / "training_state.pt"
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

            # LoRA
            if lora_rank > 0 and "lora_weights" in state:
                from peft import set_peft_model_state_dict
                set_peft_model_state_dict(pipeline.unet, state["lora_weights"])
                for p in pipeline.unet.parameters():
                    if p.requires_grad:
                        p.data = p.data.float()
                print(f"  Loaded LoRA weights")

            # Unfrozen W_Q/W_O
            if unfreeze_qo and "qo_weights" in state:
                unet_params = dict(pipeline.unet.named_parameters())
                loaded = 0
                for name, saved_tensor in state["qo_weights"].items():
                    if name in unet_params:
                        unet_params[name].data = saved_tensor.float().to(device)
                        loaded += 1
                print(f"  Loaded {loaded} unfrozen W_Q/W_O tensors")

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
            generate_samples(
                pipeline, ip_adapter, full_dataset,
                save_dir / "samples_0_pretrained.png", device,
                band_mode=band_mode, t2i_adapter=t2i_adapter,
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
        "training_type": "clip_consistency",
        "clip_gamma": clip_gamma,
        "visual_mode": visual_mode,
        "batch_size": batch_size,
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
        "drop_image_prob": drop_image_prob,
        "drop_text_prob": drop_text_prob,
        "drop_both_prob": drop_both_prob,
        "diag_interval": diag_interval,
        "drift_interval": drift_interval,
        "grad_split_interval": grad_split_interval,
        "clip_grad_checkpoint": clip_grad_checkpoint,
        "clip_alpha_cutoff": clip_alpha_cutoff,
        "ip_adapter_scale": ip_adapter_scale,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_lr": lora_lr,
        "lora_mode": lora_mode,
        "unfreeze_qo": unfreeze_qo,
        "unfreeze_qo_lr": unfreeze_qo_lr,
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

    # Stats CSV
    stats_file = save_dir / "stats.csv"
    stats_fh = open(stats_file, "a")
    if stats_fh.tell() == 0:
        stats_fh.write(
            "step,loss,L_diffusion,L_clip,L_clip_scaled,"
            "core_loss,band_loss,grad_norm,attn_gate,ff_gate,"
            "lr_pretrained,lr_scratch,l2sp,progress,"
            "cos_sim_anomaly,cos_sim_all,"
            "ip_attn_norm,h_pre_norm,"
            "ip_k_norm,ip_v_norm,text_k_norm,text_v_norm,"
            "ip_entropy,feature_drift,"
            "g_diff_norm,g_clip_norm\n"
        )

    # Precompute uncond text embedding
    uncond_text_emb = pipeline.encode_text([""], enable_grad=False)

    pbar = tqdm(range(start_step, n_steps), desc="Training", initial=start_step, total=n_steps)
    skipped_nan = 0
    prev_ckpt_dir = None

    # Timestep bucket counters for diagnostic D
    cos_sim_buckets = defaultdict(list)  # bucket_idx -> list of cos_sim

    for step in pbar:
        is_last_step = (step == start_step + remaining_steps - 1)
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

        B = images.shape[0]
        optimizer.zero_grad()
        progress = step / max(n_steps - 1, 1)
        _do_phase_timing = (step - start_step < 20)

        # ── Forward pass (single-B, NOT 2B packed) ──
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            # VAE encode
            latents = pipeline.encode_image(images)

            # Timestep sampling
            if timestep_sampling == "logit_normal":
                u = torch.sigmoid(torch.randn(B, device=device) * logit_normal_std + logit_normal_mean)
                timesteps = (u * 1000).long().clamp(0, 999)
            else:
                timesteps = torch.randint(0, 1000, (B,), device=device)

            # Noise
            noise = torch.randn_like(latents)
            if noise_offset > 0:
                noise += noise_offset * torch.randn(B, latents.shape[1], 1, 1, device=device)

            # Noisy latents
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)

            # UNet masks: maxpool to 64x64
            kernel_mask = masks.shape[-1] // latents.shape[-1]
            core_mask_64 = F.max_pool2d(masks, kernel_size=kernel_mask)
            core_mask_64 = (core_mask_64 > 0.5).float()

            dilated_binary_64, alpha_map_64, weight_map_64, band_mask_64 = \
                create_latent_band_mask(core_mask_64, band_mode)

            # Text encoding
            prompts = []
            for i_b in range(B):
                cap = batch_captions[i_b] if isinstance(batch_captions, list) else batch_captions[i_b]
                if cap:
                    prompts.append(cap)
                else:
                    atype = batch_types[i_b]
                    type_word = atype.replace("_", " ")
                    prompts.append(f"a photo of a {type_word} defect")
            text_emb = pipeline.encode_text(prompts, enable_grad=False)

            # ── 3-category conditioning dropout (per-sample) ──
            ref_01 = (references + 1.0) / 2.0
            is_cond = torch.ones(B, dtype=torch.bool, device=device)
            drop_threshold = drop_image_prob + drop_text_prob + drop_both_prob

            for i_b in range(B):
                r = random.random()
                if r < drop_image_prob:
                    # Drop image only
                    ref_01[i_b] = 0.0
                    is_cond[i_b] = False
                elif r < drop_image_prob + drop_text_prob:
                    # Drop text only
                    text_emb[i_b] = uncond_text_emb[0]
                elif r < drop_threshold:
                    # Drop both
                    ref_01[i_b] = 0.0
                    text_emb[i_b] = uncond_text_emb[0]
                    is_cond[i_b] = False

            # Visual encoding
            if _do_phase_timing:
                torch.cuda.synchronize()
                _tp0_start = time.time()
            ip_image_embeds = ip_adapter.encode_image(ref_01, mask=clip_masks, core_mask=clip_core_masks)

            # Multi-crop: encode second crop
            null_token_mask = None
            if references_2 is not None:
                ref_2_01 = (references_2 + 1.0) / 2.0
                # Zero out second crop ref for dropped samples
                for i_b in range(B):
                    if not is_cond[i_b]:
                        ref_2_01[i_b] = 0.0
                ip_embeds_2 = ip_adapter.encode_image(ref_2_01, mask=clip_masks_2, core_mask=clip_core_masks_2)
                K = ip_image_embeds.shape[1]
                ip_image_embeds = torch.cat([ip_image_embeds, ip_embeds_2], dim=1)
                if group_valid is not None:
                    mask_1 = group_valid[:, 0:1].expand(-1, K)
                    mask_2 = group_valid[:, 1:2].expand(-1, K)
                    null_token_mask = torch.cat([mask_1, mask_2], dim=1)

            # Inpainting input
            mask_latents = dilated_binary_64
            unet_mask_512 = F.interpolate(dilated_binary_64, size=masks.shape[-2:], mode='nearest')
            masked_image = images * (1 - unet_mask_512)
            masked_image_latents = pipeline.encode_image(masked_image)
            model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

            # T2I-Adapter
            t2i_kwargs = {}
            if t2i_adapter is not None:
                t2i_input = torch.cat([core_mask_64, band_mask_64], dim=1)
                t2i_features = t2i_adapter(t2i_input, mask=dilated_binary_64)
                t2i_kwargs = t2i_adapter.prepare_unet_kwargs(t2i_features)

            # Cross-attention kwargs
            cross_attn_kwargs = {"ip_adapter_image_embeds": ip_image_embeds}
            if binary_cross_attn_mask:
                cross_attn_kwargs["ip_adapter_mask"] = dilated_binary_64
            else:
                cross_attn_kwargs["ip_adapter_mask"] = alpha_map_64
            if null_token_mask is not None:
                cross_attn_kwargs["null_token_mask"] = null_token_mask

            # Diagnostic A/F/H: attention norms
            diagnostics = None
            if step % diag_interval == 0 or is_last_step:
                diagnostics = {}
                cross_attn_kwargs["diagnostics"] = diagnostics

            # UNet forward (single-B)
            eps_pred = pipeline.unet(
                model_input, timesteps,
                encoder_hidden_states=text_emb,
                cross_attention_kwargs=cross_attn_kwargs,
                **t2i_kwargs,
            ).sample.float()

        if _do_phase_timing:
            torch.cuda.synchronize()
            _vram_free, _vram_total = torch.cuda.mem_get_info()
            _tp0_end = time.time()
            print(f"    [Phase0] encode+UNet fwd: {(_tp0_end-_tp0_start)*1000:.0f}ms  "
                  f"VRAM: {(_vram_total-_vram_free)/1e9:.1f}/{_vram_total/1e9:.1f} GiB", flush=True)

        # ── Loss computation (outside autocast for fp32), per-sample averaging ──
        weight_expanded = weight_map_64.expand(-1, 4, -1, -1).float()
        per_w = weight_expanded.sum(dim=(1, 2, 3)).clamp(min=1e-8)  # [B]
        diff_sq = (eps_pred.float() - noise.float()) ** 2

        L_diffusion = ((diff_sq * weight_expanded).sum(dim=(1, 2, 3)) / per_w).mean()

        # Core/band diagnostics (detached), per-sample
        with torch.no_grad():
            core_exp = (core_mask_64 > 0.5).float().expand(-1, 4, -1, -1)
            band_w_exp = (weight_map_64 * (1.0 - (core_mask_64 > 0.5).float())).expand(-1, 4, -1, -1)
            core_per = (diff_sq * core_exp).sum(dim=(1, 2, 3)) / core_exp.sum(dim=(1, 2, 3)).clamp(min=1e-8)
            band_per = (diff_sq * band_w_exp).sum(dim=(1, 2, 3)) / band_w_exp.sum(dim=(1, 2, 3)).clamp(min=1e-8)
            core_loss_val = core_per.mean().item()
            band_loss_val = band_per.mean().item()

        # ── Two-pass approach for CLIP consistency loss ──
        # The UNet graph (~6-8 GiB of stored activations) cannot coexist in
        # memory with the VAE+CLIP computation for the CLIP loss.
        # Solution: backward L_diffusion first (frees UNet graph), then compute
        # CLIP loss, then do a second UNet forward for CLIP gradient injection.
        L_clip_val = float('nan')  # NaN until actually computed (avoids deflating trend)
        L_clip_scaled_val = float('nan')
        clip_extras = {"cos_sim_anomaly": float('nan'), "cos_sim_all": float('nan')}
        clip_grad_eps = None
        L_diff_val = L_diffusion.item()

        # L2-SP regularization (included in Phase 1 backward)
        l2sp_val = 0.0
        loss_phase1 = L_diffusion
        if l2sp_reg is not None:
            l2sp_loss = l2sp_reg.compute()
            l2sp_val = l2sp_loss.item()
            loss_phase1 = loss_phase1 + l2sp_loss

        # NaN check
        if torch.isnan(L_diffusion):
            skipped_nan += 1
            del L_diffusion, eps_pred
            gc.collect()
            torch.cuda.empty_cache()
            continue

        # Save eps_pred values for CLIP computation (before freeing UNet graph)
        eps_pred_vals = eps_pred.detach()

        # ── Phase 1: backward L_diffusion + L2-SP (frees UNet graph) ──
        g_diff_norm_val = 0.0
        g_clip_norm_val = 0.0
        if _do_phase_timing:
            torch.cuda.synchronize()
            _tp1_start = time.time()
        loss_phase1.backward()
        del loss_phase1, L_diffusion, eps_pred
        if _do_phase_timing:
            torch.cuda.synchronize()
            _vram_free, _vram_total = torch.cuda.mem_get_info()
            _tp1_end = time.time()
            print(f"    [Phase1] backward L_diff: {(_tp1_end-_tp1_start)*1000:.0f}ms  "
                  f"VRAM: {(_vram_total-_vram_free)/1e9:.1f}/{_vram_total/1e9:.1f} GiB", flush=True)

        # Diagnostic C: measure L_diff gradient norm (before CLIP contribution)
        # Cheap — just reads existing .grad tensors, no extra forward/backward
        g_diff_norm_val = torch.nn.utils.clip_grad_norm_(
            trainable_params, max_norm=float('inf'),
        ).item()

        torch.cuda.empty_cache()

        # ── Phase 2: CLIP consistency (UNet graph freed, ~8 GiB available) ──
        if _do_phase_timing:
            torch.cuda.synchronize()
            _tp2_start = time.time()
        if is_cond.any():
            batch_meta = {
                "clip_flip_h": batch.get("clip_flip_h", torch.zeros(B)),
                "clip_flip_v": batch.get("clip_flip_v", torch.zeros(B)),
                "clip_rotation": batch.get("clip_rotation", torch.zeros(B)),
                "orig_h": batch.get("orig_h", torch.full((B,), 512.0)),
                "orig_w": batch.get("orig_w", torch.full((B,), 512.0)),
                "clip_crop_top": batch.get("clip_crop_top", torch.zeros(B)),
                "clip_crop_left": batch.get("clip_crop_left", torch.zeros(B)),
                "clip_crop_h": batch.get("clip_crop_h", torch.full((B,), 224.0)),
                "clip_crop_w": batch.get("clip_crop_w", torch.full((B,), 224.0)),
                "clip_resized": batch.get("clip_resized", torch.zeros(B)),
                "ref_01_1": (references + 1.0) / 2.0,
                "patch_mask_1": make_patch_mask(
                    batch["clip_mask"].float() if "clip_mask" in batch
                    else torch.ones(B, 1, 224, 224),
                ),
            }
            if multi_crop:
                batch_meta["clip_crop_top_2"] = batch.get("clip_crop_top_2", torch.zeros(B))
                batch_meta["clip_crop_left_2"] = batch.get("clip_crop_left_2", torch.zeros(B))
                batch_meta["clip_crop_h_2"] = batch.get("clip_crop_h_2", torch.zeros(B))
                batch_meta["clip_crop_w_2"] = batch.get("clip_crop_w_2", torch.zeros(B))
                batch_meta["clip_resized_2"] = batch.get("clip_resized_2", torch.zeros(B))
                batch_meta["group_valid"] = group_valid
                if references_2 is not None:
                    batch_meta["ref_01_2"] = (references_2 + 1.0) / 2.0
                    batch_meta["patch_mask_2"] = make_patch_mask(
                        batch["clip_mask_2"].float() if "clip_mask_2" in batch
                        else torch.ones(B, 1, 224, 224),
                    )

            L_clip_val, clip_grad_eps, clip_extras = compute_clip_consistency_loss(
                pipeline, ip_adapter,
                noisy_latents.detach(), noise.detach(), timesteps.detach(),
                eps_pred_vals,  # detached values (UNet graph already freed)
                is_cond, batch_meta, multi_crop,
                clip_grad_checkpoint=clip_grad_checkpoint,
                clip_alpha_cutoff=clip_alpha_cutoff,
            )
            L_clip_scaled_val = clip_gamma * L_clip_val
            # When gamma=0, discard grad to skip Phase 3 (expensive second UNet forward)
            if clip_gamma == 0:
                clip_grad_eps = None

        if _do_phase_timing:
            torch.cuda.synchronize()
            _vram_free, _vram_total = torch.cuda.mem_get_info()
            _tp2_end = time.time()
            print(f"    [Phase2] CLIP loss: {(_tp2_end-_tp2_start)*1000:.0f}ms  "
                  f"VRAM: {(_vram_total-_vram_free)/1e9:.1f}/{_vram_total/1e9:.1f} GiB", flush=True)

        # ── Phase 3: second UNet forward + CLIP gradient injection ──
        # Re-encode and re-run UNet to get a fresh computation graph for
        # propagating the CLIP gradient to IP-Adapter and T2I-Adapter params.
        # Guard: skip if clip_grad_eps contains NaN/Inf (numerical issues in VAE/CLIP)
        if clip_grad_eps is not None and not torch.isfinite(clip_grad_eps).all():
            print(f"  [WARN step {step}] clip_grad_eps has NaN/Inf, skipping CLIP gradient", flush=True)
            clip_grad_eps = None
        if clip_grad_eps is not None:
            torch.cuda.empty_cache()
            if _do_phase_timing:
                torch.cuda.synchronize()
                _tp3_start = time.time()
            with torch.amp.autocast('cuda'):
                # Re-encode reference (trainable parts need fresh graph)
                ip_image_embeds_2 = ip_adapter.encode_image(
                    ref_01, mask=clip_masks, core_mask=clip_core_masks,
                )
                null_token_mask_2 = None
                if references_2 is not None:
                    ip_embeds_2_crop2 = ip_adapter.encode_image(
                        ref_2_01, mask=clip_masks_2, core_mask=clip_core_masks_2,
                    )
                    K2 = ip_image_embeds_2.shape[1]
                    ip_image_embeds_2 = torch.cat(
                        [ip_image_embeds_2, ip_embeds_2_crop2], dim=1,
                    )
                    if group_valid is not None:
                        m1 = group_valid[:, 0:1].expand(-1, K2)
                        m2 = group_valid[:, 1:2].expand(-1, K2)
                        null_token_mask_2 = torch.cat([m1, m2], dim=1)

                cross_attn_kwargs_2 = {
                    "ip_adapter_image_embeds": ip_image_embeds_2,
                }
                if binary_cross_attn_mask:
                    cross_attn_kwargs_2["ip_adapter_mask"] = dilated_binary_64
                else:
                    cross_attn_kwargs_2["ip_adapter_mask"] = alpha_map_64
                if null_token_mask_2 is not None:
                    cross_attn_kwargs_2["null_token_mask"] = null_token_mask_2

                t2i_kwargs_2 = {}
                if t2i_adapter is not None:
                    t2i_input_2 = torch.cat([core_mask_64, band_mask_64], dim=1)
                    t2i_features_2 = t2i_adapter(
                        t2i_input_2, mask=dilated_binary_64,
                    )
                    t2i_kwargs_2 = t2i_adapter.prepare_unet_kwargs(t2i_features_2)

                eps_pred_2 = pipeline.unet(
                    model_input, timesteps,
                    encoder_hidden_states=text_emb,
                    cross_attention_kwargs=cross_attn_kwargs_2,
                    **t2i_kwargs_2,
                ).sample.float()

            # Proxy tensor: grad w.r.t. eps_pred_2 = clip_grad_eps
            clip_proxy = (eps_pred_2 * clip_grad_eps).sum()
            (clip_gamma * clip_proxy).backward()

            # Diagnostic C: CLIP gradient magnitude (in eps_pred space)
            g_clip_norm_val = (clip_gamma * clip_grad_eps.norm()).item()

            del eps_pred_2, clip_proxy, clip_grad_eps
            if _do_phase_timing:
                torch.cuda.synchronize()
                _vram_free, _vram_total = torch.cuda.mem_get_info()
                _tp3_end = time.time()
                print(f"    [Phase3] UNet2+clip_proxy backward: {(_tp3_end-_tp3_start)*1000:.0f}ms  "
                      f"VRAM: {(_vram_total-_vram_free)/1e9:.1f}/{_vram_total/1e9:.1f} GiB", flush=True)

        del eps_pred_vals
        _t_fwd = time.time()

        torch.cuda.synchronize()
        _t_bwd = time.time()
        _raw_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        grad_norm = _raw_norm if math.isfinite(_raw_norm) else 0.0
        optimizer.step()
        _t_opt = time.time()

        # Total loss: L_diff + L2-SP always; add L_clip only when computed (not NaN)
        loss_val = L_diff_val + l2sp_val
        if math.isfinite(L_clip_scaled_val):
            loss_val += L_clip_scaled_val
        losses.append(loss_val)

        loss_fh.write(f"{loss_val}\n")
        loss_fh.flush()

        # ── Aggregate diagnostics ──
        extras = {}

        # A/F/H: attention norms (aggregated across layers)
        ip_attn_norm_val = 0.0
        h_pre_norm_val = 0.0
        ip_k_norm_val = 0.0
        ip_v_norm_val = 0.0
        text_k_norm_val = 0.0
        text_v_norm_val = 0.0
        ip_entropy_val = 0.0
        if diagnostics:
            n_layers = 0
            for key, val in diagnostics.items():
                if key.endswith("/ip_out_norm"):
                    ip_attn_norm_val += val; n_layers += 1
                elif key.endswith("/h_pre_norm"):
                    h_pre_norm_val += val
                elif key.endswith("/ip_k_norm"):
                    ip_k_norm_val += val
                elif key.endswith("/ip_v_norm"):
                    ip_v_norm_val += val
                elif key.endswith("/text_k_norm"):
                    text_k_norm_val += val
                elif key.endswith("/text_v_norm"):
                    text_v_norm_val += val
                elif key.endswith("/ip_entropy"):
                    ip_entropy_val += val
            if n_layers > 0:
                ip_attn_norm_val /= n_layers
                h_pre_norm_val /= n_layers
                ip_k_norm_val /= n_layers
                ip_v_norm_val /= n_layers
                text_k_norm_val /= n_layers
                text_v_norm_val /= n_layers
                ip_entropy_val /= n_layers

        # B: Feature drift
        feature_drift_val = 0.0
        if (step % drift_interval == 0 or is_last_step) and step > 0:
            try:
                feature_drift_val = compute_feature_drift(
                    pipeline, ip_adapter, model_input.detach(),
                    timesteps.detach(), text_emb.detach(),
                    ip_image_embeds.detach(), alpha_map_64.detach(),
                    {k: [f.detach() for f in v] if isinstance(v, list) else v
                     for k, v in t2i_kwargs.items()},
                    null_token_mask.detach() if null_token_mask is not None else None,
                )
            except RuntimeError:
                pass  # OOM — skip this diagnostic

        # D: Cosine sim vs timestep bucket
        if clip_extras.get("cos_sim_anomaly", 0) > 0:
            for i_b in range(B):
                if is_cond[i_b]:
                    bucket = min(int(timesteps[i_b].item() / 100), 9)
                    cos_sim_buckets[bucket].append(clip_extras["cos_sim_anomaly"])

        # Gate values
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

        # Write stats CSV
        stats_fh.write(
            f"{step},{loss_val},{L_diff_val},{L_clip_val},{L_clip_scaled_val},"
            f"{core_loss_val},{band_loss_val},{grad_norm},{attn_gate},{ff_gate},"
            f"{lr_pre},{lr_scr},{l2sp_val},{progress},"
            f"{clip_extras.get('cos_sim_anomaly', 0.0)},{clip_extras.get('cos_sim_all', 0.0)},"
            f"{ip_attn_norm_val},{h_pre_norm_val},"
            f"{ip_k_norm_val},{ip_v_norm_val},{text_k_norm_val},{text_v_norm_val},"
            f"{ip_entropy_val},{feature_drift_val},"
            f"{g_diff_norm_val},{g_clip_norm_val}\n"
        )
        stats_fh.flush()

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(loss=f"{avg:.4f}", clip=f"{L_clip_val:.4f}")

        # Timing diagnostics (first 20 steps)
        if step - start_step < 20:
            _t_end = time.time()
            print(f"  [TIMING step {step}] data={(_t_data-_t0)*1000:.0f}ms "
                  f"transfer={(_t_transfer-_t_data)*1000:.0f}ms "
                  f"fwd={(_t_fwd-_t_transfer)*1000:.0f}ms "
                  f"bwd={(_t_bwd-_t_fwd)*1000:.0f}ms "
                  f"opt={(_t_opt-_t_bwd)*1000:.0f}ms "
                  f"TOTAL={(_t_end-_t0)*1000:.0f}ms", flush=True)

        # Free CUDA tensors (eps_pred, L_diffusion already freed in Phase 1)
        del images, masks, references, clip_masks, clip_core_masks
        del references_2, clip_masks_2, clip_core_masks_2, group_valid
        del model_input, noisy_latents, latents, noise

        # Early sample snapshots
        if (step + 1) in (500, 1000, 2000) and (step + 1) % save_every != 0:
            torch.cuda.empty_cache()
            try:
                generate_samples(
                    pipeline, ip_adapter, full_dataset,
                    save_dir / f"samples_{step + 1}.png", device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
                )
            except RuntimeError as e:
                print(f"  Warning: sample generation failed ({e}), continuing training...")
            torch.cuda.empty_cache()
            save_loss_plot(losses, stats_file, save_dir / f"loss_{step + 1}.png")

        # Checkpoints + samples
        if (step + 1) % save_every == 0:
            print(f"\n  Checkpoint at step {step + 1}...", flush=True)
            ckpt_dir = save_dir / f"checkpoint_{step + 1}"
            save_checkpoint(
                ip_adapter, optimizer, ckpt_dir,
                band_mode=band_mode,
                t2i_adapter=t2i_adapter,
                step=step + 1,
                losses=losses,
                unet=pipeline.unet if lora_rank > 0 else None,
                qo_params_named=qo_params_named if qo_params_named else None,
            )
            # Delete previous checkpoint (keep only latest + final)
            if prev_ckpt_dir is not None and prev_ckpt_dir.exists():
                import shutil
                shutil.rmtree(prev_ckpt_dir)
            prev_ckpt_dir = ckpt_dir
            torch.cuda.empty_cache()
            try:
                generate_samples(
                    pipeline, ip_adapter, full_dataset,
                    save_dir / f"samples_{step + 1}.png", device,
                    band_mode=band_mode, t2i_adapter=t2i_adapter,
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

    save_checkpoint(
        ip_adapter, optimizer,
        save_dir / "checkpoint_final",
        band_mode=band_mode,
        t2i_adapter=t2i_adapter,
        step=n_steps,
        losses=losses,
        unet=pipeline.unet if lora_rank > 0 else None,
        qo_params_named=qo_params_named if qo_params_named else None,
    )

    torch.cuda.empty_cache()
    try:
        generate_samples(
            pipeline, ip_adapter, full_dataset,
            save_dir / "final_samples.png", device,
            band_mode=band_mode, t2i_adapter=t2i_adapter,
        )
    except RuntimeError as e:
        print(f"  Warning: final samples failed ({e})")

    print(f"\nResults saved to: {save_dir}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CLIP Consistency Training — standard diffusion + CLIP consistency loss",
    )
    project_root = Path(__file__).parent.parent

    # CLIP consistency
    parser.add_argument("--clip-gamma", type=float, default=1.0,
                        help="Weight for CLIP consistency loss (default: 1.0)")
    parser.add_argument("--no-clip-grad-checkpoint", action="store_true",
                        help="Disable gradient checkpointing on CLIP encoder")
    parser.add_argument("--clip-alpha-cutoff", type=float, default=0.5,
                        help="Skip CLIP loss for samples with alpha_bar < cutoff (0=no cutoff)")

    # Diagnostics
    parser.add_argument("--diag-interval", type=int, default=50,
                        help="Attention diagnostics (A/F/H) frequency (default: 50)")
    parser.add_argument("--drift-interval", type=int, default=500,
                        help="Feature drift (B) frequency (default: 500)")
    parser.add_argument("--grad-split-interval", type=int, default=100,
                        help="Gradient split (C) frequency (default: 100)")

    # Standard training
    parser.add_argument("--splits-dir", type=str, default="data/concepts")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--captions-file", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="results/clip_consistency")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-pretrained", type=float, default=1e-4)
    parser.add_argument("--lambda-sp", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=10)

    # Conditioning dropout
    parser.add_argument("--drop-image-prob", type=float, default=0.10)
    parser.add_argument("--drop-text-prob", type=float, default=0.10)
    parser.add_argument("--drop-both-prob", type=float, default=0.05)

    # IP-Adapter
    parser.add_argument("--ip-adapter-type", type=str, default="plus")
    parser.add_argument("--ip-adapter-k", type=int, default=16)
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--no-mask-visual", action="store_true")
    parser.add_argument("--visual-mode", type=int, default=3)
    parser.add_argument("--no-learnable-gates", action="store_true")
    parser.add_argument("--force-gates", action="store_true")
    parser.add_argument("--sa-num-layers", type=int, default=3)
    parser.add_argument("--sa-num-heads", type=int, default=12)

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=0,
                        help="LoRA rank for UNet attention layers (0=disabled)")
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-lr", type=float, default=5e-5)
    parser.add_argument("--lora-mode", type=str, default="mid_up",
                        choices=["all", "cross", "mid_up"])

    # UNet unfreezing
    parser.add_argument("--unfreeze-qo", type=str, default="",
                        choices=["", "mid_up", "all_cross", "all"])
    parser.add_argument("--unfreeze-qo-lr", type=float, default=1e-5)

    # Data
    parser.add_argument("--no-multi-crop", action="store_true")
    parser.add_argument("--no-clip-align", action="store_true")
    parser.add_argument("--band-mode", type=int, default=2)
    parser.add_argument("--loss-core-ratio", type=float, default=0.8)
    parser.add_argument("--binary-cross-attn-mask", action="store_true")
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"])
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--types", type=str, nargs="+", default=None)
    parser.add_argument("--exclude-sources", type=str, nargs="+", default=None)
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--timestep-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--logit-normal-mean", type=float, default=0.0)
    parser.add_argument("--logit-normal-std", type=float, default=1.0)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-live-viewer", action="store_true")
    parser.add_argument("--pilot-subset-path", type=str, default=None)
    parser.add_argument("--pilot-subset-size", type=int, default=0)

    args = parser.parse_args()

    # Resolve paths
    if not os.path.isabs(args.splits_dir):
        args.splits_dir = str(project_root / args.splits_dir)
    if args.data_root and not os.path.isabs(args.data_root):
        args.data_root = str(project_root / args.data_root)
    if not os.path.isabs(args.captions_file):
        args.captions_file = str(project_root / args.captions_file)
    if not os.path.isabs(args.save_dir):
        args.save_dir = str(project_root / args.save_dir)

    train_clip_consistency(
        splits_dir=Path(args.splits_dir),
        save_dir=Path(args.save_dir),
        captions_file=Path(args.captions_file),
        clip_gamma=args.clip_gamma,
        clip_grad_checkpoint=not args.no_clip_grad_checkpoint,
        clip_alpha_cutoff=args.clip_alpha_cutoff,
        diag_interval=args.diag_interval,
        drift_interval=args.drift_interval,
        grad_split_interval=args.grad_split_interval,
        n_steps=args.steps,
        batch_size=args.batch_size,
        save_every=args.save_every,
        lr=args.lr,
        lr_pretrained=args.lr_pretrained,
        lambda_sp=args.lambda_sp,
        drop_image_prob=args.drop_image_prob,
        drop_text_prob=args.drop_text_prob,
        drop_both_prob=args.drop_both_prob,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        mask_visual=not args.no_mask_visual,
        visual_mode=args.visual_mode,
        learnable_gates=not args.no_learnable_gates,
        force_gates=args.force_gates,
        sa_num_layers=args.sa_num_layers,
        sa_num_heads=args.sa_num_heads,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_lr=args.lora_lr,
        lora_mode=args.lora_mode,
        unfreeze_qo=args.unfreeze_qo,
        unfreeze_qo_lr=args.unfreeze_qo_lr,
        data_root=Path(args.data_root) if args.data_root else None,
        anomaly_types=args.types,
        exclude_sources=args.exclude_sources,
        augment=args.augment,
        multi_crop=not args.no_multi_crop,
        clip_align=not args.no_clip_align,
        band_mode=args.band_mode,
        loss_core_ratio=args.loss_core_ratio,
        binary_cross_attn_mask=args.binary_cross_attn_mask,
        t2i_adapter_mode=args.t2i_adapter_mode,
        noise_offset=args.noise_offset,
        timestep_sampling=args.timestep_sampling,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        resume_dir=args.resume,
        no_live_viewer=args.no_live_viewer,
        pilot_subset_path=args.pilot_subset_path,
        pilot_subset_size=args.pilot_subset_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
