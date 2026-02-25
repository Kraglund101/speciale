"""
Shared mask utilities for dilation and soft loss weighting.

Two distinct dilation contexts:
1. CLIP masked self-attention: tight dilation (min_r=2, max_r=10)
   → maximize anomaly signal in visual token extraction
2. UNet latent-space band dilation (1-2 Chebyshev pixels at 64×64)
   → define regeneration region + route visual info to transition zone

Band-based loss: core=1.0, band=flat alpha weights, bg=0.0
Ratio mechanism fixes gradient split (e.g., 80% core / 20% band).
Pure PyTorch — no scipy dependency.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def adaptive_dilation_radius(mask: torch.Tensor, min_r: int = 2, max_r: int = 10) -> int:
    """Compute adaptive dilation radius: max(min_r, min(max_r, 0.5 * sqrt(area))).

    Args:
        mask: Binary mask [1, H, W] or [B, 1, H, W]
        min_r: Minimum dilation radius in pixels
        max_r: Maximum dilation radius in pixels

    Returns:
        Dilation radius in pixels at mask resolution
    """
    area = mask.sum().item()
    if area == 0:
        return 0
    r = 0.5 * math.sqrt(area)
    return max(min_r, min(max_r, int(r)))


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation of binary mask via max pooling (square structuring element).

    Args:
        mask: Binary mask [1, H, W] or [B, 1, H, W]
        radius: Dilation radius in pixels

    Returns:
        Dilated binary mask, same shape as input
    """
    if radius <= 0:
        return mask

    had_batch = mask.dim() == 4
    if not had_batch:
        mask = mask.unsqueeze(0)  # [1, 1, H, W]

    kernel = 2 * radius + 1
    padded = F.pad(mask, [radius] * 4, mode="constant", value=0)
    dilated = F.max_pool2d(padded, kernel_size=kernel, stride=1)
    dilated = (dilated > 0.5).float()

    if not had_batch:
        dilated = dilated.squeeze(0)  # [1, H, W]

    return dilated


def dilate_mask_batch(masks: torch.Tensor, min_r: int, max_r: int) -> torch.Tensor:
    """Dilate a batch of masks with per-sample adaptive radius.

    Args:
        masks: Binary masks [B, 1, H, W]
        min_r: Minimum dilation radius
        max_r: Maximum dilation radius

    Returns:
        Dilated masks [B, 1, H, W]
    """
    batch_size = masks.shape[0]
    dilated = torch.zeros_like(masks)
    for i in range(batch_size):
        r = adaptive_dilation_radius(masks[i], min_r=min_r, max_r=max_r)
        dilated[i] = dilate_mask(masks[i], r)
    return dilated


def compute_soft_loss_mask(
    core_mask: torch.Tensor,
    dilated_mask: torch.Tensor,
    decay: str = "gaussian",
    min_weight: float = 0.1,
) -> torch.Tensor:
    """Compute soft loss weight mask: core=1.0, ring=decaying, bg=0.0.

    Operates at latent resolution (e.g. 64x64). Uses distance transform
    from core boundary to compute decay within the dilation ring.

    Args:
        core_mask: Binary core mask [B, 1, H, W] at latent resolution
        dilated_mask: Binary dilated mask [B, 1, H, W] at latent resolution
        decay: "gaussian", "linear", or "cosine"
        min_weight: Minimum weight for any ring pixel (prevents zero-weight)

    Returns:
        Soft weight mask [B, 1, H, W] with values in [min_weight, 1.0] for
        masked pixels, 0.0 for background
    """
    batch_size = core_mask.shape[0]
    device = core_mask.device
    soft = torch.zeros_like(core_mask)

    for i in range(batch_size):
        core_i = core_mask[i, 0]   # [H, W]
        dil_i = dilated_mask[i, 0]  # [H, W]
        ring_i = (dil_i - core_i).clamp(0, 1)

        # Core always gets weight 1.0
        weight_i = core_i.clone()

        if ring_i.sum() > 0 and core_i.sum() > 0:
            # Distance transform from core boundary (on CPU for scipy)
            core_np = core_i.cpu().numpy()
            from scipy import ndimage
            dist = ndimage.distance_transform_edt(1 - core_np)
            dist_t = torch.from_numpy(dist).to(device).float()

            # Normalize: 0 at core edge, 1 at ring edge
            ring_dist = dist_t * ring_i
            max_dist = ring_dist.max()
            if max_dist > 0:
                norm_dist = (dist_t / max_dist).clamp(0, 1)
            else:
                norm_dist = torch.zeros_like(dist_t)

            # Compute decay
            if decay == "gaussian":
                # sigma such that weight ≈ 0.05 at ring edge
                sigma = 1.0 / math.sqrt(2 * math.log(20))
                ring_weight = torch.exp(-0.5 * (norm_dist / sigma) ** 2)
            elif decay == "linear":
                ring_weight = 1.0 - norm_dist
            elif decay == "cosine":
                ring_weight = 0.5 * (1 + torch.cos(math.pi * norm_dist))
            else:
                ring_weight = 1.0 - norm_dist

            # Apply minimum weight floor
            ring_weight = ring_weight.clamp(min=min_weight)

            # Only apply to ring pixels
            weight_i = weight_i + ring_i * ring_weight
            weight_i = weight_i.clamp(0, 1)
        elif ring_i.sum() > 0:
            # No core but has ring — give flat min_weight
            weight_i = weight_i + ring_i * min_weight

        soft[i, 0] = weight_i

    return soft


def compute_ratio_loss_mask(
    core_mask: torch.Tensor,
    dilated_mask: torch.Tensor,
    core_ratio: float = 0.9,
    min_weight: float = 0.01,
) -> torch.Tensor:
    """Compute loss weight mask with fixed core/ring gradient ratio.

    Total gradient split is fixed: e.g. core_ratio=0.9 means 90% of aggregate
    gradient goes to core, 10% to ring. Ring weights are gaussian-shaped
    (distance-based) but scaled so their sum hits the target ratio.

    Formula: ring_total = n_core * (1 - core_ratio) / core_ratio
    Then per-ring-cell weight = gaussian_shape * (ring_total / sum_of_gaussian).

    Args:
        core_mask: Binary core mask [B, 1, H, W] at latent resolution
        dilated_mask: Binary dilated mask [B, 1, H, W] at latent resolution
        core_ratio: Fraction of total gradient going to core (e.g. 0.9 = 90%)
        min_weight: Absolute minimum weight per ring cell

    Returns:
        Weight mask [B, 1, H, W] with core=1.0, ring=scaled gaussian, bg=0.0
    """
    from scipy import ndimage

    batch_size = core_mask.shape[0]
    device = core_mask.device
    soft = torch.zeros_like(core_mask)

    for i in range(batch_size):
        core_i = core_mask[i, 0]   # [H, W]
        dil_i = dilated_mask[i, 0]  # [H, W]
        ring_i = (dil_i - core_i).clamp(0, 1)

        # Core always gets weight 1.0
        weight_i = core_i.clone()

        n_core = core_i.sum().item()
        n_ring = ring_i.sum().item()

        if n_core > 0 and n_ring > 0:
            # Target total ring weight
            ring_total_target = n_core * (1 - core_ratio) / core_ratio

            # Distance transform from core boundary
            core_np = core_i.cpu().numpy()
            dist = ndimage.distance_transform_edt(1 - core_np)
            dist_t = torch.from_numpy(dist).to(device).float()

            # Normalize distances within ring
            ring_dist = dist_t * ring_i
            max_dist = ring_dist.max()
            if max_dist > 0:
                norm_dist = (dist_t / max_dist).clamp(0, 1)
            else:
                norm_dist = torch.zeros_like(dist_t)

            # Gaussian shape (unnormalized)
            sigma = 1.0 / math.sqrt(2 * math.log(20))
            raw_weights = torch.exp(-0.5 * (norm_dist / sigma) ** 2)

            # Scale so ring sum = ring_total_target
            raw_ring_sum = (raw_weights * ring_i).sum()
            if raw_ring_sum > 0:
                scale = ring_total_target / raw_ring_sum
                ring_weights = raw_weights * scale
            else:
                ring_weights = torch.full_like(raw_weights, ring_total_target / n_ring)

            # Clamp to minimum
            ring_weights = ring_weights.clamp(min=min_weight)

            weight_i = weight_i + ring_i * ring_weights
        elif n_ring > 0:
            # No core but has ring — flat min_weight
            weight_i = weight_i + ring_i * min_weight

        soft[i, 0] = weight_i

    return soft


def unet_roundtrip_masks(
    mask: torch.Tensor,
    band_mode: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute UNet-resolution masks roundtripped to original resolution.

    Pipeline: mask → maxpool to 512 → maxpool to 64 → dilate → nearest to 512 → nearest to (H,W)

    This ensures the CLIP masked self-attention sees the SAME region that the
    UNet must reconstruct (core + band), eliminating the "blind zone" where
    UNet needs to reconstruct pixels that CLIP never saw.

    Args:
        mask: Binary mask [1, H, W] in {0, 1}
        band_mode: 1 = 1px Chebyshev dilation, 2 = 2px (default 2)

    Returns:
        (core_native, dilated_native) both [1, H, W] binary float tensors.
        core_native: the core anomaly region after UNet quantization roundtrip.
        dilated_native: core + band region after UNet quantization + dilation roundtrip.
    """
    _, H, W = mask.shape

    # Down to 512 (SD 1.5 input resolution)
    mask_512 = downsample_mask_maxpool(mask, 512)  # [1, 512, 512]

    # Down to 64 (latent resolution) — this is where UNet operates
    core_64 = downsample_mask_maxpool(mask_512.unsqueeze(0), 64)  # [1, 1, 64, 64]

    # Chebyshev dilation at latent resolution (same as create_latent_band_mask)
    dilated_64 = core_64
    for _ in range(band_mode):
        dilated_64 = F.max_pool2d(dilated_64, kernel_size=3, stride=1, padding=1)
    dilated_64 = (dilated_64 > 0.5).float()
    core_64 = (core_64 > 0.5).float()

    # Nearest upsample back: 64 → 512 → (H, W)
    core_512 = F.interpolate(core_64, size=(512, 512), mode='nearest')
    dilated_512 = F.interpolate(dilated_64, size=(512, 512), mode='nearest')

    core_native = F.interpolate(core_512, size=(H, W), mode='nearest').squeeze(0)  # [1, H, W]
    dilated_native = F.interpolate(dilated_512, size=(H, W), mode='nearest').squeeze(0)  # [1, H, W]

    return core_native, dilated_native


def dilate_mask_latent(
    mask: torch.Tensor, radius: int, target_size: int = 64
) -> torch.Tensor:
    """Downsample mask to latent resolution, then dilate by N latent pixels.

    Order: maxpool 512→target_size, then iterative 8-connected dilation
    (grow all 8 neighbours, repeated ``radius`` times).

    Args:
        mask: Binary mask [B, 1, H, W] or [1, H, W] at image resolution
        radius: Dilation radius in latent pixels (e.g. 1 or 2)
        target_size: Latent spatial resolution (default 64)

    Returns:
        Dilated mask at latent resolution [B, 1, target_size, target_size]
    """
    core = downsample_mask_maxpool(mask, target_size)
    if radius <= 0:
        return core
    # Iterative 8-connected dilation: grow by 1 cell in all directions, N times
    dilated = core
    for _ in range(radius):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
    return (dilated > 0.5).float()


def downsample_mask_maxpool(mask: torch.Tensor, target_size: int) -> torch.Tensor:
    """Downsample mask to target resolution via max pooling.

    Preserves small anomalies — any anomaly pixel in a cell marks the whole cell.
    Handles arbitrary input resolutions by nearest-upscaling to the next exact
    multiple of target_size before maxpooling (nearest upscale of binary masks
    is lossless pixel replication).

    Args:
        mask: Binary mask [B, 1, H, W] or [1, H, W]
        target_size: Target spatial resolution (e.g., 64 for latent space, 512)

    Returns:
        Downsampled binary mask at [B, 1, target_size, target_size]
    """
    had_no_batch = mask.dim() == 3
    if had_no_batch:
        mask = mask.unsqueeze(0)

    _, _, h, w = mask.shape

    # Use ceil so non-exact ratios (e.g. 800->512) upscale to the next exact
    # multiple first, then maxpool.  Nearest upscale is lossless for binary masks.
    kernel_h = max(1, -(-h // target_size))  # ceil division
    kernel_w = max(1, -(-w // target_size))  # ceil division

    # Resize to exact multiple of target_size (upscale if needed)
    pool_h = target_size * kernel_h
    pool_w = target_size * kernel_w
    if h != pool_h or w != pool_w:
        mask = F.interpolate(mask, size=(pool_h, pool_w), mode="nearest")

    pooled = F.max_pool2d(mask, kernel_size=(kernel_h, kernel_w))
    pooled = (pooled > 0.5).float()

    if had_no_batch:
        pooled = pooled.squeeze(0)

    return pooled


def create_latent_band_mask(
    core_mask: torch.Tensor,
    band_mode: int = 1,
    core_ratio: float = 0.8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create latent-space band masks via Chebyshev dilation (max_pool2d).

    Returns 4 tensors at 64×64 latent resolution:

    - dilated_binary: core + band = 1, bg = 0
        Used for UNet inpainting conditional channel and T2I-Adapter mask.
    - alpha_map: core = 1.0, band = fractional alpha, bg = 0.0
        Used for inference blending and soft cross-attention routing.
    - weight_map: core = 1.0, band = ratio-scaled, bg = 0.0
        Used for training loss weighting (core_ratio controls gradient split).
    - band_mask: band = 1, else = 0
        Used as channel 1 in T2I-Adapter 2-channel input.

    Band creation uses Chebyshev distance (8-connected dilation via max_pool2d):
    - Mode 1: 1-pixel band. Alpha = 0.5.
    - Mode 2: 2-pixel band (inner + outer). Inner alpha = 2/3, outer alpha = 1/3.

    Weight map uses ratio scaling so aggregate gradient is exactly
    ``core_ratio`` for core and ``1 - core_ratio`` for band.

    Args:
        core_mask: Binary mask [B, 1, 64, 64] (maxpooled from image-space).
        band_mode: 1 (single 1-pixel band) or 2 (inner + outer band).
        core_ratio: Fraction of total gradient going to core (e.g. 0.8 = 80%).

    Returns:
        (dilated_binary, alpha_map, weight_map, band_mask) each [B, 1, 64, 64].
    """
    assert band_mode in (1, 2), f"band_mode must be 1 or 2, got {band_mode}"
    assert 0.0 < core_ratio < 1.0, f"core_ratio must be in (0, 1), got {core_ratio}"

    core = core_mask.float()

    if band_mode == 1:
        # Single 1-pixel Chebyshev dilation
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        dilated_1 = (dilated_1 > 0.5).float()
        band = (dilated_1 - core).clamp(0, 1)

        dilated_binary = dilated_1
        alpha_map = core + 0.5 * band
        band_mask = band

        # Weight map: ratio-scaled per sample
        weight_map = _compute_band_weights_mode1(core, band, core_ratio)

    else:  # band_mode == 2
        # Two-ring Chebyshev dilation
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        dilated_1 = (dilated_1 > 0.5).float()
        dilated_2 = F.max_pool2d(dilated_1, kernel_size=3, stride=1, padding=1)
        dilated_2 = (dilated_2 > 0.5).float()

        inner = (dilated_1 - core).clamp(0, 1)
        outer = (dilated_2 - dilated_1).clamp(0, 1)
        band = (dilated_2 - core).clamp(0, 1)

        dilated_binary = dilated_2
        alpha_map = core + (2.0 / 3.0) * inner + (1.0 / 3.0) * outer
        band_mask = band

        # Weight map: ratio-scaled per sample
        weight_map = _compute_band_weights_mode2(core, inner, outer, core_ratio)

    return dilated_binary, alpha_map, weight_map, band_mask


def _compute_band_weights_mode1(
    core: torch.Tensor,
    band: torch.Tensor,
    core_ratio: float,
) -> torch.Tensor:
    """Compute per-sample ratio-scaled weight map for mode 1 (single band).

    Core pixels get weight 1.0. Band pixels get a uniform weight scaled so
    that ``sum(core_weights) / sum(all_weights) == core_ratio``.

    Args:
        core: [B, 1, H, W] binary core mask.
        band: [B, 1, H, W] binary band mask.
        core_ratio: Target fraction for core gradient.

    Returns:
        weight_map [B, 1, H, W]: core=1.0, band=scaled weight, bg=0.0.
    """
    B = core.shape[0]
    weight_map = core.clone()

    for i in range(B):
        n_core = core[i].sum()
        n_band = band[i].sum()

        if n_core > 0 and n_band > 0:
            # band_total_target = n_core * (1 - core_ratio) / core_ratio
            # per_band_weight = band_total_target / n_band
            band_w = (n_core * (1 - core_ratio) / core_ratio) / n_band
            weight_map[i] = weight_map[i] + band[i] * band_w

    return weight_map


def _compute_band_weights_mode2(
    core: torch.Tensor,
    inner: torch.Tensor,
    outer: torch.Tensor,
    core_ratio: float,
) -> torch.Tensor:
    """Compute per-sample ratio-scaled weight map for mode 2 (inner + outer band).

    Inner band gets 2/3 of the total band gradient budget, outer gets 1/3.

    Args:
        core: [B, 1, H, W] binary core mask.
        inner: [B, 1, H, W] binary inner band mask.
        outer: [B, 1, H, W] binary outer band mask.
        core_ratio: Target fraction for core gradient.

    Returns:
        weight_map [B, 1, H, W]: core=1.0, inner/outer=scaled weights, bg=0.0.
    """
    B = core.shape[0]
    weight_map = core.clone()

    for i in range(B):
        n_core = core[i].sum()
        n_inner = inner[i].sum()
        n_outer = outer[i].sum()

        if n_core > 0:
            band_total = n_core * (1 - core_ratio) / core_ratio

            if n_inner > 0:
                inner_w = (2.0 / 3.0 * band_total) / n_inner
                weight_map[i] = weight_map[i] + inner[i] * inner_w

            if n_outer > 0:
                outer_w = (1.0 / 3.0 * band_total) / n_outer
                weight_map[i] = weight_map[i] + outer[i] * outer_w

    return weight_map
