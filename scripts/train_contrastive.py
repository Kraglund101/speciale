"""
Contrastive CLIP-conditioning regularization training script.

Variant-batch training: packs multiple CLIP-conditioning variants (pos1, pos2,
neg, null) into a single UNet forward pass per host anchor, enabling contrastive
losses (L_inv, L_rank, L_triplet) alongside the standard diffusion loss.

This is a STANDALONE script — it does NOT modify train_anomagic.py.

Strategy A: L_diff + L_inv + L_rank
Strategy B: L_diff + L_triplet
"""
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
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


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class HostAnchor:
    """One anchor in the host batch, with pre-sampled variant indices."""
    anchor_idx: int          # index into dataset
    anchor_type: str         # "typed" or "untyped"
    group: str               # group name
    product: str             # product name
    weight: float            # sample weight from JSON
    pos2_idx: Optional[int] = None   # same-group different-product
    neg_idx: Optional[int] = None    # different-group
    neg_is_null: bool = False        # True if neg replaced by null CLIP
    neg_source: str = ""             # "same_host", "cross_host", "ip_zero"
    variant_roles: List[str] = field(default_factory=list)
    # Filled after CLIP crop loading
    caption: str = ""


@dataclass
class VariantBatch:
    """Packed tensors for a single UNet forward pass across all variants."""
    # UNet inputs [V, ...]
    model_input: torch.Tensor      # [V, 9, 64, 64]
    timesteps: torch.Tensor        # [V]
    text_emb: torch.Tensor         # [V, 77, 768]
    ip_image_embeds: torch.Tensor  # [V, K, dim]
    cross_attn_kwargs: dict
    t2i_kwargs: dict

    # Ground truth
    noise: torch.Tensor            # [V, 4, 64, 64]
    latents: torch.Tensor          # [V, 4, 64, 64] clean

    # Masks
    weight_map_64: torch.Tensor    # [V, 1, 64, 64]
    core_mask_64: torch.Tensor     # [V, 1, 64, 64]
    band_mask_64: torch.Tensor     # [V, 1, 64, 64]

    # Index mapping
    anchor_ids: List[int]          # [V] which host anchor
    anchor_types: List[str]        # [V] "typed"/"untyped"
    variant_roles: List[str]       # [V] "pos1","pos2","neg","true","null"
    anchor_groups: List[str]       # [V] group name
    sample_weights: torch.Tensor   # [V]
    gets_ldiff: List[bool] = field(default_factory=list)  # [V] which variants get L_diff
    neg_sources: List[str] = field(default_factory=list)  # [V] neg source per variant

    # For diagnostics
    null_token_mask: Optional[torch.Tensor] = None  # [V, 2K] or None


# ======================================================================
# ContrastiveHostSampler
# ======================================================================

class ContrastiveHostSampler:
    """Wraps the contrastive dataset JSON to produce structured host batches."""

    def __init__(
        self,
        data_json_path: str,
        data_root: str,
        captions_file: Optional[str] = None,
        typed_frac: float = 0.912,
    ):
        with open(data_json_path, "r", encoding="utf-8") as f:
            all_samples_raw = json.load(f)

        self.data_root = Path(data_root)
        self.typed_frac = typed_frac

        # Filter out samples with empty masks (e.g. BTech plate 0028-0037)
        # These cause UNet NaN under fp16 autocast
        _ZERO_MASK_PATHS = {
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0028.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0029.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0030.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0031.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0032.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0033.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0034.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0035.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0036.bmp",
            "AnomVerse_data_filtered/BTech/BTech/03/test/ko/0037.bmp",
        }
        self.all_samples = [s for s in all_samples_raw if s["image_path"] not in _ZERO_MASK_PATHS]
        n_filtered = len(all_samples_raw) - len(self.all_samples)
        if n_filtered > 0:
            print(f"Filtered {n_filtered} zero-mask samples")

        # Load captions
        self.captions: Dict[str, str] = {}
        if captions_file and Path(captions_file).exists():
            with open(captions_file, "r", encoding="utf-8") as f:
                captions_data = json.load(f)
            for entry in captions_data:
                key = entry.get("image_path") or entry.get("filename", "")
                self.captions[key] = entry["caption"]

        # Build indices
        self.group_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.group_product_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.product_group_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # {(product, group): [idx]}
        self.typed_indices: List[int] = []
        self.untyped_indices: List[int] = []
        self.groups: List[str] = []

        for i, s in enumerate(self.all_samples):
            g = s["group"]
            p = s["product"]
            self.group_to_indices[g].append(i)
            self.group_product_to_indices[(g, p)].append(i)
            if g != "other":
                self.product_group_to_indices[(p, g)].append(i)
            if g == "other":
                self.untyped_indices.append(i)
            else:
                self.typed_indices.append(i)

        self.groups = [g for g in self.group_to_indices if g != "other"]
        self.typed_groups = set(self.groups)

        # For cross-class pairing: products per group
        self.group_products: Dict[str, List[str]] = defaultdict(list)
        for (g, p) in self.group_product_to_indices:
            if g != "other" and p not in self.group_products[g]:
                self.group_products[g].append(p)

        # For same-host neg pairing: groups per product
        self.product_to_groups: Dict[str, List[str]] = defaultdict(list)
        for (p, g) in self.product_group_to_indices:
            if g not in self.product_to_groups[p]:
                self.product_to_groups[p].append(g)

        print(f"ContrastiveHostSampler: {len(self.all_samples)} samples, "
              f"{len(self.typed_indices)} typed ({len(self.groups)} groups), "
              f"{len(self.untyped_indices)} untyped")

    def _resolve_path(self, rel_path: str) -> str:
        """Resolve a relative image/mask path to absolute."""
        p = self.data_root / rel_path
        return str(p)

    def get_sample_info(self, idx: int) -> dict:
        """Get sample metadata by index."""
        return self.all_samples[idx]

    def sample_host_batch(
        self,
        host_batch_size: int = 4,
        p_null_typed_neg: float = 0.25,
        typed_neg_mode: str = "same_host_diff_type",
        p_neg_same_host: float = 0.6,
        p_neg_cross_host: float = 0.3,
    ) -> List[HostAnchor]:
        """Return a list of HostAnchors with pre-sampled variant indices."""
        anchors = []

        # Decide typed vs untyped for each slot
        n_typed = 0
        slot_types = []
        for _ in range(host_batch_size):
            if random.random() < self.typed_frac:
                slot_types.append("typed")
                n_typed += 1
            else:
                slot_types.append("untyped")

        # Guarantee at least 1 typed
        if n_typed == 0:
            slot_types[random.randint(0, host_batch_size - 1)] = "typed"

        for st in slot_types:
            if st == "typed":
                anchor = self._sample_typed_anchor(
                    p_null_typed_neg, typed_neg_mode,
                    p_neg_same_host=p_neg_same_host,
                    p_neg_cross_host=p_neg_cross_host,
                )
            else:
                anchor = self._sample_untyped_anchor()
            anchors.append(anchor)

        return anchors

    def _sample_typed_anchor(
        self,
        p_null_typed_neg: float,
        typed_neg_mode: str = "same_host_diff_type",
        p_neg_same_host: float = 0.6,
        p_neg_cross_host: float = 0.3,
    ) -> HostAnchor:
        """Sample a typed anchor with pos2 and neg variants.

        Args:
            typed_neg_mode: How to sample the negative reference.
                "same_host_diff_type": same product, different anomaly group
                "cross_host_diff_type": any product, different anomaly group (old behavior)
                "hybrid": try same_host first, fall back to cross_host
                "weighted": explicit probability weights for same/cross host
        """
        anchor_idx = random.choice(self.typed_indices)
        s = self.all_samples[anchor_idx]
        group = s["group"]
        product = s["product"]

        # pos2: same group, different product
        other_products = [
            p for p in self.group_products[group]
            if p != product and self.group_product_to_indices.get((group, p))
        ]
        pos2_idx = None
        if other_products:
            other_prod = random.choice(other_products)
            candidates = self.group_product_to_indices[(group, other_prod)]
            pos2_idx = random.choice(candidates)
        else:
            # Fallback: same group, any sample (including same product)
            candidates = [i for i in self.group_to_indices[group] if i != anchor_idx]
            if candidates:
                pos2_idx = random.choice(candidates)
            else:
                pos2_idx = anchor_idx  # degenerate: only 1 sample in group

        # neg: different group (or null)
        neg_is_null = random.random() < p_null_typed_neg
        neg_idx = None
        neg_source = ""
        if not neg_is_null:
            if typed_neg_mode == "weighted":
                # Weighted neg source selection — no cross_host fallback
                # If selected source fails, fall through to ip_zero (line below)
                p_sh = p_neg_same_host / (p_neg_same_host + p_neg_cross_host + 1e-8)
                if random.random() < p_sh:
                    neg_idx, neg_source = self._sample_neg_by_mode(
                        product, group, "same_host_diff_type")
                else:
                    neg_idx, neg_source = self._sample_neg_by_mode(
                        product, group, "cross_host_diff_type")
            else:
                # Legacy modes: same_host_diff_type, cross_host_diff_type, hybrid
                neg_idx, neg_source = self._sample_neg_by_mode(
                    product, group, typed_neg_mode,
                )
            if neg_idx is None:
                neg_is_null = True
                neg_source = "ip_zero"
        else:
            neg_source = "ip_zero"

        caption = self.captions.get(s["image_path"], s.get("caption_short", ""))

        return HostAnchor(
            anchor_idx=anchor_idx,
            anchor_type="typed",
            group=group,
            product=product,
            weight=s.get("weight", 1.0),
            pos2_idx=pos2_idx,
            neg_idx=neg_idx,
            neg_is_null=neg_is_null,
            neg_source=neg_source,
            variant_roles=["pos1", "pos2", "neg"],
            caption=caption,
        )

    def _sample_neg_by_mode(
        self, product: str, group: str, mode: str,
    ) -> Tuple[Optional[int], str]:
        """Sample a negative index based on typed_neg_mode.

        Returns (neg_idx, neg_source) where neg_source is "same_host" or "cross_host".
        Returns (None, "") if no valid neg found.
        """
        def _try_same_host() -> Optional[int]:
            other_groups = [
                g for g in self.product_to_groups[product]
                if g != group and self.product_group_to_indices.get((product, g))
            ]
            if other_groups:
                neg_group = random.choice(other_groups)
                return random.choice(self.product_group_to_indices[(product, neg_group)])
            return None

        def _try_cross_host() -> Optional[int]:
            other_groups = [
                g for g in self.groups
                if g != group and self.group_to_indices.get(g)
            ]
            if other_groups:
                neg_group = random.choice(other_groups)
                return random.choice(self.group_to_indices[neg_group])
            return None

        if mode == "same_host_diff_type":
            idx = _try_same_host()
            if idx is not None:
                return idx, "same_host"
            # No cross_host fallback — caller falls through to ip_zero
            return (None, "")

        elif mode == "cross_host_diff_type":
            idx = _try_cross_host()
            return (idx, "cross_host") if idx is not None else (None, "")

        elif mode == "hybrid":
            idx = _try_same_host()
            if idx is not None:
                return idx, "same_host"
            # No cross_host fallback — caller falls through to ip_zero
            return (None, "")

        else:
            raise ValueError(f"Unknown typed_neg_mode: {mode}")

    def _sample_untyped_anchor(self) -> HostAnchor:
        """Sample an untyped anchor with true + null variants."""
        anchor_idx = random.choice(self.untyped_indices)
        s = self.all_samples[anchor_idx]
        caption = self.captions.get(s["image_path"], s.get("caption_short", ""))

        return HostAnchor(
            anchor_idx=anchor_idx,
            anchor_type="untyped",
            group=s["group"],
            product=s["product"],
            weight=s.get("weight", 1.0),
            variant_roles=["true", "null"],
            caption=caption,
        )


# ======================================================================
# Pilot subset generation
# ======================================================================

def generate_pilot_subset(
    sampler: ContrastiveHostSampler,
    target_size: int = 3000,
    min_samples_per_product: int = 10,
    seed: int = 42,
) -> List[int]:
    """Generate a balanced pilot subset for fast iteration.

    Filters to groups with 2+ products having >= min_samples_per_product,
    then samples balanced across products. Includes proportional untyped samples.
    """
    rng = random.Random(seed)

    # Find viable groups: 2+ products with enough samples
    viable_groups: Dict[str, List[str]] = {}
    for g in sampler.groups:
        viable_prods = []
        for p in sampler.group_products[g]:
            if len(sampler.group_product_to_indices[(g, p)]) >= min_samples_per_product:
                viable_prods.append(p)
        if len(viable_prods) >= 2:
            viable_groups[g] = viable_prods

    if not viable_groups:
        print("WARNING: No viable groups for pilot subset, using all typed indices")
        return sampler.typed_indices[:target_size]

    # Budget allocation
    typed_budget = int(target_size * sampler.typed_frac)
    untyped_budget = target_size - typed_budget
    per_group = typed_budget // len(viable_groups)

    indices = []
    for g, prods in viable_groups.items():
        per_prod = per_group // len(prods)
        for p in prods:
            pool = sampler.group_product_to_indices[(g, p)]
            n = min(per_prod, len(pool))
            indices.extend(rng.sample(pool, n))

    # Untyped
    if sampler.untyped_indices and untyped_budget > 0:
        n_ut = min(untyped_budget, len(sampler.untyped_indices))
        indices.extend(rng.sample(sampler.untyped_indices, n_ut))

    rng.shuffle(indices)
    print(f"Pilot subset: {len(indices)} samples from {len(viable_groups)} groups")
    return indices


# ======================================================================
# Image/mask loading helpers
# ======================================================================

def _load_image_mask(
    sample_info: dict,
    data_root: Path,
    image_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load image and mask from sample info, resize to image_size.

    Returns:
        image: [3, H, W] in [-1, 1]
        mask: [1, H, W] in {0, 1}
    """
    img_path = data_root / sample_info["image_path"]
    mask_path = data_root / sample_info["mask_path"]

    try:
        img_pil = Image.open(img_path).convert("RGB").resize(
            (image_size, image_size), Image.LANCZOS,
        )
    except Exception:
        img_pil = Image.new("RGB", (image_size, image_size), (128, 128, 128))

    try:
        mask_pil = Image.open(mask_path).convert("L").resize(
            (image_size, image_size), Image.NEAREST,
        )
    except Exception:
        mask_pil = Image.new("L", (image_size, image_size), 0)

    img_np = np.array(img_pil).astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]

    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    mask = torch.from_numpy(mask_np).unsqueeze(0)  # [1, H, W]
    mask = (mask > 0.5).float()

    return image, mask


def _load_clip_reference(
    sample_info: dict,
    data_root: Path,
    band_mode: int = 2,
    multi_crop: bool = True,
    clip_align: bool = True,
    crop_size: int = 224,
) -> dict:
    """Load and prepare CLIP reference crop(s) for a sample.

    Returns dict with keys matching AnomalyDataset output:
        reference, clip_mask, clip_core_mask, [reference_2, clip_mask_2, ...]
    """
    import torchvision.transforms.functional as TF

    img_path = data_root / sample_info["image_path"]
    mask_path = data_root / sample_info["mask_path"]

    try:
        img_pil = Image.open(img_path).convert("RGB")
    except Exception:
        img_pil = Image.new("RGB", (512, 512), (128, 128, 128))

    try:
        mask_pil = Image.open(mask_path).convert("L")
    except Exception:
        mask_pil = Image.new("L", (512, 512), 0)

    img_t = TF.to_tensor(img_pil)  # [3, H, W] in [0, 1]
    mask_t = TF.to_tensor(mask_pil)  # [1, H, W]
    mask_t = (mask_t > 0.5).float()

    result = {}

    if multi_crop:
        if clip_align:
            core_native, dil_native = unet_roundtrip_masks(mask_t, band_mode)
            crops, crop_masks, valid, extra = clip_crop_multi(
                img_t, mask_t, crop_size=crop_size, n_groups=2,
                clip_masks=[dil_native, core_native],
            )
            result["reference"] = crops[0] * 2.0 - 1.0  # [3, 224, 224] in [-1, 1]
            result["clip_mask"] = (extra[0][0] > 0.5).float()
            result["clip_core_mask"] = (extra[0][1] > 0.5).float()
            result["reference_2"] = crops[1] * 2.0 - 1.0
            result["clip_mask_2"] = (extra[1][0] > 0.5).float()
            result["clip_core_mask_2"] = (extra[1][1] > 0.5).float()
            result["group_valid"] = torch.tensor([float(valid[0]), float(valid[1])])
        else:
            crops, crop_masks, valid = clip_crop_multi(
                img_t, mask_t, crop_size=crop_size, n_groups=2,
            )
            result["reference"] = crops[0] * 2.0 - 1.0
            result["clip_mask"] = (crop_masks[0] > 0.5).float()
            result["reference_2"] = crops[1] * 2.0 - 1.0
            result["clip_mask_2"] = (crop_masks[1] > 0.5).float()
            result["group_valid"] = torch.tensor([float(valid[0]), float(valid[1])])
    else:
        from src.utils.crop_utils import clip_crop
        if clip_align:
            core_native, dil_native = unet_roundtrip_masks(mask_t, band_mode)
            crop_img, crop_mask, extra_c = clip_crop(
                img_t, mask_t, crop_size=crop_size,
                clip_masks=[dil_native, core_native],
            )
            result["reference"] = crop_img * 2.0 - 1.0
            result["clip_mask"] = (extra_c[0] > 0.5).float()
            result["clip_core_mask"] = (extra_c[1] > 0.5).float()
        else:
            crop_img, crop_mask = clip_crop(img_t, mask_t, crop_size=crop_size)
            result["reference"] = crop_img * 2.0 - 1.0
            result["clip_mask"] = (crop_mask > 0.5).float()

    return result


# ======================================================================
# Variant batch construction
# ======================================================================

def build_variant_batch(
    host_anchors: List[HostAnchor],
    sampler: ContrastiveHostSampler,
    pipeline,
    ip_adapter,
    t2i_adapter,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    multi_crop: bool = True,
    clip_align: bool = True,
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    device: str = "cuda",
) -> VariantBatch:
    """Build a packed variant batch for a single UNet forward pass.

    For each host anchor:
    1. Load host image/mask, compute latents, noise, x_t, timestep, masks, text_emb, T2I
    2. Load variant reference images and CLIP-encode them
    3. Repeat host tensors per variant, pack into a single batch

    No random conditioning dropout. All variants use clean embeddings.
    CFG/null robustness comes from explicit ip_zero variants:
      - typed: neg_is_null with p_null_typed_neg (~20%)
      - untyped: explicit "null" role = ip_zero
    """
    n_hosts = len(host_anchors)
    data_root = sampler.data_root

    # --- Phase 1: Load host images + masks ---
    host_images = []     # [3, 512, 512] in [-1, 1]
    host_masks = []      # [1, 512, 512] in {0, 1}
    host_captions = []

    for ha in host_anchors:
        info = sampler.get_sample_info(ha.anchor_idx)
        img, msk = _load_image_mask(info, data_root)
        host_images.append(img)
        host_masks.append(msk)
        cap = ha.caption or info.get("caption_short", "")
        if not cap:
            type_word = ha.group.replace("_", " ")
            cap = f"a photo of a {type_word} defect"
        host_captions.append(cap)

    host_images_t = torch.stack(host_images).to(device).float()  # [H, 3, 512, 512]
    host_masks_t = torch.stack(host_masks).to(device).float()    # [H, 1, 512, 512]

    # --- Phase 2: Compute per-host latents, noise, timestep, masks ---
    with torch.no_grad():
        host_latents = pipeline.encode_image(host_images_t)  # [H, 4, 64, 64]

    H = n_hosts
    latent_h = host_latents.shape[-1]  # 64

    # Timestep sampling
    if timestep_sampling == "logit_normal":
        u = torch.sigmoid(logit_normal_mean + logit_normal_std * torch.randn(H, device=device))
        host_timesteps = (u * 1000).long().clamp(0, 999)
    else:
        host_timesteps = torch.randint(0, 1000, (H,), device=device, dtype=torch.long)

    host_noise = torch.randn_like(host_latents)
    if noise_offset > 0:
        host_noise += noise_offset * torch.randn(
            H, host_latents.shape[1], 1, 1, device=device, dtype=host_latents.dtype,
        )
    host_noisy_latents = pipeline.scheduler.add_noise(host_latents, host_noise, host_timesteps)

    # UNet masks
    kernel = host_masks_t.shape[-1] // latent_h
    host_core_64 = F.max_pool2d(host_masks_t, kernel_size=kernel)
    host_core_64 = (host_core_64 > 0.5).float()  # [H, 1, 64, 64]

    host_dilated_list = []
    host_alpha_list = []
    host_weight_list = []
    host_band_list = []
    for i in range(H):
        d, a, w, b = create_latent_band_mask(
            host_core_64[i:i+1], band_mode, core_ratio=loss_core_ratio,
        )
        host_dilated_list.append(d)
        host_alpha_list.append(a)
        host_weight_list.append(w)
        host_band_list.append(b)
    host_dilated_64 = torch.cat(host_dilated_list, dim=0)
    host_alpha_64 = torch.cat(host_alpha_list, dim=0)
    host_weight_64 = torch.cat(host_weight_list, dim=0)
    host_band_64 = torch.cat(host_band_list, dim=0)

    # Inpainting inputs
    unet_mask_512 = F.interpolate(host_dilated_64, size=host_images_t.shape[-2:], mode='nearest')
    masked_image = host_images_t * (1 - unet_mask_512)
    with torch.no_grad():
        masked_image_latents = pipeline.encode_image(masked_image)

    host_model_input = torch.cat(
        [host_noisy_latents, host_dilated_64, masked_image_latents], dim=1,
    )  # [H, 9, 64, 64]

    # Text embeddings
    text_emb = pipeline.encode_text(host_captions, enable_grad=False)  # [H, 77, 768]

    # T2I features
    host_t2i_features = None
    if t2i_adapter is not None:
        t2i_input = torch.cat([host_core_64, host_band_64], dim=1)  # [H, 2, 64, 64]
        host_t2i_features = t2i_adapter(t2i_input, mask=host_dilated_64)

    # --- Phase 3: Load all variant CLIP references ---
    # Collect unique sample indices we need to CLIP-encode
    clip_refs_needed: Dict[int, dict] = {}  # idx -> clip ref data

    for ha in host_anchors:
        # Anchor's own reference (pos1 or true)
        if ha.anchor_idx not in clip_refs_needed:
            info = sampler.get_sample_info(ha.anchor_idx)
            clip_refs_needed[ha.anchor_idx] = _load_clip_reference(
                info, data_root, band_mode=band_mode,
                multi_crop=multi_crop, clip_align=clip_align,
            )
        # pos2 reference
        if ha.pos2_idx is not None and ha.pos2_idx not in clip_refs_needed:
            info = sampler.get_sample_info(ha.pos2_idx)
            clip_refs_needed[ha.pos2_idx] = _load_clip_reference(
                info, data_root, band_mode=band_mode,
                multi_crop=multi_crop, clip_align=clip_align,
            )
        # neg reference
        if ha.neg_idx is not None and ha.neg_idx not in clip_refs_needed:
            info = sampler.get_sample_info(ha.neg_idx)
            clip_refs_needed[ha.neg_idx] = _load_clip_reference(
                info, data_root, band_mode=band_mode,
                multi_crop=multi_crop, clip_align=clip_align,
            )

    # --- Phase 4: Batch CLIP encode all unique references ---
    ref_idx_list = sorted(clip_refs_needed.keys())
    idx_to_embeds: Dict[int, torch.Tensor] = {}

    if ref_idx_list:
        # Encode crop 1
        refs_1 = torch.stack([clip_refs_needed[i]["reference"] for i in ref_idx_list]).to(device)
        refs_1_01 = (refs_1 + 1.0) / 2.0
        masks_1 = torch.stack([clip_refs_needed[i]["clip_mask"] for i in ref_idx_list]).to(device)
        core_masks_1 = None
        if "clip_core_mask" in clip_refs_needed[ref_idx_list[0]]:
            core_masks_1 = torch.stack(
                [clip_refs_needed[i]["clip_core_mask"] for i in ref_idx_list]
            ).to(device)

        embeds_1 = ip_adapter.encode_image(refs_1_01, mask=masks_1, core_mask=core_masks_1)

        # Encode crop 2 if multi-crop
        if multi_crop and "reference_2" in clip_refs_needed[ref_idx_list[0]]:
            refs_2 = torch.stack([clip_refs_needed[i]["reference_2"] for i in ref_idx_list]).to(device)
            refs_2_01 = (refs_2 + 1.0) / 2.0
            masks_2 = torch.stack([clip_refs_needed[i]["clip_mask_2"] for i in ref_idx_list]).to(device)
            core_masks_2 = None
            if "clip_core_mask_2" in clip_refs_needed[ref_idx_list[0]]:
                core_masks_2 = torch.stack(
                    [clip_refs_needed[i]["clip_core_mask_2"] for i in ref_idx_list]
                ).to(device)

            embeds_2 = ip_adapter.encode_image(refs_2_01, mask=masks_2, core_mask=core_masks_2)

            K = embeds_1.shape[1]
            all_embeds = torch.cat([embeds_1, embeds_2], dim=1)  # [N, 2K, dim]

            # Build null token masks
            for j, idx in enumerate(ref_idx_list):
                gv = clip_refs_needed[idx].get("group_valid")
                if gv is not None:
                    m1 = gv[0:1].expand(K)
                    m2 = gv[1:2].expand(K)
                    ntm = torch.cat([m1, m2], dim=0)  # [2K]
                else:
                    ntm = torch.ones(2 * K, device=device)
                idx_to_embeds[idx] = (all_embeds[j], ntm)
        else:
            for j, idx in enumerate(ref_idx_list):
                idx_to_embeds[idx] = (embeds_1[j], None)

    # --- Phase 5: Build packed variant tensors ---
    variant_model_inputs = []
    variant_timesteps = []
    variant_text_embs = []
    variant_ip_embeds = []
    variant_noise = []
    variant_latents = []
    variant_weights = []
    variant_core = []
    variant_band = []
    variant_alpha = []
    variant_null_token_masks = []
    variant_anchor_ids = []
    variant_anchor_types = []
    variant_roles = []
    variant_groups = []
    variant_sample_weights = []
    variant_gets_ldiff = []
    variant_neg_sources = []

    for host_i, ha in enumerate(host_anchors):
        for role in ha.variant_roles:
            # Repeat host tensors
            variant_model_inputs.append(host_model_input[host_i])
            variant_timesteps.append(host_timesteps[host_i])
            variant_text_embs.append(text_emb[host_i])
            variant_noise.append(host_noise[host_i])
            variant_latents.append(host_latents[host_i])
            variant_weights.append(host_weight_64[host_i])
            variant_core.append(host_core_64[host_i])
            variant_band.append(host_band_64[host_i])
            variant_alpha.append(host_alpha_64[host_i])

            # Determine which CLIP embeddings to use + L_diff eligibility
            # L_diff targets: pos1, true, null (ip_zero), neg-when-ip_zero
            # NOT: pos2 (invariance probe), neg-when-wrong-type (regularizer only)
            if role in ("pos1", "true"):
                embeds_data = idx_to_embeds[ha.anchor_idx]
                gets_ldiff = True
            elif role == "pos2":
                embeds_data = idx_to_embeds[ha.pos2_idx]
                gets_ldiff = False  # invariance probe only
            elif role == "neg":
                if ha.neg_is_null:
                    # ip_zero: trains CFG baseline → gets L_diff
                    ref_embeds = idx_to_embeds[ha.anchor_idx]
                    embeds_data = (
                        torch.zeros_like(ref_embeds[0]),
                        ref_embeds[1] if ref_embeds[1] is not None else None,
                    )
                    gets_ldiff = True
                else:
                    # wrong-type CLIP: regularizer probe only → NO L_diff
                    embeds_data = idx_to_embeds[ha.neg_idx]
                    gets_ldiff = False
            elif role == "null":
                # Untyped null (ip_zero): trains CFG baseline → gets L_diff
                ref_embeds = idx_to_embeds[ha.anchor_idx]
                embeds_data = (
                    torch.zeros_like(ref_embeds[0]),
                    ref_embeds[1] if ref_embeds[1] is not None else None,
                )
                gets_ldiff = True
            else:
                raise ValueError(f"Unknown variant role: {role}")

            variant_ip_embeds.append(embeds_data[0])
            variant_null_token_masks.append(embeds_data[1])

            variant_anchor_ids.append(host_i)
            variant_anchor_types.append(ha.anchor_type)
            variant_roles.append(role)
            variant_groups.append(ha.group)
            variant_sample_weights.append(ha.weight)
            variant_gets_ldiff.append(gets_ldiff)
            variant_neg_sources.append(ha.neg_source if role == "neg" else "")

    V = len(variant_model_inputs)

    # Pack variant embeddings (no random dropout — contrastive families are deterministic)
    # CFG/null robustness comes from explicit ip_zero variants:
    #   - typed: neg_is_null with p_null_typed_neg (~20%)
    #   - untyped: explicit "null" role = ip_zero
    packed_ip_embeds = torch.stack(variant_ip_embeds)  # [V, K_total, dim]
    packed_text_emb = torch.stack(variant_text_embs)    # [V, 77, 768]

    # Build cross-attention kwargs
    cross_attn_kwargs = {"ip_adapter_image_embeds": packed_ip_embeds}
    cross_attn_kwargs["ip_adapter_mask"] = torch.stack(variant_alpha)

    # Null token mask
    has_ntm = any(ntm is not None for ntm in variant_null_token_masks)
    packed_ntm = None
    if has_ntm:
        ntm_list = []
        for ntm in variant_null_token_masks:
            if ntm is not None:
                ntm_list.append(ntm)
            else:
                ntm_list.append(torch.ones(packed_ip_embeds.shape[1], device=device))
        packed_ntm = torch.stack(ntm_list).to(device)  # [V, 2K]
        cross_attn_kwargs["null_token_mask"] = packed_ntm

    # T2I features
    t2i_kwargs = {}
    if t2i_adapter is not None and host_t2i_features is not None:
        # Repeat T2I features per variant
        expanded_features = []
        for feat in host_t2i_features:
            expanded = []
            for host_i, ha in enumerate(host_anchors):
                for _ in ha.variant_roles:
                    expanded.append(feat[host_i])
            expanded_features.append(torch.stack(expanded))
        t2i_kwargs = t2i_adapter.prepare_unet_kwargs(expanded_features)

    return VariantBatch(
        model_input=torch.stack(variant_model_inputs),
        timesteps=torch.stack(variant_timesteps),
        text_emb=packed_text_emb,
        ip_image_embeds=packed_ip_embeds,
        cross_attn_kwargs=cross_attn_kwargs,
        t2i_kwargs=t2i_kwargs,
        noise=torch.stack(variant_noise),
        latents=torch.stack(variant_latents),
        weight_map_64=torch.stack(variant_weights),
        core_mask_64=torch.stack(variant_core),
        band_mask_64=torch.stack(variant_band),
        anchor_ids=variant_anchor_ids,
        anchor_types=variant_anchor_types,
        variant_roles=variant_roles,
        anchor_groups=variant_groups,
        sample_weights=torch.tensor(variant_sample_weights, device=device),
        gets_ldiff=variant_gets_ldiff,
        neg_sources=variant_neg_sources,
        null_token_mask=packed_ntm,
    )


# ======================================================================
# Loss computation
# ======================================================================

def _weighted_l2(diff: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted L2 norm: sqrt(sum(diff^2 * weight) / sum(weight))."""
    w = weight.expand_as(diff)
    return ((diff ** 2 * w).sum() / w.sum().clamp(min=1e-8) + 1e-8).sqrt()


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted MSE: sum((pred - target)^2 * weight) / sum(weight)."""
    diff = (pred - target) ** 2
    w = weight.expand_as(diff)
    return (diff * w).sum() / w.sum().clamp(min=1e-8)


def compute_contrastive_loss(
    eps_pred: torch.Tensor,  # [V, 4, 64, 64]
    batch: VariantBatch,
    strategy: str = "A",
    lambda_inv: float = 0.1,
    lambda_rank: float = 0.1,
    lambda_rank_untyped: float = 0.1,
    lambda_triplet: float = 0.1,
    gamma: float = 0.0,
    triplet_margin_m: float = 0.10,
    use_untyped_rank_null: bool = True,
    triplet_core_only: bool = True,
    triplet_mode: str = "hinge",
    triplet_softplus_k: float = 20.0,
    triplet_softplus_offset: float = 0.0,
) -> Tuple[torch.Tensor, dict]:
    """Compute combined diffusion + contrastive losses.

    Returns:
        (total_loss, extras_dict) where extras_dict contains diagnostics.
    """
    V = eps_pred.shape[0]
    noise = batch.noise.float()
    weight = batch.weight_map_64.expand_as(eps_pred)
    core = batch.core_mask_64.expand_as(eps_pred)

    # --- L_diff: diffusion loss only for eligible variants ---
    # pos1, true, null (ip_zero), neg-when-ip_zero get L_diff
    # pos2 and wrong-type neg do NOT (regularizer probes only)
    diff = (eps_pred - noise) ** 2
    weighted_diff = diff * weight

    # Per-variant L_diff (all V, used for diagnostics + L_rank)
    per_variant_diff = (weighted_diff.reshape(V, -1).sum(dim=1) /
                        weight.reshape(V, -1).sum(dim=1).clamp(min=1e-8))

    # L_diff mask: only eligible variants contribute (uniform weighting)
    ldiff_mask = torch.tensor(batch.gets_ldiff, device=eps_pred.device, dtype=torch.float32)
    L_diff = (per_variant_diff * ldiff_mask).sum() / ldiff_mask.sum().clamp(min=1e-8)

    # Diagnostics: core/band (over L_diff-eligible variants only)
    with torch.no_grad():
        band = batch.band_mask_64.expand_as(eps_pred)
        core_n = core.reshape(V, -1).sum(dim=1).clamp(min=1)
        band_n = band.reshape(V, -1).sum(dim=1).clamp(min=1)
        core_loss_vals = (diff * core).reshape(V, -1).sum(dim=1) / core_n
        band_loss_vals = (diff * band).reshape(V, -1).sum(dim=1) / band_n
        # Average only over L_diff-eligible variants
        ldiff_idx = [i for i in range(V) if batch.gets_ldiff[i]]
        if ldiff_idx:
            L_diff_core = core_loss_vals[ldiff_idx].mean().item()
            L_diff_band = band_loss_vals[ldiff_idx].mean().item()
        else:
            L_diff_core = core_loss_vals.mean().item()
            L_diff_band = band_loss_vals.mean().item()

    # --- Build role→index mapping ---
    # Group by anchor_id
    anchor_variants: Dict[int, Dict[str, int]] = defaultdict(dict)
    for vi in range(V):
        anchor_variants[batch.anchor_ids[vi]][batch.variant_roles[vi]] = vi

    # --- Per-anchor terms + sample weights ---
    L_inv_terms = [];       L_inv_weights = []
    L_rank_typed_terms = []; L_rank_typed_weights = []
    L_rank_untyped_terms = [];L_rank_untyped_weights = []
    L_triplet_terms = [];   L_triplet_weights = []
    d_same_sem_vals = []   # only anchors with semantic neg (same_host)
    d_same_null_vals = []  # only anchors with ip_zero neg
    d_diff_vals = []       # ALL typed negs (semantic + ip_zero)
    d_diff_sem_vals = []   # semantic negs only (same_host)
    d_diff_null_vals = []  # ip_zero negs only
    L_true_vals = []
    L_wrong_vals = []
    L_true_untyped_vals = []
    L_null_untyped_vals = []
    rank_satisfied_typed = []
    rank_satisfied_untyped = []

    # Split rank tracking by neg source
    rank_sat_by_source: Dict[str, List[float]] = defaultdict(list)
    # Margin diagnostics
    typed_margins: List[float] = []
    untyped_margins: List[float] = []
    # Conditioning strength: ||eps_cond - eps_null|| (cond vs null, not cond vs cond)
    cond_null_dists: List[float] = []
    # CFG probe: L_diff of eps_cfg = eps_null + scale * (eps_cond - eps_null)
    cfg_probe_scale = 3.0
    cfg_L_cond_vals: List[float] = []   # L_diff of conditioned (baseline)
    cfg_L_null_vals: List[float] = []   # L_diff of null
    cfg_L_cfg_vals: List[float] = []    # L_diff of CFG-amplified

    # Triplet geometry tracking (Strategy A + B)
    triplet_gap_vals: List[float] = []     # d_diff - d_same_core (excludes ip_zero)
    triplet_gap_same_host: List[float] = []   # d_diff - d_same for same_host negs
    triplet_gap_cross_host: List[float] = []  # d_diff - d_same for cross_host negs
    triplet_residuals: List[float] = []    # raw r = margin + d_same_core - d_diff
    triplet_n_same_host = 0
    triplet_n_cross_host = 0
    triplet_n_skipped = 0

    # Per-anchor sample weight lookup
    anchor_weight = {}  # anchor_id -> float
    for vi in range(V):
        aid = batch.anchor_ids[vi]
        if aid not in anchor_weight:
            anchor_weight[aid] = batch.sample_weights[vi].item()

    # Neg source lookup for typed anchors
    anchor_neg_source: Dict[int, str] = {}
    for vi in range(V):
        if batch.variant_roles[vi] == "neg" and batch.neg_sources[vi]:
            anchor_neg_source[batch.anchor_ids[vi]] = batch.neg_sources[vi]

    for anchor_id, roles in anchor_variants.items():
        atype = batch.anchor_types[roles[list(roles.keys())[0]]]
        aw = anchor_weight[anchor_id]

        if atype == "typed":
            pos1_vi = roles.get("pos1")
            pos2_vi = roles.get("pos2")
            neg_vi = roles.get("neg")

            if pos1_vi is None or pos2_vi is None:
                continue

            eps_pos1 = eps_pred[pos1_vi]
            eps_pos2 = eps_pred[pos2_vi]
            w_i = weight[pos1_vi]  # Same host, same weight map (80/20)
            c_i = core[pos1_vi]    # Core-only mask

            # d_same_inv: relative L2 distance between same-type variants (80/20 weighted → L_inv)
            diff_same = eps_pos1 - eps_pos2
            norm_pos1 = _weighted_l2(eps_pos1, w_i)
            norm_pos2 = _weighted_l2(eps_pos2, w_i)
            d_same_inv = _weighted_l2(diff_same, w_i) / (norm_pos1 + norm_pos2 + 1e-6)

            # d_same_core: core-only (for triplet + logging, comparable to d_diff_core)
            d_same_core = _weighted_l2(diff_same, c_i) / (
                _weighted_l2(eps_pos1, c_i) + _weighted_l2(eps_pos2, c_i) + 1e-6
            )
            neg_src = anchor_neg_source.get(anchor_id, "")
            if neg_src == "ip_zero":
                d_same_null_vals.append(d_same_core.item())
            elif neg_src:  # same_host (or any semantic source)
                d_same_sem_vals.append(d_same_core.item())

            if strategy == "A":
                # L_inv: same-type invariance (80/20 weighted) — UNCHANGED
                L_inv_terms.append(d_same_inv)
                L_inv_weights.append(aw)

                # L_rank: ranking hinge (core-only)
                if neg_vi is not None:
                    eps_neg = eps_pred[neg_vi]
                    L_true = _masked_mse(eps_pos1, noise[pos1_vi], c_i)
                    L_wrong = _masked_mse(eps_neg, noise[neg_vi], c_i)
                    margin = L_wrong.item() - L_true.item()
                    L_rank_typed_terms.append(F.relu(gamma + L_true - L_wrong))
                    L_rank_typed_weights.append(aw)
                    L_true_vals.append(L_true.item())
                    L_wrong_vals.append(L_wrong.item())
                    satisfied = float(L_true.item() < L_wrong.item())
                    rank_satisfied_typed.append(satisfied)
                    typed_margins.append(margin)

                    # Split rank satisfaction by neg source
                    if neg_src:
                        rank_sat_by_source[neg_src].append(satisfied)

                    # d_diff_core: separation distance (core-only)
                    diff_diff = eps_pos1 - eps_neg
                    norm_neg = _weighted_l2(eps_neg, c_i)
                    d_diff_core = _weighted_l2(diff_diff, c_i) / (
                        _weighted_l2(eps_pos1, c_i) + norm_neg + 1e-6
                    )
                    d_diff_vals.append(d_diff_core.item())
                    if neg_src == "ip_zero":
                        d_diff_null_vals.append(d_diff_core.item())
                        # Conditioning strength: raw ||eps_cond - eps_null|| (core-only)
                        cond_null_dists.append(_weighted_l2(diff_diff, c_i).item())
                        # CFG probe: amplify visual signal
                        noise_i = noise[pos1_vi]
                        eps_cfg = eps_neg + cfg_probe_scale * (eps_pos1 - eps_neg)
                        cfg_L_cond_vals.append(_masked_mse(eps_pos1, noise_i, c_i).item())
                        cfg_L_null_vals.append(_masked_mse(eps_neg, noise[neg_vi], c_i).item())
                        cfg_L_cfg_vals.append(_masked_mse(eps_cfg, noise_i, c_i).item())
                    else:
                        d_diff_sem_vals.append(d_diff_core.item())

                    # L_triplet (skip ip_zero — not semantic, too easy)
                    if neg_src != "ip_zero":
                        gap = d_diff_core.item() - d_same_core.item()
                        triplet_gap_vals.append(gap)
                        if neg_src == "same_host":
                            triplet_gap_same_host.append(gap)
                        elif neg_src == "cross_host":
                            triplet_gap_cross_host.append(gap)

                        if triplet_mode == "hinge":
                            r = triplet_margin_m + d_same_core - d_diff_core
                            triplet_residuals.append(r.item())
                            L_triplet_terms.append(F.relu(r))
                        elif triplet_mode == "softplus":
                            raw = d_same_core - d_diff_core + triplet_softplus_offset
                            triplet_residuals.append(raw.item())
                            L_triplet_terms.append(F.softplus(triplet_softplus_k * raw) / triplet_softplus_k)

                        L_triplet_weights.append(aw)
                        if neg_src == "same_host":
                            triplet_n_same_host += 1
                        elif neg_src == "cross_host":
                            triplet_n_cross_host += 1
                    else:
                        triplet_n_skipped += 1
                else:
                    triplet_n_skipped += 1

            elif strategy == "B":
                # L_triplet (skip ip_zero — not semantic)
                if neg_vi is not None:
                    eps_neg = eps_pred[neg_vi]
                    diff_diff = eps_pos1 - eps_neg
                    d_diff_core = _weighted_l2(diff_diff, c_i) / (
                        _weighted_l2(eps_pos1, c_i) + _weighted_l2(eps_neg, c_i) + 1e-6
                    )
                    d_diff_vals.append(d_diff_core.item())
                    if neg_src == "ip_zero":
                        d_diff_null_vals.append(d_diff_core.item())
                    else:
                        d_diff_sem_vals.append(d_diff_core.item())

                    if neg_src != "ip_zero":
                        if triplet_mode == "hinge":
                            L_triplet_terms.append(F.relu(triplet_margin_m + d_same_core - d_diff_core))
                        elif triplet_mode == "softplus":
                            raw = d_same_core - d_diff_core + triplet_softplus_offset
                            L_triplet_terms.append(F.softplus(triplet_softplus_k * raw) / triplet_softplus_k)

                        L_triplet_weights.append(aw)

        elif atype == "untyped" and use_untyped_rank_null:
            true_vi = roles.get("true")
            null_vi = roles.get("null")

            if true_vi is None or null_vi is None:
                continue

            eps_true = eps_pred[true_vi]
            eps_null = eps_pred[null_vi]
            w_i = weight[true_vi]
            c_i = core[true_vi]

            L_true_u = _masked_mse(eps_true, noise[true_vi], c_i)
            L_null_u = _masked_mse(eps_null, noise[null_vi], c_i)
            margin_u = L_null_u.item() - L_true_u.item()

            L_rank_untyped_terms.append(F.relu(gamma + L_true_u - L_null_u))
            L_rank_untyped_weights.append(aw)
            L_true_untyped_vals.append(L_true_u.item())
            L_null_untyped_vals.append(L_null_u.item())
            satisfied_u = float(L_true_u.item() < L_null_u.item())
            rank_satisfied_untyped.append(satisfied_u)
            untyped_margins.append(margin_u)
            rank_sat_by_source["untyped"].append(satisfied_u)
            # Conditioning strength: raw ||eps_cond - eps_null|| (core-only)
            cond_null_dists.append(_weighted_l2(eps_true - eps_null, c_i).item())
            # CFG probe: amplify visual signal
            noise_i = noise[true_vi]
            eps_cfg_u = eps_null + cfg_probe_scale * (eps_true - eps_null)
            cfg_L_cond_vals.append(L_true_u.item())
            cfg_L_null_vals.append(L_null_u.item())
            cfg_L_cfg_vals.append(_masked_mse(eps_cfg_u, noise_i, c_i).item())

    # --- Aggregate losses (sample-weight–aware) ---
    loss_total = L_diff

    L_inv_val = 0.0
    L_rank_typed_val = 0.0
    L_rank_untyped_val = 0.0
    L_triplet_val = 0.0

    def _weighted_mean(terms, weights):
        """Weighted mean of loss terms using per-anchor sample weights."""
        t = torch.stack(terms)
        w = torch.tensor(weights, device=t.device, dtype=t.dtype)
        return (t * w).sum() / w.sum().clamp(min=1e-8)

    if strategy == "A":
        if L_inv_terms:
            L_inv_t = _weighted_mean(L_inv_terms, L_inv_weights)
            L_inv_val = L_inv_t.item()
            loss_total = loss_total + lambda_inv * L_inv_t

        if L_rank_typed_terms:
            L_rank_t = _weighted_mean(L_rank_typed_terms, L_rank_typed_weights)
            L_rank_typed_val = L_rank_t.item()
            loss_total = loss_total + lambda_rank * L_rank_t

        if L_triplet_terms:
            L_trip = _weighted_mean(L_triplet_terms, L_triplet_weights)
            L_triplet_val = L_trip.item()
            loss_total = loss_total + lambda_triplet * L_trip

        if L_rank_untyped_terms:
            L_rank_u = _weighted_mean(L_rank_untyped_terms, L_rank_untyped_weights)
            L_rank_untyped_val = L_rank_u.item()
            loss_total = loss_total + lambda_rank_untyped * L_rank_u

    elif strategy == "B":
        if L_triplet_terms:
            L_trip = _weighted_mean(L_triplet_terms, L_triplet_weights)
            L_triplet_val = L_trip.item()
            loss_total = loss_total + lambda_triplet * L_trip

        if L_rank_untyped_terms:
            L_rank_u = _weighted_mean(L_rank_untyped_terms, L_rank_untyped_weights)
            L_rank_untyped_val = L_rank_u.item()
            loss_total = loss_total + lambda_rank_untyped * L_rank_u

    # --- Margin diagnostics ---
    def _margin_stats(margins: List[float], gamma_val: float) -> dict:
        if not margins:
            return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "frac_above_gamma": 0.0}
        arr = np.array(margins)
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "frac_above_gamma": float(np.mean(arr > gamma_val)) if gamma_val > 0 else 0.0,
        }

    typed_mstats = _margin_stats(typed_margins, gamma)
    untyped_mstats = _margin_stats(untyped_margins, gamma)

    # --- ||Δε|| in sigma bins (early/mid/late timestep) ---
    # Per-variant ||eps_pred - noise|| on core, binned by timestep.
    # early = t < 333 (low noise, hard ε-prediction), mid = 333-666, late = t > 666 (high noise, easy ε-prediction)
    with torch.no_grad():
        t = batch.timesteps  # [V]
        per_var_eps_norm = ((diff * core).reshape(V, -1).sum(dim=1) /
                           core.reshape(V, -1).sum(dim=1).clamp(min=1)).sqrt()
        ldiff_set = set(i for i in range(V) if batch.gets_ldiff[i])
        bins = {"early": [], "mid": [], "late": []}
        for vi in range(V):
            if vi not in ldiff_set:
                continue
            tv = t[vi].item()
            if tv < 333:
                bins["early"].append(per_var_eps_norm[vi].item())
            elif tv < 667:
                bins["mid"].append(per_var_eps_norm[vi].item())
            else:
                bins["late"].append(per_var_eps_norm[vi].item())

    # Split rank satisfaction by neg source
    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    extras = {
        "L_diff": L_diff.item(),
        "L_diff_core": L_diff_core,
        "L_diff_band": L_diff_band,
        "L_inv": L_inv_val,
        "L_rank_typed": L_rank_typed_val,
        "L_rank_untyped": L_rank_untyped_val,
        "L_triplet": L_triplet_val,
        "d_same_mean": float(np.mean(d_same_sem_vals)) if d_same_sem_vals else 0.0,
        "d_same_std": float(np.std(d_same_sem_vals)) if d_same_sem_vals else 0.0,
        "d_same_null_mean": float(np.mean(d_same_null_vals)) if d_same_null_vals else 0.0,
        "d_diff_mean": float(np.mean(d_diff_sem_vals)) if d_diff_sem_vals else 0.0,
        "d_diff_std": float(np.std(d_diff_sem_vals)) if d_diff_sem_vals else 0.0,
        "d_diff_null_mean": float(np.mean(d_diff_null_vals)) if d_diff_null_vals else 0.0,
        "pct_rank_satisfied_typed": float(np.mean(rank_satisfied_typed)) if rank_satisfied_typed else 0.0,
        "pct_rank_satisfied_untyped": float(np.mean(rank_satisfied_untyped)) if rank_satisfied_untyped else 0.0,
        "L_true_mean": float(np.mean(L_true_vals)) if L_true_vals else 0.0,
        "L_wrong_mean": float(np.mean(L_wrong_vals)) if L_wrong_vals else 0.0,
        "L_true_untyped_mean": float(np.mean(L_true_untyped_vals)) if L_true_untyped_vals else 0.0,
        "L_null_untyped_mean": float(np.mean(L_null_untyped_vals)) if L_null_untyped_vals else 0.0,
        "n_typed": sum(1 for t in batch.anchor_types if t == "typed"),
        "n_untyped": sum(1 for t in batch.anchor_types if t == "untyped"),
        # Split rank by neg source
        "rank_sat_same_host": _safe_mean(rank_sat_by_source.get("same_host", [])),
        "rank_sat_cross_host": _safe_mean(rank_sat_by_source.get("cross_host", [])),
        "rank_sat_ip_zero": _safe_mean(rank_sat_by_source.get("ip_zero", [])),
        "rank_sat_untyped": _safe_mean(rank_sat_by_source.get("untyped", [])),
        "n_rank_same_host": float(len(rank_sat_by_source.get("same_host", []))),
        "n_rank_cross_host": float(len(rank_sat_by_source.get("cross_host", []))),
        "n_rank_ip_zero": float(len(rank_sat_by_source.get("ip_zero", []))),
        "n_rank_untyped": float(len(rank_sat_by_source.get("untyped", []))),
        # Margin diagnostics
        "margin_typed_mean": typed_mstats["mean"],
        "margin_typed_median": typed_mstats["median"],
        "margin_typed_p10": typed_mstats["p10"],
        "margin_typed_p90": typed_mstats["p90"],
        "margin_typed_frac_above_gamma": typed_mstats["frac_above_gamma"],
        "margin_untyped_mean": untyped_mstats["mean"],
        "margin_untyped_median": untyped_mstats["median"],
        "margin_untyped_p10": untyped_mstats["p10"],
        "margin_untyped_p90": untyped_mstats["p90"],
        "margin_untyped_frac_above_gamma": untyped_mstats["frac_above_gamma"],
        # Triplet geometry diagnostics
        "triplet_margin": triplet_margin_m,
        "L_triplet_weighted": lambda_triplet * L_triplet_val,
        "triplet_violation_rate": float(np.mean([r > 0 for r in triplet_residuals])) if triplet_residuals else 0.0,
        "triplet_n_same_host": float(triplet_n_same_host),
        "triplet_n_cross_host": float(triplet_n_cross_host),
        "triplet_n_skipped": float(triplet_n_skipped),
        "gap_mean": float(np.mean(triplet_gap_vals)) if triplet_gap_vals else 0.0,
        # Aggregate gap positive rate (% d_diff > d_same, semantic negs only)
        "gap_positive_rate": float(np.mean([g > 0 for g in triplet_gap_vals])) if triplet_gap_vals else 0.0,
        # Per-source gap positive rate
        "gap_pos_rate_same_host": float(np.mean([g > 0 for g in triplet_gap_same_host])) if triplet_gap_same_host else 0.0,
        "gap_pos_rate_cross_host": float(np.mean([g > 0 for g in triplet_gap_cross_host])) if triplet_gap_cross_host else 0.0,
        # Per-source gap mean (rate = sign, mean = margin/strength)
        "gap_mean_same_host": float(np.mean(triplet_gap_same_host)) if triplet_gap_same_host else 0.0,
        "gap_mean_cross_host": float(np.mean(triplet_gap_cross_host)) if triplet_gap_cross_host else 0.0,
        # ||Δε|| sigma bins (early t<333, mid 333-666, late >666)
        "eps_norm_early": float(np.mean(bins["early"])) if bins["early"] else 0.0,
        "eps_norm_mid": float(np.mean(bins["mid"])) if bins["mid"] else 0.0,
        "eps_norm_late": float(np.mean(bins["late"])) if bins["late"] else 0.0,
        # Conditioning strength: ||eps_cond - eps_null|| (cond vs null baseline)
        "cond_null_dist_mean": float(np.mean(cond_null_dists)) if cond_null_dists else 0.0,
        "cond_null_dist_std": float(np.std(cond_null_dists)) if cond_null_dists else 0.0,
        # CFG visual probe (scale=3.0): L_diff under null / cond / CFG-amplified
        "cfg_L_null": float(np.mean(cfg_L_null_vals)) if cfg_L_null_vals else 0.0,
        "cfg_L_cond": float(np.mean(cfg_L_cond_vals)) if cfg_L_cond_vals else 0.0,
        "cfg_L_cfg": float(np.mean(cfg_L_cfg_vals)) if cfg_L_cfg_vals else 0.0,
    }

    return loss_total, extras


# ======================================================================
# Loss plot
# ======================================================================

def _ema_smooth(values, smoothing: float = 0.99, seed=None):
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    ema = np.empty_like(arr)
    if seed is not None:
        for i in range(len(arr)):
            if np.isnan(arr[i]):
                ema[i] = ema[i - 1] if i > 0 else seed
            else:
                prev = ema[i - 1] if i > 0 else seed
                ema[i] = smoothing * prev + (1.0 - smoothing) * arr[i]
    else:
        first = 0
        while first < len(arr) and np.isnan(arr[first]):
            first += 1
        if first >= len(arr):
            return np.full_like(arr, np.nan)
        for i in range(first):
            ema[i] = arr[first]
        ema[first] = arr[first]
        for i in range(first + 1, len(arr)):
            if np.isnan(arr[i]):
                ema[i] = ema[i - 1]
            else:
                ema[i] = smoothing * ema[i - 1] + (1.0 - smoothing) * arr[i]
    return ema


def save_contrastive_loss_plot(stats_file: Path, save_path: Path):
    """Render 4x3 contrastive training diagnostics plot."""
    import matplotlib.pyplot as plt

    # Parse stats CSV
    rows = []
    try:
        with open(stats_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})
    except (FileNotFoundError, ValueError):
        return

    if len(rows) < 2:
        return

    # Load run config for lambda values
    run_dir = stats_file.parent
    cfg = {}
    try:
        with open(run_dir / "run_config.json") as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    lam_inv = cfg.get("lambda_inv", 1.0)
    lam_rank = cfg.get("lambda_rank", 5.0)
    lam_trip = cfg.get("lambda_triplet", 0.25)

    steps = np.array([r["step"] for r in rows])
    SM = 0.99
    SF = 0.6

    def _get(key):
        return np.array([r.get(key, 0.0) for r in rows])

    fig, axes = plt.subplots(4, 3, figsize=(21, 20))

    # [0,0] Loss trend (L_diff) + weighted contribution overlay
    L_diff = _get("L_diff")
    L_inv = _get("L_inv")
    L_rank_t = _get("L_rank_typed")
    ax = axes[0, 0]
    if len(L_diff) > 1:
        ax.plot(steps, _ema_smooth(L_diff, SF), color="red", linewidth=0.8, alpha=0.3, label=f"EMA({SF})")
        ema_slow = _ema_smooth(L_diff, SM)
        ax.plot(steps, ema_slow, color="darkblue", linewidth=2.5, label=f"L_diff EMA({SM})")
        # Weighted contribution traces (weighted only, no raw)
        if L_inv.any():
            inv_w = _ema_smooth(lam_inv * L_inv, SM)
            ax.plot(steps, inv_w, color="purple", linewidth=1.5, label=f"{lam_inv}\u00d7L_inv ({inv_w[-1]:.4f})")
        if L_rank_t.any():
            rk_w = _ema_smooth(lam_rank * L_rank_t, SM)
            ax.plot(steps, rk_w, color="green", linewidth=1.5, label=f"{lam_rank}\u00d7L_rank ({rk_w[-1]:.4f})")
        L_trip_raw = _get("L_triplet")
        if L_trip_raw.any():
            trip_w = _ema_smooth(lam_trip * L_trip_raw, SM)
            ax.plot(steps, trip_w, color="orange", linewidth=1.5, label=f"{lam_trip}\u00d7L_trip ({trip_w[-1]:.4f})")
        ax.set_title(f"L_diff \u2014 trend: {ema_slow[-1]:.4f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # [0,1] Core vs Band (stacked with weighted total)
    core = _get("L_diff_core")
    band = _get("L_diff_band")
    ax = axes[0, 1]
    if core.any():
        ce = _ema_smooth(core, SM)
        be = _ema_smooth(band, SM)
        ax.fill_between(steps, 0, ce, color="#2196F3", alpha=0.4)
        ax.fill_between(steps, ce, ce + be, color="#FF9800", alpha=0.4)
        ax.plot(steps, ce, color="#1565C0", linewidth=1.5, label="Core")
        ax.plot(steps, ce + be, color="#E65100", linewidth=1.5, label="Band (stacked)")
        weighted = 0.8 * ce + 0.2 * be
        ax.plot(steps, weighted, color="black", linewidth=2, linestyle="--",
                label=f"0.8\u00d7core+0.2\u00d7band = {weighted[-1]:.4f}")
        last_x = steps[-1]
        ax.annotate(f"{ce[-1]:.3f}", xy=(last_x, ce[-1] / 2),
                    fontsize=9, fontweight="bold", color="#1565C0", ha="right")
        ax.annotate(f"{be[-1]:.3f}",
                    xy=(last_x, ce[-1] + be[-1] / 2),
                    fontsize=9, fontweight="bold", color="#E65100", ha="right")
        ax.set_title(f"Core vs Band, EMA({SM}) \u2014 weighted: {weighted[-1]:.4f}")
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Per-pixel MSE")
    ax.grid(True, alpha=0.3)

    # [0,2] Role Embedding Norms
    emb_g = _get("emb_global_norm")
    emb_a = _get("emb_anomaly_norm")
    emb_n = _get("emb_normal_norm")
    ax = axes[0, 2]
    has_emb = emb_g.any() or emb_a.any() or emb_n.any()
    if has_emb:
        if emb_g.any():
            ax.plot(steps, emb_g, color="green", linewidth=1.5, label=f"Global ({emb_g[-1]:.4f})")
        if emb_a.any():
            ax.plot(steps, emb_a, color="red", linewidth=1.5, label=f"Anomaly ({emb_a[-1]:.4f})")
        if emb_n.any():
            ax.plot(steps, emb_n, color="blue", linewidth=1.5, label=f"Band ({emb_n[-1]:.4f})")
        ax.set_title("Role Embedding Norms")
    else:
        ax.set_title("Role Embedding Norms (no data)")
    ax.set_xlabel("Step"); ax.set_ylabel("L2 Norm")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Row 1: Contrastive diagnostics ---

    # [1,0] Contrastive Distances (semantic only + null separate)
    ds = _get("d_same_mean")
    dd = _get("d_diff_mean")       # now semantic-only
    dd_null = _get("d_diff_null_mean")
    ax = axes[1, 0]
    if ds.any():
        ds_e = _ema_smooth(ds, SM)
        ax.plot(steps, ds_e, color="blue", linewidth=2, label=f"d_same ({ds_e[-1]:.4f})")
    if dd.any():
        dd_e = _ema_smooth(dd, SM)
        ax.plot(steps, dd_e, color="red", linewidth=2, label=f"d_diff_sem ({dd_e[-1]:.4f})")
    if dd_null.any():
        dd_null_e = _ema_smooth(dd_null, SM)
        ax.plot(steps, dd_null_e, color="orange", linewidth=1.5, linestyle=":", label=f"d_diff_null ({dd_null_e[-1]:.4f})")
    gap_raw = _get("gap_mean")
    if gap_raw.any():
        gap_ema = _ema_smooth(gap_raw, SM)
        ax.plot(steps, gap_ema, color="green", linewidth=1.5, linestyle="--", label=f"gap ({gap_ema[-1]:.4f})")
    ax.set_title("Contrastive Distances")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Normalized distance")
    ax.grid(True, alpha=0.3)

    # [1,1] Gap Satisfaction Rate (same_host only)
    gpr_sh = _get("gap_pos_rate_same_host")
    ax = axes[1, 1]
    if gpr_sh.any():
        gpr_sh_ema = _ema_smooth(gpr_sh, SM)
        ax.plot(steps, gpr_sh_ema, color="green", linewidth=2, label=f"gap > 0 ({gpr_sh_ema[-1]:.0%})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Gap Satisfaction (same_host)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Rate")
    ax.grid(True, alpha=0.3)

    # [1,2] Normalized Contrastive Diagnostics
    ax = axes[1, 2]
    if ds.any() and dd.any():
        ds_e = _ema_smooth(ds, SM)
        dd_e = _ema_smooth(dd, SM)
        gap_rel = (dd_e - ds_e) / (dd_e + ds_e + 1e-6)
        ratio = dd_e / (ds_e + 1e-6)
        ax.plot(steps, gap_rel, color="purple", linewidth=2, label=f"gap_rel ({gap_rel[-1]:.4f})")
        ax.plot(steps, ratio, color="teal", linewidth=2, label=f"d_diff/d_same ({ratio[-1]:.4f})")
    ax.set_title("Normalized Contrastive Diagnostics")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    # --- Row 2: Regularizer losses ---

    # [2,0] L_inv / L_triplet
    L_inv = _get("L_inv")
    L_trip = _get("L_triplet")
    ax = axes[2, 0]
    if L_inv.any():
        ax.plot(steps, _ema_smooth(L_inv, SM), color="purple", linewidth=2, label=f"L_inv ({_ema_smooth(L_inv, SM)[-1]:.4f})")
    if L_trip.any():
        ax.plot(steps, _ema_smooth(L_trip, SM), color="orange", linewidth=2, label=f"L_triplet ({_ema_smooth(L_trip, SM)[-1]:.4f})")
    viol = _get("triplet_violation_rate")
    if viol.any():
        ax2 = ax.twinx()
        viol_ema = _ema_smooth(viol, SM)
        ax2.plot(steps, viol_ema, color="gray", linewidth=1, linestyle=":", label=f"viol% ({viol_ema[-1]:.0%})")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Violation rate", fontsize=8, color="gray")
        ax2.legend(loc="upper right", fontsize=7)
    ax.set_title("Regularizer Losses")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # [2,1] L_rank typed + untyped
    L_rt = _get("L_rank_typed")
    L_ru = _get("L_rank_untyped")
    ax = axes[2, 1]
    if L_rt.any():
        ax.plot(steps, _ema_smooth(L_rt, SM), color="green", linewidth=2, label=f"Typed ({_ema_smooth(L_rt, SM)[-1]:.4f})")
    if L_ru.any():
        ax.plot(steps, _ema_smooth(L_ru, SM), color="brown", linewidth=2, label=f"Untyped ({_ema_smooth(L_ru, SM)[-1]:.4f})")
    ax.set_title("L_rank")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)

    # [2,2] Rank satisfaction % — split by neg source
    def _mask_rs(sat_key, n_key):
        sat = _get(sat_key).copy()
        n = _get(n_key)
        if len(sat) > 0 and len(n) > 0:
            sat[n == 0] = np.nan
        return sat

    rs_same = _mask_rs("rank_sat_same_host", "n_rank_same_host")
    rs_ipz = _mask_rs("rank_sat_ip_zero", "n_rank_ip_zero")
    rs_untyped = _mask_rs("rank_sat_untyped", "n_rank_untyped")
    ax = axes[2, 2]
    for arr, color, name in [
        (rs_same, "green", "same_host"),
        (rs_ipz, "red", "ip_zero"),
        (rs_untyped, "brown", "untyped"),
    ]:
        if len(arr) > 0 and np.any(~np.isnan(arr)):
            e = _ema_smooth(arr, SM)
            ax.plot(steps, e, color=color, linewidth=2, label=f"{name} ({e[-1]:.0%})")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rank Satisfaction by Source")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel("Step"); ax.set_ylabel("Fraction")
    ax.grid(True, alpha=0.3)

    # --- Row 3: Model internals ---

    # [3,0] Gates
    ag = _get("attn_gate")
    fg = _get("ff_gate")
    ax = axes[3, 0]
    if ag.any():
        ax.plot(steps, ag, color="blue", linewidth=1.5, label=f"Attn ({ag[-1]:.4f})")
        ax.plot(steps, fg, color="green", linewidth=1.5, label=f"FF ({fg[-1]:.4f})")
    ax.set_title("Gates")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("Gate value")
    ax.grid(True, alpha=0.3)

    # [3,1] Grad norm
    gn = _get("grad_norm")
    ax = axes[3, 1]
    if gn.any():
        ax.plot(steps, _ema_smooth(gn, SM), color="darkblue", linewidth=2.5, label=f"EMA({SM})")
        ax.set_title(f"Grad Norm \u2014 {_ema_smooth(gn, SM)[-1]:.4f}")
    ax.set_xlabel("Step"); ax.set_ylabel("Norm")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # [3,2] L_true vs L_wrong
    lt = _get("L_true_mean")
    lw = _get("L_wrong_mean")
    ax = axes[3, 2]
    if lt.any():
        ax.plot(steps, _ema_smooth(lt, SM), color="blue", linewidth=2, label=f"L_true ({_ema_smooth(lt, SM)[-1]:.4f})")
    if lw.any():
        ax.plot(steps, _ema_smooth(lw, SM), color="red", linewidth=2, label=f"L_wrong ({_ema_smooth(lw, SM)[-1]:.4f})")
    gamma_vals = _get("gamma")
    gamma_str = f" \u2014 \u03b3={gamma_vals[-1]:.4f}" if gamma_vals.any() and gamma_vals[-1] > 0 else ""
    ax.set_title(f"True vs Wrong Loss{gamma_str}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("Step"); ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
# Checkpoint save
# ======================================================================

def save_contrastive_checkpoint(
    ip_adapter, optimizer, save_dir: Path,
    band_mode: int = 2, t2i_adapter=None,
    step: int = 0, losses: list = None,
):
    """Save IP-Adapter + T2I-Adapter + optimizer state."""
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
    torch.save(state, save_dir / "training_state.pt")


# ======================================================================
# Checkpoint eval probes
# ======================================================================

def run_eval_probes(
    pipeline, ip_adapter, sampler: ContrastiveHostSampler,
    save_dir: Path, step: int, device: str,
    band_mode: int = 2, t2i_adapter=None,
):
    """Run fixed-seed diagnostic probes at checkpoint time.

    Generates a 4-anchor × 4-variant qualitative grid.
    """
    import matplotlib.pyplot as plt

    # Fixed seed for reproducibility across checkpoints
    probe_rng = random.Random(42)

    # Pick 4 fixed typed anchors from different groups
    probe_groups = probe_rng.sample(sampler.groups, min(4, len(sampler.groups)))
    probe_anchors = []
    for g in probe_groups:
        idx = probe_rng.choice(sampler.group_to_indices[g])
        probe_anchors.append((idx, g))

    if not probe_anchors:
        return

    data_root = sampler.data_root
    n_anchors = len(probe_anchors)

    # Rows: real, mask, true_clip_gen, same_type_swap_gen, diff_type_neg_gen, null_gen
    fig, axes = plt.subplots(6, n_anchors, figsize=(3.5 * n_anchors, 18), squeeze=False)

    for col, (idx, group) in enumerate(probe_anchors):
        info = sampler.get_sample_info(idx)
        img, msk = _load_image_mask(info, data_root)
        clip_data = _load_clip_reference(info, data_root, band_mode=band_mode)

        image_t = img.unsqueeze(0).to(device)
        mask_t = msk.unsqueeze(0).to(device)
        ref_t = clip_data["reference"].unsqueeze(0).to(device)

        img_np = ((img.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        mask_np = msk[0].numpy()

        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f"Real ({group})", fontsize=7)
        axes[0, col].axis("off")

        axes[1, col].imshow(mask_np, cmap="gray")
        axes[1, col].set_title(f"Mask ({mask_np.mean()*100:.1f}%)", fontsize=7)
        axes[1, col].axis("off")

        caption = sampler.captions.get(info["image_path"], f"a photo of a {group} defect")

        gen_kwargs = dict(
            num_steps=20, noise_strength=1.0, reference_mode="crop",
            band_mode=band_mode, t2i_adapter=t2i_adapter, seed=42 + col,
        )

        def _to_np(t):
            return ((t[0].cpu().float().permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)

        # Gen with true CLIP
        with torch.no_grad():
            gen_true = generate_anomagic_single(
                pipeline, ip_adapter, image_t, mask_t, ref_t,
                group, caption, guidance_scale=1.0, cfg_mode="visual", **gen_kwargs,
            )
        axes[2, col].imshow(_to_np(gen_true))
        axes[2, col].set_title("True CLIP", fontsize=7)
        axes[2, col].axis("off")

        # Same-type swap
        other_prods = [p for p in sampler.group_products[group] if p != info["product"]]
        if other_prods:
            swap_prod = probe_rng.choice(other_prods)
            swap_idx = probe_rng.choice(sampler.group_product_to_indices[(group, swap_prod)])
            swap_info = sampler.get_sample_info(swap_idx)
            swap_clip = _load_clip_reference(swap_info, data_root, band_mode=band_mode)
            swap_ref = swap_clip["reference"].unsqueeze(0).to(device)
            with torch.no_grad():
                gen_swap = generate_anomagic_single(
                    pipeline, ip_adapter, image_t, mask_t, swap_ref,
                    group, caption, guidance_scale=1.0, cfg_mode="visual", **gen_kwargs,
                )
            axes[3, col].imshow(_to_np(gen_swap))
            axes[3, col].set_title(f"Same-type swap ({swap_prod[:12]})", fontsize=6)
        else:
            axes[3, col].set_title("No swap available", fontsize=7)
        axes[3, col].axis("off")

        # Different-type neg
        other_groups = [g for g in sampler.groups if g != group]
        if other_groups:
            neg_group = probe_rng.choice(other_groups)
            neg_idx = probe_rng.choice(sampler.group_to_indices[neg_group])
            neg_info = sampler.get_sample_info(neg_idx)
            neg_clip = _load_clip_reference(neg_info, data_root, band_mode=band_mode)
            neg_ref = neg_clip["reference"].unsqueeze(0).to(device)
            with torch.no_grad():
                gen_neg = generate_anomagic_single(
                    pipeline, ip_adapter, image_t, mask_t, neg_ref,
                    group, caption, guidance_scale=1.0, cfg_mode="visual", **gen_kwargs,
                )
            axes[4, col].imshow(_to_np(gen_neg))
            axes[4, col].set_title(f"Diff-type ({neg_group[:12]})", fontsize=6)
        else:
            axes[4, col].set_title("No neg available", fontsize=7)
        axes[4, col].axis("off")

        # Null CLIP
        with torch.no_grad():
            # Temporarily zero CLIP embeddings
            zero_ref = torch.zeros_like(ref_t)
            gen_null = generate_anomagic_single(
                pipeline, ip_adapter, image_t, mask_t, zero_ref,
                group, caption, guidance_scale=1.0, cfg_mode="visual", **gen_kwargs,
            )
        axes[5, col].imshow(_to_np(gen_null))
        axes[5, col].set_title("Null CLIP", fontsize=7)
        axes[5, col].axis("off")

    plt.suptitle(f"Eval Probes — Step {step}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_dir / f"probes_{step}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ======================================================================
# Live loss viewer launcher
# ======================================================================

def _launch_live_loss_contrastive(stats_file: Path):
    """Launch contrastive live loss viewer as a detached subprocess."""
    import subprocess
    script = Path(__file__).parent / "live_loss_contrastive.py"
    try:
        subprocess.Popen(
            [sys.executable, str(script), str(stats_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  Live loss viewer launched (reading {stats_file})")
    except Exception as e:
        print(f"  Warning: could not launch live loss viewer: {e}")


# ======================================================================
# Main training loop
# ======================================================================

def train_contrastive(
    data_json: Path,
    save_dir: Path,
    captions_file: Path,
    data_root: Path,
    n_steps: int = 5000,
    save_every: int = 1000,
    host_batch_size: int = 4,
    lr: float = 1e-4,
    lr_pretrained: float = 1e-4,
    lambda_sp: float = 0.0,
    device: str = "cuda",
    strategy: str = "A",
    lambda_inv: float = 0.1,
    lambda_rank: float = 0.1,
    lambda_rank_untyped: float = 0.1,
    lambda_triplet: float = 0.1,
    rank_gamma_scale: float = 0.05,
    triplet_margin_m: float = 0.10,
    regularizer_warmup_frac: float = 0.05,
    p_null_typed_neg: float = 0.0,
    typed_neg_mode: str = "same_host_diff_type",
    use_untyped_rank_null: bool = True,
    ip_adapter_type: str = "plus",
    ip_adapter_k: int = 16,
    ip_adapter_scale: float = 1.0,
    mask_visual: bool = True,
    band_mode: int = 2,
    loss_core_ratio: float = 0.8,
    t2i_adapter_mode: str = "cascade",
    augment: bool = False,
    multi_crop: bool = True,
    clip_align: bool = True,
    visual_mode: int = 3,
    learnable_gates: bool = True,
    force_gates: bool = False,
    sa_num_layers: int = 3,
    sa_num_heads: int = 12,
    cfg_mode: str = "visual",
    no_live_viewer: bool = False,
    noise_offset: float = 0.05,
    timestep_sampling: str = "logit_normal",
    logit_normal_mean: float = 0.0,
    logit_normal_std: float = 1.0,
    resume_dir: Path = None,
    variant_chunk_size: int = 0,
    pilot_subset_path: Path = None,
    pilot_subset_size: int = 3000,
    triplet_margin_auto: bool = True,
    triplet_margin_alpha: float = 0.5,
    triplet_margin: Optional[float] = None,
    triplet_core_only: bool = True,
    triplet_mode: str = "hinge",
    triplet_softplus_k: float = 20.0,
    triplet_softplus_offset: float = 0.0,
    p_neg_same_host: float = 0.6,
    p_neg_cross_host: float = 0.3,
):
    """Train with contrastive CLIP-conditioning regularization."""
    # Seed everything
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    torch.cuda.manual_seed_all(_SEED)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONTRASTIVE TRAINING — Variant-Batch CLIP Regularization")
    print("=" * 70)
    print(f"Strategy: {strategy}")
    print(f"Host batch size: {host_batch_size}")
    print(f"Steps: {n_steps}, Save every: {save_every}")
    print(f"LR (scratch): {lr}, LR (pretrained): {lr_pretrained}")
    if strategy == "A":
        print(f"Lambda: inv={lambda_inv}, rank={lambda_rank}, triplet={lambda_triplet}, rank_untyped={lambda_rank_untyped}")
        print(f"Rank gamma scale: {rank_gamma_scale}")
        margin_desc = f"manual={triplet_margin}" if triplet_margin is not None else f"auto (alpha={triplet_margin_alpha})" if triplet_margin_auto else f"fixed={triplet_margin_m}"
        print(f"Triplet margin: {margin_desc}")
        print(f"Core-only (rank+triplet): {triplet_core_only}")
    else:
        print(f"Lambda: triplet={lambda_triplet}, rank_untyped={lambda_rank_untyped}")
        print(f"Triplet margin: {triplet_margin_m}")
    print(f"Regularizer warmup: {regularizer_warmup_frac:.0%}")
    print(f"p_null_typed_neg: {p_null_typed_neg}")
    print(f"Multi-crop: {multi_crop}, CLIP-align: {clip_align}")
    print(f"Visual mode: {visual_mode}, SA layers: {sa_num_layers}, heads: {sa_num_heads}")
    print(f"Gates: {'forced=1' if force_gates else ('learnable' if learnable_gates else 'fixed')}")
    print(f"Typed neg mode: {typed_neg_mode}")
    print(f"Triplet mode: {triplet_mode}" + (f" (k={triplet_softplus_k}, offset={triplet_softplus_offset})" if triplet_mode == "softplus" else ""))
    if typed_neg_mode == "weighted":
        print(f"Neg sampling: weighted (sh={p_neg_same_host}, ch={p_neg_cross_host}, null={p_null_typed_neg})")
    print()

    # L_diff routing summary
    print("L_diff routing:")
    print("  typed pos1          -> L_diff YES")
    print("  typed pos2          -> L_diff NO  (invariance probe)")
    print("  typed neg (wrong)   -> L_diff NO  (regularizer probe)")
    print("  typed neg (ip_zero) -> L_diff YES (CFG baseline)")
    print("  untyped true        -> L_diff YES  (text=shared, ip=real)")
    print("  untyped null        -> L_diff YES  (text=shared, ip=zero)")
    print("  Untyped rank = visual/IP CFG: real IP vs ip_zero (text held fixed)")
    print()

    # =========================================
    # Initialize Sampler
    # =========================================
    sampler = ContrastiveHostSampler(
        str(data_json), str(data_root), captions_file=str(captions_file),
    )

    # Pilot subset
    pilot_indices = None
    if pilot_subset_path and Path(pilot_subset_path).exists():
        with open(pilot_subset_path) as f:
            pilot_indices = json.load(f)
        print(f"Loaded pilot subset: {len(pilot_indices)} indices from {pilot_subset_path}")
    elif pilot_subset_size > 0 and pilot_subset_size < len(sampler.all_samples):
        pilot_indices = generate_pilot_subset(sampler, target_size=pilot_subset_size)
        # Save for reproducibility
        pilot_path = save_dir / "pilot_indices.json"
        with open(pilot_path, "w") as f:
            json.dump(pilot_indices, f)
        print(f"Saved pilot indices to {pilot_path}")

    if pilot_indices is not None:
        # Restrict sampler to pilot subset
        pilot_set = set(pilot_indices)
        sampler.typed_indices = [i for i in sampler.typed_indices if i in pilot_set]
        sampler.untyped_indices = [i for i in sampler.untyped_indices if i in pilot_set]
        # Rebuild group indices
        for g in list(sampler.group_to_indices.keys()):
            sampler.group_to_indices[g] = [i for i in sampler.group_to_indices[g] if i in pilot_set]
        for key in list(sampler.group_product_to_indices.keys()):
            sampler.group_product_to_indices[key] = [
                i for i in sampler.group_product_to_indices[key] if i in pilot_set
            ]
        # Filter product_group_to_indices
        for key in list(sampler.product_group_to_indices.keys()):
            sampler.product_group_to_indices[key] = [
                i for i in sampler.product_group_to_indices[key] if i in pilot_set
            ]
        # Rebuild group_products (remove products with empty index lists)
        sampler.group_products = defaultdict(list)
        for (g, p), idxs in sampler.group_product_to_indices.items():
            if g != "other" and idxs and p not in sampler.group_products[g]:
                sampler.group_products[g].append(p)
        # Rebuild product_to_groups
        sampler.product_to_groups = defaultdict(list)
        for (p, g), idxs in sampler.product_group_to_indices.items():
            if idxs and g not in sampler.product_to_groups[p]:
                sampler.product_to_groups[p].append(g)
        # Remove empty groups
        sampler.groups = [g for g in sampler.groups if sampler.group_to_indices.get(g)]
        print(f"Pilot subset active: {len(sampler.typed_indices)} typed, {len(sampler.untyped_indices)} untyped")

    # =========================================
    # Initialize Pipeline
    # =========================================
    print("\nInitializing models...")
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    # IP-Adapter
    print(f"\nLoading IP-Adapter ({ip_adapter_type}, K={ip_adapter_k}, visual_mode={visual_mode})...")
    ip_adapter = create_ip_adapter(
        pipeline, adapter_type=ip_adapter_type, num_tokens=ip_adapter_k,
        scale=ip_adapter_scale, load_pretrained=True, mask_visual=mask_visual,
        visual_mode=visual_mode, learnable_gates=learnable_gates,
        force_gates=force_gates, sa_num_layers=sa_num_layers,
        sa_num_heads=sa_num_heads,
    )
    ip_adapter.freeze_image_encoder()

    # T2I-Adapter
    t2i_adapter = None
    if t2i_adapter_mode != "off":
        from src.models.t2i_adapter import T2IAdapter
        t2i_adapter = T2IAdapter(in_channels=2, injection_mode=t2i_adapter_mode).to(device)
        n_t2i = sum(p.numel() for p in t2i_adapter.parameters())
        print(f"\nT2I-Adapter ({t2i_adapter_mode}): {n_t2i:,} params")

    # Ensure fp32
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    # =========================================
    # Optimizer
    # =========================================
    group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
    group_b_modules = []
    if hasattr(ip_adapter, 'masked_self_attn'):
        group_b_modules.append(ip_adapter.masked_self_attn)
    if t2i_adapter is not None:
        group_b_modules.append(t2i_adapter)

    norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)
    a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)

    # Group C: gates/scales
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

    b_decay_raw, b_no_decay_raw = split_decay_no_decay(group_b_modules, norm_ids)
    b_decay = [p for p in b_decay_raw if id(p) not in group_c_ids]
    b_no_decay = [p for p in b_no_decay_raw if id(p) not in group_c_ids]

    param_groups = [
        {"params": a_decay,     "lr": lr_pretrained, "weight_decay": 1e-4, "label": "A_decay"},
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

    trainable_params = [p for pg in param_groups for p in pg["params"]]
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"\n  Trainable parameters: {n_trainable:,}")
    for pg in param_groups:
        n = sum(p.numel() for p in pg["params"])
        print(f"    {pg['label']:12s}: {n:>10,} params  (lr={pg['lr']:.1e}, wd={pg['weight_decay']:.1e})")

    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    # L2-SP
    l2sp_reg = None
    if lambda_sp > 0 and len(a_decay) > 0:
        l2sp_reg = L2SPRegularizer(a_decay, lambda_sp=lambda_sp)
        print(f"\n  L2-SP: {l2sp_reg.total_elements:,} elements, lambda={lambda_sp:.1e}")

    # bf16 autocast: same exponent range as fp32 (max ~3.4e38), eliminates
    # fp16 overflow (65504 ceiling) in attention dot products and sums.
    use_amp = True
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pipeline.text_encoder.float()
    pipeline.vae.float()
    pipeline.dtype = torch.float32

    # =========================================
    # Resume
    # =========================================
    start_step = 0
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
            print(f"  Loaded IP-Adapter weights")

        state_path = resume_dir / "training_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=device)
            try:
                optimizer.load_state_dict(state["optimizer"])
                print("  Loaded optimizer state")
            except Exception as e:
                print(f"  WARNING: Could not load optimizer ({e})")
            start_step = state.get("step", 0)
            print(f"  Resuming from step {start_step}")
            if t2i_adapter is not None and "t2i_adapter" in state:
                t2i_adapter.load_state_dict(state["t2i_adapter"])
                print("  Loaded T2I-Adapter weights")
            del state
            torch.cuda.empty_cache()

    remaining_steps = n_steps - start_step
    if remaining_steps <= 0:
        print(f"\nAlready completed {start_step}/{n_steps} steps.")
        return

    # =========================================
    # Save run config
    # =========================================
    run_config = {
        "n_steps": n_steps,
        "strategy": strategy,
        "host_batch_size": host_batch_size,
        "lambda_inv": lambda_inv,
        "lambda_rank": lambda_rank,
        "lambda_rank_untyped": lambda_rank_untyped,
        "lambda_triplet": lambda_triplet,
        "rank_gamma_scale": rank_gamma_scale,
        "triplet_margin_m": triplet_margin_m,
        "regularizer_warmup_frac": regularizer_warmup_frac,
        "visual_mode": visual_mode,
        "sa_num_layers": sa_num_layers,
        "sa_num_heads": sa_num_heads,
        "learnable_gates": learnable_gates,
        "force_gates": force_gates,
        "lr": lr,
        "lr_pretrained": lr_pretrained,
        "multi_crop": multi_crop,
        "band_mode": band_mode,
        "noise_offset": noise_offset,
        "timestep_sampling": timestep_sampling,
        "typed_neg_mode": typed_neg_mode,
        "p_null_typed_neg": p_null_typed_neg,
        "triplet_margin_auto": triplet_margin_auto,
        "triplet_margin_alpha": triplet_margin_alpha,
        "triplet_core_only": triplet_core_only,
        "triplet_mode": triplet_mode,
        "triplet_softplus_k": triplet_softplus_k,
        "triplet_softplus_offset": triplet_softplus_offset,
        "p_neg_same_host": p_neg_same_host,
        "p_neg_cross_host": p_neg_cross_host,
    }
    with open(save_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # Launch live loss viewer
    if not no_live_viewer:
        _launch_live_loss_contrastive(save_dir / "stats.csv")

    # =========================================
    # Training Loop
    # =========================================
    # Preflight summary
    print()
    print("+" + "=" * 44 + "+")
    print("|" + "         PREFLIGHT CHECK".ljust(44) + "|")
    print("+" + "=" * 44 + "+")
    print(f"| triplet_mode: {triplet_mode}".ljust(45) + "|")
    if triplet_mode == "softplus":
        print(f"| softplus k={triplet_softplus_k}, offset={triplet_softplus_offset}".ljust(45) + "|")
    print(f"| neg mode: {typed_neg_mode}".ljust(45) + "|")
    if typed_neg_mode == "weighted":
        print(f"| neg weights: sh={p_neg_same_host} ch={p_neg_cross_host}".ljust(45) + "|")
    print(f"| p_null_typed_neg: {p_null_typed_neg}".ljust(45) + "|")
    if strategy == "A":
        print(f"| lambda_inv: {lambda_inv}, lambda_rank: {lambda_rank}".ljust(45) + "|")
        print(f"| lambda_triplet: {lambda_triplet}".ljust(45) + "|")
        margin_desc = f"manual={triplet_margin}" if triplet_margin is not None else f"auto (alpha={triplet_margin_alpha})" if triplet_margin_auto else f"fixed={triplet_margin_m}"
        print(f"| triplet_margin: {margin_desc}".ljust(45) + "|")
        print(f"| core-only (rank+triplet): {triplet_core_only}".ljust(45) + "|")
    else:
        print(f"| lambda_triplet: {lambda_triplet}".ljust(45) + "|")
    print(f"| L_diff weighting: uniform".ljust(45) + "|")
    print(f"| L_diff routing: see above".ljust(45) + "|")
    print(f"| Weighted traces in loss plot: YES".ljust(45) + "|")
    print("+" + "=" * 44 + "+")
    print()

    print(f"Starting training for {remaining_steps} steps ({start_step}/{n_steps} done)...")

    stats_file = save_dir / "stats.csv"
    stats_fh = open(stats_file, "a")
    if stats_fh.tell() == 0:
        stats_fh.write(
            "step,loss_total,L_diff,L_diff_core,L_diff_band,"
            "L_inv,L_rank_typed,L_rank_untyped,L_triplet,"
            "d_same_mean,d_same_std,d_same_null_mean,d_diff_mean,d_diff_std,d_diff_null_mean,"
            "pct_rank_satisfied_typed,pct_rank_satisfied_untyped,"
            "L_true_mean,L_wrong_mean,L_true_untyped_mean,L_null_untyped_mean,"
            "gamma,grad_norm,attn_gate,ff_gate,lr_pretrained,lr_scratch,"
            "n_typed,n_untyped,"
            "rank_sat_same_host,rank_sat_cross_host,rank_sat_ip_zero,rank_sat_untyped,"
            "n_rank_same_host,n_rank_cross_host,n_rank_ip_zero,n_rank_untyped,"
            "margin_typed_mean,margin_typed_median,margin_typed_p10,margin_typed_p90,"
            "margin_typed_frac_above_gamma,"
            "margin_untyped_mean,margin_untyped_median,margin_untyped_p10,margin_untyped_p90,"
            "margin_untyped_frac_above_gamma,"
            "triplet_margin,L_triplet_weighted,triplet_violation_rate,"
            "triplet_n_same_host,triplet_n_cross_host,triplet_n_skipped,gap_mean,"
            "gap_positive_rate,gap_pos_rate_same_host,gap_pos_rate_cross_host,"
            "gap_mean_same_host,gap_mean_cross_host,"
            "emb_global_norm,emb_anomaly_norm,emb_normal_norm,"
            "eps_norm_early,eps_norm_mid,eps_norm_late,"
            "cond_null_dist_mean,cond_null_dist_std,"
            "cfg_L_null,cfg_L_cond,cfg_L_cfg\n"
        )

    losses = []
    warmup_L_diff_accum = []
    warmup_gap_accum = []
    gamma = 0.0
    warmup_steps = int(n_steps * regularizer_warmup_frac)
    prev_ckpt_dir = None

    pbar = tqdm(range(start_step, n_steps), desc="Training", initial=start_step, total=n_steps)
    skipped_nan = 0

    for step in pbar:
        # Sample host batch
        host_anchors = sampler.sample_host_batch(
            host_batch_size=host_batch_size,
            p_null_typed_neg=p_null_typed_neg,
            typed_neg_mode=typed_neg_mode,
            p_neg_same_host=p_neg_same_host,
            p_neg_cross_host=p_neg_cross_host,
        )

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            # Build variant batch
            vbatch = build_variant_batch(
                host_anchors, sampler, pipeline, ip_adapter, t2i_adapter,
                band_mode=band_mode, loss_core_ratio=loss_core_ratio,
                multi_crop=multi_crop, clip_align=clip_align,
                noise_offset=noise_offset,
                timestep_sampling=timestep_sampling,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
                device=device,
            )
            # UNet forward (single pass for all variants)
            if variant_chunk_size > 0 and vbatch.model_input.shape[0] > variant_chunk_size:
                # Chunked forward to avoid OOM
                V = vbatch.model_input.shape[0]
                eps_chunks = []
                for chunk_start in range(0, V, variant_chunk_size):
                    chunk_end = min(chunk_start + variant_chunk_size, V)
                    chunk_kwargs = {"ip_adapter_image_embeds": vbatch.ip_image_embeds[chunk_start:chunk_end]}
                    chunk_kwargs["ip_adapter_mask"] = vbatch.cross_attn_kwargs["ip_adapter_mask"][chunk_start:chunk_end]
                    if "null_token_mask" in vbatch.cross_attn_kwargs:
                        chunk_kwargs["null_token_mask"] = vbatch.cross_attn_kwargs["null_token_mask"][chunk_start:chunk_end]

                    chunk_t2i = {}
                    if vbatch.t2i_kwargs:
                        for k, v in vbatch.t2i_kwargs.items():
                            if isinstance(v, list):
                                chunk_t2i[k] = [feat[chunk_start:chunk_end] for feat in v]
                            elif isinstance(v, torch.Tensor):
                                chunk_t2i[k] = v[chunk_start:chunk_end]
                            else:
                                chunk_t2i[k] = v

                    chunk_pred = pipeline.unet(
                        vbatch.model_input[chunk_start:chunk_end],
                        vbatch.timesteps[chunk_start:chunk_end],
                        encoder_hidden_states=vbatch.text_emb[chunk_start:chunk_end],
                        cross_attention_kwargs=chunk_kwargs,
                        **chunk_t2i,
                    ).sample.float()
                    torch.cuda.synchronize()  # TDR prevention
                    eps_chunks.append(chunk_pred)
                eps_pred = torch.cat(eps_chunks, dim=0)
            else:
                eps_pred = pipeline.unet(
                    vbatch.model_input,
                    vbatch.timesteps,
                    encoder_hidden_states=vbatch.text_emb,
                    cross_attention_kwargs=vbatch.cross_attn_kwargs,
                    **vbatch.t2i_kwargs,
                ).sample.float()
                torch.cuda.synchronize()  # TDR prevention

        # Loss computed OUTSIDE autocast — sum of 16K squared fp16 values
        # can overflow fp16 in _weighted_l2; fp32 loss is essentially free
        in_warmup = step < warmup_steps

        if in_warmup:
            # Only L_diff during warmup (but compute distances for calibration)
            loss_total, extras = compute_contrastive_loss(
                eps_pred, vbatch,
                strategy=strategy,
                lambda_inv=0.0, lambda_rank=0.0,
                lambda_rank_untyped=0.0, lambda_triplet=0.0,
                gamma=0.0, triplet_margin_m=triplet_margin_m,
                use_untyped_rank_null=use_untyped_rank_null,
                triplet_core_only=triplet_core_only,
                triplet_mode=triplet_mode,
                triplet_softplus_k=triplet_softplus_k,
                triplet_softplus_offset=triplet_softplus_offset,
            )
            warmup_L_diff_accum.append(extras["L_diff"])
            if extras.get("gap_mean", 0.0) != 0.0:
                warmup_gap_accum.append(extras["gap_mean"])
        else:
            # Freeze gamma + triplet margin at end of warmup
            if step == warmup_steps and warmup_L_diff_accum:
                L_ref = np.mean(warmup_L_diff_accum)
                gamma = rank_gamma_scale * L_ref
                print(f"\n  Warmup done. L_ref={L_ref:.4f}, gamma={gamma:.4f}")

                # Triplet margin auto-calibration (hinge only; softplus uses offset)
                if triplet_mode == "hinge":
                    if triplet_margin is not None:
                        triplet_margin_m = triplet_margin
                    elif triplet_margin_auto and warmup_gap_accum:
                        gap_ref = float(np.median(warmup_gap_accum))
                        triplet_margin_m = max(0.005, triplet_margin_alpha * gap_ref)
                    print(f"  Triplet margin: {triplet_margin_m:.4f}")
                else:
                    print(f"  Softplus triplet (k={triplet_softplus_k}, offset={triplet_softplus_offset})")

            loss_total, extras = compute_contrastive_loss(
                eps_pred, vbatch,
                strategy=strategy,
                lambda_inv=lambda_inv,
                lambda_rank=lambda_rank,
                lambda_rank_untyped=lambda_rank_untyped,
                lambda_triplet=lambda_triplet,
                gamma=gamma,
                triplet_margin_m=triplet_margin_m,
                use_untyped_rank_null=use_untyped_rank_null,
                triplet_core_only=triplet_core_only,
                triplet_mode=triplet_mode,
                triplet_softplus_k=triplet_softplus_k,
                triplet_softplus_offset=triplet_softplus_offset,
            )

        # L2-SP
        l2sp_val = 0.0
        if l2sp_reg is not None:
            l2sp_loss = l2sp_reg.compute()
            l2sp_val = l2sp_loss.item()
            loss_total = loss_total + l2sp_loss

        if torch.isnan(loss_total) or torch.isinf(loss_total):
            skipped_nan += 1
            del eps_pred, loss_total, vbatch
            gc.collect()
            torch.cuda.empty_cache()
            continue

        scaler.scale(loss_total).backward()
        torch.cuda.synchronize()  # TDR prevention
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0).item()
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss_total.item()
        losses.append(extras["L_diff"])

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

        # Role embedding norms
        emb_g = emb_a = emb_n = 0.0
        if hasattr(ip_adapter, 'masked_self_attn'):
            sa = ip_adapter.masked_self_attn
            emb_g = sa.emb_global.data.norm().item()
            emb_a = sa.emb_anomaly.data.norm().item()
            emb_n = sa.emb_normal.data.norm().item()

        lr_pre = optimizer.param_groups[0]["lr"]
        lr_scr = optimizer.param_groups[2]["lr"]

        # Write stats
        stats_fh.write(
            f"{step},{loss_val},{extras['L_diff']},{extras['L_diff_core']},{extras['L_diff_band']},"
            f"{extras['L_inv']},{extras['L_rank_typed']},{extras['L_rank_untyped']},{extras['L_triplet']},"
            f"{extras['d_same_mean']},{extras['d_same_std']},{extras['d_same_null_mean']},{extras['d_diff_mean']},{extras['d_diff_std']},{extras['d_diff_null_mean']},"
            f"{extras['pct_rank_satisfied_typed']},{extras['pct_rank_satisfied_untyped']},"
            f"{extras['L_true_mean']},{extras['L_wrong_mean']},{extras['L_true_untyped_mean']},{extras['L_null_untyped_mean']},"
            f"{gamma},{grad_norm},{attn_gate},{ff_gate},{lr_pre},{lr_scr},"
            f"{extras['n_typed']},{extras['n_untyped']},"
            f"{extras['rank_sat_same_host']},{extras['rank_sat_cross_host']},"
            f"{extras['rank_sat_ip_zero']},{extras['rank_sat_untyped']},"
            f"{extras['n_rank_same_host']},{extras['n_rank_cross_host']},"
            f"{extras['n_rank_ip_zero']},{extras['n_rank_untyped']},"
            f"{extras['margin_typed_mean']},{extras['margin_typed_median']},"
            f"{extras['margin_typed_p10']},{extras['margin_typed_p90']},"
            f"{extras['margin_typed_frac_above_gamma']},"
            f"{extras['margin_untyped_mean']},{extras['margin_untyped_median']},"
            f"{extras['margin_untyped_p10']},{extras['margin_untyped_p90']},"
            f"{extras['margin_untyped_frac_above_gamma']},"
            f"{extras['triplet_margin']},{extras['L_triplet_weighted']},{extras['triplet_violation_rate']},"
            f"{extras['triplet_n_same_host']},{extras['triplet_n_cross_host']},"
            f"{extras['triplet_n_skipped']},{extras['gap_mean']},"
            f"{extras['gap_positive_rate']},{extras['gap_pos_rate_same_host']},"
            f"{extras['gap_pos_rate_cross_host']},"
            f"{extras['gap_mean_same_host']},{extras['gap_mean_cross_host']},"
            f"{emb_g},{emb_a},{emb_n},"
            f"{extras['eps_norm_early']},{extras['eps_norm_mid']},{extras['eps_norm_late']},"
            f"{extras['cond_null_dist_mean']},{extras['cond_null_dist_std']},"
            f"{extras['cfg_L_null']},{extras['cfg_L_cond']},{extras['cfg_L_cfg']}\n"
        )
        stats_fh.flush()

        if step % 50 == 0:
            avg = sum(losses[-100:]) / max(len(losses[-100:]), 1)
            pbar.set_postfix(
                L_diff=f"{avg:.4f}",
                d_same=f"{extras['d_same_mean']:.3f}",
                rank_sat=f"{extras['pct_rank_satisfied_typed']:.0%}",
            )

        # Cleanup
        del vbatch, eps_pred, loss_total, extras

        # Early snapshots
        if (step + 1) in (500, 1000, 2000) and (step + 1) % save_every != 0:
            torch.cuda.empty_cache()
            save_contrastive_loss_plot(stats_file, save_dir / f"loss_{step + 1}.png")

        # Checkpoint
        if (step + 1) % save_every == 0:
            print(f"\n  Checkpoint at step {step + 1}...")
            ckpt_dir = save_dir / f"checkpoint_{step + 1}"
            save_contrastive_checkpoint(
                ip_adapter, optimizer, ckpt_dir,
                band_mode=band_mode, t2i_adapter=t2i_adapter,
                step=step + 1, losses=losses,
            )
            if prev_ckpt_dir is not None and prev_ckpt_dir.exists():
                import shutil
                shutil.rmtree(prev_ckpt_dir)
            prev_ckpt_dir = ckpt_dir

            save_contrastive_loss_plot(stats_file, save_dir / f"loss_{step + 1}.png")

            torch.cuda.empty_cache()
            try:
                run_eval_probes(
                    pipeline, ip_adapter, sampler, save_dir,
                    step + 1, device, band_mode=band_mode,
                    t2i_adapter=t2i_adapter,
                )
            except RuntimeError as e:
                print(f"  Warning: eval probes failed ({e})")
            torch.cuda.empty_cache()

    stats_fh.close()

    # =========================================
    # Final outputs
    # =========================================
    print("\nTraining complete!")
    print(f"  Total steps: {n_steps}")
    print(f"  Skipped (NaN/Inf): {skipped_nan}")

    save_contrastive_loss_plot(stats_file, save_dir / "training_loss.png")

    save_contrastive_checkpoint(
        ip_adapter, optimizer, save_dir / "checkpoint_final",
        band_mode=band_mode, t2i_adapter=t2i_adapter,
        step=n_steps, losses=losses,
    )

    torch.cuda.empty_cache()
    try:
        run_eval_probes(
            pipeline, ip_adapter, sampler, save_dir,
            n_steps, device, band_mode=band_mode,
            t2i_adapter=t2i_adapter,
        )
    except RuntimeError as e:
        print(f"  Warning: final eval probes failed ({e})")

    print(f"\nResults saved to: {save_dir}")


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Contrastive CLIP-conditioning regularization training",
    )

    # Data
    parser.add_argument("--data-json", type=str, required=True,
                        help="Path to contrastive_training.json")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory for resolving relative image paths")
    parser.add_argument("--captions-file", type=str, required=True,
                        help="Path to captions.json")

    # Strategy
    parser.add_argument("--strategy", type=str, default="A", choices=["A", "B"],
                        help="A = L_inv + L_rank, B = L_triplet")
    parser.add_argument("--host-batch-size", type=int, default=4,
                        help="Number of host anchors per step")
    parser.add_argument("--p-null-typed-neg", type=float, default=0.20,
                        help="Fraction of typed negs replaced by null CLIP")
    parser.add_argument("--typed-neg-mode", type=str, default="same_host_diff_type",
                        choices=["same_host_diff_type", "cross_host_diff_type", "hybrid", "weighted"],
                        help="How to sample typed negatives")
    parser.add_argument("--no-untyped-rank-null", action="store_true",
                        help="Disable L_rank for untyped (true vs null)")

    # Loss weights
    parser.add_argument("--lambda-inv", type=float, default=1.0,
                        help="L_inv weight (Strategy A)")
    parser.add_argument("--lambda-rank", type=float, default=5.0,
                        help="L_rank weight (Strategy A, typed)")
    parser.add_argument("--lambda-rank-untyped", type=float, default=0.1,
                        help="L_rank weight (untyped)")
    parser.add_argument("--lambda-triplet", type=float, default=0.25,
                        help="L_triplet weight (Strategy A + B)")

    # Margins / schedules
    parser.add_argument("--rank-gamma-scale", type=float, default=0.05,
                        help="gamma = scale * L_ref")
    parser.add_argument("--triplet-margin-m", type=float, default=0.10,
                        help="Normalized triplet margin (fallback if auto disabled)")
    parser.add_argument("--triplet-margin", type=float, default=None,
                        help="Manual triplet margin override (None = auto)")
    parser.add_argument("--no-triplet-margin-auto", dest="triplet_margin_auto",
                        action="store_false", default=True,
                        help="Disable auto-calibration of triplet margin")
    parser.add_argument("--triplet-margin-alpha", type=float, default=0.5,
                        help="Fraction of median gap for auto margin")
    parser.add_argument("--no-triplet-core-only", dest="triplet_core_only",
                        action="store_false", default=True,
                        help="Use 80/20 weighting instead of core-only for rank+triplet")
    parser.add_argument("--regularizer-warmup-frac", type=float, default=0.05,
                        help="Warmup fraction (L_diff only, no regularizers)")

    # Triplet mode
    parser.add_argument("--triplet-mode", type=str, default="hinge",
                        choices=["hinge", "softplus"],
                        help="Triplet loss mode: hinge (margin) or softplus (smooth)")
    parser.add_argument("--triplet-softplus-k", type=float, default=20.0,
                        help="Softplus sharpness (higher = sharper)")
    parser.add_argument("--triplet-softplus-offset", type=float, default=0.0,
                        help="Optional soft margin offset for softplus mode (default 0.0)")

    # Weighted neg sampling
    parser.add_argument("--p-neg-same-host", type=float, default=0.6,
                        help="Probability weight for same-host negatives (weighted mode)")
    parser.add_argument("--p-neg-cross-host", type=float, default=0.3,
                        help="Probability weight for cross-host negatives (weighted mode)")

    # Variant batching
    parser.add_argument("--variant-chunk-size", type=int, default=0,
                        help="0 = full parallel, >0 = chunk into sequential UNet forwards")

    # Pilot subset
    parser.add_argument("--pilot-subset-path", type=str, default=None,
                        help="Path to curated subset indices JSON")
    parser.add_argument("--pilot-subset-size", type=int, default=0,
                        help="Auto-generate curated subset of this size (0=use all)")

    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-pretrained", type=float, default=1e-4)
    parser.add_argument("--lambda-sp", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=12,
                        help="(Unused — host-batch-size controls batch. Kept for compatibility.)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default="results/contrastive_training")

    # Model
    parser.add_argument("--ip-adapter-type", type=str, default="plus", choices=["standard", "plus"])
    parser.add_argument("--ip-adapter-k", type=int, default=16, choices=[1, 4, 16])
    parser.add_argument("--ip-adapter-scale", type=float, default=1.0)
    parser.add_argument("--no-mask-visual", action="store_true")
    parser.add_argument("--visual-mode", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--no-learnable-gates", action="store_true")
    parser.add_argument("--force-gates", action="store_true")
    parser.add_argument("--sa-num-layers", type=int, default=3)
    parser.add_argument("--sa-num-heads", type=int, default=12)

    # Conditioning
    parser.add_argument("--band-mode", type=int, default=2, choices=[1, 2])
    parser.add_argument("--loss-core-ratio", type=float, default=0.8)
    parser.add_argument("--t2i-adapter-mode", type=str, default="cascade",
                        choices=["cascade", "skip_only", "off"])
    parser.add_argument("--cfg-mode", type=str, default="visual",
                        choices=["text", "visual", "both"])

    # Data
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--no-multi-crop", action="store_true")
    parser.add_argument("--no-clip-align", action="store_true")
    parser.add_argument("--no-live-viewer", action="store_true")
    parser.add_argument("--noise-offset", type=float, default=0.05)
    parser.add_argument("--timestep-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--logit-normal-mean", type=float, default=0.0)
    parser.add_argument("--logit-normal-std", type=float, default=1.0)

    parser.add_argument("--diff-only", action="store_true",
                        help="L_diff only: zero all regularizer lambdas (inv/rank/triplet). "
                             "Distances still logged for observation.")

    args = parser.parse_args()

    # --diff-only: override all regularizer lambdas to 0
    if args.diff_only:
        args.lambda_inv = 0.0
        args.lambda_rank = 0.0
        args.lambda_rank_untyped = 0.0
        args.lambda_triplet = 0.0

    project_root = Path(__file__).parent.parent

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    train_contrastive(
        data_json=_resolve(args.data_json),
        save_dir=_resolve(args.save_dir),
        captions_file=_resolve(args.captions_file),
        data_root=_resolve(args.data_root),
        n_steps=args.steps,
        save_every=args.save_every,
        host_batch_size=args.host_batch_size,
        lr=args.lr,
        lr_pretrained=args.lr_pretrained,
        lambda_sp=args.lambda_sp,
        device=args.device,
        strategy=args.strategy,
        lambda_inv=args.lambda_inv,
        lambda_rank=args.lambda_rank,
        lambda_rank_untyped=args.lambda_rank_untyped,
        lambda_triplet=args.lambda_triplet,
        rank_gamma_scale=args.rank_gamma_scale,
        triplet_margin_m=args.triplet_margin_m,
        regularizer_warmup_frac=args.regularizer_warmup_frac,
        p_null_typed_neg=args.p_null_typed_neg,
        typed_neg_mode=args.typed_neg_mode,
        use_untyped_rank_null=not args.no_untyped_rank_null,
        ip_adapter_type=args.ip_adapter_type,
        ip_adapter_k=args.ip_adapter_k,
        ip_adapter_scale=args.ip_adapter_scale,
        mask_visual=not args.no_mask_visual,
        band_mode=args.band_mode,
        loss_core_ratio=args.loss_core_ratio,
        t2i_adapter_mode=args.t2i_adapter_mode,
        augment=args.augment,
        multi_crop=not args.no_multi_crop,
        clip_align=not args.no_clip_align,
        visual_mode=args.visual_mode,
        learnable_gates=not args.no_learnable_gates,
        force_gates=args.force_gates,
        sa_num_layers=args.sa_num_layers,
        sa_num_heads=args.sa_num_heads,
        cfg_mode=args.cfg_mode,
        no_live_viewer=args.no_live_viewer,
        noise_offset=args.noise_offset,
        timestep_sampling=args.timestep_sampling,
        logit_normal_mean=args.logit_normal_mean,
        logit_normal_std=args.logit_normal_std,
        resume_dir=_resolve(args.resume) if args.resume else None,
        variant_chunk_size=args.variant_chunk_size,
        pilot_subset_path=_resolve(args.pilot_subset_path) if args.pilot_subset_path else None,
        pilot_subset_size=args.pilot_subset_size,
        triplet_margin_auto=args.triplet_margin_auto,
        triplet_margin_alpha=args.triplet_margin_alpha,
        triplet_margin=args.triplet_margin,
        triplet_core_only=args.triplet_core_only,
        triplet_mode=args.triplet_mode,
        triplet_softplus_k=args.triplet_softplus_k,
        triplet_softplus_offset=args.triplet_softplus_offset,
        p_neg_same_host=args.p_neg_same_host,
        p_neg_cross_host=args.p_neg_cross_host,
    )
