"""Trace exactly where fp16 overflow happens with zero-mask vs normal-mask samples.

Hooks every UNet module to record max |activation| values.
Compares zero-mask sample (BTech plate 0032) vs a normal sample.
"""
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_contrastive import (
    ContrastiveHostSampler,
    _load_image_mask,
    _load_clip_reference,
    build_variant_batch,
)
from src.utils.mask_utils import create_latent_band_mask


def trace_nan():
    device = "cuda"
    data_json = "anomverse_extension/datasets/full_training_dataset/contrastive_training.json"
    captions_file = "anomverse_extension/datasets/full_training_dataset/captions_from_master.json"
    data_root = "anomverse_extension/datasets/full_training_dataset"

    # --- Load models ---
    print("Loading models...")
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter

    pipeline = create_pipeline("sd_1.5", device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()

    ip_adapter = create_ip_adapter(
        pipeline, adapter_type="plus", num_tokens=16,
        scale=1.0, load_pretrained=True, mask_visual=True,
        visual_mode=3, learnable_gates=True,
        force_gates=False, sa_num_layers=3, sa_num_heads=12,
    )
    ip_adapter.freeze_image_encoder()
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    from src.models.t2i_adapter import T2IAdapter
    t2i_adapter = T2IAdapter(in_channels=2, injection_mode="cascade").to(device)

    sampler = ContrastiveHostSampler(data_json, data_root, captions_file=captions_file)

    # --- Find the zero-mask sample and a normal sample ---
    zero_idx = None
    normal_idx = None
    for i, s in enumerate(sampler.all_samples):
        if "BTech/BTech/03/test/ko/0032" in s.get("image_path", ""):
            zero_idx = i
        if normal_idx is None and s.get("dataset") != "BTech":
            img, msk = _load_image_mask(s, Path(data_root))
            if msk.sum() > 100:  # decent mask
                normal_idx = i
        if zero_idx is not None and normal_idx is not None:
            break

    print(f"Zero-mask sample index: {zero_idx}")
    print(f"Normal sample index: {normal_idx}")

    # --- Compare CLIP embeddings ---
    for label, idx in [("ZERO-MASK", zero_idx), ("NORMAL", normal_idx)]:
        s = sampler.all_samples[idx]
        img, msk = _load_image_mask(s, Path(data_root))
        clip_ref = _load_clip_reference(s, Path(data_root), band_mode=2, multi_crop=True, clip_align=True)

        print(f"\n{'='*60}")
        print(f"  {label}: {s['image_path'][-50:]}")
        print(f"  mask coverage: {msk.mean().item():.6f}, nonzero: {msk.sum().item():.0f}")

        # CLIP encode
        ref = clip_ref["reference"].unsqueeze(0).to(device)  # [1, 3, 224, 224]
        ref_01 = (ref + 1.0) / 2.0
        clip_mask = clip_ref["clip_mask"].unsqueeze(0).to(device)
        core_mask = clip_ref.get("clip_core_mask")
        if core_mask is not None:
            core_mask = core_mask.unsqueeze(0).to(device)

        with torch.amp.autocast(device_type="cuda"):
            embeds = ip_adapter.encode_image(ref_01, mask=clip_mask, core_mask=core_mask)

        print(f"  CLIP embeds shape: {embeds.shape}")
        print(f"  CLIP embeds: min={embeds.min().item():.4f}, max={embeds.max().item():.4f}, "
              f"mean={embeds.mean().item():.4f}, std={embeds.std().item():.4f}")
        print(f"  CLIP embeds abs max: {embeds.abs().max().item():.4f}")
        print(f"  Any NaN: {embeds.isnan().any().item()}, Any Inf: {embeds.isinf().any().item()}")

    # --- Now run full UNet forward with hooks to trace max values ---
    print(f"\n\n{'='*60}")
    print("TRACING UNet FORWARD PASS WITH HOOKS")
    print(f"{'='*60}")

    # Hook storage
    max_vals = OrderedDict()
    nan_locations = []
    inf_locations = []

    def make_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                t = output
            elif isinstance(output, tuple) and len(output) > 0:
                t = output[0] if isinstance(output[0], torch.Tensor) else None
            else:
                t = None

            if t is not None:
                abs_max = t.abs().max().item()
                has_nan = t.isnan().any().item()
                has_inf = t.isinf().any().item()
                max_vals[name] = abs_max
                if has_nan:
                    nan_locations.append((name, abs_max, t.shape, t.dtype))
                if has_inf:
                    inf_locations.append((name, abs_max, t.shape, t.dtype))
        return hook

    # Register hooks on ALL UNet submodules
    hooks = []
    for name, module in pipeline.unet.named_modules():
        if name:  # skip root
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Build a minimal batch for each sample type
    import random, numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    for label, idx in [("NORMAL", normal_idx), ("ZERO-MASK", zero_idx)]:
        max_vals.clear()
        nan_locations.clear()
        inf_locations.clear()

        s = sampler.all_samples[idx]
        img, msk = _load_image_mask(s, Path(data_root))
        clip_ref = _load_clip_reference(s, Path(data_root), band_mode=2, multi_crop=True, clip_align=True)

        # Build single-sample UNet inputs manually
        img_t = img.unsqueeze(0).to(device)  # [1, 3, 512, 512]
        msk_t = msk.unsqueeze(0).to(device)  # [1, 1, 512, 512]

        # Encode to latents
        with torch.no_grad():
            latents = pipeline.encode_image(img_t)  # [1, 4, 64, 64]

        # Masks at 64x64
        msk_64 = F.max_pool2d(msk_t, kernel_size=8)  # [1, 1, 64, 64]
        dilated, alpha, weight_map, band = create_latent_band_mask(msk_64, band_mode=2)

        # Noise + timestep
        noise = torch.randn_like(latents)
        t_val = torch.tensor([500], device=device)
        noisy = pipeline.scheduler.add_noise(latents, noise, t_val)

        # Masked image
        masked_img = img_t * (1 - F.interpolate(msk_t, size=(512, 512), mode="nearest"))
        with torch.no_grad():
            masked_latents = pipeline.encode_image(masked_img)

        model_input = torch.cat([noisy, dilated, masked_latents], dim=1)  # [1, 9, 64, 64]

        # Text embedding
        caption = s.get("caption_short", "anomaly")
        text_emb = pipeline.encode_text([caption])  # [1, 77, 768]

        # CLIP embedding
        ref = clip_ref["reference"].unsqueeze(0).to(device)
        ref_01 = (ref + 1.0) / 2.0
        clip_mask_t = clip_ref["clip_mask"].unsqueeze(0).to(device)
        core_mask_t = clip_ref.get("clip_core_mask")
        if core_mask_t is not None:
            core_mask_t = core_mask_t.unsqueeze(0).to(device)

        with torch.amp.autocast(device_type="cuda"):
            ip_embeds = ip_adapter.encode_image(ref_01, mask=clip_mask_t, core_mask=core_mask_t)

        # Cross-attn kwargs
        cross_attn_kwargs = {
            "ip_adapter_image_embeds": ip_embeds,
            "ip_adapter_mask": alpha.to(device),
        }

        # T2I adapter
        t2i_input = torch.cat([msk_64, band], dim=1).to(device)
        t2i_features = t2i_adapter(t2i_input, mask=dilated.to(device))
        t2i_kwargs = {"down_intrablock_additional_residuals": t2i_features}

        print(f"\n--- {label} UNet Forward ---")
        print(f"  model_input: {model_input.shape}, dtype={model_input.dtype}")
        print(f"  ip_embeds: min={ip_embeds.min().item():.4f}, max={ip_embeds.max().item():.4f}, abs_max={ip_embeds.abs().max().item():.4f}")
        print(f"  text_emb: abs_max={text_emb.abs().max().item():.4f}")
        print(f"  alpha_map: sum={alpha.sum().item():.2f}")

        with torch.amp.autocast(device_type="cuda"):
            eps_pred = pipeline.unet(
                model_input, t_val,
                encoder_hidden_states=text_emb,
                cross_attention_kwargs=cross_attn_kwargs,
                **t2i_kwargs,
            ).sample

        print(f"  eps_pred: min={eps_pred.min().item():.4f}, max={eps_pred.max().item():.4f}")
        print(f"  eps_pred NaN: {eps_pred.isnan().any().item()}, Inf: {eps_pred.isinf().any().item()}")

        # Report top-20 max activation layers
        sorted_vals = sorted(max_vals.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 20 max |activation| layers:")
        for name, val in sorted_vals[:20]:
            flag = " *** OVERFLOW ***" if val >= 65504 else ""
            print(f"    {val:>12.2f}  {name}{flag}")

        if nan_locations:
            print(f"\n  *** NaN detected in {len(nan_locations)} layers: ***")
            for name, val, shape, dtype in nan_locations:
                print(f"    {name}: abs_max={val}, shape={shape}, dtype={dtype}")

        if inf_locations:
            print(f"\n  *** Inf detected in {len(inf_locations)} layers: ***")
            for name, val, shape, dtype in inf_locations:
                print(f"    {name}: abs_max={val}, shape={shape}, dtype={dtype}")

        if not nan_locations and not inf_locations:
            print(f"\n  No NaN/Inf in any layer (all clean)")

    # Cleanup hooks
    for h in hooks:
        h.remove()

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    trace_nan()
