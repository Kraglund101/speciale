#!/usr/bin/env python
"""Generate 100 diverse synthetic anomalies for visual inspection.

Samples from 10 diverse defect types (10 per type), loads the baseline_cosine_1e4_2e5_k1
checkpoint, and generates each with CFG=3.5 (visual mode). Saves comparison
grids (original | mask | generated) plus individual outputs.

Usage:
    python scripts/generate_100.py \
        --checkpoint important_results/baseline_cosine_1e4_2e5_k1/checkpoint_final \
        --output-dir results/synthetic_100 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generate import generate_anomagic_single
from src.utils.crop_utils import clip_crop_multi
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool, unet_roundtrip_masks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Config ───────────────────────────────────────────────────────────────────

# 10 diverse defect types × 10 samples each = 100 total
# Chosen to span surface defects, structural, contamination, missing parts, etc.
DEFECT_TYPES = [
    "scratch",          # Surface — 8850 samples
    "contamination",    # Surface — 14467 samples
    "pit",              # Surface — 6763 samples
    "crack",            # Structural — 189 samples
    "hole",             # Structural — 236 samples
    "deformation",      # Structural — 2569 samples
    "damage",           # Structural — 2120 samples
    "abrasion",         # Surface — 2909 samples
    "color",            # Appearance — 180 samples
    "cut",              # Structural — 201 samples
]

SAMPLES_PER_TYPE = 10
NUM_STEPS = 30
GUIDANCE_SCALE = 3.5
CFG_MODE = "visual"
NOISE_STRENGTH = 0.7
BAND_MODE = 2


def _load_image_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Load image → [1, 3, H, W] in [-1, 1]."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_mask_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    """Load mask → [1, 1, H, W] binary (maxpool downsample)."""
    pil = Image.open(path).convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy((arr > 0.5).astype(np.float32)).unsqueeze(0)  # [1, H, W]
    t = downsample_mask_maxpool(t, size)  # [1, size, size]
    return t.unsqueeze(0).float().to(device)  # [1, 1, size, size]


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """[1, 3, H, W] in [-1, 1] → PIL."""
    arr = ((t[0].detach().cpu().clamp(-1, 1).float() + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _prepare_clip_crops(
    img_path: Path, mask_path: Path, band_mode: int, clip_align: bool, device: str,
) -> Dict[str, Any]:
    """Load image + mask, run clip_crop_multi, return dict of tensors.

    Matches the validation_suite.py _prepare_clip_crops implementation exactly.
    """
    ref_pil = Image.open(img_path).convert("RGB")
    ref_mask_np = (
        np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0 > 0.5
    ).astype(np.float32)
    ref_tensor = torch.from_numpy(
        np.array(ref_pil).astype(np.float32) / 255.0
    ).permute(2, 0, 1)  # [3, H, W] in [0, 1]
    ref_mask_tensor = torch.from_numpy(ref_mask_np).unsqueeze(0).float()  # [1, H, W]

    if clip_align:
        core_native, dil_native = unet_roundtrip_masks(ref_mask_tensor, band_mode)
        crops, crop_masks, valid, extra = clip_crop_multi(
            ref_tensor, ref_mask_tensor, n_groups=2,
            clip_masks=[dil_native, core_native],
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (extra[0][0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": (extra[0][1] > 0.5).float().unsqueeze(0).to(device),
            "clip_ref_2": (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_2": (extra[1][0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_2": (extra[1][1] > 0.5).float().unsqueeze(0).to(device),
        }
    else:
        crops, crop_masks, valid = clip_crop_multi(
            ref_tensor, ref_mask_tensor, n_groups=2,
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (crop_masks[0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": None,
            "clip_ref_2": (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_2": (crop_masks[1] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_2": None,
        }

    result["group_valid"] = torch.tensor(
        [[float(valid[0]), float(valid[1])]], device=device,
    )
    # For visualisation (both crops)
    result["crop_np"] = (crops[0].permute(1, 2, 0).numpy()).clip(0, 1)
    result["crop_mask_np"] = crop_masks[0][0].numpy()
    result["crop_np_2"] = (crops[1].permute(1, 2, 0).numpy()).clip(0, 1)
    result["crop_mask_np_2"] = crop_masks[1][0].numpy()
    return result


def _load_checkpoint(checkpoint_dir: Path, device: str):
    """Load pipeline + IP-Adapter + T2I-Adapter from checkpoint."""
    from src.models.base import create_pipeline
    from src.models.ip_adapter import create_ip_adapter
    from src.models.t2i_adapter import T2IAdapter

    config_path = checkpoint_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    pipeline = create_pipeline(config.get("sd_version", "sd_1.5"), device=device)
    pipeline.load_pipeline()
    pipeline.freeze_all()
    pipeline.vae.float()

    ip_ckpt = checkpoint_dir / "ip_adapter.pt"
    if not ip_ckpt.exists():
        raise FileNotFoundError(f"No ip_adapter.pt in {checkpoint_dir}")

    state = torch.load(ip_ckpt, map_location="cpu", weights_only=True)
    sa_layers = 0
    for k in state.keys():
        if k.startswith("masked_self_attn.layers."):
            layer_idx = int(k.split(".")[2])
            sa_layers = max(sa_layers, layer_idx + 1)
    if sa_layers == 0:
        sa_layers = 3

    force_gates = config.get("force_gates", False)
    learnable_gates = config.get("learnable_gates", True)
    if not force_gates and "force_gates" not in config:
        has_gate_params = any("gate.gate" in k for k in state.keys())
        if not has_gate_params:
            force_gates = True
            learnable_gates = False
            print("  Detected force_gates=True (no gate params in checkpoint)")

    ip_adapter = create_ip_adapter(
        pipeline,
        adapter_type=config.get("adapter_type", "plus"),
        num_tokens=config.get("num_tokens", 16),
        scale=config.get("scale", 1.0),
        load_pretrained=False,
        anomaly_residual=config.get("anomaly_residual", False),
        mask_visual=config.get("mask_visual", True),
        visual_mode=config.get("visual_mode", 3),
        sa_num_layers=sa_layers,
        force_gates=force_gates,
        learnable_gates=learnable_gates,
    )
    ip_adapter.load_state_dict(state, strict=False)
    ip_adapter.to(device)
    ip_adapter.masked_self_attn.float()
    ip_adapter.image_projection.float()
    for proc in ip_adapter.attn_processors.values():
        proc.float()

    t2i_adapter = None
    training_state_path = checkpoint_dir / "training_state.pt"
    training_state = None
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)

    if training_state and "t2i_adapter" in training_state:
        t2i_state = training_state["t2i_adapter"]
        in_ch = 2
        for k, v in t2i_state.items():
            if "conv_in" in k and v.dim() == 4:
                in_ch = v.shape[1]
                break
        t2i_adapter = T2IAdapter(in_channels=in_ch).to(device).float()
        t2i_adapter.load_state_dict(t2i_state)
        print("  T2I-Adapter loaded from training_state.pt")

    # Apply EMA if available
    if training_state and "ema" in training_state:
        from src.utils.optim_utils import build_norm_param_id_set, split_decay_no_decay
        ema_state = training_state["ema"]
        shadow = ema_state["shadow"]

        group_a_modules = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
        group_b_modules = []
        if hasattr(ip_adapter, 'masked_self_attn'):
            group_b_modules.append(ip_adapter.masked_self_attn)
        if t2i_adapter is not None:
            group_b_modules.append(t2i_adapter)

        norm_ids = build_norm_param_id_set(group_a_modules + group_b_modules)
        a_decay, a_no_decay = split_decay_no_decay(group_a_modules, norm_ids)
        b_decay, b_no_decay = split_decay_no_decay(group_b_modules, norm_ids)

        gate_params = []
        if hasattr(ip_adapter, 'masked_self_attn'):
            for name, p in ip_adapter.masked_self_attn.named_parameters():
                if "gate" in name.lower():
                    gate_params.append(p)

        all_params = a_decay + a_no_decay + b_decay + b_no_decay + gate_params

        if len(shadow) == len(all_params):
            for p, s in zip(all_params, shadow):
                p.data.copy_(s.to(p.device))
            print("  EMA weights applied")
        else:
            print(f"  EMA size mismatch: {len(shadow)} vs {len(all_params)}, skipping")

    return pipeline, ip_adapter, t2i_adapter


def _sample_from_splits(
    splits_dir: Path, data_root: Path, defect_types: List[str],
    samples_per_type: int, seed: int,
) -> List[Dict[str, Any]]:
    """Sample training images from split files, ensuring diversity."""
    rng = random.Random(seed)
    samples = []

    for dtype in defect_types:
        split_path = splits_dir / f"{dtype}.json"
        if not split_path.exists():
            print(f"  WARNING: {split_path} not found, skipping {dtype}")
            continue

        with open(split_path) as f:
            data = json.load(f)
        images = data.get("images", [])

        # Filter to only images that exist on disk
        valid = []
        for img in images:
            img_path = data_root / img["image_path"]
            mask_path = data_root / img["mask_path"]
            if img_path.exists() and mask_path.exists():
                valid.append(img)

        if len(valid) == 0:
            print(f"  WARNING: no valid images for {dtype}")
            continue

        # Try to get diverse products
        by_product: Dict[str, list] = {}
        for img in valid:
            prod = img.get("product", "unknown")
            by_product.setdefault(prod, []).append(img)

        selected = []
        products = list(by_product.keys())
        rng.shuffle(products)

        # Round-robin across products
        idx = 0
        while len(selected) < samples_per_type and idx < 1000:
            prod = products[idx % len(products)]
            pool = by_product[prod]
            if pool:
                choice = rng.choice(pool)
                pool.remove(choice)
                selected.append(choice)
            idx += 1

        for img in selected:
            samples.append({
                "defect_type": dtype,
                "image_path": str(data_root / img["image_path"]),
                "mask_path": str(data_root / img["mask_path"]),
                "rel_image_path": img["image_path"],
                "product": img.get("product", "unknown"),
                "source_dataset": img.get("source_dataset", "unknown"),
            })

    return samples


def _create_comparison_grid(
    original: Image.Image, mask: Image.Image, generated: Image.Image,
    ref_crop: np.ndarray, ref_mask: np.ndarray,
) -> Image.Image:
    """Create a 5-column comparison: original | mask | reference | generated | overlay."""
    size = 512
    orig_r = original.resize((size, size), Image.LANCZOS)
    mask_r = mask.resize((size, size), Image.NEAREST).convert("RGB")
    gen_r = generated.resize((size, size), Image.LANCZOS)

    # Reference with mask overlay
    ref_pil = Image.fromarray((ref_crop * 255).clip(0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)

    # Overlay: blend original with generated using mask
    orig_arr = np.array(orig_r).astype(np.float32)
    gen_arr = np.array(gen_r).astype(np.float32)
    mask_arr = np.array(mask.resize((size, size), Image.NEAREST)).astype(np.float32) / 255.0
    mask_arr = mask_arr[..., None]
    overlay_arr = (gen_arr * mask_arr + orig_arr * (1 - mask_arr)).clip(0, 255).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr)

    grid = Image.new("RGB", (size * 5, size))
    grid.paste(orig_r, (0, 0))
    grid.paste(mask_r, (size, 0))
    grid.paste(ref_pil, (size * 2, 0))
    grid.paste(gen_r, (size * 3, 0))
    grid.paste(overlay, (size * 4, 0))

    return grid


def main():
    parser = argparse.ArgumentParser(description="Generate 100 diverse synthetic anomalies")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--output-dir", type=str, default="results/synthetic_100",
                        help="Output directory")
    parser.add_argument("--data-root", type=str,
                        default="anomverse_extension/datasets/full_training_dataset",
                        help="Root for training data")
    parser.add_argument("--captions-file", type=str,
                        default="anomverse_extension/datasets/full_training_dataset/captions_from_master.json",
                        help="Path to captions JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--band-mode", type=int, default=2, choices=[1, 2])
    parser.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE)
    parser.add_argument("--noise-strength", type=float, default=NOISE_STRENGTH)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--no-clip-align", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    checkpoint_dir = Path(args.checkpoint)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root

    splits_dir = data_root / "splits_by_type"
    clip_align = not args.no_clip_align

    # Load captions
    captions: Dict[str, str] = {}
    captions_path = Path(args.captions_file)
    if not captions_path.is_absolute():
        captions_path = project_root / captions_path
    if captions_path.exists():
        with open(captions_path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            captions = {entry["image_path"]: entry["caption"] for entry in raw}
        elif isinstance(raw, dict):
            captions = raw
        print(f"Loaded {len(captions)} captions")

    # Sample 100 diverse examples
    print(f"Sampling {len(DEFECT_TYPES)} × {SAMPLES_PER_TYPE} = {len(DEFECT_TYPES) * SAMPLES_PER_TYPE} examples...")
    samples = _sample_from_splits(splits_dir, data_root, DEFECT_TYPES, SAMPLES_PER_TYPE, args.seed)
    print(f"  Got {len(samples)} valid samples")

    if len(samples) == 0:
        print("ERROR: no valid samples found. Check --data-root.")
        sys.exit(1)

    # Load model
    print(f"\nLoading checkpoint from {checkpoint_dir}...")
    pipeline, ip_adapter, t2i_adapter = _load_checkpoint(checkpoint_dir, args.device)
    ip_adapter.eval()

    # Generate
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []

    for idx, sample in enumerate(samples):
        sample_t0 = time.time()
        dtype = sample["defect_type"]
        img_path = Path(sample["image_path"])
        mask_path = Path(sample["mask_path"])

        print(f"[{idx+1:3d}/{len(samples)}] {dtype:20s} {sample['product']:20s} ", end="", flush=True)

        try:
            # Create per-type output directory
            type_dir = output_dir / dtype
            type_dir.mkdir(parents=True, exist_ok=True)

            # Load canvas (the anomaly image IS the canvas — self-reconstruction)
            canvas_t = _load_image_tensor(img_path, device=args.device)
            mask_t = _load_mask_tensor(mask_path, device=args.device)

            # Check mask has content at 512
            if mask_t.sum() < 1:
                print("SKIP (empty mask)")
                continue

            # Prepare CLIP crops from the anomaly image + mask
            clip_data = _prepare_clip_crops(
                img_path, mask_path, args.band_mode, clip_align, args.device,
            )

            # Get caption
            rel_path = sample["rel_image_path"]
            caption = captions.get(rel_path, f"a photo of a {dtype.replace('_', ' ')} defect")

            # Generate
            seed = args.seed + idx
            random.seed(seed)

            with torch.no_grad():
                out = generate_anomagic_single(
                    pipeline, ip_adapter,
                    canvas_t, mask_t, clip_data["clip_ref"],
                    dtype, caption,
                    num_steps=args.num_steps,
                    guidance_scale=args.guidance_scale,
                    noise_strength=args.noise_strength,
                    reference_mode="crop",
                    band_mode=args.band_mode,
                    t2i_adapter=t2i_adapter,
                    seed=seed,
                    clip_mask=clip_data["clip_mask_1"],
                    clip_core_mask=clip_data["clip_core_1"],
                    reference_2=clip_data["clip_ref_2"],
                    clip_mask_2=clip_data["clip_mask_2"],
                    clip_core_mask_2=clip_data["clip_core_2"],
                    group_valid=clip_data["group_valid"],
                    cfg_mode=CFG_MODE,
                    inference_mode="same",  # Self-reconstruction
                )
            torch.cuda.synchronize()

            # Save individual outputs
            sample_name = f"{dtype}_{idx:03d}_{sample['product']}"
            gen_pil = _tensor_to_pil(out)
            gen_pil.save(type_dir / f"{sample_name}.png")

            # Save comparison grid
            orig_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")
            grid = _create_comparison_grid(
                orig_pil, mask_pil, gen_pil,
                clip_data["crop_np"], clip_data["crop_mask_np"],
            )
            grid.save(type_dir / f"{sample_name}_grid.png")

            elapsed = time.time() - sample_t0
            print(f"OK ({elapsed:.1f}s)")

            results.append({
                "index": idx, "defect_type": dtype, "product": sample["product"],
                "source_dataset": sample["source_dataset"],
                "image_path": sample["rel_image_path"],
                "caption": caption, "seed": seed, "status": "ok",
            })

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "index": idx, "defect_type": dtype, "product": sample["product"],
                "status": "error", "error": str(e),
            })

        torch.cuda.empty_cache()

    total_time = time.time() - t0
    ok_count = sum(1 for r in results if r["status"] == "ok")

    # Save manifest
    manifest = {
        "total_samples": len(samples),
        "ok": ok_count,
        "errors": len(results) - ok_count,
        "defect_types": DEFECT_TYPES,
        "samples_per_type": SAMPLES_PER_TYPE,
        "guidance_scale": args.guidance_scale,
        "noise_strength": args.noise_strength,
        "cfg_mode": CFG_MODE,
        "num_steps": args.num_steps,
        "band_mode": args.band_mode,
        "seed": args.seed,
        "checkpoint": str(checkpoint_dir),
        "total_time_s": round(total_time, 1),
        "results": results,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Create overview grid (10x10 — one row per defect type, 10 examples per row)
    print(f"\nCreating overview grid...")
    try:
        fig, axes = plt.subplots(len(DEFECT_TYPES), SAMPLES_PER_TYPE,
                                 figsize=(2 * SAMPLES_PER_TYPE, 2 * len(DEFECT_TYPES)))
        for row, dtype in enumerate(DEFECT_TYPES):
            type_dir = output_dir / dtype
            type_results = [r for r in results if r["defect_type"] == dtype and r["status"] == "ok"]
            for col in range(SAMPLES_PER_TYPE):
                ax = axes[row, col] if len(DEFECT_TYPES) > 1 else axes[col]
                if col < len(type_results):
                    r = type_results[col]
                    sample_name = f"{dtype}_{r['index']:03d}_{r['product']}"
                    img_path = type_dir / f"{sample_name}.png"
                    if img_path.exists():
                        img = Image.open(img_path)
                        ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(dtype.replace("_", "\n"), fontsize=8, rotation=0,
                                  ha="right", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle(f"100 Synthetic Anomalies — CFG={args.guidance_scale}, {CFG_MODE} mode",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "overview_grid.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved overview_grid.png")
    except Exception as e:
        print(f"  Grid creation failed: {e}")

    print(f"\nDone! {ok_count}/{len(samples)} generated in {total_time:.0f}s")
    print(f"Output: {output_dir}")
    print(f"  Per-type folders with individual images + comparison grids")
    print(f"  overview_grid.png — 10×10 grid of all generated images")
    print(f"  manifest.json — full metadata")


if __name__ == "__main__":
    main()
