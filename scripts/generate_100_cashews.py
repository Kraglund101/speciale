#!/usr/bin/env python
"""Generate 100 synthetic cashew anomalies — one per real anomaly reference.

For each of the 100 real cashew anomaly images (94 easy + 6 hard):
  1. Pick a random normal cashew canvas
  2. Place the anomaly mask onto it (using birefnet FG masks)
  3. Generate a synthetic anomaly using the trained model

Placement modes:
  rp (random position) — random transforms + random position (default)
  mo (mimic original)  — no transforms, spiral search around original centroid

Usage:
    # Prep only (no GPU needed):
    python scripts/generate_100_cashews.py --prep-only --placement-mode rp \
        --output-dir results/cashew_100_rp --seed 42

    python scripts/generate_100_cashews.py --prep-only --placement-mode mo \
        --output-dir results/cashew_100_mo --seed 42

    # Full generation:
    python scripts/generate_100_cashews.py \
        --checkpoint important_results/baseline_cosine_1e4_2e5_k1/checkpoint_final \
        --output-dir results/cashew_100_rp --reuse-prep results/cashew_100_rp/prep
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.generate import generate_anomagic_single
from src.utils.crop_utils import clip_crop_multi
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool, unet_roundtrip_masks
from src.utils.placement_utils import load_binary_mask, place_aligned_mask, place_easy_mask, place_hard_mask

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────

CASHEW_BASE = "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew"
EASY_IMG_DIR = "Data/Images/Anomaly/easy_imgs"
HARD_IMG_DIR = "Data/Images/Anomaly/hard_imgs"
MASK_DIR = "Data/Masks/Anomaly"
NORMAL_DIR = "Data/Images/Normal"
FG_MASKS_DIR = "birefnet_masks/Normal"
CAPTION_DIR = "anomverse_extension/datasets/VisA_validation_dataset/prompt_short/cashew/Data/Images/Anomaly"

NUM_STEPS = 50
# BAND_MODE is no longer a module constant — resolved from run_config.json at runtime.
CFG_VARIANTS = [
    {"name": "cfg1_vis",   "guidance_scale": 1.0, "cfg_mode": "visual"},
    {"name": "cfg35_vis",  "guidance_scale": 3.5, "cfg_mode": "visual"},
    {"name": "cfg7_vis",   "guidance_scale": 7.0, "cfg_mode": "visual"},
]

# Hard anomaly caption overrides — auto-generated captions misidentify
# deformed cashews as "Cookie" or "Biscuit"
HARD_CAPTION_OVERRIDES = {
    94: "Cashew nut has a misshapen form with uneven and fused-together parts.",
    95: "Cashew nut has a deformation with an irregular, merged-looking form.",
    96: "Cashew nut has a misshapen, multi-lobed form with an abnormal split shape.",
    97: "Cashew nut has a misshapen defect with an irregular, fused-together form.",
    98: "Cashew nut has a misshapen form with an abnormal curvature.",
    99: "Cashew nut has a misshapen form with an irregular, twisted appearance.",
}


# ── Helpers (from validation_suite.py) ───────────────────────────────────────

def _load_image_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _load_mask_tensor(path: Path, size: int = 512, device: str = "cpu") -> torch.Tensor:
    pil = Image.open(path).convert("L")
    arr = np.array(pil).astype(np.float32) / 255.0
    t = torch.from_numpy((arr > 0.5).astype(np.float32)).unsqueeze(0)
    t = downsample_mask_maxpool(t, size)
    return t.unsqueeze(0).float().to(device)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = ((t[0].detach().cpu().clamp(-1, 1).float() + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def _prepare_clip_crops(
    img_path: Path, mask_path: Path, band_mode: int, clip_align: bool, device: str,
    clip_core_only: bool = False, n_groups: int = 2,
) -> Dict[str, Any]:
    ref_pil = Image.open(img_path).convert("RGB")
    ref_mask_np = (
        np.array(Image.open(mask_path).convert("L")).astype(np.float32) / 255.0 > 0.5
    ).astype(np.float32)
    ref_tensor = torch.from_numpy(
        np.array(ref_pil).astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    ref_mask_tensor = torch.from_numpy(ref_mask_np).unsqueeze(0).float()

    if clip_align:
        core_native, dil_native = unet_roundtrip_masks(ref_mask_tensor, band_mode)
        attn_mask = core_native if clip_core_only else dil_native
        crops, crop_masks, valid, extra = clip_crop_multi(
            ref_tensor, attn_mask, n_groups=n_groups,
            clip_masks=[attn_mask, core_native],
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (extra[0][0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": (extra[0][1] > 0.5).float().unsqueeze(0).to(device),
        }
        if n_groups >= 2:
            result["clip_ref_2"] = (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device)
            result["clip_mask_2"] = (extra[1][0] > 0.5).float().unsqueeze(0).to(device)
            result["clip_core_2"] = (extra[1][1] > 0.5).float().unsqueeze(0).to(device)
        else:
            result["clip_ref_2"] = result["clip_mask_2"] = result["clip_core_2"] = None
    else:
        crops, crop_masks, valid = clip_crop_multi(
            ref_tensor, ref_mask_tensor, n_groups=n_groups,
        )
        result = {
            "clip_ref": (crops[0] * 2.0 - 1.0).unsqueeze(0).to(device),
            "clip_mask_1": (crop_masks[0] > 0.5).float().unsqueeze(0).to(device),
            "clip_core_1": None,
        }
        if n_groups >= 2:
            result["clip_ref_2"] = (crops[1] * 2.0 - 1.0).unsqueeze(0).to(device)
            result["clip_mask_2"] = (crop_masks[1] > 0.5).float().unsqueeze(0).to(device)
            result["clip_core_2"] = None
        else:
            result["clip_ref_2"] = result["clip_mask_2"] = result["clip_core_2"] = None

    gv = [float(valid[0]), float(valid[1]) if n_groups >= 2 else 0.0]
    result["group_valid"] = torch.tensor([gv], device=device)
    result["crop_np"] = (crops[0].permute(1, 2, 0).numpy()).clip(0, 1)
    result["crop_mask_np"] = crop_masks[0][0].numpy()
    if n_groups >= 2:
        result["crop_np_2"] = (crops[1].permute(1, 2, 0).numpy()).clip(0, 1)
        result["crop_mask_np_2"] = crop_masks[1][0].numpy()
    else:
        result["crop_np_2"] = result["crop_np"]  # same as crop 1 for viz
        result["crop_mask_np_2"] = result["crop_mask_np"]
    return result


def _load_training_config(checkpoint_dir: Path) -> Dict[str, Any]:
    """Load run_config.json saved alongside the checkpoint (in training save_dir).

    Training saves it at `<save_dir>/run_config.json`. Checkpoint dirs are
    `<save_dir>/checkpoint_N` or `checkpoint_final`, so we look in the parent.
    """
    cfg_path = checkpoint_dir.parent / "run_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f)
    print(f"WARNING: no run_config.json at {cfg_path} — inference defaults will not track training.")
    return {}


# ── Checkpoint loading (from validation_suite.py) ───────────────────────────

def _load_checkpoint(checkpoint_dir: Path, device: str):
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
    state = torch.load(ip_ckpt, map_location="cpu", weights_only=True)
    sa_layers = 0
    for k in state.keys():
        if k.startswith("masked_self_attn.layers."):
            sa_layers = max(sa_layers, int(k.split(".")[2]) + 1)
    if sa_layers == 0:
        sa_layers = 3

    force_gates = config.get("force_gates", False)
    learnable_gates = config.get("learnable_gates", True)
    if not force_gates and "force_gates" not in config:
        if not any("gate.gate" in k for k in state.keys()):
            force_gates, learnable_gates = True, False

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
        print("  T2I-Adapter loaded")

    if training_state and "ema" in training_state:
        from src.utils.optim_utils import build_norm_param_id_set, split_decay_no_decay
        shadow = training_state["ema"]["shadow"]
        group_a = [ip_adapter.image_projection] + list(ip_adapter.attn_processors.values())
        group_b = []
        if hasattr(ip_adapter, 'masked_self_attn'):
            group_b.append(ip_adapter.masked_self_attn)
        if t2i_adapter is not None:
            group_b.append(t2i_adapter)
        norm_ids = build_norm_param_id_set(group_a + group_b)
        a_d, a_nd = split_decay_no_decay(group_a, norm_ids)

        # Group C: gate/scale params (must be excluded from B, matching training)
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

        b_d_raw, b_nd_raw = split_decay_no_decay(group_b, norm_ids)
        b_d = [p for p in b_d_raw if id(p) not in group_c_ids]
        b_nd = [p for p in b_nd_raw if id(p) not in group_c_ids]

        all_params = a_d + a_nd + b_d + b_nd + group_c_params
        if len(shadow) == len(all_params):
            for p, s in zip(all_params, shadow):
                p.data.copy_(s.to(p.device))
            print(f"  EMA weights applied ({len(shadow)} params)")
        else:
            print(f"  WARNING: EMA param count mismatch: model={len(all_params)}, shadow={len(shadow)}")

    return pipeline, ip_adapter, t2i_adapter


# ── Prep: place anomaly onto normal canvas ──────────────────────────────────

def _prep_one_easy(
    idx: int,
    anomaly_img_path: Path,
    anomaly_mask_path: Path,
    normal_path: Path,
    fg_mask_path: Path,
    seed: int,
    out_dir: Path,
) -> Optional[Path]:
    """Place easy anomaly mask onto normal canvas. Returns prep dir or None."""
    prep_dir = out_dir / f"{idx:03d}"
    if (prep_dir / "meta.json").exists():
        return prep_dir

    # VisA masks use label values (0=bg, nonzero=anomaly), not 0/255
    raw = np.array(Image.open(anomaly_mask_path).convert("L")).astype(np.float32)
    anomaly_mask = (raw > 0).astype(np.float32)
    H, W = anomaly_mask.shape

    canvas_pil = Image.open(normal_path).convert("RGB")
    if canvas_pil.size != (W, H):
        canvas_pil = canvas_pil.resize((W, H), Image.LANCZOS)

    fg_mask = load_binary_mask(fg_mask_path)
    if fg_mask.shape != (H, W):
        fg_pil = Image.fromarray((fg_mask * 255).astype(np.uint8))
        fg_pil = fg_pil.resize((W, H), Image.BILINEAR)
        fg_mask = (np.array(fg_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    placed = place_easy_mask(anomaly_mask, fg_mask, max_attempts=500, seed=seed)
    if placed is None:
        return None

    prep_dir.mkdir(parents=True, exist_ok=True)
    canvas_pil.save(prep_dir / "canvas.png")
    Image.fromarray((placed * 255).astype(np.uint8)).save(prep_dir / "placed_mask.png")
    Image.open(anomaly_img_path).convert("RGB").save(prep_dir / "ref_image.png")
    Image.fromarray((anomaly_mask * 255).astype(np.uint8)).save(prep_dir / "ref_mask.png")
    json.dump({"idx": idx, "normal": normal_path.name, "is_hard": False, "seed": seed},
              open(prep_dir / "meta.json", "w"), indent=2)
    return prep_dir


def _prep_one_easy_aligned(
    idx: int,
    anomaly_img_path: Path,
    anomaly_mask_path: Path,
    normal_path: Path,
    fg_mask_path: Path,
    out_dir: Path,
) -> Optional[Path]:
    """Place easy anomaly at aligned position (no random transforms). Returns prep dir or None."""
    prep_dir = out_dir / f"{idx:03d}"
    if (prep_dir / "meta.json").exists():
        return prep_dir

    # VisA masks use label values (0=bg, nonzero=anomaly), not 0/255
    raw = np.array(Image.open(anomaly_mask_path).convert("L")).astype(np.float32)
    anomaly_mask = (raw > 0).astype(np.float32)
    H, W = anomaly_mask.shape

    canvas_pil = Image.open(normal_path).convert("RGB")
    if canvas_pil.size != (W, H):
        canvas_pil = canvas_pil.resize((W, H), Image.LANCZOS)

    fg_mask = load_binary_mask(fg_mask_path)
    if fg_mask.shape != (H, W):
        fg_pil = Image.fromarray((fg_mask * 255).astype(np.uint8))
        fg_pil = fg_pil.resize((W, H), Image.BILINEAR)
        fg_mask = (np.array(fg_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)

    result = place_aligned_mask(anomaly_mask, fg_mask, max_search_radius=200)
    if result is None:
        # Roundtripped mask too large — fall back to original mask
        result = place_aligned_mask(anomaly_mask, fg_mask, max_search_radius=200)
    if result is None:
        # Edge-hugging anomaly: relax FG constraint to 90%
        result = place_aligned_mask(anomaly_mask, fg_mask, max_search_radius=200, min_inside_frac=0.9)
    if result is None:
        return None

    placed, target_centroid, actual_centroid = result

    prep_dir.mkdir(parents=True, exist_ok=True)
    canvas_pil.save(prep_dir / "canvas.png")
    Image.fromarray((placed * 255).astype(np.uint8)).save(prep_dir / "placed_mask.png")
    Image.open(anomaly_img_path).convert("RGB").save(prep_dir / "ref_image.png")
    Image.fromarray((anomaly_mask * 255).astype(np.uint8)).save(prep_dir / "ref_mask.png")
    json.dump({
        "idx": idx, "normal": normal_path.name, "is_hard": False, "seed": 0,
        "aligned": True,
        "target_centroid": list(target_centroid),
        "actual_centroid": list(actual_centroid),
    }, open(prep_dir / "meta.json", "w"), indent=2)
    return prep_dir


def _prep_one_hard(
    idx: int,
    anomaly_img_path: Path,
    anomaly_mask_path: Path,
    canvas_fg_dict: Dict[str, Tuple[Path, np.ndarray]],
    seed: int,
    out_dir: Path,
) -> Optional[Path]:
    """Place hard anomaly mask onto a canvas (searches multiple canvases). Returns prep dir or None."""
    prep_dir = out_dir / f"{idx:03d}"
    if (prep_dir / "meta.json").exists():
        return prep_dir

    # VisA masks use label values (0=bg, nonzero=anomaly), not 0/255
    raw = np.array(Image.open(anomaly_mask_path).convert("L")).astype(np.float32)
    anomaly_mask = (raw > 0).astype(np.float32)

    # Build fg-only dict for place_hard_mask
    fg_only = {cid: fg for cid, (_, fg) in canvas_fg_dict.items()}
    result = place_hard_mask(anomaly_mask, fg_only, scale_range=(0.8, 1.2), seed=seed)

    if result is not None:
        canvas_id, placed, angle, scale, cov = result
        normal_path, _ = canvas_fg_dict[canvas_id]
        meta_extra = {"angle": float(angle), "scale": float(scale), "coverage": float(cov)}
        print(f"    (hard placement: canvas={canvas_id}, angle={angle:.0f}, scale={scale:.2f}, cov={cov:.1f}%)")
    else:
        # Fallback: use FG mask of first available canvas (hard anomalies cover full object)
        canvas_id = next(iter(canvas_fg_dict))
        normal_path, fg = canvas_fg_dict[canvas_id]
        placed = fg.copy()
        meta_extra = {"fallback": True}
        print(f"    (hard fallback: using FG mask of canvas {canvas_id})")

    H, W = anomaly_mask.shape
    canvas_pil = Image.open(normal_path).convert("RGB")
    if canvas_pil.size != (W, H):
        canvas_pil = canvas_pil.resize((W, H), Image.LANCZOS)

    prep_dir.mkdir(parents=True, exist_ok=True)
    canvas_pil.save(prep_dir / "canvas.png")
    Image.fromarray((placed * 255).astype(np.uint8)).save(prep_dir / "placed_mask.png")
    Image.open(anomaly_img_path).convert("RGB").save(prep_dir / "ref_image.png")
    Image.fromarray((anomaly_mask * 255).astype(np.uint8)).save(prep_dir / "ref_mask.png")
    json.dump({"idx": idx, "normal": normal_path.name, "is_hard": True, "seed": seed, **meta_extra},
              open(prep_dir / "meta.json", "w"), indent=2)
    return prep_dir


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate 100 synthetic cashew anomalies")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/cashew_100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--reuse-prep", type=str, default=None,
                        help="Path to existing prep dir to skip placement")
    parser.add_argument("--placement-mode", type=str, choices=["rp", "mo"], default="rp",
                        help="Placement mode: rp=random position, mo=mimic original")
    parser.add_argument("--prep-only", action="store_true",
                        help="Run placement step only (no model loading or generation)")
    parser.add_argument("--inpaint-mask-mode", type=str, default="dilated",
                        choices=["dilated", "core_binary"],
                        help="UNet inpainting mask channel. 'dilated' (default) or 'core_binary' "
                             "(ablation: only core is flagged as masked, band becomes 'preserve canvas').")
    parser.add_argument("--zero-t2i-band", action="store_true",
                        help="Zero the T2I band input channel at inference (ablation: T2I only sees core).")
    parser.add_argument("--cfg-only", type=str, default=None,
                        choices=["cfg1_vis", "cfg35_vis", "cfg7_vis"],
                        help="Limit generation to a single CFG variant (default: all three).")
    parser.add_argument("--cross-attn-mask-mode", type=str, default="alpha",
                        choices=["alpha", "core_binary"],
                        help="IP-Adapter cross-attention spatial mask. "
                             "'alpha' (default) = soft alpha map (core=1, inner=2/3, outer=1/3). "
                             "'core_binary' = binary core-only (zeros IP contribution in band).")
    parser.add_argument("--clip-core-only", action=argparse.BooleanOptionalAction, default=None,
        help="CLIP core-only attn mask at inference. Default True (no normal pixels attended). "
             "Use --no-clip-core-only to match training's clip_core_only for matched-training ablations.")
    parser.add_argument("--n-groups", type=int, default=None,
        help="Number of CLIP crops. Default: inherit from training's multi_crop (2 if True else 1).")
    parser.add_argument("--band-mode", type=int, default=None,
        help="Band mode. Default: inherit from training's run_config.json.")
    parser.add_argument("--set-alpha-to-one", action="store_true",
                        help="Override DDIM scheduler: force final_alpha_cumprod=1.0 so the last "
                             "step outputs pure x_0_pred with zero residual noise.")
    args = parser.parse_args()

    if not args.prep_only and args.checkpoint is None:
        parser.error("--checkpoint is required unless --prep-only is set")

    project_root = Path(__file__).parent.parent
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = project_root / checkpoint_dir
    else:
        checkpoint_dir = None
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    cashew_base = project_root / CASHEW_BASE
    caption_dir = project_root / CAPTION_DIR

    # ── Discover all 100 anomalies ──────────────────────────────────────────
    easy_dir = cashew_base / EASY_IMG_DIR
    hard_dir = cashew_base / HARD_IMG_DIR
    mask_dir = cashew_base / MASK_DIR
    normal_dir = cashew_base / NORMAL_DIR
    fg_dir = cashew_base / FG_MASKS_DIR

    easy_imgs = sorted(easy_dir.glob("*.JPG"))
    hard_imgs = sorted(hard_dir.glob("*.JPG"))
    normals = sorted(p for p in normal_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    # Build list: (idx, img_path, mask_path, is_hard)
    anomalies = []
    for img_path in easy_imgs:
        idx = int(img_path.stem)
        anomalies.append((idx, img_path, mask_dir / f"{idx:03d}.png", False))
    for img_path in hard_imgs:
        idx = int(img_path.stem)
        anomalies.append((idx, img_path, mask_dir / f"{idx:03d}.png", True))
    anomalies.sort(key=lambda x: x[0])

    print(f"Found {len(anomalies)} cashew anomalies ({len(easy_imgs)} easy + {len(hard_imgs)} hard)")
    print(f"Found {len(normals)} normal canvases")

    # Load captions
    captions = {}
    for txt_path in caption_dir.glob("*_analysis_short.txt"):
        idx_str = txt_path.stem.replace("_analysis_short", "")
        captions[int(idx_str)] = txt_path.read_text().strip()
    print(f"Loaded {len(captions)} captions")

    # ── Step 1: Prep (place masks onto normal canvases) ─────────────────────
    reuse_prep = Path(args.reuse_prep) if args.reuse_prep else None
    if reuse_prep and not reuse_prep.is_absolute():
        reuse_prep = project_root / reuse_prep

    if reuse_prep and reuse_prep.exists():
        print(f"\n=== Step 1: Reusing prep from {reuse_prep} ===")
        prep_dir = reuse_prep
        prepped = []
        for idx, img_path, mask_path, is_hard in anomalies:
            p = prep_dir / f"{idx:03d}"
            if not (p / "meta.json").exists():
                print(f"  [{idx:03d}] SKIP — not in reuse prep")
                continue
            caption = HARD_CAPTION_OVERRIDES.get(idx, captions.get(idx, "Cashew nut has a surface defect."))
            noise_strength = 1.0 if is_hard else 0.7
            prepped.append((idx, p, is_hard, caption, noise_strength))
        print(f"  Reused {len(prepped)}/{len(anomalies)} prep dirs")
    else:
        print(f"\n=== Step 1: Prep (placing anomaly masks onto normal canvases) ===")
        prep_dir = output_dir / "prep"
        prep_dir.mkdir(parents=True, exist_ok=True)

        rng = random.Random(args.seed)
        prepped = []  # (idx, prep_path, is_hard, caption)

        # Build canvas FG dict for hard placement (sample ~50 to keep search tractable)
        has_hard = any(is_hard for _, _, _, is_hard in anomalies)
        canvas_fg_dict: Dict[str, Tuple[Path, np.ndarray]] = {}
        if has_hard:
            print("  Loading FG masks for hard placement...")
            hard_rng = random.Random(args.seed + 9999)
            hard_pool = list(normals)
            hard_rng.shuffle(hard_pool)
            hard_pool = hard_pool[:10]
            for normal_path in hard_pool:
                fg_path = fg_dir / f"{normal_path.stem}_binary.png"
                if not fg_path.exists():
                    continue
                fg = load_binary_mask(fg_path)
                canvas_fg_dict[normal_path.stem] = (normal_path, fg)
            print(f"  Loaded {len(canvas_fg_dict)} canvas FG masks for hard placement")

        for idx, img_path, mask_path, is_hard in anomalies:
            if is_hard:
                prep_path = _prep_one_hard(
                    idx, img_path, mask_path, canvas_fg_dict,
                    seed=args.seed + idx, out_dir=prep_dir,
                )
            else:
                # Try up to 10 different canvases for easy placement
                prep_path = None
                for attempt in range(10):
                    normal_idx = rng.randint(0, len(normals) - 1)
                    normal_path = normals[normal_idx]
                    fg_mask_path = fg_dir / f"{normal_path.stem}_binary.png"
                    if not fg_mask_path.exists():
                        continue
                    if args.placement_mode == "mo":
                        prep_path = _prep_one_easy_aligned(
                            idx, img_path, mask_path, normal_path, fg_mask_path,
                            out_dir=prep_dir,
                        )
                    else:
                        prep_path = _prep_one_easy(
                            idx, img_path, mask_path, normal_path, fg_mask_path,
                            seed=args.seed + idx + attempt * 1000, out_dir=prep_dir,
                        )
                    if prep_path is not None:
                        break

            if prep_path is None:
                print(f"  [{idx:03d}] SKIP — placement failed")
                continue

            caption = HARD_CAPTION_OVERRIDES.get(idx, captions.get(idx, "Cashew nut has a surface defect."))
            noise_strength = 1.0 if is_hard else 0.7
            prepped.append((idx, prep_path, is_hard, caption, noise_strength))
            if not is_hard:
                print(f"  [{idx:03d}] OK (easy) → normal {normal_path.stem}")

    print(f"\nPrepped {len(prepped)}/{len(anomalies)} examples")

    if args.prep_only:
        json.dump({
            "total": len(prepped), "ok": len(prepped),
            "placement_mode": args.placement_mode,
            "seed": args.seed,
            "prep_only": True,
        }, open(output_dir / "manifest.json", "w"), indent=2)
        print(f"\n--prep-only: done. Prep dir: {output_dir / 'prep'}")
        return

    # ── Step 2: Load model ──────────────────────────────────────────────────
    print(f"\n=== Step 2: Loading checkpoint ===")
    pipeline, ip_adapter, t2i_adapter = _load_checkpoint(checkpoint_dir, args.device)
    ip_adapter.eval()

    # Resolve inference settings.
    #   - band_mode, n_groups: inherit from training's run_config.json.
    #   - clip_core_only: fixed True by default (inference always "no normal").
    #     Use --no-clip-core-only to match training instead.
    #   - scheduler: CLI-selected; default ddim.
    run_cfg = _load_training_config(checkpoint_dir)
    band_mode = args.band_mode if args.band_mode is not None else run_cfg.get("band_mode", 2)
    clip_core_only = args.clip_core_only if args.clip_core_only is not None else True
    if args.n_groups is not None:
        n_groups = args.n_groups
    else:
        n_groups = 2 if run_cfg.get("multi_crop", False) else 1

    # Cross-attn mask mode: inherit from training's binary_cross_attn_mask_core.
    # CLI --cross-attn-mask-mode still overrides if explicitly passed.
    if args.cross_attn_mask_mode == "alpha" and run_cfg.get("binary_cross_attn_mask_core", False):
        args.cross_attn_mask_mode = "core_binary"
        print(f"  Inheriting cross_attn_mask_mode='core_binary' from training run_config.")

    # mask_visual is carried by the IP-Adapter checkpoint config.json and
    # auto-applied by IPAdapterAttnProcessor (ip_adapter.py:598) — no inference
    # override needed. We just surface it here for auditability.
    print(
        f"  Inference settings: band_mode={band_mode}, "
        f"clip_core_only={clip_core_only}, n_groups={n_groups}, "
        f"mask_visual={ip_adapter.config.mask_visual}  "
        f"(training clip_core_only={run_cfg.get('clip_core_only', '?')}, "
        f"multi_crop={run_cfg.get('multi_crop', '?')}, "
        f"mask_visual={run_cfg.get('mask_visual', '?')}, "
        f"timestep_sampling={run_cfg.get('timestep_sampling', '?')}, "
        f"uniform_mask_loss={run_cfg.get('uniform_mask_loss', '?')})"
    )

    if args.set_alpha_to_one:
        sched = pipeline.scheduler
        sched.register_to_config(set_alpha_to_one=True)
        sched.final_alpha_cumprod = torch.tensor(
            1.0, device=sched.final_alpha_cumprod.device, dtype=sched.final_alpha_cumprod.dtype,
        )
        print(f"  DDIM scheduler: set_alpha_to_one=True (final_alpha_cumprod={sched.final_alpha_cumprod.item():.6f})")

    # ── Step 3: Generate ────────────────────────────────────────────────────
    cfg_variants = CFG_VARIANTS
    if args.cfg_only is not None:
        cfg_variants = [v for v in CFG_VARIANTS if v["name"] == args.cfg_only]
        print(f"\n--cfg-only {args.cfg_only}: limited to {len(cfg_variants)} variant")
    print(f"\n=== Step 3: Generating {len(prepped)} synthetic anomalies ({len(cfg_variants)} CFG variants each) ===")
    for v in cfg_variants:
        (output_dir / "generated" / v["name"]).mkdir(parents=True, exist_ok=True)
    grid_dir = output_dir / "grids"
    grid_dir.mkdir(parents=True, exist_ok=True)

    from PIL import ImageDraw
    from scipy.ndimage import binary_dilation

    t0 = time.time()
    results = []

    for i, (idx, prep_path, is_hard, caption, noise_strength) in enumerate(prepped):
        print(f"  [{i+1:3d}/{len(prepped)}] anomaly {idx:03d} {'hard' if is_hard else 'easy':4s} ", end="", flush=True)
        try:
            canvas_t = _load_image_tensor(prep_path / "canvas.png", device=args.device)
            mask_t = _load_mask_tensor(prep_path / "placed_mask.png", device=args.device)

            clip_data = _prepare_clip_crops(
                prep_path / "ref_image.png", prep_path / "ref_mask.png",
                band_mode, True, args.device,
                clip_core_only=clip_core_only, n_groups=n_groups,
            )

            seed = args.seed + idx

            # Generate all CFG variants
            gen_pils = {}
            for v in cfg_variants:
                random.seed(seed)
                with torch.no_grad():
                    out = generate_anomagic_single(
                        pipeline, ip_adapter,
                        canvas_t, mask_t, clip_data["clip_ref"],
                        "cashew", caption,
                        num_steps=args.num_steps,
                        guidance_scale=v["guidance_scale"],
                        noise_strength=noise_strength,
                        reference_mode="crop",
                        band_mode=band_mode,
                        t2i_adapter=t2i_adapter,
                        seed=seed,
                        clip_mask=clip_data["clip_mask_1"],
                        clip_core_mask=clip_data["clip_core_1"],
                        reference_2=clip_data["clip_ref_2"],
                        clip_mask_2=clip_data["clip_mask_2"],
                        clip_core_mask_2=clip_data["clip_core_2"],
                        group_valid=clip_data["group_valid"],
                        cfg_mode=v["cfg_mode"],
                        inference_mode="different",
                        cross_attn_mask_mode=args.cross_attn_mask_mode,
                        zero_t2i_band=args.zero_t2i_band,
                        inpaint_mask_mode=args.inpaint_mask_mode,
                    )
                torch.cuda.synchronize()
                gen_pil = _tensor_to_pil(out)
                gen_pil.save(output_dir / "generated" / v["name"] / f"{idx:03d}.png")
                gen_pils[v["name"]] = gen_pil

            # ── Comparison grid ──
            # Row 1: Ref | Canvas+mask | CFG=1 | CFG=3.5 | CFG=7.0
            # Row 2: CLIP crop 1 | CLIP crop 2 | (empty cells)
            S = 512
            CS = 224  # CLIP crop native size
            label_h = 28
            n_cols = 2 + len(cfg_variants)  # ref + canvas + variants

            canvas_pil = Image.open(prep_path / "canvas.png").convert("RGB").resize((S, S), Image.LANCZOS)
            ref_pil = Image.open(prep_path / "ref_image.png").convert("RGB").resize((S, S), Image.LANCZOS)
            ref_mask_arr = np.array(Image.open(prep_path / "ref_mask.png").convert("L").resize((S, S), Image.NEAREST)).astype(np.float32) / 255.0
            ref_mask_bin = (ref_mask_arr > 0.5)

            # Reference: red edge contour of original anomaly mask
            struct = np.ones((5, 5), dtype=bool)
            ref_edge = binary_dilation(ref_mask_bin, struct).astype(np.float32) - ref_mask_bin.astype(np.float32)
            ref_rgba = ref_pil.convert("RGBA")
            red_edge = np.zeros((S, S, 4), dtype=np.uint8)
            red_edge[:, :, 0] = 255; red_edge[:, :, 3] = (ref_edge * 255).astype(np.uint8)
            ref_overlay = Image.alpha_composite(ref_rgba, Image.fromarray(red_edge)).convert("RGB")

            # Canvas: roundtripped core (red) + band (yellow) contours
            # This shows exactly what the UNet sees after maxpool to 64 + dilation + nearest up
            from scipy import ndimage as _ndi
            mask_t = _load_mask_tensor(prep_path / "placed_mask.png", device="cpu")
            kernel = S // 64  # = 8
            core_64 = F.max_pool2d(mask_t, kernel_size=kernel)
            core_64 = (core_64 > 0.5).float()
            _, _, _, band_64 = create_latent_band_mask(core_64, band_mode)
            core_512 = F.interpolate(core_64, size=(S, S), mode="nearest")[0, 0].numpy() > 0.5
            band_512 = F.interpolate(band_64, size=(S, S), mode="nearest")[0, 0].numpy() > 0.5
            dilated_512 = core_512 | band_512
            # 1px contour edges via erode-XOR
            core_edge = core_512 ^ _ndi.binary_erosion(core_512, iterations=1)
            dilated_edge = dilated_512 ^ _ndi.binary_erosion(dilated_512, iterations=1)
            band_edge = dilated_edge & ~core_512
            # RGBA overlay: red=core, yellow=band
            zone_overlay = np.zeros((S, S, 4), dtype=np.uint8)
            zone_overlay[band_edge] = [255, 217, 0, 204]    # yellow
            zone_overlay[core_edge] = [255, 0, 0, 204]      # red
            canvas_rgba = canvas_pil.convert("RGBA")
            canvas_overlay = Image.alpha_composite(canvas_rgba, Image.fromarray(zone_overlay)).convert("RGB")

            # CLIP crop overlays (red tint on anomaly region)
            def _make_crop_overlay(crop_np, mask_np):
                tint = np.array([1.0, 0.2, 0.2])
                m = mask_np[..., None]
                overlay = (crop_np * (1 - 0.3 * m) + tint * 0.3 * m).clip(0, 1)
                return Image.fromarray((overlay * 255).astype(np.uint8)).resize((S, S), Image.LANCZOS)

            crop1_pil = _make_crop_overlay(clip_data["crop_np"], clip_data["crop_mask_np"])
            crop2_pil = _make_crop_overlay(clip_data["crop_np_2"], clip_data["crop_mask_np_2"])
            crop2_valid = clip_data["group_valid"][0, 1].item() > 0.5

            # Helper: overlay core/band contours on a PIL image
            def _add_zone_contour(pil_img):
                return Image.alpha_composite(
                    pil_img.resize((S, S), Image.LANCZOS).convert("RGBA"),
                    Image.fromarray(zone_overlay),
                ).convert("RGB")

            # Row 1 data
            row1_data = [
                (ref_overlay, f"Reference #{idx:03d} ({'hard' if is_hard else 'easy'})"),
                (canvas_overlay, "Canvas (red=core, yellow=band)"),
            ]
            for v in cfg_variants:
                lbl = f"CFG={v['guidance_scale']:.1f}"
                row1_data.append((_add_zone_contour(gen_pils[v["name"]]), lbl))

            # Row 2 data
            row2_data = [
                (crop1_pil, "CLIP crop 1"),
                (crop2_pil, f"CLIP crop 2{'' if crop2_valid else ' (null)'}"),
            ]
            # Pad remaining columns with empty cells
            for _ in range(n_cols - 2):
                row2_data.append((None, ""))

            # Build grid: 2 rows
            grid_w = S * n_cols
            grid_h = 2 * (S + label_h)
            grid = Image.new("RGB", (grid_w, grid_h), (20, 20, 20))
            draw_grid = ImageDraw.Draw(grid)

            for row_i, row_data in enumerate([row1_data, row2_data]):
                y_off = row_i * (S + label_h)
                for col_i, (panel_img, lbl) in enumerate(row_data):
                    x_off = col_i * S
                    if panel_img is not None:
                        grid.paste(panel_img, (x_off, y_off + label_h))
                    draw_grid = ImageDraw.Draw(grid)
                    if lbl:
                        draw_grid.text((x_off + 8, y_off + 5), lbl, fill=(255, 255, 200))

            grid.save(grid_dir / f"{idx:03d}_grid.png")

            elapsed = time.time() - t0
            avg = elapsed / (i + 1)
            print(f"OK ({avg:.1f}s/img)")
            results.append({"idx": idx, "is_hard": is_hard, "status": "ok"})

        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"idx": idx, "is_hard": is_hard, "status": "error", "error": str(e)})

        torch.cuda.empty_cache()

    total = time.time() - t0
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\nDone! {ok}/{len(prepped)} generated in {total:.0f}s")

    # ── Overview grid (10×10) ───────────────────────────────────────────────
    print("Creating overview grid...")
    try:
        ncols = 10
        nrows = (ok + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
        axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else list(axes)
        overview_dir = output_dir / "generated" / "cfg7_vis"
        gen_images = sorted(overview_dir.glob("*.png"))
        for i, ax in enumerate(axes):
            if i < len(gen_images):
                ax.imshow(Image.open(gen_images[i]))
                ax.set_title(gen_images[i].stem, fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.suptitle("100 Synthetic Cashew Anomalies (CFG=7 visual)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "overview.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved overview.png")
    except Exception as e:
        print(f"  Overview failed: {e}")

    # Save manifest
    json.dump({
        "total": len(prepped), "ok": ok,
        "cfg_variants": [v["name"] for v in cfg_variants],
        "num_steps": args.num_steps, "seed": args.seed,
        "checkpoint": str(checkpoint_dir),
        "set_alpha_to_one": bool(args.set_alpha_to_one),
        "elapsed_s": round(total, 1),
        "results": results,
    }, open(output_dir / "manifest.json", "w"), indent=2)

    print(f"\nOutput: {output_dir}")
    print(f"  generated/  — 100 synthetic anomaly images")
    print(f"  grids/      — comparison grids (canvas | mask | reference | generated)")
    print(f"  prep/       — placed canvases + masks")
    print(f"  overview.png")


if __name__ == "__main__":
    main()
