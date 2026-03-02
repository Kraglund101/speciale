"""Generic offline prep script for the validation suite.

Iterates ALL_PREP_EXAMPLES from validation_suite.py and generates
(canvas, placed_mask, fg_mask, ref_image, ref_mask) for each one.

Each example spec contains:
    id, image_path, mask_path   — the anomaly reference
    normal_dir, normal_index    — which normal image to use as canvas
    fg_mode                     — "ones" (texture) or "birefnet" (object)

For fg_mode="birefnet", BiRefNet runs on-the-fly by default.
Use --no-birefnet to disable (falls back to all-ones FG with a warning).
Alternatively, --fg-masks-dir provides pre-generated FG masks.

Usage:
    # Default (BiRefNet enabled):
    python scripts/prep_validation_data.py \
        --data-root anomverse_extension/datasets/full_training_dataset \
        --output-dir anomverse_extension/datasets/validation_suite \
        --seed 42

    # With pre-generated FG masks:
    python scripts/prep_validation_data.py \
        --data-root ... --output-dir ... \
        --fg-masks-dir path/to/fg_masks --seed 42

    # Dry run (textures work, objects get all-ones FG):
    python scripts/prep_validation_data.py \
        --data-root ... --output-dir ... --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.placement_utils import load_binary_mask, place_easy_mask, place_hard_mask
from scripts.validation_suite import ALL_PREP_EXAMPLES


# ── BiRefNet integration ──────────────────────────────────────────────────

_birefnet_model = None
_birefnet_transform = None


def _load_birefnet(device: str = "cuda") -> None:
    """Lazy-load BiRefNet model (only when --run-birefnet is used)."""
    global _birefnet_model, _birefnet_transform
    if _birefnet_model is not None:
        return

    import torch
    from torchvision import transforms
    from transformers import AutoModelForImageSegmentation

    print("  Loading BiRefNet (ZhengPeng7/BiRefNet)...")
    _birefnet_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True,
    ).to(device).eval()
    _birefnet_transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print("  BiRefNet loaded.")


def _run_birefnet_single(image_pil: Image.Image, device: str = "cuda") -> np.ndarray:
    """Run BiRefNet on a single PIL image. Returns binary mask [H, W] float32."""
    import torch
    _load_birefnet(device)
    original_size = image_pil.size  # (W, H)
    inp = _birefnet_transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = _birefnet_model(inp)
        pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
        pred = pred.sigmoid()
    mask = pred[0].squeeze().cpu().numpy()
    binary = (mask > 0.5).astype(np.float32)
    binary_pil = Image.fromarray((binary * 255).astype(np.uint8))
    binary_pil = binary_pil.resize(original_size, Image.BILINEAR)
    return (np.array(binary_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)


# ── Core logic ────────────────────────────────────────────────────────────


def _pick_normal(base: Path, normal_dir: str, index: int) -> Path:
    """Pick the i-th image (sorted) from a normal directory."""
    d = base / normal_dir
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not imgs:
        raise FileNotFoundError(f"No images in {d}")
    return imgs[index % len(imgs)]


def _get_fg_mask(
    canvas_pil: Image.Image,
    H: int, W: int,
    fg_mode: str,
    run_birefnet: bool,
    fg_masks_dir: Optional[Path],
    normal_stem: str,
    device: str,
    fg_mask_suffix: str = "",
) -> np.ndarray:
    """Get foreground mask for canvas placement.

    Args:
        fg_mask_suffix: Suffix appended to stem when looking up pre-generated
            masks. E.g. "_binary" looks for "{stem}_binary.png".

    Returns binary FG mask [H, W] float32.
    """
    if fg_mode == "ones":
        return np.ones((H, W), dtype=np.float32)

    # fg_mode == "birefnet" — try pre-generated first, then live inference
    if fg_masks_dir is not None:
        fg_path = fg_masks_dir / f"{normal_stem}{fg_mask_suffix}.png"
        if fg_path.exists():
            fg = load_binary_mask(fg_path)
            if fg.shape != (H, W):
                fg_pil = Image.fromarray((fg * 255).astype(np.uint8))
                fg_pil = fg_pil.resize((W, H), Image.BILINEAR)
                fg = (np.array(fg_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
            return fg

    if run_birefnet:
        fg = _run_birefnet_single(canvas_pil, device)
        if fg.shape != (H, W):
            fg_pil = Image.fromarray((fg * 255).astype(np.uint8))
            fg_pil = fg_pil.resize((W, H), Image.BILINEAR)
            fg = (np.array(fg_pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
        # If BiRefNet returns empty mask (no object found → pure texture), use all-ones
        if fg.sum() == 0:
            print(f"    BiRefNet returned empty FG (texture/no object), using all-ones")
            return np.ones((H, W), dtype=np.float32)
        return fg

    # Fallback
    print(f"    WARNING: fg_mode=birefnet but no --run-birefnet or --fg-masks-dir, using all-ones")
    return np.ones((H, W), dtype=np.float32)


def prep_example(
    ex: dict,
    data_root: Path,
    project_root: Path,
    output_dir: Path,
    seed: int,
    force: bool = False,
    run_birefnet: bool = False,
    global_fg_masks_dir: Optional[Path] = None,
    device: str = "cuda",
) -> Optional[str]:
    """Prepare a single example: canvas + placed_mask + fg_mask + ref copies.

    Args:
        ex: Example spec dict (from ALL_PREP_EXAMPLES).
        data_root: Default root for resolving paths (training data).
        project_root: Project root for resolving per-example base_dir.
        output_dir: Where to write {ex['id']}/... output.
        seed: Deterministic seed for placement.
        force: Overwrite existing.
        run_birefnet: Whether to run BiRefNet for fg_mode="birefnet".
        global_fg_masks_dir: Global pre-generated FG masks directory (fallback).
        device: CUDA device.

    Returns:
        Example ID on success, None on failure.
    """
    ex_dir = output_dir / ex["id"]
    meta_path = ex_dir / "meta.json"

    if meta_path.exists() and not force:
        print(f"  {ex['id']}: already exists, skipping")
        return ex["id"]

    # Resolve base directory: per-example base_dir overrides data_root
    if "base_dir" in ex:
        base = Path(ex["base_dir"])
        if not base.is_absolute():
            base = project_root / base
    else:
        base = data_root

    # Resolve paths relative to base
    image_path = base / ex["image_path"]
    mask_path = base / ex["mask_path"]
    if not image_path.exists():
        print(f"  {ex['id']}: image not found ({image_path}), skipping")
        return None
    if not mask_path.exists():
        print(f"  {ex['id']}: mask not found ({mask_path}), skipping")
        return None

    # Load anomaly mask
    if ex.get("mask_is_label", False):
        # Label mask: values are class IDs (0=bg, 1+=anomaly). Binarize at >0.
        raw = np.array(Image.open(mask_path).convert("L")).astype(np.float32)
        anomaly_mask = (raw > 0).astype(np.float32)
    else:
        anomaly_mask = load_binary_mask(mask_path)
    H, W = anomaly_mask.shape

    # Pick normal canvas
    normal_path = _pick_normal(base, ex["normal_dir"], ex["normal_index"])
    canvas_pil = Image.open(normal_path).convert("RGB")
    if canvas_pil.size != (W, H):
        canvas_pil = canvas_pil.resize((W, H), Image.LANCZOS)

    # Resolve per-example fg_masks_dir (relative to base), fall back to global
    ex_fg_masks_dir = global_fg_masks_dir
    if "fg_masks_dir" in ex:
        ex_fg_masks_dir = base / ex["fg_masks_dir"]
    fg_mask_suffix = ex.get("fg_mask_suffix", "")

    # Get foreground mask
    fg_mask = _get_fg_mask(
        canvas_pil, H, W,
        fg_mode=ex["fg_mode"],
        run_birefnet=run_birefnet,
        fg_masks_dir=ex_fg_masks_dir,
        normal_stem=normal_path.stem,
        device=device,
        fg_mask_suffix=fg_mask_suffix,
    )

    # Place anomaly mask within FG region
    placement_mode = ex.get("placement_mode", "easy")
    scale_range = tuple(ex.get("scale_range", [0.8, 1.2]))
    if placement_mode == "hard":
        # Hard: anomaly mask must CONTAIN entire FG (for deformation anomalies)
        result = place_hard_mask(
            anomaly_mask,
            {"canvas": fg_mask},
            scale_range=scale_range,
            seed=seed,
        )
        placed = result[1] if result is not None else None
    else:
        # Easy: anomaly mask must be entirely INSIDE FG
        placed = place_easy_mask(
            anomaly_mask, fg_mask,
            max_attempts=500,
            scale_range=scale_range,
            seed=seed,
        )
    if placed is None:
        print(f"  {ex['id']}: placement failed ({placement_mode} mode), skipping")
        return None

    # Save outputs
    ex_dir.mkdir(parents=True, exist_ok=True)
    canvas_pil.save(ex_dir / "canvas.png")
    Image.fromarray((placed * 255).astype(np.uint8)).save(ex_dir / "placed_mask.png")
    Image.fromarray((fg_mask * 255).astype(np.uint8)).save(ex_dir / "fg_mask.png")

    ref_pil = Image.open(image_path).convert("RGB")
    ref_pil.save(ex_dir / "ref_image.png")
    Image.fromarray((anomaly_mask * 255).astype(np.uint8)).save(ex_dir / "ref_mask.png")

    meta = {
        "id": ex["id"],
        "image_path": ex["image_path"],
        "mask_path": ex["mask_path"],
        "base_dir": str(base),
        "normal_path": str(normal_path.relative_to(base)),
        "fg_mode": ex["fg_mode"],
        "seed": seed,
        "placement_coverage_pct": float(placed.sum() / (H * W) * 100),
        "fg_coverage_pct": float(fg_mask.sum() / (H * W) * 100),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  {ex['id']}: OK "
          f"(coverage={meta['placement_coverage_pct']:.1f}%, "
          f"fg={meta['fg_coverage_pct']:.1f}%)")
    return ex["id"]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare validation suite data (generic — processes all examples that need prep)",
    )
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root directory for training data")
    parser.add_argument("--output-dir", type=str,
                        default="anomverse_extension/datasets/validation_suite",
                        help="Output directory for prepped validation data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Re-generate existing outputs")
    parser.add_argument("--no-birefnet", action="store_true",
                        help="Disable BiRefNet (uses all-ones FG for birefnet examples)")
    parser.add_argument("--fg-masks-dir", type=str, default=None,
                        help="Directory of pre-generated FG masks ({stem}.png)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    fg_masks_dir = Path(args.fg_masks_dir) if args.fg_masks_dir else None
    if fg_masks_dir and not fg_masks_dir.is_absolute():
        fg_masks_dir = project_root / fg_masks_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing validation data in {output_dir}")
    print(f"Data root: {data_root}")
    print(f"Examples to prep: {len(ALL_PREP_EXAMPLES)}")
    run_birefnet = not args.no_birefnet
    if run_birefnet:
        print("BiRefNet: enabled (on-the-fly FG segmentation)")
    elif fg_masks_dir:
        print(f"FG masks dir: {fg_masks_dir}")

    # Check if any examples need BiRefNet without their own fg_masks_dir
    needs_global_birefnet = any(
        ex["fg_mode"] == "birefnet" and "fg_masks_dir" not in ex
        for ex in ALL_PREP_EXAMPLES
    )
    if needs_global_birefnet and not run_birefnet and fg_masks_dir is None:
        print("WARNING: Some examples need fg_mode=birefnet but no "
              "BiRefNet or --fg-masks-dir provided. They will use all-ones FG.")

    # Process all examples
    results = []
    for i, ex in enumerate(ALL_PREP_EXAMPLES):
        result = prep_example(
            ex, data_root, project_root, output_dir,
            seed=args.seed + i,
            force=args.force,
            run_birefnet=run_birefnet,
            global_fg_masks_dir=fg_masks_dir,
            device=args.device,
        )
        if result is not None:
            results.append(result)

    # Write manifest
    manifest = {
        "seed": args.seed,
        "prepped_ids": results,
        "total": len(ALL_PREP_EXAMPLES),
        "success": len(results),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path} ({len(results)}/{len(ALL_PREP_EXAMPLES)} prepped)")
    print("Done.")


if __name__ == "__main__":
    main()
