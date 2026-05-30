#!/usr/bin/env python
"""Run MetaUAS mask refinement on generated cashew anomalies.

For each generated image, compares it against the normal canvas using MetaUAS
to produce a refined anomaly mask — the mask of where the anomaly *actually*
ended up, rather than where we told it to go.

Usage:
    python scripts/run_metauas_cashews.py \
        --data-dir results/cashew_100_rp \
        --cfg-variant cfg7_vis \
        --threshold 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "MetaUAS"))

from src.utils.mask_utils import create_latent_band_mask

from metauas import MetaUAS, safely_load_state_dict
import kornia as K

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METAUAS_WEIGHTS = Path(__file__).parent.parent / "MetaUAS" / "weights" / "metauas-512.ckpt"
IMG_SIZE = 512


def load_model(weights_path: Path, img_size: int = 512) -> torch.nn.Module:
    model = MetaUAS("efficientnet-b4", "unet", 5, 5, 3, "sa", "cat")
    model = safely_load_state_dict(model, str(weights_path))
    model.cuda().eval()
    return model


def load_image_tensor(path: Path, size: int = 512) -> torch.Tensor:
    """Load image as [1, 3, H, W] float32 tensor in [0, 1]."""
    pil = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_mask_np(path: Path, size: int = 512) -> np.ndarray:
    """Load mask as binary [H, W] array."""
    pil = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(pil).astype(np.float32) / 255.0 > 0.5).astype(np.float32)


def compute_roundtripped_dilated(mask_512: np.ndarray, band_mode: int = 2) -> np.ndarray:
    """Roundtrip mask through latent space and return dilated (core+band) at 512."""
    mask_t = torch.from_numpy(mask_512).unsqueeze(0).unsqueeze(0).float()
    core_64 = F.max_pool2d(mask_t, kernel_size=8)
    core_64 = (core_64 > 0.5).float()
    _, _, _, band_64 = create_latent_band_mask(core_64, band_mode=band_mode)
    core_512 = F.interpolate(core_64, size=(512, 512), mode='nearest')[0, 0].numpy()
    band_512 = F.interpolate(band_64, size=(512, 512), mode='nearest')[0, 0].numpy()
    return ((core_512 > 0.5) | (band_512 > 0.5)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str,
                        default="results/cashew_100_rp")
    parser.add_argument("--cfg-variant", type=str, default="cfg7_vis")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for MetaUAS probability map (default 0.5, Anomagic uses 0.9)")
    parser.add_argument("--weights", type=str, default=str(METAUAS_WEIGHTS))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    prep_dir = data_dir / "prep"
    gen_dir = data_dir / "generated" / args.cfg_variant
    out_dir = data_dir / "metauas_masks" / args.cfg_variant
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading MetaUAS from {args.weights}")
    model = load_model(Path(args.weights))

    sample_dirs = sorted(prep_dir.iterdir())
    print(f"Processing {len(sample_dirs)} samples...")
    print(f"Threshold: {args.threshold}")

    stats = []

    for sample_dir in sample_dirs:
        idx = sample_dir.name
        canvas_path = sample_dir / "canvas.png"
        mask_path = sample_dir / "placed_mask.png"
        gen_path = gen_dir / f"{idx}.png"

        if not gen_path.exists():
            print(f"  SKIP {idx}: no generated image")
            continue

        # Load images
        canvas_t = load_image_tensor(canvas_path, IMG_SIZE).cuda()
        gen_t = load_image_tensor(gen_path, IMG_SIZE).cuda()
        raw_mask = load_mask_np(mask_path, IMG_SIZE)
        # Roundtripped dilated mask = what the UNet actually modifies (core + band)
        dilated_mask = compute_roundtripped_dilated(raw_mask, band_mode=2)

        # MetaUAS: prompt=normal canvas, query=generated anomaly
        with torch.no_grad():
            test_data = {
                "prompt_image": canvas_t,
                "query_image": gen_t,
            }
            pred = model(test_data)  # [1, 1, 512, 512]

        # MetaUAS output: sigmoid, low=normal, high=anomalous
        anomaly_prob = pred[0, 0].cpu().numpy()
        refined_mask = (anomaly_prob > args.threshold).astype(np.float32)

        # Stats — compare against roundtripped dilated mask (core+band)
        dilated_area = dilated_mask.sum()
        refined_area = refined_mask.sum()
        overlap = (dilated_mask * refined_mask).sum()
        iou = overlap / max(dilated_mask.sum() + refined_mask.sum() - overlap, 1)

        # How much of the refined mask is outside the dilated zone?
        outside = ((refined_mask > 0.5) & (dilated_mask < 0.5)).sum()
        outside_frac = outside / max(refined_area, 1)

        stats.append({
            "idx": idx,
            "dilated_area": int(dilated_area),
            "refined_area": int(refined_area),
            "iou": float(iou),
            "outside_frac": float(outside_frac),
        })

        # Save refined mask
        mask_pil = Image.fromarray((refined_mask * 255).astype(np.uint8), mode="L")
        mask_pil.save(out_dir / f"{idx}_mask.png")

        # Save probability map
        prob_pil = Image.fromarray((anomaly_prob * 255).clip(0, 255).astype(np.uint8), mode="L")
        prob_pil.save(out_dir / f"{idx}_prob.png")

        print(f"  {idx}: dilated={int(dilated_area):6d}px  refined={int(refined_area):6d}px  "
              f"IoU={iou:.3f}  outside={outside_frac:.2%}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    ious = np.array([s["iou"] for s in stats])
    outside_fracs = np.array([s["outside_frac"] for s in stats])
    dilated_areas = np.array([s["dilated_area"] for s in stats])
    ref_areas = np.array([s["refined_area"] for s in stats])

    print(f"  IoU (dilated vs MetaUAS):   mean={ious.mean():.3f}  median={np.median(ious):.3f}  "
          f"min={ious.min():.3f}  max={ious.max():.3f}")
    print(f"  Outside fraction:           mean={outside_fracs.mean():.2%}  "
          f"median={np.median(outside_fracs):.2%}  max={outside_fracs.max():.2%}")
    print(f"  Area ratio (refined/dilated): mean={np.mean(ref_areas/np.maximum(dilated_areas, 1)):.2f}  "
          f"median={np.median(ref_areas/np.maximum(dilated_areas, 1)):.2f}")

    # Breakdown easy vs hard
    easy = [s for s in stats if int(s["idx"]) < 94]
    hard = [s for s in stats if int(s["idx"]) >= 94]
    if easy:
        e_iou = np.array([s["iou"] for s in easy])
        e_out = np.array([s["outside_frac"] for s in easy])
        print(f"\n  Easy (000-093): IoU mean={e_iou.mean():.3f}, outside mean={e_out.mean():.2%}")
    if hard:
        h_iou = np.array([s["iou"] for s in hard])
        h_out = np.array([s["outside_frac"] for s in hard])
        print(f"  Hard (094-099): IoU mean={h_iou.mean():.3f}, outside mean={h_out.mean():.2%}")

    # Save overview grid (first 20 samples)
    n_show = min(20, len(stats))
    fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_show):
        s = stats[i]
        idx = s["idx"]

        canvas = np.array(Image.open(prep_dir / idx / "canvas.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        generated = np.array(Image.open(gen_dir / f"{idx}.png").convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        raw_m = load_mask_np(prep_dir / idx / "placed_mask.png", IMG_SIZE)
        dilated_m = compute_roundtripped_dilated(raw_m, band_mode=2)
        ref_m = np.array(Image.open(out_dir / f"{idx}_mask.png")) / 255.0
        prob = np.array(Image.open(out_dir / f"{idx}_prob.png")) / 255.0

        axes[i, 0].imshow(canvas); axes[i, 0].set_title(f"{idx} Canvas")
        axes[i, 1].imshow(generated); axes[i, 1].set_title("Generated")

        # Overlay: red=dilated zone, green=MetaUAS mask, yellow=overlap
        overlay = generated.astype(np.float32) / 255.0
        red = np.zeros_like(overlay)
        red[dilated_m > 0.5] = [1, 0, 0]
        green = np.zeros_like(overlay)
        green[ref_m > 0.5] = [0, 1, 0]
        combined = overlay * 0.5 + red * 0.25 + green * 0.25
        axes[i, 2].imshow(combined.clip(0, 1))
        axes[i, 2].set_title(f"Red=dilated, Green=MetaUAS\nIoU={s['iou']:.3f}")

        axes[i, 3].imshow(prob, cmap="hot", vmin=0, vmax=1)
        axes[i, 3].set_title("MetaUAS prob map")

        # Difference: MetaUAS minus dilated
        diff = ref_m - dilated_m  # +1=added, -1=removed
        axes[i, 4].imshow(diff, cmap="RdBu", vmin=-1, vmax=1)
        axes[i, 4].set_title(f"Diff (blue=removed, red=added)\noutside={s['outside_frac']:.1%}")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    grid_path = out_dir / "overview_grid.png"
    fig.savefig(grid_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"\nSaved overview grid: {grid_path}")


if __name__ == "__main__":
    main()
