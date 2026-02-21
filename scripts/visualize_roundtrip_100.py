"""
Roundtrip dilation visualization: 100 samples stratified across products.

5 columns per sample:
  Col 0: 512x512 with 4-zone coloring (GT, core RT overflow, ring, dilated RT overflow)
  Col 1-4: 64/32/16/8 maxpooled (shown at 512 scale) with grid lines

4 colors:
  Red    = GT anomaly
  Yellow = Core RT overflow (8x8 rounding beyond GT)
  Blue   = Dilation ring
  Green  = Dilated RT overflow (8x8 rounding beyond dilation)

Loads directly from master_training.json for product info.
"""
import sys
import json
import random
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.ndimage import binary_dilation

random.seed(123)
np.random.seed(123)

ROOT = Path(__file__).parent.parent
DATA_BASE = ROOT / "anomverse_extension" / "datasets" / "full_training_dataset"
RESOLUTIONS = [64, 32, 16, 8]


def adaptive_dilation_radius(mask_np, min_r=8, max_r=16):
    area = mask_np.sum()
    r = 0.5 * math.sqrt(area)
    return int(max(min_r, min(max_r, round(r))))


def dilate_mask(mask_np, radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    struct = (x**2 + y**2) <= radius**2
    return binary_dilation(mask_np > 0.5, structure=struct).astype(np.float32)


def stratified_sample(samples, n_target, rng):
    by_product = defaultdict(list)
    for i, s in enumerate(samples):
        by_product[s["product"]].append(i)
    products = sorted(by_product.keys())
    n_products = len(products)
    base = n_target // n_products
    remainder = n_target - base * n_products
    shuffled = list(products)
    rng.shuffle(shuffled)
    selected = []
    for j, prod in enumerate(shuffled):
        count = base + (1 if j < remainder else 0)
        pool = by_product[prod]
        count = min(count, len(pool))
        selected.extend(rng.sample(pool, count))
    if len(selected) < n_target:
        used = set(selected)
        remaining = [i for i in range(len(samples)) if i not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[:n_target - len(selected)])
    rng.shuffle(selected)
    return selected[:n_target]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="results/unet/roundtrip_100")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--dataset-filter", default="realiad")
    args = parser.parse_args()

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(123)

    print(f"Loading master_training.json...")
    with open(DATA_BASE / "master_training.json", encoding="utf-8") as f:
        all_samples = json.load(f)
    print(f"Total: {len(all_samples)}")

    if args.dataset_filter:
        all_samples = [s for s in all_samples if s["dataset"] == args.dataset_filter]
        print(f"After filter '{args.dataset_filter}': {len(all_samples)}")

    # Filter to samples with non-empty masks
    valid = []
    for i, s in enumerate(all_samples):
        mask_path = DATA_BASE / s["mask_path"]
        if mask_path.exists():
            valid.append(s)
    print(f"Valid (mask exists): {len(valid)}")

    indices = stratified_sample(valid, args.n_samples, rng)
    print(f"Selected {len(indices)} across {len(set(valid[i]['product'] for i in indices))} products")

    # Stats tracking
    stats_by_product = defaultdict(lambda: {
        "core_64": [], "ring_64": [], "r_values": [], "coverage_pct": [],
        "core_exp": [], "dil_exp": [],
    })

    for si, idx in enumerate(indices):
        s = valid[idx]
        product = s["product"]
        defect = s.get("defect_type", "?")

        img_pil = Image.open(str(DATA_BASE / s["image_path"])).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(DATA_BASE / s["mask_path"])).convert("L").resize((512, 512), Image.NEAREST)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        coverage_pct = 100 * mask_np.sum() / (512 * 512)
        area_px = int(mask_np.sum())

        if area_px == 0:
            print(f"  Skip {si}: empty mask")
            continue

        r = adaptive_dilation_radius(mask_np, min_r=8, max_r=16)
        dilated_np = dilate_mask(mask_np, r)
        dilated_t = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(0)

        # Roundtrip CORE: 512 -> 64 -> 512
        core_64 = F.max_pool2d(mask_t, kernel_size=8)
        core_rt = F.interpolate(core_64, size=(512, 512), mode="nearest").squeeze().numpy()
        core_expansion = (core_rt > 0.5).sum() / max(area_px, 1)

        # Roundtrip DILATED: 512 -> 64 -> 512
        dil_64 = F.max_pool2d(dilated_t, kernel_size=8)
        dil_rt = F.interpolate(dil_64, size=(512, 512), mode="nearest").squeeze().numpy()
        dil_expansion = (dil_rt > 0.5).sum() / max(dilated_np.sum(), 1)

        # Stats
        core_64_cells = int((core_64.squeeze() > 0.5).sum().item())
        dil_64_cells = int((dil_64.squeeze() > 0.5).sum().item())
        ring_64_cells = dil_64_cells - core_64_cells
        stats_by_product[product]["core_64"].append(core_64_cells)
        stats_by_product[product]["ring_64"].append(ring_64_cells)
        stats_by_product[product]["r_values"].append(r)
        stats_by_product[product]["coverage_pct"].append(coverage_pct)
        stats_by_product[product]["core_exp"].append(core_expansion)
        stats_by_product[product]["dil_exp"].append(dil_expansion)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        # Col 0: 4-zone coloring at 512
        zone = np.zeros((512, 512), dtype=np.int32)
        zone[dil_rt > 0.5] = 4       # green: dilated RT overflow
        zone[dilated_np > 0.5] = 3    # blue: dilation ring + core
        zone[core_rt > 0.5] = 2       # yellow: core RT overflow
        zone[mask_np > 0.5] = 1       # red: GT core

        colors = {
            1: (1.0, 0.0, 0.0),
            2: (1.0, 0.8, 0.0),
            3: (0.2, 0.4, 1.0),
            4: (0.0, 0.9, 0.0),
        }
        overlay = img_np.copy()
        for z, col in colors.items():
            mask_z = (zone == z)
            for c in range(3):
                overlay[:, :, c] = np.where(mask_z, img_np[:, :, c] * 0.4 + col[c] * 0.6, overlay[:, :, c])

        axes[0].imshow(overlay)
        axes[0].set_title(
            f"512x512 | dilation r={r}\n"
            f"Core RT: {core_expansion:.1f}x | Dilated RT: {dil_expansion:.1f}x",
            fontsize=7,
        )
        axes[0].axis("off")

        # Cols 1-4: maxpooled at each resolution with grid
        for ri, res in enumerate(RESOLUTIONS):
            pool_k = 512 // res

            core_down = F.max_pool2d(mask_t, kernel_size=pool_k)
            dilated_down = F.max_pool2d(dilated_t, kernel_size=pool_k)

            core_cells = int((core_down.squeeze() > 0.5).sum().item())
            dilated_cells = int((dilated_down.squeeze() > 0.5).sum().item())
            ring_cells = dilated_cells - core_cells

            core_up = F.interpolate(core_down, size=(512, 512), mode="nearest").squeeze().numpy()
            dilated_up = F.interpolate(dilated_down, size=(512, 512), mode="nearest").squeeze().numpy()
            ring_up = (dilated_up > 0.5) & (core_up < 0.5)

            axes[ri + 1].imshow(img_np)
            ov_c = np.zeros((512, 512, 4), dtype=np.float32)
            ov_c[:, :, 0] = 1.0
            ov_c[:, :, 3] = (core_up > 0.5).astype(np.float32) * 0.6
            axes[ri + 1].imshow(ov_c)
            ov_r = np.zeros((512, 512, 4), dtype=np.float32)
            ov_r[:, :, 2] = 1.0
            ov_r[:, :, 3] = ring_up.astype(np.float32) * 0.6
            axes[ri + 1].imshow(ov_r)

            # Grid lines
            cell_px = 512.0 / res
            for g in range(res + 1):
                axes[ri + 1].axhline(y=g * cell_px, color="yellow", linewidth=0.3, alpha=0.4)
                axes[ri + 1].axvline(x=g * cell_px, color="yellow", linewidth=0.3, alpha=0.4)

            axes[ri + 1].set_title(
                f"{res}x{res}\n{core_cells}c + {ring_cells}r / {res*res}",
                fontsize=9,
            )
            axes[ri + 1].axis("off")

        legend_elements = [
            Patch(facecolor=(1, 0, 0), label="GT anomaly"),
            Patch(facecolor=(1, 0.8, 0), label="Core RT overflow (8x8 rounding beyond GT)"),
            Patch(facecolor=(0.2, 0.4, 1), label="Dilation ring"),
            Patch(facecolor=(0, 0.9, 0), label="Dilated RT overflow (8x8 rounding beyond dilation)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=6.5,
                   bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(
            f"Sample {si} \u2014 {product} / {defect} \u2014 {coverage_pct:.2f}% coverage\n"
            f"Maxpool downsampling \u2022 Dilation at 512 (r=clamp(0.5\u00b7\u221a{area_px}, 8, 16) = {r})",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.88])
        plt.savefig(save_dir / f"roundtrip_{si:04d}_{product}_{defect}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        if (si + 1) % 20 == 0:
            print(f"  {si + 1}/{len(indices)} done")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Product':<22} {'N':>3} {'Core64':>7} {'Ring64':>7} {'r':>4} {'Cov%':>7} {'CoreExp':>8} {'DilExp':>8}")
    print("-" * 75)
    for prod in sorted(stats_by_product.keys()):
        st = stats_by_product[prod]
        n = len(st["core_64"])
        print(f"{prod:<22} {n:>3} {np.median(st['core_64']):>7.0f} {np.median(st['ring_64']):>7.0f} "
              f"{np.median(st['r_values']):>4.0f} {np.median(st['coverage_pct']):>6.2f}% "
              f"{np.median(st['core_exp']):>7.1f}x {np.median(st['dil_exp']):>7.1f}x")

    # Summary figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ri, res in enumerate(RESOLUTIONS):
        exp_core = []
        exp_dil = []
        for idx in indices:
            s = valid[idx]
            mask_pil = Image.open(str(DATA_BASE / s["mask_path"])).convert("L").resize((512, 512), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            gt_area = mask_np.sum()
            if gt_area == 0:
                continue
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            r = adaptive_dilation_radius(mask_np)
            dilated_np = dilate_mask(mask_np, r)
            dilated_t = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(0)

            pool_k = 512 // res
            core_down = F.max_pool2d(mask_t, kernel_size=pool_k)
            dilated_down = F.max_pool2d(dilated_t, kernel_size=pool_k)
            core_up = F.interpolate(core_down, size=(512, 512), mode="nearest").squeeze().numpy()
            dilated_up = F.interpolate(dilated_down, size=(512, 512), mode="nearest").squeeze().numpy()
            exp_core.append((core_up > 0.5).sum() / gt_area)
            exp_dil.append((dilated_up > 0.5).sum() / gt_area)

        axes[ri].bar(["Core", "Dilated"], [np.median(exp_core), np.median(exp_dil)],
                     color=["red", "blue"], alpha=0.6)
        axes[ri].set_title(f"{res}x{res}", fontsize=10)
        axes[ri].set_ylabel("Median expansion x")
        axes[ri].axhline(y=1.0, color="gray", ls="--", lw=0.8)

    plt.suptitle(f"Median expansion factor (roundtrip area / GT area) across {len(indices)} samples",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved {len(indices)} + summary to {save_dir}/")


if __name__ == "__main__":
    main()
