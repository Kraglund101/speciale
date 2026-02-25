"""
Latent-space dilation visualization: 100 samples stratified across products.

3 rows per sample (one per dilation method):
  Row 0: Current image-space — dilate at 512 (r=clamp(0.5*sqrt(area),8,16)), maxpool→64
  Row 1: Latent 1px — maxpool 512→64, then dilate(r=1) at 64
  Row 2: Latent 2px — maxpool 512→64, then dilate(r=2) at 64

Each row has 2 columns (same style as roundtrip_100):
  Col 0: 512x512 with 4-zone coloring (GT, core RT overflow, ring, dilated RT overflow)
  Col 1: 64x64 roundtrip (shown at 512 scale) with RGBA overlays + grid lines

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
sys.path.insert(0, str(ROOT))
from src.utils.mask_utils import downsample_mask_maxpool
DATA_BASE = ROOT / "anomverse_extension" / "datasets" / "full_training_dataset"


def adaptive_dilation_radius(mask_np, min_r=8, max_r=16):
    area = mask_np.sum()
    r = 0.5 * math.sqrt(area)
    return int(max(min_r, min(max_r, round(r))))


def dilate_mask_image(mask_np, radius):
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    struct = (x**2 + y**2) <= radius**2
    return binary_dilation(mask_np > 0.5, structure=struct).astype(np.float32)


def dilate_latent(mask_64, radius):
    """Dilate a [1,1,64,64] mask by radius latent pixels via iterative 8-connected growth.

    Grows all 8 neighbours once per iteration, repeated `radius` times.
    Simple and predictable: 1px = immediate neighbours, 2px = two steps out.
    """
    if radius <= 0:
        return mask_64
    dilated = mask_64
    for _ in range(radius):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
    return (dilated > 0.5).float()


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
    parser.add_argument("--save-dir", default="results/unet/latent_dilation_100")
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
        "core_64": [], "lat1_ring": [], "lat2_ring": [], "img_ring": [],
        "r_values": [], "coverage_pct": [],
        "core_exp": [], "img_dil_exp": [], "lat1_exp": [], "lat2_exp": [],
    })

    for si, idx in enumerate(indices):
        s = valid[idx]
        product = s["product"]
        defect = s.get("defect_type", "?")

        img_pil = Image.open(str(DATA_BASE / s["image_path"])).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(DATA_BASE / s["mask_path"])).convert("L")
        mask_raw = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0)
        mask_raw = (mask_raw > 0.5).float().unsqueeze(0)  # [1, H, W]
        mask_raw = downsample_mask_maxpool(mask_raw, 512)  # [1, 512, 512]
        mask_np = mask_raw.squeeze(0).numpy()
        mask_t = mask_raw.unsqueeze(0)  # [1, 1, 512, 512]

        img_np = np.array(img_pil).astype(np.float32) / 255.0

        coverage_pct = 100 * mask_np.sum() / (512 * 512)
        area_px = int(mask_np.sum())

        if area_px == 0:
            print(f"  Skip {si}: empty mask")
            continue

        # Core: maxpool 512→64
        core_64 = F.max_pool2d(mask_t, kernel_size=8)
        core_rt = F.interpolate(core_64, size=(512, 512), mode="nearest").squeeze().numpy()
        core_cells = int((core_64.squeeze() > 0.5).sum().item())
        core_expansion = (core_rt > 0.5).sum() / max(area_px, 1)

        # --- Image-space dilation (old approach) ---
        r = adaptive_dilation_radius(mask_np, min_r=8, max_r=16)
        dilated_np = dilate_mask_image(mask_np, r)
        dilated_t = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(0)
        img_dil_64 = F.max_pool2d(dilated_t, kernel_size=8)
        img_dil_rt = F.interpolate(img_dil_64, size=(512, 512), mode="nearest").squeeze().numpy()
        img_dil_cells = int((img_dil_64.squeeze() > 0.5).sum().item())
        img_dil_ring = img_dil_cells - core_cells
        img_dil_expansion = (img_dil_rt > 0.5).sum() / max(dilated_np.sum(), 1)

        # --- Latent 1px dilation ---
        lat1_64 = dilate_latent(core_64, radius=1)
        lat1_rt = F.interpolate(lat1_64, size=(512, 512), mode="nearest").squeeze().numpy()
        lat1_cells = int((lat1_64.squeeze() > 0.5).sum().item())
        lat1_ring = lat1_cells - core_cells
        lat1_expansion = (lat1_rt > 0.5).sum() / max(area_px, 1)

        # --- Latent 2px dilation ---
        lat2_64 = dilate_latent(core_64, radius=2)
        lat2_rt = F.interpolate(lat2_64, size=(512, 512), mode="nearest").squeeze().numpy()
        lat2_cells = int((lat2_64.squeeze() > 0.5).sum().item())
        lat2_ring = lat2_cells - core_cells
        lat2_expansion = (lat2_rt > 0.5).sum() / max(area_px, 1)

        # Stats
        stats_by_product[product]["core_64"].append(core_cells)
        stats_by_product[product]["lat1_ring"].append(lat1_ring)
        stats_by_product[product]["lat2_ring"].append(lat2_ring)
        stats_by_product[product]["img_ring"].append(img_dil_ring)
        stats_by_product[product]["r_values"].append(r)
        stats_by_product[product]["coverage_pct"].append(coverage_pct)
        stats_by_product[product]["core_exp"].append(core_expansion)
        stats_by_product[product]["img_dil_exp"].append(img_dil_expansion)
        stats_by_product[product]["lat1_exp"].append(lat1_expansion)
        stats_by_product[product]["lat2_exp"].append(lat2_expansion)

        # --- 3 rows: one per method ---
        # Each row: (dilated_np at 512, dilated_rt at 512, dil_64, label, ring_cells, total_cells, expansion)
        rows = [
            (dilated_np,  img_dil_rt,  img_dil_64, f"Image-space (r={r})",  img_dil_ring, img_dil_cells, img_dil_expansion),
            (lat1_rt,     lat1_rt,     lat1_64,    "Latent 1px",            lat1_ring,    lat1_cells,    lat1_expansion),
            (lat2_rt,     lat2_rt,     lat2_64,    "Latent 2px",            lat2_ring,    lat2_cells,    lat2_expansion),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(9, 13))

        zone_colors = {
            1: (1.0, 0.0, 0.0),       # red: GT
            2: (1.0, 0.8, 0.0),       # yellow: core RT overflow
            3: (0.2, 0.4, 1.0),       # blue: dilation ring
            4: (0.0, 0.9, 0.0),       # green: dilated RT overflow
        }

        for ri, (dil_at_512, dil_rt, dil_64, label, ring_cells, total_cells, expansion) in enumerate(rows):
            # --- Col 0: 4-zone coloring at 512 ---
            zone = np.zeros((512, 512), dtype=np.int32)
            zone[dil_rt > 0.5] = 4        # green: dilated RT overflow
            zone[dil_at_512 > 0.5] = 3    # blue: dilation ring + core
            zone[core_rt > 0.5] = 2       # yellow: core RT overflow
            zone[mask_np > 0.5] = 1       # red: GT core

            overlay = img_np.copy()
            for z, col in zone_colors.items():
                mask_z = (zone == z)
                for c in range(3):
                    overlay[:, :, c] = np.where(mask_z, img_np[:, :, c] * 0.4 + col[c] * 0.6, overlay[:, :, c])

            axes[ri, 0].imshow(overlay, interpolation="nearest")
            axes[ri, 0].set_title(
                f"{label}\n"
                f"Core RT: {core_expansion:.1f}x | Dilated RT: {expansion:.1f}x",
                fontsize=7,
            )
            axes[ri, 0].axis("off")

            # --- Col 1: solid cell-fill at 64x64 resolution ---
            # Paint each latent cell as a solid color block — no alpha blending issues
            core_np = core_64.squeeze().numpy()      # [64,64]
            dil_np_64 = dil_64.squeeze().numpy()     # [64,64]
            ring_np_64 = ((dil_np_64 > 0.5) & (core_np < 0.5)).astype(np.float32)

            # Build a colored image at 64x64, then upscale
            cell_img = np.zeros((64, 64, 3), dtype=np.float32)
            # Start with downsampled image as background
            img_small = np.array(img_pil.resize((64, 64), Image.BILINEAR)).astype(np.float32) / 255.0
            cell_img[:] = img_small

            # Paint core cells solid red (with slight image bleed-through)
            core_bool = core_np > 0.5
            for c, val in enumerate([1.0, 0.0, 0.0]):
                cell_img[:, :, c] = np.where(core_bool, img_small[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])

            # Paint ring cells solid blue
            ring_bool = ring_np_64 > 0.5
            for c, val in enumerate([0.2, 0.4, 1.0]):
                cell_img[:, :, c] = np.where(ring_bool, img_small[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])

            # Upscale with nearest to 512 — crisp cell boundaries
            axes[ri, 1].imshow(cell_img, interpolation="nearest", extent=[0, 512, 512, 0])

            # Grid lines at 64x64 latent cell boundaries
            cell_px = 512.0 / 64
            for g in range(64 + 1):
                axes[ri, 1].axhline(y=g * cell_px, color="white", linewidth=0.15, alpha=0.3)
                axes[ri, 1].axvline(x=g * cell_px, color="white", linewidth=0.15, alpha=0.3)

            axes[ri, 1].set_title(
                f"64x64\n{core_cells}c + {ring_cells}r / 4096",
                fontsize=9,
            )
            axes[ri, 1].set_xlim(0, 512)
            axes[ri, 1].set_ylim(512, 0)
            axes[ri, 1].axis("off")

        legend_elements = [
            Patch(facecolor=(1, 0, 0), label="GT anomaly"),
            Patch(facecolor=(1, 0.8, 0), label="Core RT overflow (8x8 rounding beyond GT)"),
            Patch(facecolor=(0.2, 0.4, 1), label="Dilation ring"),
            Patch(facecolor=(0, 0.9, 0), label="Dilated RT overflow (8x8 rounding beyond dilation)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=6.5,
                   bbox_to_anchor=(0.5, -0.01))

        plt.suptitle(
            f"Sample {si} \u2014 {product} / {defect} \u2014 {coverage_pct:.2f}% coverage\n"
            f"Maxpool 512\u219264 \u2022 Core: {core_cells}c \u2022 "
            f"Img r={r}: +{img_dil_ring}r \u2022 Lat 1px: +{lat1_ring}r \u2022 Lat 2px: +{lat2_ring}r",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.92])
        plt.savefig(save_dir / f"latdil_{si:04d}_{product}_{defect}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()

        if (si + 1) % 20 == 0:
            print(f"  {si + 1}/{len(indices)} done")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Product':<22} {'N':>3} {'Core64':>7} {'L1 rng':>7} {'L2 rng':>7} {'Img rng':>7} {'r':>4} {'Cov%':>7}")
    print("-" * 75)
    all_core = []
    all_lat1 = []
    all_lat2 = []
    all_img = []
    all_cov = []
    for prod in sorted(stats_by_product.keys()):
        st = stats_by_product[prod]
        n = len(st["core_64"])
        all_core.extend(st["core_64"])
        all_lat1.extend(st["lat1_ring"])
        all_lat2.extend(st["lat2_ring"])
        all_img.extend(st["img_ring"])
        all_cov.extend(st["coverage_pct"])
        print(f"{prod:<22} {n:>3} {np.median(st['core_64']):>7.0f} {np.median(st['lat1_ring']):>7.0f} "
              f"{np.median(st['lat2_ring']):>7.0f} {np.median(st['img_ring']):>7.0f} "
              f"{np.median(st['r_values']):>4.0f} {np.median(st['coverage_pct']):>6.2f}%")
    print("-" * 75)
    print(f"{'OVERALL':<22} {len(all_core):>3} {np.median(all_core):>7.0f} {np.median(all_lat1):>7.0f} "
          f"{np.median(all_lat2):>7.0f} {np.median(all_img):>7.0f} "
          f"{'':>4} {np.median(all_cov):>6.2f}%")

    # Summary figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    labels = ["Core", "Lat 1px", "Lat 2px", "Img-space"]
    totals = [
        np.array(all_core),
        np.array(all_core) + np.array(all_lat1),
        np.array(all_core) + np.array(all_lat2),
        np.array(all_core) + np.array(all_img),
    ]
    rings = [
        np.zeros(len(all_core)),
        np.array(all_lat1),
        np.array(all_lat2),
        np.array(all_img),
    ]

    axes[0].bar(labels, [np.median(t) for t in totals],
                color=["red", "#66aaff", "#2266cc", "green"], alpha=0.6)
    axes[0].set_ylabel("Median total cells (64x64)")
    axes[0].set_title("Total masked cells")

    axes[1].bar(labels[1:], [np.median(r) for r in rings[1:]],
                color=["#66aaff", "#2266cc", "green"], alpha=0.6)
    axes[1].set_ylabel("Median ring cells")
    axes[1].set_title("Dilation ring cells")

    core_arr = np.maximum(np.array(all_core), 1)
    ratios = [r / core_arr for r in rings[1:]]
    axes[2].bar(labels[1:], [np.median(r) for r in ratios],
                color=["#66aaff", "#2266cc", "green"], alpha=0.6)
    axes[2].set_ylabel("Median ring/core ratio")
    axes[2].set_title("Ring relative to core")
    axes[2].axhline(y=1.0, color="gray", ls="--", lw=0.8)

    gt_areas = np.array(all_cov) / 100 * 512 * 512
    gt_areas = np.maximum(gt_areas, 1)
    exp_factors = [(t * 64) / gt_areas for t in totals]
    axes[3].bar(labels, [np.median(e) for e in exp_factors],
                color=["red", "#66aaff", "#2266cc", "green"], alpha=0.6)
    axes[3].set_ylabel("Median expansion x")
    axes[3].set_title("Roundtrip area / GT area")
    axes[3].axhline(y=1.0, color="gray", ls="--", lw=0.8)

    plt.suptitle(
        f"Latent-space vs image-space dilation across {len(all_core)} samples\n"
        f"Median core: {np.median(all_core):.0f} | "
        f"Lat 1px ring: {np.median(all_lat1):.0f} | "
        f"Lat 2px ring: {np.median(all_lat2):.0f} | "
        f"Img-space ring: {np.median(all_img):.0f}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved {len(all_core)} + summary to {save_dir}/")


if __name__ == "__main__":
    main()
