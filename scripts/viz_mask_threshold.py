"""Visualize mask downsampling threshold impact through the full pipeline.

Compares three strategies for resizing masks from original resolution to 512:
  1. nearest + > 0.5  (OLD — the bug we're fixing)
  2. bilinear + > 0.5
  3. bilinear + > 0    (current fix)

Then traces each through:
  512→64 maxpool → band dilation (mode 2) → total affected area

Outputs:
  - Per-sample visual comparison (mask at 512, 64, 64+band)
  - Aggregate stats: pixel counts, % inflation, by resolution bucket
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import downsample_mask_maxpool, create_latent_band_mask


def resize_mask_nearest(mask_pil: Image.Image, size: int = 512) -> torch.Tensor:
    """Old pipeline: nearest resize + > 0.5 threshold."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > 0.5).float()  # [1, H, W]


def resize_mask_bilinear(mask_pil: Image.Image, size: int = 512,
                         threshold: float = 0.5) -> torch.Tensor:
    """New pipeline: bilinear resize + threshold."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > threshold).float()  # [1, H, W]


def full_pipeline(mask_512: torch.Tensor, band_mode: int = 2):
    """Run mask through 512→64 maxpool + band dilation. Returns core, dilated, band pixel counts."""
    mask_4d = mask_512.unsqueeze(0)  # [1, 1, 512, 512]
    core_64 = downsample_mask_maxpool(mask_4d, 64)  # [1, 1, 64, 64]
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=band_mode
    )
    n_core = core_64.sum().item()
    n_band = band_mask.sum().item()
    n_total = dilated_binary.sum().item()
    return core_64.squeeze(), dilated_binary.squeeze(), band_mask.squeeze(), n_core, n_band, n_total


def main():
    root = Path("anomverse_extension/datasets/full_training_dataset")
    master_path = root / "master_training.json"

    with open(master_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("images") or data.get("samples", [])

    # Collect samples by resolution
    by_res = defaultdict(list)
    for e in entries:
        mp = e.get("mask_path_full") or e.get("mask_path_abs") or e.get("mask_path", "")
        if mp and not os.path.isabs(mp):
            mp = str(root / mp)
        if os.path.exists(mp):
            by_res[mp] = mp  # dedup

    # Rebuild grouped by resolution
    res_groups = defaultdict(list)
    seen = set()
    for e in entries:
        mp = e.get("mask_path_full") or e.get("mask_path_abs") or e.get("mask_path", "")
        if mp and not os.path.isabs(mp):
            mp = str(root / mp)
        if mp in seen or not os.path.exists(mp):
            continue
        seen.add(mp)
        try:
            m = Image.open(mp)
            res = f"{m.size[0]}x{m.size[1]}"
            if len(res_groups[res]) < 200:  # cap per resolution
                res_groups[res].append(mp)
        except Exception:
            pass

    print(f"Resolutions found: {sorted(res_groups.keys(), key=lambda x: -len(res_groups[x]))}")
    for res in sorted(res_groups.keys(), key=lambda x: -len(res_groups[x])):
        print(f"  {res}: {len(res_groups[res])} samples")

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------
    strategies = ["nearest>0.5", "bilinear>0.5", "bilinear>0"]
    # Per-resolution stats: {res: {strategy: [n_core_64, ...], ...}}
    stats = defaultdict(lambda: defaultdict(lambda: {
        "n_512": [], "n_core_64": [], "n_band_64": [], "n_total_64": [],
        "lost_pixels": [],  # pixels in ground truth (bilinear>0) but not in this strategy at 512
    }))

    all_masks_for_viz = []  # store a few for visual comparison

    for res, paths in sorted(res_groups.items(), key=lambda x: -len(x[1])):
        for mp in paths:
            mask_pil = Image.open(mp).convert("L")
            orig_px = (np.array(mask_pil) > 127).sum()
            if orig_px == 0:
                continue

            # Three strategies
            m_nearest = resize_mask_nearest(mask_pil)
            m_bilinear_05 = resize_mask_bilinear(mask_pil, threshold=0.5)
            m_bilinear_00 = resize_mask_bilinear(mask_pil, threshold=0.0)

            results = {}
            for name, m512 in [("nearest>0.5", m_nearest),
                               ("bilinear>0.5", m_bilinear_05),
                               ("bilinear>0", m_bilinear_00)]:
                n512 = m512.sum().item()
                core64, dil64, band64, nc, nb, nt = full_pipeline(m512)
                results[name] = {
                    "m512": m512, "core64": core64, "dil64": dil64,
                    "n512": n512, "n_core_64": nc, "n_band_64": nb, "n_total_64": nt,
                }
                stats[res][name]["n_512"].append(n512)
                stats[res][name]["n_core_64"].append(nc)
                stats[res][name]["n_band_64"].append(nb)
                stats[res][name]["n_total_64"].append(nt)

            # Pixel loss compared to bilinear>0 at 512 (most conservative)
            ref_512 = m_bilinear_00
            for name, m512 in [("nearest>0.5", m_nearest),
                               ("bilinear>0.5", m_bilinear_05),
                               ("bilinear>0", m_bilinear_00)]:
                lost = ((ref_512 > 0.5) & (m512 < 0.5)).sum().item()
                stats[res][name]["lost_pixels"].append(lost)

            # Save a few for visual comparison
            if len(all_masks_for_viz) < 12:
                all_masks_for_viz.append({
                    "path": mp, "res": res, "orig_px": orig_px,
                    "results": results,
                })

    # ------------------------------------------------------------------
    # Print aggregate table
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AGGREGATE STATS BY RESOLUTION")
    print("=" * 100)

    for res in sorted(stats.keys(), key=lambda x: -len(res_groups.get(x, []))):
        n_samples = len(stats[res]["nearest>0.5"]["n_512"])
        if n_samples == 0:
            continue
        print(f"\n--- {res} ({n_samples} non-empty samples) ---")
        print(f"{'Strategy':<16} {'mean@512':>10} {'mean_core@64':>14} {'mean_band@64':>14} "
              f"{'mean_total@64':>15} {'mean_lost@512':>15}")
        for strat in strategies:
            s = stats[res][strat]
            print(f"{strat:<16} {np.mean(s['n_512']):>10.1f} {np.mean(s['n_core_64']):>14.1f} "
                  f"{np.mean(s['n_band_64']):>14.1f} {np.mean(s['n_total_64']):>15.1f} "
                  f"{np.mean(s['lost_pixels']):>15.1f}")

    # Inflation: bilinear>0 vs bilinear>0.5 at 64x64
    print("\n" + "=" * 100)
    print("INFLATION: bilinear>0 vs bilinear>0.5 (at 64x64 post-dilation)")
    print("=" * 100)
    for res in sorted(stats.keys(), key=lambda x: -len(res_groups.get(x, []))):
        s05 = stats[res]["bilinear>0.5"]
        s00 = stats[res]["bilinear>0"]
        n = len(s05["n_total_64"])
        if n == 0:
            continue
        # Per-sample inflation
        inflations_core = []
        inflations_total = []
        for i in range(n):
            c05 = s05["n_core_64"][i]
            c00 = s00["n_core_64"][i]
            t05 = s05["n_total_64"][i]
            t00 = s00["n_total_64"][i]
            if c05 > 0:
                inflations_core.append((c00 - c05) / c05 * 100)
            if t05 > 0:
                inflations_total.append((t00 - t05) / t05 * 100)
        if inflations_core:
            print(f"{res} ({n} samples):")
            print(f"  Core@64:  mean +{np.mean(inflations_core):.1f}%, "
                  f"median +{np.median(inflations_core):.1f}%, "
                  f"max +{np.max(inflations_core):.1f}%, "
                  f"p95 +{np.percentile(inflations_core, 95):.1f}%")
            print(f"  Total@64: mean +{np.mean(inflations_total):.1f}%, "
                  f"median +{np.median(inflations_total):.1f}%, "
                  f"max +{np.max(inflations_total):.1f}%, "
                  f"p95 +{np.percentile(inflations_total, 95):.1f}%")

    # Also: bilinear>0 vs nearest>0.5 (what we fixed)
    print("\n" + "=" * 100)
    print("PIXELS LOST: nearest>0.5 vs bilinear>0.5 (at 512, relative to bilinear>0)")
    print("=" * 100)
    for res in sorted(stats.keys(), key=lambda x: -len(res_groups.get(x, []))):
        s_near = stats[res]["nearest>0.5"]
        s_bi05 = stats[res]["bilinear>0.5"]
        n = len(s_near["lost_pixels"])
        if n == 0:
            continue
        near_lost = [x for x in s_near["lost_pixels"] if x > 0]
        bi05_lost = [x for x in s_bi05["lost_pixels"] if x > 0]
        print(f"{res} ({n} samples):")
        print(f"  nearest>0.5:  {len(near_lost)}/{n} samples lost pixels, "
              f"mean lost={np.mean(s_near['lost_pixels']):.1f}")
        print(f"  bilinear>0.5: {len(bi05_lost)}/{n} samples lost pixels, "
              f"mean lost={np.mean(s_bi05['lost_pixels']):.1f}")

    # ------------------------------------------------------------------
    # Visual comparison figure
    # ------------------------------------------------------------------
    n_viz = min(8, len(all_masks_for_viz))
    if n_viz == 0:
        print("No samples to visualize!")
        return

    fig, axes = plt.subplots(n_viz, 7, figsize=(28, 4 * n_viz))
    if n_viz == 1:
        axes = axes[np.newaxis, :]

    col_titles = [
        "nearest>0.5 @512", "bilinear>0.5 @512", "bilinear>0 @512",
        "nearest core@64", "bilinear>0.5 core@64", "bilinear>0 core@64",
        "bilinear>0 dil@64"
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i, item in enumerate(all_masks_for_viz[:n_viz]):
        r = item["results"]
        res = item["res"]

        # @512 masks
        axes[i, 0].imshow(r["nearest>0.5"]["m512"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(r["bilinear>0.5"]["m512"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(r["bilinear>0"]["m512"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)

        # Core @64
        axes[i, 3].imshow(r["nearest>0.5"]["core64"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 4].imshow(r["bilinear>0.5"]["core64"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
        axes[i, 5].imshow(r["bilinear>0"]["core64"].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)

        # Dilated @64 (bilinear>0 only, to show band)
        dil = r["bilinear>0"]["dil64"].squeeze().numpy()
        core = r["bilinear>0"]["core64"].squeeze().numpy()
        # Color: core=white, band=gray, bg=black
        viz = np.zeros((*dil.shape, 3))
        viz[core > 0.5] = [1, 1, 1]
        viz[(dil > 0.5) & (core < 0.5)] = [0.5, 0.5, 0.8]  # band = blueish
        axes[i, 6].imshow(viz)

        # Row label
        nc_near = int(r["nearest>0.5"]["n_core_64"])
        nc_bi05 = int(r["bilinear>0.5"]["n_core_64"])
        nc_bi00 = int(r["bilinear>0"]["n_core_64"])
        nt_bi05 = int(r["bilinear>0.5"]["n_total_64"])
        nt_bi00 = int(r["bilinear>0"]["n_total_64"])
        axes[i, 0].set_ylabel(
            f"{res}\norig={item['orig_px']}\n"
            f"c64: {nc_near}/{nc_bi05}/{nc_bi00}\n"
            f"tot: -/{nt_bi05}/{nt_bi00}",
            fontsize=8, rotation=0, labelpad=80, va="center",
        )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("Mask Threshold Comparison: nearest>0.5 vs bilinear>0.5 vs bilinear>0\n"
                 "Through full pipeline: resize→512 → maxpool→64 → band dilation (mode 2)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = "results/viz_mask_threshold.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visual comparison to {out_path}")
    plt.close()

    # ------------------------------------------------------------------
    # Histogram: inflation distribution
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all per-sample inflations
    all_core_infl = []
    all_total_infl = []
    for res in stats:
        s05 = stats[res]["bilinear>0.5"]
        s00 = stats[res]["bilinear>0"]
        for i in range(len(s05["n_core_64"])):
            c05 = s05["n_core_64"][i]
            c00 = s00["n_core_64"][i]
            t05 = s05["n_total_64"][i]
            t00 = s00["n_total_64"][i]
            if c05 > 0:
                all_core_infl.append((c00 - c05) / c05 * 100)
            if t05 > 0:
                all_total_infl.append((t00 - t05) / t05 * 100)

    axes[0].hist(all_core_infl, bins=50, edgecolor="black", alpha=0.7)
    axes[0].set_title(f"Core@64 inflation: bilinear>0 vs bilinear>0.5\n"
                      f"mean={np.mean(all_core_infl):.1f}%, median={np.median(all_core_infl):.1f}%")
    axes[0].set_xlabel("% inflation")
    axes[0].set_ylabel("Count")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.5)

    axes[1].hist(all_total_infl, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_title(f"Total@64 (core+band) inflation: bilinear>0 vs bilinear>0.5\n"
                      f"mean={np.mean(all_total_infl):.1f}%, median={np.median(all_total_infl):.1f}%")
    axes[1].set_xlabel("% inflation")
    axes[1].set_ylabel("Count")
    axes[1].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path2 = "results/viz_mask_threshold_hist.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved inflation histograms to {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()
