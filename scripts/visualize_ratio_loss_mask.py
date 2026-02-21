"""
Visualize ratio-based soft loss mask: fixed gradient split between core and ring.

Formula:
  ring_total = n_core * (1 - core_ratio) / core_ratio
  Per-ring-cell weight (flat): w = ring_total / n_ring
  Per-ring-cell weight (shaped): gaussian shape, scaled so sum = ring_total

Compares ratios: 70/30, 80/20, 90/10
Shows per-cell weights and actual gradient contribution percentages.
"""
import math
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset

SEED = 42

RATIOS = [0.7, 0.8, 0.9]
RATIO_LABELS = ["70/30", "80/20", "90/10"]


def tensor_to_np01(t):
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    padded = F.pad(mask.unsqueeze(0), [radius] * 4, mode="constant", value=0)
    dilated = F.max_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0)
    return (dilated > 0.5).float()


def adaptive_dilation_radius(mask, min_r=2, max_r=10):
    area = mask.sum().item()
    if area == 0:
        return 0
    r = 0.5 * math.sqrt(area)
    return max(min_r, min(max_r, int(r)))


def downsample_mask_maxpool(mask, size):
    _, h, w = mask.shape
    kernel_h = h // size
    kernel_w = w // size
    target_h = size * kernel_h
    target_w = size * kernel_w
    if h != target_h or w != target_w:
        mask_resized = F.interpolate(
            mask.unsqueeze(0), size=(target_h, target_w), mode="nearest"
        ).squeeze(0)
    else:
        mask_resized = mask
    pooled = F.max_pool2d(mask_resized.unsqueeze(0), kernel_size=(kernel_h, kernel_w)).squeeze(0)
    return (pooled > 0.5).float()


def compute_ratio_mask(core_64, dilated_64, core_ratio=0.8, min_weight=0.01):
    """Compute loss weight mask with fixed core/ring gradient ratio.

    Args:
        core_64: Binary core mask [64, 64]
        dilated_64: Binary dilated mask [64, 64]
        core_ratio: Fraction of total gradient going to core (e.g. 0.8 = 80%)
        min_weight: Absolute minimum weight per ring cell

    Returns:
        soft: Weight mask [64, 64]
        actual_ratio: Actual core ratio achieved (may differ due to min_weight clamping)
    """
    ring = np.clip(dilated_64 - core_64, 0, 1)
    n_core = int(core_64.sum())
    n_ring = int(ring.sum())

    soft = np.zeros_like(core_64)
    soft[core_64 > 0.5] = 1.0

    if n_core == 0 or n_ring == 0:
        actual_ratio = 1.0 if n_core > 0 else 0.0
        return soft, actual_ratio

    # Target ring total weight
    ring_total_target = n_core * (1 - core_ratio) / core_ratio

    # Compute gaussian-shaped ring weights, then scale to hit target total
    dist = ndimage.distance_transform_edt(1 - core_64)
    ring_dist = dist * ring
    max_dist = ring_dist.max()
    if max_dist > 0:
        norm_dist = np.clip(dist / max_dist, 0, 1)
    else:
        norm_dist = np.zeros_like(dist)

    # Gaussian shape (unnormalized)
    sigma = 1.0 / math.sqrt(2 * math.log(20))
    raw_weights = np.exp(-0.5 * (norm_dist / sigma) ** 2)

    # Sum of raw weights in ring
    raw_ring_sum = (raw_weights * ring).sum()
    if raw_ring_sum > 0:
        scale = ring_total_target / raw_ring_sum
        ring_weights = raw_weights * scale
    else:
        ring_weights = np.full_like(raw_weights, ring_total_target / n_ring)

    # Clamp to minimum
    ring_weights = np.maximum(ring_weights, min_weight)

    soft[ring > 0.5] = ring_weights[ring > 0.5]

    # Compute actual ratio
    core_total = n_core * 1.0
    ring_total_actual = soft[ring > 0.5].sum()
    actual_ratio = core_total / (core_total + ring_total_actual)

    return soft, actual_ratio


def draw_dual_grid(ax, core, dilated, title):
    gs = core.shape[0]
    ring = np.clip(dilated - core, 0, 1)
    bg = np.clip(1 - dilated, 0, 1)
    img = np.zeros((gs, gs, 3), dtype=np.float32)
    img[:, :, 0] += core * 0.9; img[:, :, 1] += core * 0.15; img[:, :, 2] += core * 0.15
    img[:, :, 0] += ring * 0.15; img[:, :, 1] += ring * 0.3; img[:, :, 2] += ring * 0.9
    img[:, :, 0] += bg * 0.12; img[:, :, 1] += bg * 0.12; img[:, :, 2] += bg * 0.12
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def overlay_dual_mask(ax, img_np, orig_mask_np, dilated_mask_np, title):
    ax.imshow(img_np)
    h, w = img_np.shape[:2]
    orig_2d = orig_mask_np.squeeze()
    dil_2d = dilated_mask_np.squeeze()
    if orig_2d.shape != (h, w):
        orig_2d = np.array(Image.fromarray((orig_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    if dil_2d.shape != (h, w):
        dil_2d = np.array(Image.fromarray((dil_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    ring = np.clip(dil_2d - orig_2d, 0, 1)
    ov = np.zeros((h, w, 4), dtype=np.float32)
    ov[:, :, 2] = 1.0; ov[:, :, 3] = ring * 0.5
    ax.imshow(ov)
    ov2 = np.zeros((h, w, 4), dtype=np.float32)
    ov2[:, :, 0] = 1.0; ov2[:, :, 3] = orig_2d * 0.5
    ax.imshow(ov2)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_soft(ax, soft, title, vmax=None):
    cmap = LinearSegmentedColormap.from_list("w", [
        (0.0, (0.10, 0.10, 0.10)),
        (0.01, (0.10, 0.10, 0.10)),
        (0.02, (0.12, 0.20, 0.55)),
        (0.3, (0.70, 0.25, 0.10)),
        (0.7, (0.95, 0.70, 0.10)),
        (1.0, (1.0, 1.0, 0.85)),
    ])
    vm = vmax if vmax else max(soft.max(), 0.01)
    im = ax.imshow(soft, interpolation="nearest", cmap=cmap, vmin=0, vmax=vm)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def visualize(
    splits_dir, save_dir, n_samples=20,
    exclude_sources=None, data_root=None,
    min_r=8, max_r=16,
):
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = AnomalyDataset(
        splits_dir, image_size=512,
        exclude_sources=exclude_sources,
        data_root=data_root,
        return_reference=False,
    )
    if len(dataset) == 0:
        print("No data!")
        return

    n_samples = min(n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)

    # Track stats
    all_stats = {r: {"actual_ratio": [], "ring_w_min": [], "ring_w_max": [],
                      "ring_w_mean": [], "n_core": [], "n_ring": []}
                 for r in RATIOS}

    print(f"Processing {n_samples} samples (r=max({min_r}, min({max_r}, 0.5*sqrt(area))))...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]
        atype = sample["anomaly_type"]

        pct = mask.sum().item() / mask.numel() * 100
        area = int(mask.sum().item())
        r = adaptive_dilation_radius(mask, min_r=min_r, max_r=max_r)
        dilated = dilate_mask(mask, r)

        img_np = tensor_to_np01(image)
        mask_np = mask.squeeze(0).numpy()
        dilated_np = dilated.squeeze(0).numpy()

        core_64 = downsample_mask_maxpool(mask, 64).squeeze(0).numpy()
        dil_64 = downsample_mask_maxpool(dilated, 64).squeeze(0).numpy()
        ring_64 = np.clip(dil_64 - core_64, 0, 1)
        n_core = int(core_64.sum())
        n_ring = int(ring_64.sum())

        # Compute for each ratio
        ratio_masks = {}
        for cr in RATIOS:
            soft, actual = compute_ratio_mask(core_64, dil_64, core_ratio=cr)
            ratio_masks[cr] = (soft, actual)

        # Find zoom bbox
        ys, xs = np.where(dil_64 > 0.5)
        if len(ys) > 0:
            pad = 5
            y0, y1 = max(0, ys.min() - pad), min(64, ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(64, xs.max() + pad + 1)
            side = max(y1 - y0, x1 - x0)
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            y0 = max(0, cy - side // 2)
            y1 = min(64, y0 + side)
            if y1 - y0 < side:
                y0 = max(0, y1 - side)
            x0 = max(0, cx - side // 2)
            x1 = min(64, x0 + side)
            if x1 - x0 < side:
                x0 = max(0, x1 - side)
        else:
            y0, y1, x0, x1 = 0, 64, 0, 64

        # === Figure: 3 rows (ratios) x 4 cols ===
        fig, axes = plt.subplots(len(RATIOS), 4, figsize=(20, 4 * len(RATIOS)))

        for row, (cr, label) in enumerate(zip(RATIOS, RATIO_LABELS)):
            soft, actual = ratio_masks[cr]
            ring_weights = soft[ring_64 > 0.5]
            rw_min = ring_weights.min() if len(ring_weights) > 0 else 0
            rw_max = ring_weights.max() if len(ring_weights) > 0 else 0
            rw_mean = ring_weights.mean() if len(ring_weights) > 0 else 0

            # Track stats
            all_stats[cr]["actual_ratio"].append(actual)
            all_stats[cr]["ring_w_min"].append(rw_min)
            all_stats[cr]["ring_w_max"].append(rw_max)
            all_stats[cr]["ring_w_mean"].append(rw_mean)
            all_stats[cr]["n_core"].append(n_core)
            all_stats[cr]["n_ring"].append(n_ring)

            # Col 0: image + mask overlay
            overlay_dual_mask(axes[row, 0], img_np, mask_np, dilated_np,
                              f"{atype} (r={r})\n{n_core} core + {n_ring} ring")

            # Col 1: dual grid (red=core, blue=ring)
            draw_dual_grid(axes[row, 1], core_64, dil_64,
                           f"64x64 grid\n{n_core} core + {n_ring} ring")

            # Col 2: full soft mask
            draw_soft(axes[row, 2], soft,
                      f"Target {label} — actual {actual*100:.0f}/{(1-actual)*100:.0f}")

            # Col 3: zoomed soft mask
            crop = soft[y0:y1, x0:x1]
            draw_soft(axes[row, 3], crop,
                      f"Zoomed — ring w: {rw_min:.3f}–{rw_max:.2f} (mean={rw_mean:.3f})")

            axes[row, 0].set_ylabel(f"{label}\ncore/ring",
                                     fontsize=13, fontweight="bold",
                                     rotation=0, labelpad=70, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core (weight=1.0)"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Ring (scaled weight)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=10, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} — {atype} — {pct:.2f}% — r={r} — "
            f"Ratio-based soft loss (core=1.0, ring scaled to hit target ratio)",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary plot ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for j, (cr, label) in enumerate(zip(RATIOS, RATIO_LABELS)):
        s = all_stats[cr]

        # Top row: actual ratio achieved
        axes[0, j].hist(s["actual_ratio"], bins=20, alpha=0.7, color="steelblue", edgecolor="white")
        axes[0, j].axvline(x=cr, color="red", linestyle="--", linewidth=2, label=f"Target {cr:.0%}")
        axes[0, j].set_xlabel("Actual core ratio")
        axes[0, j].set_title(f"Target {label}\nmean actual = {np.mean(s['actual_ratio']):.1%}", fontsize=11)
        axes[0, j].legend()
        axes[0, j].set_xlim(0.5, 1.0)

        # Bottom row: ring weight distribution
        all_means = s["ring_w_mean"]
        all_mins = s["ring_w_min"]
        axes[1, j].scatter(s["n_core"], all_means, alpha=0.5, s=30, label="Mean ring w", color="steelblue")
        axes[1, j].scatter(s["n_core"], all_mins, alpha=0.5, s=20, label="Min ring w", color="orange", marker="v")
        axes[1, j].set_xlabel("Core cells (n_core)")
        axes[1, j].set_ylabel("Ring weight")
        axes[1, j].set_title(f"{label}: ring weight vs anomaly size", fontsize=11)
        axes[1, j].legend(fontsize=8)
        axes[1, j].axhline(y=0.01, color="red", linestyle=":", alpha=0.5, label="min_weight floor")

    plt.suptitle(
        "Ratio-based soft loss: actual ratios and ring weights\n"
        "Formula: ring_total = n_core × (1-ratio)/ratio, gaussian-shaped, scaled to total",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY — Ratio-based soft loss mask")
    print(f"{'='*80}")
    for cr, label in zip(RATIOS, RATIO_LABELS):
        s = all_stats[cr]
        print(f"\nTarget {label}:")
        print(f"  Actual ratio: mean={np.mean(s['actual_ratio']):.1%}  "
              f"min={np.min(s['actual_ratio']):.1%}  max={np.max(s['actual_ratio']):.1%}")
        print(f"  Ring weight:  mean={np.mean(s['ring_w_mean']):.4f}  "
              f"min_of_mins={np.min(s['ring_w_min']):.4f}  max_of_means={np.max(s['ring_w_mean']):.4f}")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", default="results/unet/ratio_loss_mask_viz")
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--exclude-sources", nargs="*", default=[])
    parser.add_argument("--min-r", type=int, default=8)
    parser.add_argument("--max-r", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    dr = Path(args.data_root) if args.data_root else None
    if dr and not dr.is_absolute():
        dr = root / dr

    visualize(
        splits_dir=root / args.splits_dir,
        save_dir=root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=dr,
        min_r=args.min_r,
        max_r=args.max_r,
    )
