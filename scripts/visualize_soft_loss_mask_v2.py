"""
Visualize soft loss mask v2 — clearer dilation visualization + minimum weight floor.

Per sample (1 row x 6 cols):
  Col 0: Original image + dual overlay (red=core, blue=ring) at 512x512
  Col 1: Dual-color grid at 64x64 (red=core, blue=ring, gray=bg)
  Col 2: Soft loss mask — linear
  Col 3: Soft loss mask — cosine
  Col 4: Soft loss mask — flat ring (e.g. 0.3)
  Col 5: Zoomed crop of the linear soft mask
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


def compute_soft_mask(core_64, dilated_64, decay="linear", min_weight=0.0):
    """Compute soft weight mask. Core=1.0, ring decays, bg=0.0.

    Args:
        min_weight: Minimum weight for any ring pixel (prevents zero-weight in ring)
    """
    ring = np.clip(dilated_64 - core_64, 0, 1)

    if core_64.sum() == 0:
        return dilated_64.copy()

    if decay == "flat":
        # Flat weight for entire ring
        soft = np.zeros_like(core_64)
        soft[core_64 > 0.5] = 1.0
        soft[ring > 0.5] = min_weight if min_weight > 0 else 0.3
        return soft

    # Distance from core boundary outward
    dist_from_core = ndimage.distance_transform_edt(1 - core_64)

    # Max distance within ring
    ring_distances = dist_from_core * ring
    max_ring_dist = ring_distances.max()
    if max_ring_dist == 0:
        max_ring_dist = 1.0

    norm_dist = np.clip(dist_from_core / max_ring_dist, 0, 1)

    if decay == "linear":
        ring_weight = 1.0 - norm_dist
    elif decay == "cosine":
        ring_weight = 0.5 * (1 + np.cos(np.pi * norm_dist))
    else:
        ring_weight = 1.0 - norm_dist

    # Apply minimum weight floor
    ring_weight = np.maximum(ring_weight, min_weight)

    soft = np.zeros_like(core_64)
    soft[core_64 > 0.5] = 1.0
    soft[ring > 0.5] = ring_weight[ring > 0.5]

    return soft


def draw_dual_grid(ax, core, dilated, title, grid_size=64):
    """Red=core, blue=ring, dark gray=bg at grid resolution."""
    ring = np.clip(dilated - core, 0, 1)
    bg = np.clip(1 - dilated, 0, 1)

    img = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    # Red for core
    img[:, :, 0] += core * 0.9
    img[:, :, 1] += core * 0.15
    img[:, :, 2] += core * 0.15
    # Blue for ring
    img[:, :, 0] += ring * 0.15
    img[:, :, 1] += ring * 0.3
    img[:, :, 2] += ring * 0.9
    # Dark gray for bg
    img[:, :, 0] += bg * 0.12
    img[:, :, 1] += bg * 0.12
    img[:, :, 2] += bg * 0.12

    ax.imshow(img, interpolation="nearest")
    n_core = int(core.sum())
    n_ring = int(ring.sum())
    ax.set_title(f"{title}\n{n_core} core + {n_ring} ring = {n_core + n_ring} cells", fontsize=9)
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
    # Blue ring
    ov_ring = np.zeros((h, w, 4), dtype=np.float32)
    ov_ring[:, :, 2] = 1.0
    ov_ring[:, :, 3] = ring * 0.5
    ax.imshow(ov_ring)
    # Red core
    ov_core = np.zeros((h, w, 4), dtype=np.float32)
    ov_core[:, :, 0] = 1.0
    ov_core[:, :, 3] = orig_2d * 0.5
    ax.imshow(ov_core)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_soft(ax, soft, title, show_cb=True):
    """Draw soft mask with perceptual colormap."""
    cmap = LinearSegmentedColormap.from_list("w", [
        (0.0, (0.10, 0.10, 0.10)),
        (0.01, (0.10, 0.10, 0.10)),
        (0.02, (0.12, 0.20, 0.55)),
        (0.3, (0.70, 0.25, 0.10)),
        (0.7, (0.95, 0.70, 0.10)),
        (1.0, (1.0, 1.0, 0.85)),
    ])
    im = ax.imshow(soft, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    # Annotate min/max weight in ring
    ring_vals = soft[(soft > 0) & (soft < 1.0)]
    if len(ring_vals) > 0:
        rmin, rmax = ring_vals.min(), ring_vals.max()
        ax.set_title(f"{title}\nring: {rmin:.2f}–{rmax:.2f}", fontsize=9)
    else:
        ax.set_title(title, fontsize=9)
    ax.axis("off")
    if show_cb:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


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

        # Compute soft masks
        soft_linear = compute_soft_mask(core_64, dil_64, "linear", min_weight=0.1)
        soft_cosine = compute_soft_mask(core_64, dil_64, "cosine", min_weight=0.1)
        soft_flat = compute_soft_mask(core_64, dil_64, "flat", min_weight=0.3)

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

        fig, axes = plt.subplots(2, 6, figsize=(26, 8),
                                  gridspec_kw={"height_ratios": [1, 1]})

        # === Top row: full 64x64 view ===
        overlay_dual_mask(axes[0, 0], img_np, mask_np, dilated_np,
                          f"{atype} — {pct:.2f}%\nr={r} (area={area})")

        draw_dual_grid(axes[0, 1], core_64, dil_64, "64x64 (red=core, blue=ring)")

        draw_soft(axes[0, 2], soft_linear, "Linear (min=0.1)")
        draw_soft(axes[0, 3], soft_cosine, "Cosine (min=0.1)")
        draw_soft(axes[0, 4], soft_flat, "Flat ring (0.3)")

        # Col 5: dual grid upscaled onto image
        dil_up = F.interpolate(
            torch.from_numpy(dil_64).unsqueeze(0).unsqueeze(0).float(),
            size=(512, 512), mode="nearest"
        ).squeeze().numpy()
        core_up = F.interpolate(
            torch.from_numpy(core_64).unsqueeze(0).unsqueeze(0).float(),
            size=(512, 512), mode="nearest"
        ).squeeze().numpy()
        overlay_dual_mask(axes[0, 5], img_np, core_up, dil_up,
                          "64x64 grid → upscaled\n(red=core, blue=ring)")

        # === Bottom row: zoomed crops ===
        # Zoom of dual grid
        crop_core = core_64[y0:y1, x0:x1]
        crop_dil = dil_64[y0:y1, x0:x1]
        crop_size = crop_core.shape[0]

        draw_dual_grid(axes[1, 0], crop_core, crop_dil,
                       f"Zoomed dual grid\n({crop_size}x{crop_size} region)", crop_size)

        # Annotate cell values on zoomed dual grid
        # (skip, too small to read)

        draw_dual_grid(axes[1, 1], crop_core, crop_dil,
                       "Zoomed (red=core, blue=ring)", crop_size)

        draw_soft(axes[1, 2], soft_linear[y0:y1, x0:x1], "Linear zoomed")
        draw_soft(axes[1, 3], soft_cosine[y0:y1, x0:x1], "Cosine zoomed")
        draw_soft(axes[1, 4], soft_flat[y0:y1, x0:x1], "Flat zoomed")

        # Annotated zoom: show actual weight values per cell
        crop_lin = soft_linear[y0:y1, x0:x1]
        im = axes[1, 5].imshow(crop_lin, interpolation="nearest",
                                cmap="YlOrRd", vmin=0, vmax=1)
        # Annotate each cell with its weight value
        for cy_i in range(crop_lin.shape[0]):
            for cx_i in range(crop_lin.shape[1]):
                val = crop_lin[cy_i, cx_i]
                if val > 0.01:
                    color = "black" if val > 0.5 else "white"
                    axes[1, 5].text(cx_i, cy_i, f"{val:.2f}",
                                     ha="center", va="center",
                                     fontsize=6, color=color, fontweight="bold")
        axes[1, 5].set_title("Linear weights (annotated)", fontsize=9)
        axes[1, 5].axis("off")
        plt.colorbar(im, ax=axes[1, 5], fraction=0.046, pad=0.04)

        # Legend
        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core (weight=1.0)"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring (decaying weight)"),
            mpatches.Patch(color=(0.12, 0.12, 0.12), label="Background (weight=0.0)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=10, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} — {atype} — Soft Loss Mask (min_r={min_r}, max_r={max_r}, r={r})",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=130, bbox_inches="tight")
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # Decay profile comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 1, 200)
    ax.plot(x, np.maximum(1.0 - x, 0.1), label="Linear (min=0.1)", linewidth=2.5)
    ax.plot(x, np.maximum(0.5 * (1 + np.cos(np.pi * x)), 0.1), label="Cosine (min=0.1)", linewidth=2.5)
    ax.axhline(y=0.3, color="green", linestyle="--", linewidth=2, label="Flat (0.3)")
    ax.set_xlabel("Normalized distance from core (0=core edge, 1=ring edge)", fontsize=12)
    ax.set_ylabel("Loss weight", fontsize=12)
    ax.set_title("Decay profiles (with minimum weight floor)", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(save_dir / "decay_profiles.png", dpi=150)
    plt.close()

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", default="results/soft_loss_mask_v2")
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
