"""
Visualize CLIP 16x16 patches: full downscale vs adaptive crop.

Compares two strategies for preparing CLIP input:
  Row 0 (Full):  dilation at 512 → 512→224 downscale → 16x16 patches
  Row 1 (Crop):  dilation at 512 → adaptive crop 224 → 16x16 patches

Dilation always happens at 512x512 (original resolution) BEFORE any
downscale or crop, so radius is consistent regardless of strategy.

Shows how many more anomaly patches the crop strategy captures.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset
from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

SEED = 42


def tensor_to_np01(t):
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def downsample_mask_maxpool_16(mask: torch.Tensor) -> torch.Tensor:
    """Downsample mask to 16x16 CLIP patch grid via maxpool."""
    mask_224 = F.interpolate(
        mask.unsqueeze(0) if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0),
        size=(224, 224), mode="nearest"
    )
    if mask_224.dim() == 3:
        mask_224 = mask_224.unsqueeze(0)
    pooled = F.max_pool2d(mask_224, kernel_size=14).squeeze()
    return (pooled > 0.5).float()


def get_square_bbox(mask_np):
    """Get square bbox. Returns (cy, cx, side) or None."""
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    pad = max(1, int(side * 0.1))
    side = side + 2 * pad
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    return cy, cx, side


def adaptive_crop_tensor(image, mask, dilated_mask, target=224):
    """Crop or downscale image+mask+dilated_mask tensors.

    Returns (img_224, mask_224, dilated_224, method, bbox_side).
    """
    mask_np = mask.squeeze(0).numpy()
    bbox = get_square_bbox(mask_np)

    _, h, w = image.shape

    if bbox is not None and bbox[2] <= target:
        cy, cx, side = bbox
        crop_side = target
        y0 = max(0, cy - crop_side // 2)
        x0 = max(0, cx - crop_side // 2)
        if y0 + crop_side > h:
            y0 = max(0, h - crop_side)
        if x0 + crop_side > w:
            x0 = max(0, w - crop_side)
        img_crop = image[:, y0:y0+crop_side, x0:x0+crop_side]
        mask_crop = mask[:, y0:y0+crop_side, x0:x0+crop_side]
        dil_crop = dilated_mask[:, y0:y0+crop_side, x0:x0+crop_side]
        return img_crop, mask_crop, dil_crop, "crop", side
    else:
        img_224 = F.interpolate(
            image.unsqueeze(0), size=(target, target),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        mask_224 = F.interpolate(
            mask.unsqueeze(0).float(), size=(target, target), mode="nearest"
        ).squeeze(0)
        dil_224 = F.interpolate(
            dilated_mask.unsqueeze(0).float(), size=(target, target), mode="nearest"
        ).squeeze(0)
        side = bbox[2] if bbox else 0
        return img_224, mask_224, dil_224, "full", side


def draw_patch_grid(ax, img_np, mask_16, title, img_size=224):
    """Draw image with 16x16 grid, anomaly patches in red, ring in blue."""
    ax.imshow(img_np)
    gs = 16
    cell = img_size / gs
    m = mask_16.numpy() if torch.is_tensor(mask_16) else mask_16

    for r in range(gs):
        for c in range(gs):
            if m[r, c] > 0.5:
                ax.add_patch(plt.Rectangle(
                    (c * cell, r * cell), cell, cell,
                    linewidth=0, facecolor='red', alpha=0.35,
                ))

    for j in range(gs + 1):
        ax.axhline(y=j * cell, color='white', linewidth=0.3, alpha=0.4)
        ax.axvline(x=j * cell, color='white', linewidth=0.3, alpha=0.4)

    n = int(m.sum())
    ax.set_title(f"{title}\n{n}/256 patches", fontsize=10)
    ax.axis("off")


def draw_patch_grid_dual(ax, img_np, core_16, dilated_16, title, img_size=224):
    """Draw image with core patches (red) and ring patches (blue)."""
    ax.imshow(img_np)
    gs = 16
    cell = img_size / gs
    c16 = core_16.numpy() if torch.is_tensor(core_16) else core_16
    d16 = dilated_16.numpy() if torch.is_tensor(dilated_16) else dilated_16
    ring_16 = np.clip(d16 - c16, 0, 1)

    for r in range(gs):
        for c in range(gs):
            if c16[r, c] > 0.5:
                ax.add_patch(plt.Rectangle(
                    (c * cell, r * cell), cell, cell,
                    linewidth=0, facecolor='red', alpha=0.35,
                ))
            elif ring_16[r, c] > 0.5:
                ax.add_patch(plt.Rectangle(
                    (c * cell, r * cell), cell, cell,
                    linewidth=0, facecolor='blue', alpha=0.3,
                ))

    for j in range(gs + 1):
        ax.axhline(y=j * cell, color='white', linewidth=0.3, alpha=0.4)
        ax.axvline(x=j * cell, color='white', linewidth=0.3, alpha=0.4)

    n_core = int(c16.sum())
    n_ring = int(ring_16.sum())
    ax.set_title(f"{title}\n{n_core}c+{n_ring}r = {n_core+n_ring}/256", fontsize=10)
    ax.axis("off")


def show_masked_only(ax, img_np, mask_16, title, img_size=224):
    """Show only anomaly patches, rest blacked out."""
    gs = 16
    cell = img_size / gs
    m = mask_16.numpy() if torch.is_tensor(mask_16) else mask_16

    blocked = np.zeros((img_size, img_size), dtype=np.float32)
    for r in range(gs):
        for c in range(gs):
            if m[r, c] > 0.5:
                y0, y1 = int(r * cell), int(min((r + 1) * cell, img_size))
                x0, x1 = int(c * cell), int(min((c + 1) * cell, img_size))
                blocked[y0:y1, x0:x1] = 1.0

    ax.imshow(img_np * blocked[:, :, None])
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def visualize(
    splits_dir: Path, save_dir: Path, n_samples: int = 50,
    exclude_sources: list = None, data_root: Path = None,
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

    stats_full = {"core": [], "dilated": []}
    stats_crop = {"core": [], "dilated": []}
    methods = []

    print(f"Processing {n_samples} samples...")
    print("Comparing: Full downscale vs Adaptive crop at CLIP 16x16 patches")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]
        atype = sample["anomaly_type"]
        pct = mask.sum().item() / mask.numel() * 100

        # === Dilate at 512x512 (original resolution) FIRST ===
        r_512 = adaptive_dilation_radius(mask, min_r=2, max_r=10)
        dilated_512 = dilate_mask(mask, r_512)

        # === Full downscale path ===
        img_full = F.interpolate(
            image.unsqueeze(0), size=(224, 224),
            mode="bilinear", align_corners=False
        ).squeeze(0)
        mask_full = F.interpolate(
            mask.unsqueeze(0).float(), size=(224, 224), mode="nearest"
        ).squeeze(0)
        dilated_full = F.interpolate(
            dilated_512.unsqueeze(0).float(), size=(224, 224), mode="nearest"
        ).squeeze(0)
        full_np = tensor_to_np01(img_full)

        # Core and dilated patches from pre-dilated masks
        core_16_full = downsample_mask_maxpool_16(mask_full)
        dil_16_full = downsample_mask_maxpool_16(dilated_full)

        # === Adaptive crop path (dilation already done at 512) ===
        img_crop, mask_crop, dilated_crop, method, bbox_side = adaptive_crop_tensor(
            image, mask, dilated_512, target=224
        )
        crop_np = tensor_to_np01(img_crop)
        methods.append(method)

        core_16_crop = downsample_mask_maxpool_16(mask_crop)
        dil_16_crop = downsample_mask_maxpool_16(dilated_crop)

        # Track stats
        stats_full["core"].append(int(core_16_full.sum()))
        stats_full["dilated"].append(int(dil_16_full.sum()))
        stats_crop["core"].append(int(core_16_crop.sum()))
        stats_crop["dilated"].append(int(dil_16_crop.sum()))

        # === Figure: 2 rows x 4 cols ===
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))

        # Row 0: Full downscale — with core+ring overlay
        axes[0, 0].imshow(full_np)
        # Overlay core (red) and ring (blue) from pixel-level masks
        mask_full_np = mask_full.squeeze(0).numpy()
        dil_full_np = dilated_full.squeeze(0).numpy()
        ring_full_np = np.clip(dil_full_np - mask_full_np, 0, 1)
        h_f, w_f = mask_full_np.shape
        ov_c = np.zeros((h_f, w_f, 4), dtype=np.float32)
        ov_c[:, :, 0] = 1.0
        ov_c[:, :, 3] = (mask_full_np > 0.5).astype(np.float32) * 0.45
        axes[0, 0].imshow(ov_c)
        ov_r = np.zeros((h_f, w_f, 4), dtype=np.float32)
        ov_r[:, :, 2] = 1.0
        ov_r[:, :, 3] = (ring_full_np > 0.5).astype(np.float32) * 0.35
        axes[0, 0].imshow(ov_r)
        axes[0, 0].set_title(f"Full 512\u2192224", fontsize=10)
        axes[0, 0].axis("off")

        draw_patch_grid(axes[0, 1], full_np, core_16_full, "Core patches (no dil)")

        draw_patch_grid_dual(axes[0, 2], full_np, core_16_full, dil_16_full,
                             f"Dilated r={r_512}")

        show_masked_only(axes[0, 3], full_np, dil_16_full,
                         f"Masked CLIP tokens ({int(dil_16_full.sum())}/256)")

        # Row 1: Adaptive crop — with core+ring overlay
        axes[1, 0].imshow(crop_np)
        mask_crop_np = mask_crop.squeeze(0).numpy()
        dil_crop_np = dilated_crop.squeeze(0).numpy()
        ring_crop_np = np.clip(dil_crop_np - mask_crop_np, 0, 1)
        h_c, w_c = mask_crop_np.shape
        ov_c2 = np.zeros((h_c, w_c, 4), dtype=np.float32)
        ov_c2[:, :, 0] = 1.0
        ov_c2[:, :, 3] = (mask_crop_np > 0.5).astype(np.float32) * 0.45
        axes[1, 0].imshow(ov_c2)
        ov_r2 = np.zeros((h_c, w_c, 4), dtype=np.float32)
        ov_r2[:, :, 2] = 1.0
        ov_r2[:, :, 3] = (ring_crop_np > 0.5).astype(np.float32) * 0.35
        axes[1, 0].imshow(ov_r2)
        border_color = "lime" if method == "crop" else "orange"
        for spine in axes[1, 0].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
            spine.set_visible(True)
        axes[1, 0].set_title(f"{method.upper()} (bbox={bbox_side}px)", fontsize=10)
        axes[1, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        draw_patch_grid(axes[1, 1], crop_np, core_16_crop, "Core patches (no dil)")

        draw_patch_grid_dual(axes[1, 2], crop_np, core_16_crop, dil_16_crop,
                             f"Dilated r={r_512}")

        show_masked_only(axes[1, 3], crop_np, dil_16_crop,
                         f"Masked CLIP tokens ({int(dil_16_crop.sum())}/256)")

        # Row labels
        axes[0, 0].set_ylabel("Full\ndownscale", fontsize=12, fontweight="bold",
                               rotation=0, labelpad=60, va="center")
        axes[1, 0].set_ylabel("Adaptive\ncrop", fontsize=12, fontweight="bold",
                               rotation=0, labelpad=60, va="center")

        legend_handles = [
            mpatches.Patch(color="red", alpha=0.35, label="Core patches"),
            mpatches.Patch(color="blue", alpha=0.3, label="Dilation ring patches"),
            mpatches.Patch(facecolor="none", edgecolor="lime", linewidth=2,
                           linestyle="--", label="Cropped (bbox \u2264 224)"),
            mpatches.Patch(facecolor="none", edgecolor="orange", linewidth=2,
                           linestyle="--", label="Full downscale (bbox > 224)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=9, frameon=True, borderpad=0.5)

        gain_core = int(core_16_crop.sum()) - int(core_16_full.sum())
        gain_dil = int(dil_16_crop.sum()) - int(dil_16_full.sum())
        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {pct:.2f}% coverage\n"
            f"Adaptive dilation r={r_512} (r=clamp(0.5\u00b7\u221aarea, 2, 10)) \u2022 "
            f"Patch gain: core +{gain_core}, dilated +{gain_dil}",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0.06, 0.05, 1, 0.92])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary ===
    fc = np.array(stats_full["core"])
    fd = np.array(stats_full["dilated"])
    cc = np.array(stats_crop["core"])
    cd = np.array(stats_crop["dilated"])
    n_crop = sum(1 for m in methods if m == "crop")

    print(f"\n{'='*70}")
    print(f"SUMMARY — Full vs Adaptive Crop at CLIP 16x16")
    print(f"{'='*70}")
    print(f"Samples: {n_samples} ({n_crop} cropped, {n_samples - n_crop} full)")
    print(f"\n{'Strategy':<20} {'Core mean':>10} {'Core med':>10} {'Dil mean':>10} {'Dil med':>10}")
    print("-" * 62)
    print(f"{'Full downscale':<20} {fc.mean():>10.1f} {np.median(fc):>10.0f} {fd.mean():>10.1f} {np.median(fd):>10.0f}")
    print(f"{'Adaptive crop':<20} {cc.mean():>10.1f} {np.median(cc):>10.0f} {cd.mean():>10.1f} {np.median(cd):>10.0f}")
    print(f"\nPatch gain (crop - full):")
    print(f"  Core:    mean +{(cc-fc).mean():.1f}  median +{np.median(cc-fc):.0f}  max +{(cc-fc).max()}")
    print(f"  Dilated: mean +{(cd-fd).mean():.1f}  median +{np.median(cd-fd):.0f}  max +{(cd-fd).max()}")

    # Summary plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(fc, bins=25, alpha=0.5, label="Full (core)", color="red")
    axes[0].hist(cc, bins=25, alpha=0.5, label="Crop (core)", color="green")
    axes[0].set_xlabel("Core patches")
    axes[0].set_title("Core patch count")
    axes[0].legend()

    axes[1].hist(fd, bins=25, alpha=0.5, label="Full (dilated)", color="red")
    axes[1].hist(cd, bins=25, alpha=0.5, label="Crop (dilated)", color="green")
    axes[1].set_xlabel("Dilated patches")
    axes[1].set_title("Dilated patch count")
    axes[1].legend()

    gain = cd - fd
    axes[2].hist(gain, bins=25, alpha=0.6, color="steelblue")
    axes[2].axvline(x=0, color="red", linestyle="--")
    axes[2].set_xlabel("Patch gain (crop - full)")
    axes[2].set_title(f"Dilated patch gain\nmean={gain.mean():.1f}, median={np.median(gain):.0f}")

    plt.suptitle(f"Full vs Adaptive Crop — CLIP 16x16 patches (N={n_samples})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", default="results/clip/clip_mask_viz_adaptive")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--exclude-sources", nargs="*", default=[])
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
    )
