"""
Visualize UNet inpainting-channel masks with adaptive dilation.

Compares raw maxpool vs dilated-then-maxpool at each UNet spatial resolution.
Uses the same adaptive dilation formula as the CLIP mask:
    r = max(min_r, min(max_r, 0.5 * sqrt(area)))

Layout per sample (3 rows x 5 cols):
  Row 0: Original image with mask overlays (red=original, blue=dilation ring)
  Row 1: Maxpool only (no dilation) at each UNet resolution — red cells
  Row 2: Dilated + maxpool — red=original cells, blue=dilation-only cells

This helps decide the right dilation strategy for the UNet inpainting channel:
we want enough dilation for smooth surface integration, but not so much that
the anomaly region dominates the image.
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

SEED = 42

# UNet spatial resolutions for SD 1.5 (512x512 input → 64x64 latent → downsampled)
UNET_RESOLUTIONS = [64, 32, 16, 8]
UNET_LABELS = [
    "64x64\ndown_blocks.0\n(320-ch)",
    "32x32\ndown_blocks.1\n(640-ch)",
    "16x16\ndown_blocks.2\n(1280-ch)",
    "8x8\nmid_block\n(1280-ch)",
]


def tensor_to_np01(t: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] tensor [C,H,W] to numpy [H,W,3] in [0,1]."""
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological dilation of binary mask [1,H,W] by radius pixels."""
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    padded = F.pad(mask.unsqueeze(0), [radius] * 4, mode="constant", value=0)
    dilated = F.max_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0)
    return (dilated > 0.5).float()


def adaptive_dilation_radius(mask: torch.Tensor, min_r: int = 2, max_r: int = 10) -> int:
    """Compute adaptive dilation radius: max(min_r, min(max_r, 0.5 * sqrt(area))).

    Args:
        mask: Binary mask [1, H, W]
        min_r: Minimum dilation radius in pixels
        max_r: Maximum dilation radius in pixels

    Returns:
        Dilation radius in pixels at mask resolution
    """
    area = mask.sum().item()
    if area == 0:
        return 0
    r = 0.5 * math.sqrt(area)
    return max(min_r, min(max_r, int(r)))


def downsample_mask_maxpool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample binary mask [1,H,W] to [1,size,size] via max pooling."""
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


def overlay_mask_on_image(ax, img_np: np.ndarray, mask_np: np.ndarray, title: str):
    """Show image with red mask overlay."""
    ax.imshow(img_np)
    h, w = img_np.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    mask_2d = mask_np.squeeze()
    if mask_2d.shape != (h, w):
        mask_2d = np.array(
            Image.fromarray((mask_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        ) / 255.0
    overlay[:, :, 0] = 1.0
    overlay[:, :, 3] = mask_2d * 0.4
    ax.imshow(overlay)
    pct = mask_2d.sum() / mask_2d.size * 100
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def overlay_dual_mask_on_image(ax, img_np: np.ndarray, orig_mask_np: np.ndarray,
                                dilated_mask_np: np.ndarray, title: str):
    """Show image with original mask (red) and dilation ring (blue)."""
    ax.imshow(img_np)
    h, w = img_np.shape[:2]
    orig_2d = orig_mask_np.squeeze()
    dil_2d = dilated_mask_np.squeeze()
    if orig_2d.shape != (h, w):
        orig_2d = np.array(Image.fromarray((orig_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    if dil_2d.shape != (h, w):
        dil_2d = np.array(Image.fromarray((dil_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    # Dilation ring = dilated minus original
    ring = np.clip(dil_2d - orig_2d, 0, 1)
    # Blue overlay for dilation ring
    overlay_ring = np.zeros((h, w, 4), dtype=np.float32)
    overlay_ring[:, :, 2] = 1.0
    overlay_ring[:, :, 3] = ring * 0.45
    ax.imshow(overlay_ring)
    # Red overlay for original mask
    overlay_orig = np.zeros((h, w, 4), dtype=np.float32)
    overlay_orig[:, :, 0] = 1.0
    overlay_orig[:, :, 3] = orig_2d * 0.45
    ax.imshow(overlay_orig)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def draw_mask_grid_plain(ax, mask_2d: np.ndarray, title: str, grid_size: int):
    """Draw mask as a red/gray grid (no dilation distinction)."""
    gs = grid_size
    img = np.zeros((gs, gs, 3), dtype=np.float32)
    # Red for anomaly
    img[:, :, 0] = mask_2d * 0.9
    img[:, :, 1] = mask_2d * 0.2
    img[:, :, 2] = mask_2d * 0.2
    # Dark gray for background
    bg = (1 - mask_2d)
    img[:, :, 0] += bg * 0.15
    img[:, :, 1] += bg * 0.15
    img[:, :, 2] += bg * 0.15

    ax.imshow(img, interpolation="nearest")
    n_anomaly = int(mask_2d.sum())
    total = gs * gs
    pct = n_anomaly / total * 100
    ax.set_title(f"{title}\n{n_anomaly}/{total} ({pct:.1f}%)", fontsize=10)
    ax.axis("off")


def draw_mask_grid_dual(ax, orig_mask_2d: np.ndarray, dilated_mask_2d: np.ndarray,
                         title: str, grid_size: int):
    """Draw grid with red=original, blue=dilation-only, dark gray=background."""
    gs = grid_size
    img = np.zeros((gs, gs, 3), dtype=np.float32)

    ring = np.clip(dilated_mask_2d - orig_mask_2d, 0, 1)

    # Red for original anomaly cells
    img[:, :, 0] += orig_mask_2d * 0.9
    img[:, :, 1] += orig_mask_2d * 0.15
    img[:, :, 2] += orig_mask_2d * 0.15

    # Blue for dilation-only cells
    img[:, :, 0] += ring * 0.15
    img[:, :, 1] += ring * 0.3
    img[:, :, 2] += ring * 0.9

    # Dark gray for background
    bg = np.clip(1 - dilated_mask_2d, 0, 1)
    img[:, :, 0] += bg * 0.15
    img[:, :, 1] += bg * 0.15
    img[:, :, 2] += bg * 0.15

    ax.imshow(img, interpolation="nearest")

    n_orig = int(orig_mask_2d.sum())
    n_dil = int(dilated_mask_2d.sum())
    n_ring = n_dil - n_orig
    total = gs * gs
    pct = n_dil / total * 100
    ax.set_title(f"{title}\n{n_orig}+{n_ring}={n_dil}/{total} ({pct:.1f}%)", fontsize=10)
    ax.axis("off")


def visualize_unet_masks_dilated(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 200,
    exclude_sources: list = None,
    data_root: Path = None,
    min_r: int = 2,
    max_r: int = 10,
):
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = AnomalyDataset(
        splits_dir,
        image_size=512,
        exclude_sources=exclude_sources,
        data_root=data_root,
        return_reference=False,
    )

    if len(dataset) == 0:
        print("No data loaded!")
        return

    n_samples = min(n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)

    # Stats tracking
    stats = {res: {"maxpool": [], "dilated": [], "ring": []} for res in UNET_RESOLUTIONS}
    orig_pcts = []
    radii = []

    print(f"Processing {n_samples} samples...")
    print(f"Adaptive dilation: r = max({min_r}, min({max_r}, 0.5*sqrt(area)))")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]    # [3, 512, 512] in [-1, 1]
        mask = sample["mask"]      # [1, 512, 512] binary
        atype = sample["anomaly_type"]

        orig_pct = mask.sum().item() / mask.numel() * 100
        orig_pcts.append(orig_pct)

        area = int(mask.sum().item())
        r = adaptive_dilation_radius(mask, min_r=min_r, max_r=max_r)
        radii.append(r)
        dilated = dilate_mask(mask, r)

        img_np = tensor_to_np01(image)
        mask_np = mask.squeeze(0).numpy()
        dilated_np = dilated.squeeze(0).numpy()

        # === 3 rows x 5 cols ===
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # Row 0, Col 0: Original image + dual mask overlay
        overlay_dual_mask_on_image(
            axes[0, 0], img_np, mask_np, dilated_np,
            f"{atype}\n{orig_pct:.2f}% → r={r} (area={area})"
        )

        # Row 1, Col 0 / Row 2, Col 0: labels
        axes[1, 0].text(0.5, 0.5, "Maxpool\n(no dilation)",
                        transform=axes[1, 0].transAxes,
                        ha="center", va="center", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, f"Dilated (r={r})\n+ Maxpool",
                        transform=axes[2, 0].transAxes,
                        ha="center", va="center", fontsize=12, fontweight="bold")
        axes[2, 0].axis("off")

        for j, (res, label) in enumerate(zip(UNET_RESOLUTIONS, UNET_LABELS)):
            col = j + 1

            # Maxpool only (no dilation)
            mask_mp = downsample_mask_maxpool(mask, res)
            mask_mp_np = mask_mp.squeeze(0).numpy()
            n_mp = int(mask_mp_np.sum())
            stats[res]["maxpool"].append(n_mp)

            # Dilated + maxpool
            mask_dil_mp = downsample_mask_maxpool(dilated, res)
            mask_dil_mp_np = mask_dil_mp.squeeze(0).numpy()
            n_dil = int(mask_dil_mp_np.sum())
            n_ring = n_dil - n_mp
            stats[res]["dilated"].append(n_dil)
            stats[res]["ring"].append(n_ring)

            # Row 0: dilated mask upscaled back to 512 overlaid on image
            mask_upscaled = F.interpolate(
                mask_dil_mp.unsqueeze(0), size=(512, 512), mode="nearest"
            ).squeeze(0).squeeze(0).numpy()
            orig_upscaled = F.interpolate(
                mask_mp.unsqueeze(0), size=(512, 512), mode="nearest"
            ).squeeze(0).squeeze(0).numpy()
            overlay_dual_mask_on_image(
                axes[0, col], img_np, orig_upscaled, mask_upscaled,
                f"Dilated→{res}x{res}→upscaled"
            )

            # Row 1: maxpool grid (red only)
            draw_mask_grid_plain(axes[1, col], mask_mp_np, f"Maxpool {label}", res)

            # Row 2: dilated grid (red=original, blue=dilation ring)
            draw_mask_grid_dual(axes[2, col], mask_mp_np, mask_dil_mp_np,
                               f"Dilated {label}", res)

        # Legend
        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Original anomaly"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring"),
            mpatches.Patch(color=(0.15, 0.15, 0.15), label="Background"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {orig_pct:.2f}% coverage \u2014 "
            f"dilation r={r} (formula: max({min_r}, min({max_r}, \u00bd\u221aarea)))",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary statistics ===
    print(f"\n{'='*70}")
    print(f"SUMMARY — UNet Mask Dilation (min_r={min_r}, max_r={max_r})")
    print(f"{'='*70}")
    print(f"Original coverage: mean {np.mean(orig_pcts):.2f}%  median {np.median(orig_pcts):.2f}%")
    print(f"Dilation radius: mean {np.mean(radii):.1f}  median {np.median(radii):.0f}  "
          f"min {np.min(radii)}  max {np.max(radii)}\n")

    print(f"{'Resolution':<12} {'Method':<12} {'Mean':>8} {'Median':>8} {'Lost':>10} {'<=1':>8} {'<=2':>8}")
    print("-" * 72)
    for res in UNET_RESOLUTIONS:
        for method in ["maxpool", "dilated"]:
            arr = np.array(stats[res][method])
            lost = int((arr == 0).sum())
            le1 = int((arr <= 1).sum())
            le2 = int((arr <= 2).sum())
            print(f"{res}x{res:<8} {method:<12} {arr.mean():>8.1f} {np.median(arr):>8.1f} "
                  f"{lost:>10}/{n_samples} {le1:>8} {le2:>8}")

        ring = np.array(stats[res]["ring"])
        print(f"{'':12} {'ring gain':<12} {ring.mean():>8.1f} {np.median(ring):>8.1f}")
        print()

    # === Summary plot ===
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for j, res in enumerate(UNET_RESOLUTIONS):
        # Top row: cell count distributions
        mp = np.array(stats[res]["maxpool"])
        dil = np.array(stats[res]["dilated"])
        max_val = max(mp.max(), dil.max()) if len(mp) > 0 else 30
        bins = max(1, min(30, int(max_val)))
        axes[0, j].hist(mp, bins=bins, alpha=0.5, color="red", label="Maxpool only")
        axes[0, j].hist(dil, bins=bins, alpha=0.5, color="blue", label="Dilated+maxpool")
        axes[0, j].set_title(f"{res}x{res}\nmean: {mp.mean():.1f} → {dil.mean():.1f}", fontsize=10)
        axes[0, j].set_xlabel("Anomaly cells")
        axes[0, j].legend(fontsize=7)

        # Bottom row: ring gain distribution
        ring = np.array(stats[res]["ring"])
        axes[1, j].hist(ring, bins=max(1, min(30, int(ring.max()) if ring.max() > 0 else 10)),
                        alpha=0.6, color="steelblue")
        axes[1, j].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1, j].set_title(f"Ring gain at {res}x{res}\nmean +{ring.mean():.1f}", fontsize=10)
        axes[1, j].set_xlabel("Extra cells from dilation")

    plt.suptitle(
        f"UNet Mask Dilation Summary — r = max({min_r}, min({max_r}, \u00bd\u221aarea)) — N={n_samples}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Radius distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(radii, bins=max(1, max_r - min_r + 1), alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Dilation radius (pixels)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Radius distribution\nmin_r={min_r}, max_r={max_r}")

    axes[1].scatter(orig_pcts, radii, alpha=0.3, s=12, color="steelblue")
    axes[1].set_xlabel("Original coverage (%)")
    axes[1].set_ylabel("Dilation radius")
    axes[1].set_title("Coverage vs radius")

    plt.tight_layout()
    plt.savefig(save_dir / "radius_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize UNet inpainting masks with adaptive dilation"
    )
    parser.add_argument("--splits-dir", type=str,
                        default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str,
                        default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/unet_mask_viz_dilated")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    parser.add_argument("--min-r", type=int, default=2,
                        help="Minimum dilation radius (default: 2)")
    parser.add_argument("--max-r", type=int, default=10,
                        help="Maximum dilation radius (default: 10)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_unet_masks_dilated(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        min_r=args.min_r,
        max_r=args.max_r,
    )
