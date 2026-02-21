"""
Visualize how the anomaly mask looks at UNet spatial resolutions.

SD 1.5 UNet processes latents at 4 spatial scales (for 512x512 input):
  64x64 (down_blocks.0) → 32x32 (down_blocks.1) → 16x16 (down_blocks.2) → 8x8 (mid_block)

This script shows:
  Row 0: Original image + mask at 512x512
  Row 1: Mask at each UNet resolution via nearest-neighbor interpolation
  Row 2: Mask at each UNet resolution via max-pool (any anomaly pixel marks the cell)

Helps assess at which UNet layers the anomaly is still spatially resolved
vs. reduced to a single pixel or lost entirely.
"""
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


def downsample_mask_nearest(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample binary mask [1,H,W] to [1,size,size] via nearest interpolation."""
    return (F.interpolate(
        mask.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0) > 0.5).float()


def downsample_mask_maxpool(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Downsample binary mask [1,H,W] to [1,size,size] via max pooling.

    Each cell is 1 if ANY pixel in the corresponding region is anomalous.
    """
    _, h, w = mask.shape
    kernel_h = h // size
    kernel_w = w // size
    # Resize to exact multiple first
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


def draw_mask_grid(ax, mask_2d: np.ndarray, title: str, grid_size: int):
    """Draw a mask as a colored grid with cell borders."""
    gs = grid_size
    # Create RGB image: green=anomaly, dark gray=background
    img = np.zeros((gs, gs, 3), dtype=np.float32)
    img[:, :, 0] = mask_2d * 0.9   # red channel for anomaly
    img[:, :, 1] = mask_2d * 0.2
    img[:, :, 2] = mask_2d * 0.2
    # Background gets dark gray
    bg = (1 - mask_2d)
    img[:, :, 0] += bg * 0.15
    img[:, :, 1] += bg * 0.15
    img[:, :, 2] += bg * 0.15

    ax.imshow(img, interpolation="nearest")

    n_anomaly = int(mask_2d.sum())
    total = gs * gs
    pct = n_anomaly / total * 100
    ax.set_title(f"{title}\n{n_anomaly}/{total} cells ({pct:.1f}%)", fontsize=11)
    ax.axis("off")


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
    ax.set_title(f"{title}\n{pct:.2f}% coverage", fontsize=11)
    ax.axis("off")


def visualize_unet_masks(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 200,
    exclude_sources: list = None,
    data_root: Path = None,
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
    stats = {res: {"nearest": [], "maxpool": []} for res in UNET_RESOLUTIONS}
    orig_pcts = []
    types = []

    print(f"Processing {n_samples} samples...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]    # [3, 512, 512] in [-1, 1]
        mask = sample["mask"]      # [1, 512, 512] binary
        atype = sample["anomaly_type"]

        orig_pct = mask.sum().item() / mask.numel() * 100
        orig_pcts.append(orig_pct)
        types.append(atype)

        img_np = tensor_to_np01(image)
        mask_np = mask.squeeze(0).numpy()

        # === 3 rows x 5 cols figure ===
        # Row 0: original image + mask at each UNet res (upscaled for visual)
        # Row 1: nearest-neighbor downsampled masks as grids
        # Row 2: max-pool downsampled masks as grids
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        # Row 0, Col 0: Original image + mask
        overlay_mask_on_image(axes[0, 0], img_np, mask_np, f"Original 512x512\n{atype}")

        # Row 1, Col 0 / Row 2, Col 0: labels
        axes[1, 0].text(0.5, 0.5, "Nearest\nInterpolation", transform=axes[1, 0].transAxes,
                        ha="center", va="center", fontsize=12, fontweight="bold")
        axes[1, 0].axis("off")
        axes[2, 0].text(0.5, 0.5, "Max Pool\n(any pixel)", transform=axes[2, 0].transAxes,
                        ha="center", va="center", fontsize=12, fontweight="bold")
        axes[2, 0].axis("off")

        for j, (res, label) in enumerate(zip(UNET_RESOLUTIONS, UNET_LABELS)):
            col = j + 1

            # Nearest-neighbor downsample
            mask_nearest = downsample_mask_nearest(mask, res)
            mask_nearest_np = mask_nearest.squeeze(0).numpy()
            n_nearest = int(mask_nearest_np.sum())
            stats[res]["nearest"].append(n_nearest)

            # Max-pool downsample
            mask_maxpool = downsample_mask_maxpool(mask, res)
            mask_maxpool_np = mask_maxpool.squeeze(0).numpy()
            n_maxpool = int(mask_maxpool_np.sum())
            stats[res]["maxpool"].append(n_maxpool)

            # Row 0: mask upscaled back to 512 overlaid on image (to show spatial coverage)
            mask_upscaled = F.interpolate(
                mask_maxpool.unsqueeze(0), size=(512, 512), mode="nearest"
            ).squeeze(0).squeeze(0).numpy()
            overlay_mask_on_image(axes[0, col], img_np, mask_upscaled, f"Maxpool → {res}x{res}\n→ upscaled")

            # Row 1: nearest grid
            draw_mask_grid(axes[1, col], mask_nearest_np, f"Nearest {label}", res)

            # Row 2: maxpool grid
            draw_mask_grid(axes[2, col], mask_maxpool_np, f"Maxpool {label}", res)

        # Legend
        legend_handles = [mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.65, label="Anomaly mask")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=9,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(f"Sample {i} \u2014 {atype} \u2014 {orig_pct:.2f}% anomaly coverage at 512x512",
                     fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary statistics ===
    print(f"\n{'='*70}")
    print("SUMMARY — Anomaly cell counts at each UNet resolution")
    print(f"{'='*70}")
    print(f"Original coverage: mean {np.mean(orig_pcts):.2f}%  median {np.median(orig_pcts):.2f}%\n")

    print(f"{'Resolution':<12} {'Method':<10} {'Mean':>8} {'Median':>8} {'Lost':>8} {'<=1':>8} {'<=2':>8}")
    print("-" * 70)
    for res in UNET_RESOLUTIONS:
        for method in ["nearest", "maxpool"]:
            arr = np.array(stats[res][method])
            lost = int((arr == 0).sum())
            le1 = int((arr <= 1).sum())
            le2 = int((arr <= 2).sum())
            print(f"{res}x{res:<8} {method:<10} {arr.mean():>8.1f} {np.median(arr):>8.1f} "
                  f"{lost:>8}/{n_samples} {le1:>8} {le2:>8}")
        print()

    # Summary plot
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for j, res in enumerate(UNET_RESOLUTIONS):
        for row, method in enumerate(["nearest", "maxpool"]):
            arr = np.array(stats[res][method])
            color = "blue" if method == "nearest" else "orange"
            axes[row, j].hist(arr, bins=max(1, min(30, arr.max())), color=color, alpha=0.7)
            axes[row, j].set_title(f"{res}x{res} ({method})\nmean={arr.mean():.1f}, lost={int((arr==0).sum())}", fontsize=9)
            axes[row, j].set_xlabel("Anomaly cells")
            axes[row, j].set_ylabel("Count")

    plt.suptitle("Anomaly cell distribution at each UNet resolution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize anomaly masks at UNet spatial resolutions")
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/unet_mask_viz")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_unet_masks(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
    )
