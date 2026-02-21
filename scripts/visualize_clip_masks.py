"""
Visualize CLIP patch masks at 16x16: maxpool vs adaptive dilation + maxpool.

4x4 grid per sample comparing two downsampling strategies:
  Row 0 (Full, maxpool):    native+mask | 224 | maxpool grid | maxpool masked
  Row 1 (Full, adaptive):   native+dilated mask (r=X) | 224 | adaptive grid | adaptive masked
  Row 2 (Crop, maxpool):    crop+mask | crop 224 | crop grid | crop masked
  Row 3 (Crop, adaptive):   crop+dilated mask | crop 224 | crop adaptive grid | crop adaptive masked

Adaptive dilation formula: r = max(MIN_R, min(MAX_R, 0.5 * sqrt(area)))
  - area = number of anomaly pixels at mask resolution
  - Small anomalies get minimum dilation (MIN_R pixels)
  - Large anomalies scale with sqrt(area), capped at MAX_R

Seeded for reproducibility.
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.anomaly_dataset import AnomalyDataset

SEED = 42


def overlay_mask_native(ax, img_np, mask_np, title, color='red'):
    """Show image at native resolution with colored mask overlay."""
    ax.imshow(img_np)
    h, w = img_np.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.float32)
    mask_2d = mask_np.squeeze()
    if mask_2d.shape != (h, w):
        mask_resized = np.array(Image.fromarray((mask_2d * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)) / 255.0
    else:
        mask_resized = mask_2d
    if color == 'red':
        overlay[:, :, 0] = 1.0
    elif color == 'blue':
        overlay[:, :, 2] = 1.0
    elif color == 'green':
        overlay[:, :, 1] = 1.0
    overlay[:, :, 3] = mask_resized * 0.4
    ax.imshow(overlay)
    n_pixels = int(mask_resized.sum())
    total = h * w
    pct = n_pixels / total * 100
    ax.set_title(f"{title}\n{pct:.2f}% coverage", fontsize=11)
    ax.axis("off")


def overlay_dual_mask(ax, img_np, orig_mask_np, dilated_mask_np, title):
    """Show image with original mask (red) and dilation ring (blue) overlaid."""
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
    overlay_ring[:, :, 3] = ring * 0.4
    ax.imshow(overlay_ring)
    # Red overlay for original mask
    overlay_orig = np.zeros((h, w, 4), dtype=np.float32)
    overlay_orig[:, :, 0] = 1.0
    overlay_orig[:, :, 3] = orig_2d * 0.4
    ax.imshow(overlay_orig)
    pct = dil_2d.sum() / (h * w) * 100
    ax.set_title(f"{title}\n{pct:.2f}% coverage", fontsize=11)
    ax.axis("off")


def overlay_patch_grid(ax, img_np, mask_16, title, img_size=224):
    """Draw image with 16x16 patch grid, anomaly patches highlighted in red."""
    ax.imshow(img_np)
    gs = 16
    cell = img_size / gs
    mask_np = mask_16.squeeze().numpy() if torch.is_tensor(mask_16) else mask_16.squeeze()
    n = int(mask_np.sum())

    for r in range(gs):
        for c in range(gs):
            if mask_np[r, c] > 0.5:
                ax.add_patch(plt.Rectangle(
                    (c * cell, r * cell), cell, cell,
                    linewidth=0, facecolor='red', alpha=0.35,
                ))

    for j in range(gs + 1):
        ax.axhline(y=j * cell, color='white', linewidth=0.3, alpha=0.4)
        ax.axvline(x=j * cell, color='white', linewidth=0.3, alpha=0.4)

    ax.set_title(f"{title}\n{n}/256 patches", fontsize=11)
    ax.axis("off")


def show_masked_only(ax, img_np, mask_16, title, img_size=224):
    """Show only the anomaly patches, rest blacked out."""
    gs = 16
    cell = img_size / gs
    mask_np = mask_16.squeeze().numpy() if torch.is_tensor(mask_16) else mask_16.squeeze()

    blocked = np.zeros((img_size, img_size), dtype=np.float32)
    for r in range(gs):
        for c in range(gs):
            if mask_np[r, c] > 0.5:
                y0, y1 = int(r * cell), int(min((r + 1) * cell, img_size))
                x0, x1 = int(c * cell), int(min((c + 1) * cell, img_size))
                blocked[y0:y1, x0:x1] = 1.0

    ax.imshow(img_np * blocked[:, :, None])
    ax.set_title(f"{title}\n(anomaly only)", fontsize=11)
    ax.axis("off")


def get_crop_bbox(mask, pad_frac=0.1):
    """Get square bbox around all anomaly pixels. Returns (y_min, y_max, x_min, x_max) or None."""
    mask_2d = mask.squeeze(0)
    nonzero = torch.nonzero(mask_2d, as_tuple=False)
    if len(nonzero) == 0:
        return None

    y_min, x_min = nonzero.min(dim=0).values.tolist()
    y_max, x_max = nonzero.max(dim=0).values.tolist()
    _, h, w = mask.shape

    bbox_h, bbox_w = y_max - y_min, x_max - x_min
    pad_y = max(1, int(bbox_h * pad_frac))
    pad_x = max(1, int(bbox_w * pad_frac))
    y_min, x_min = max(0, y_min - pad_y), max(0, x_min - pad_x)
    y_max, x_max = min(h, y_max + pad_y + 1), min(w, x_max + pad_x + 1)

    # Make square
    crop_h, crop_w = y_max - y_min, x_max - x_min
    if crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_min + crop_h)
        if x_max - x_min < crop_h:
            x_min = max(0, x_max - crop_h)
    elif crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_min + crop_w)
        if y_max - y_min < crop_w:
            y_min = max(0, y_max - crop_w)

    return y_min, y_max, x_min, x_max


def tensor_to_pil(t):
    """Convert [-1,1] tensor [C,H,W] to PIL Image."""
    arr = ((t.permute(1, 2, 0).numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def tensor_to_np01(t):
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


def adaptive_dilation_radius(mask: torch.Tensor, min_r: int = 2, max_r: int = 32) -> int:
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


def downsample_mask_maxpool(mask: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Downsample binary mask [1,H,W] to [1,gs,gs] via maxpool.

    Correctly mirrors CLIP's 16x16 patch grid (each patch = 14x14 pixels at 224).
    """
    clip_size = grid_size * 14  # 16 * 14 = 224
    mask_224 = F.interpolate(
        mask.unsqueeze(0), size=(clip_size, clip_size), mode="nearest"
    ).squeeze(0)
    pooled = F.max_pool2d(mask_224.unsqueeze(0), kernel_size=14).squeeze(0)
    return (pooled > 0.5).float()


def visualize_masks(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 200,
    exclude_sources: list = None,
    data_root: Path = None,
    crop_pad_frac: float = 0.1,
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

    gs = 16  # CLIP patch grid

    # Compare: no dilation, fixed r=2, adaptive min_r=2 (3 rows for report)
    dilation_configs = [
        {"label": "No dilation (maxpool)", "min_r": 0, "max_r": 0, "fixed": None},
        {"label": "Fixed r=2", "min_r": 0, "max_r": 0, "fixed": 2},
        {"label": f"Adaptive min_r=2, max_r={max_r}", "min_r": 2, "max_r": max_r, "fixed": None},
    ]

    stats = {
        "orig_pct": [], "types": [],
    }
    for cfg in dilation_configs:
        stats[cfg["label"]] = {"patches": [], "radius": []}

    print(f"Processing {n_samples} samples...")
    print(f"Comparing 3 dilation levels: no dilation, fixed r=2, adaptive min_r=2 (max_r={max_r})")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]   # [3, 512, 512] in [-1, 1]
        mask = sample["mask"]     # [1, 512, 512] binary
        atype = sample["anomaly_type"]

        orig_pct = mask.sum().item() / mask.numel() * 100
        stats["orig_pct"].append(orig_pct)
        stats["types"].append(atype)

        native_img_np = tensor_to_np01(image)
        native_mask_np = mask.squeeze(0).numpy()
        area = int(mask.sum().item())

        img_full_224 = F.interpolate(
            image.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        ).squeeze(0)
        full_224_np = tensor_to_np01(img_full_224)

        # Compute masks for each dilation config
        row_data = []
        for cfg in dilation_configs:
            if cfg["fixed"] is not None:
                # Fixed radius dilation
                r = cfg["fixed"]
                dilated = dilate_mask(mask, r)
                mask_16 = downsample_mask_maxpool(dilated, gs)
                dilated_mask_np = dilated.squeeze(0).numpy()
            elif cfg["min_r"] == 0:
                # No dilation — pure maxpool
                mask_16 = downsample_mask_maxpool(mask, gs)
                r = 0
                dilated_mask_np = native_mask_np
            else:
                r = adaptive_dilation_radius(mask, min_r=cfg["min_r"], max_r=cfg["max_r"])
                dilated = dilate_mask(mask, r)
                mask_16 = downsample_mask_maxpool(dilated, gs)
                dilated_mask_np = dilated.squeeze(0).numpy()

            stats[cfg["label"]]["patches"].append(int(mask_16.sum().item()))
            stats[cfg["label"]]["radius"].append(r)
            row_data.append((cfg["label"], r, mask_16, dilated_mask_np))

        # === 3x4 figure ===
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        for row_idx, (label, r, mask_16, dilated_mask_np) in enumerate(row_data):
            n_patches = int(mask_16.sum().item())

            # Col 0: native image + mask overlay
            if r == 0:
                overlay_mask_native(axes[row_idx, 0], native_img_np, native_mask_np,
                                   f"{atype} ({orig_pct:.2f}%)\nNo dilation")
            else:
                overlay_dual_mask(axes[row_idx, 0], native_img_np, native_mask_np, dilated_mask_np,
                                 f"r={r} (area={area}, sqrt={math.sqrt(area):.0f})\nred=orig, blue=dilated")

            # Col 1: 224x224
            axes[row_idx, 1].imshow(full_224_np)
            axes[row_idx, 1].set_title(f"Full -> 224x224", fontsize=11)
            axes[row_idx, 1].axis("off")

            # Col 2: patch grid
            overlay_patch_grid(axes[row_idx, 2], full_224_np, mask_16,
                              f"{label.split(',')[0]}\n(r={r})" if r > 0 else "Maxpool only")

            # Col 3: masked only
            show_masked_only(axes[row_idx, 3], full_224_np, mask_16,
                            f"{n_patches} patches")

        # Legend
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.65, label="Anomaly mask")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=9,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 Dilation Comparison (max_r={max_r})\n"
            f"Row 0: no dilation | Row 1: fixed r=2 | Row 2: adaptive min_r=2",
            fontsize=15, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary ===
    print(f"\n{'='*70}")
    print(f"SUMMARY — Dilation Comparison (max_r={max_r})")
    print(f"{'='*70}")
    print(f"Original coverage: mean {np.mean(stats['orig_pct']):.2f}%  median {np.median(stats['orig_pct']):.2f}%\n")

    print(f"{'Mode':<35} {'Mean patches':>14} {'Median':>8} {'Lost':>10} {'<=2':>8} {'Mean r':>8}")
    print("-" * 85)
    for cfg in dilation_configs:
        label = cfg["label"]
        p = np.array(stats[label]["patches"])
        r = np.array(stats[label]["radius"])
        print(f"{label:<35} {p.mean():>14.1f} {np.median(p):>8.1f} "
              f"{(p==0).sum():>10}/{n_samples} {(p<=2).sum():>8} {r.mean():>8.1f}")

    # Patch gains relative to no-dilation baseline
    baseline = np.array(stats[dilation_configs[0]["label"]]["patches"])
    print(f"\nPatch gain vs no-dilation baseline:")
    for cfg in dilation_configs[1:]:
        label = cfg["label"]
        p = np.array(stats[label]["patches"])
        gain = p - baseline
        print(f"  {label}: mean +{gain.mean():.1f}  median +{np.median(gain):.0f}  max +{gain.max()}")

    # === Summary plots ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ["blue", "cyan", "orange"]
    for j, cfg in enumerate(dilation_configs):
        label = cfg["label"].split(",")[0] if "," in cfg["label"] else cfg["label"]
        p = np.array(stats[cfg["label"]]["patches"])
        axes[0].hist(p, bins=30, alpha=0.4, label=label, color=colors[j])
    axes[0].set_xlabel("Anomaly patches (of 256)")
    axes[0].set_title("Patch count distribution")
    axes[0].legend(fontsize=11)

    for j, cfg in enumerate(dilation_configs[1:]):
        label = f"min_r={cfg['min_r']}"
        p = np.array(stats[cfg["label"]]["patches"])
        gain = p - baseline
        axes[1].hist(gain, bins=30, alpha=0.5, label=label, color=colors[j + 1])
    axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1].set_xlabel("Extra patches vs no-dilation")
    axes[1].set_title("Patch gain distribution")
    axes[1].legend(fontsize=8)

    for j, cfg in enumerate(dilation_configs[1:]):
        label = f"min_r={cfg['min_r']}"
        r = np.array(stats[cfg["label"]]["radius"])
        p = np.array(stats[cfg["label"]]["patches"])
        gain = p - baseline
        axes[2].scatter(r, gain, alpha=0.4, s=15, label=label, color=colors[j + 1])
    axes[2].set_xlabel("Dilation radius (pixels)")
    axes[2].set_ylabel("Patch gain")
    axes[2].set_title("Radius vs patch gain")
    axes[2].legend(fontsize=8)

    plt.suptitle(
        f"Dilation Comparison: max_r={max_r}, N={n_samples} samples",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize CLIP masks: maxpool vs adaptive dilation")
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/clip/clip_mask_viz_adaptive")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    parser.add_argument("--max-r", type=int, default=10,
                        help="Maximum dilation radius in pixels for adaptive formula (default: 10)")
    parser.add_argument("--crop-pad-frac", type=float, default=0.1,
                        help="Fractional padding around crop bbox (default: 0.1)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_masks(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        crop_pad_frac=args.crop_pad_frac,
        max_r=args.max_r,
    )
