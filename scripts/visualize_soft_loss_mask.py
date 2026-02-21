"""
Visualize soft loss mask: core anomaly at full weight, decaying in dilation ring.

Shows per sample:
  Col 0: Original image + mask overlay (red=core, blue=ring)
  Col 1: Binary core mask at 64x64
  Col 2: Binary dilated mask at 64x64
  Col 3: Soft weight mask at 64x64 (core=1.0, ring decays, bg=0.0)
  Col 4: Soft weight mask colorbar closeup

The soft mask is computed via distance transform from the core boundary:
  - Core region: weight = 1.0
  - Ring region: weight = 1.0 - (distance_from_core / ring_width)  [linear]
  - Background: weight = 0.0

Alternative decay options: linear, gaussian, cosine.
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


def tensor_to_np01(t: torch.Tensor) -> np.ndarray:
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    padded = F.pad(mask.unsqueeze(0), [radius] * 4, mode="constant", value=0)
    dilated = F.max_pool2d(padded, kernel_size=kernel, stride=1).squeeze(0)
    return (dilated > 0.5).float()


def adaptive_dilation_radius(mask: torch.Tensor, min_r: int = 2, max_r: int = 10) -> int:
    area = mask.sum().item()
    if area == 0:
        return 0
    r = 0.5 * math.sqrt(area)
    return max(min_r, min(max_r, int(r)))


def downsample_mask_maxpool(mask: torch.Tensor, size: int) -> torch.Tensor:
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


def compute_soft_mask(core_64: np.ndarray, dilated_64: np.ndarray, decay: str = "linear",
                      min_weight: float = 0.3) -> np.ndarray:
    """Compute soft weight mask from core and dilated masks at 64x64.

    Args:
        core_64: Binary core mask [64, 64]
        dilated_64: Binary dilated mask [64, 64]
        decay: "linear", "gaussian", or "cosine"
        min_weight: Minimum weight for any ring pixel (prevents zero-weight in ring)

    Returns:
        Soft weight mask [64, 64] with values in [0, 1]
    """
    ring = np.clip(dilated_64 - core_64, 0, 1)

    # Distance from core boundary (inside the ring)
    if core_64.sum() == 0:
        return dilated_64.copy()

    # Distance transform: distance of each pixel to nearest core pixel
    # Invert core so distance is FROM core boundary outward
    dist_from_core = ndimage.distance_transform_edt(1 - core_64)

    # Max distance within the ring (for normalization)
    ring_distances = dist_from_core * ring
    max_ring_dist = ring_distances.max()
    if max_ring_dist == 0:
        max_ring_dist = 1.0

    # Normalized distance in ring: 0 at core boundary, 1 at ring edge
    norm_dist = np.clip(dist_from_core / max_ring_dist, 0, 1)

    # Compute decay
    if decay == "linear":
        ring_weight = 1.0 - norm_dist
    elif decay == "gaussian":
        # sigma such that weight ≈ 0.05 at ring edge
        sigma = 1.0 / np.sqrt(2 * np.log(20))
        ring_weight = np.exp(-0.5 * (norm_dist / sigma) ** 2)
    elif decay == "cosine":
        ring_weight = 0.5 * (1 + np.cos(np.pi * norm_dist))
    else:
        ring_weight = 1.0 - norm_dist

    # Apply minimum weight floor — no ring pixel gets zero
    ring_weight = np.maximum(ring_weight, min_weight)

    # Assemble: core=1.0, ring=decayed, bg=0.0
    soft = np.zeros_like(core_64)
    soft[core_64 > 0.5] = 1.0
    soft[ring > 0.5] = ring_weight[ring > 0.5]

    return soft


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
    overlay_ring = np.zeros((h, w, 4), dtype=np.float32)
    overlay_ring[:, :, 2] = 1.0
    overlay_ring[:, :, 3] = ring * 0.45
    ax.imshow(overlay_ring)
    overlay_orig = np.zeros((h, w, 4), dtype=np.float32)
    overlay_orig[:, :, 0] = 1.0
    overlay_orig[:, :, 3] = orig_2d * 0.45
    ax.imshow(overlay_orig)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def draw_soft_mask(ax, soft_mask: np.ndarray, title: str, show_colorbar: bool = False):
    """Draw soft weight mask with custom colormap: black→blue→red→yellow."""
    # Custom colormap: 0=black, low=blue, mid=red, high=yellow/white
    cmap = LinearSegmentedColormap.from_list("soft_mask", [
        (0.0, (0.12, 0.12, 0.12)),   # background: dark gray
        (0.01, (0.12, 0.12, 0.12)),   # background edge
        (0.02, (0.15, 0.25, 0.7)),    # ring start: blue
        (0.5, (0.9, 0.3, 0.1)),       # ring mid: orange-red
        (1.0, (1.0, 1.0, 0.2)),       # core: bright yellow
    ])
    im = ax.imshow(soft_mask, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Loss weight")
    return im


def draw_binary_grid(ax, mask_2d: np.ndarray, title: str, color="red"):
    gs = mask_2d.shape[0]
    img = np.zeros((gs, gs, 3), dtype=np.float32)
    if color == "red":
        img[:, :, 0] = mask_2d * 0.9
        img[:, :, 1] = mask_2d * 0.2
        img[:, :, 2] = mask_2d * 0.2
    bg = (1 - mask_2d)
    img[:, :, 0] += bg * 0.15
    img[:, :, 1] += bg * 0.15
    img[:, :, 2] += bg * 0.15
    ax.imshow(img, interpolation="nearest")
    n = int(mask_2d.sum())
    ax.set_title(f"{title}\n{n} cells", fontsize=10)
    ax.axis("off")


def draw_dual_grid(ax, core_2d: np.ndarray, dilated_2d: np.ndarray, title: str):
    """Red=core, blue=dilation ring, dark gray=background."""
    gs = core_2d.shape[0]
    ring = np.clip(dilated_2d - core_2d, 0, 1)
    bg = np.clip(1 - dilated_2d, 0, 1)

    img = np.zeros((gs, gs, 3), dtype=np.float32)
    img[:, :, 0] += core_2d * 0.9
    img[:, :, 1] += core_2d * 0.15
    img[:, :, 2] += core_2d * 0.15
    img[:, :, 0] += ring * 0.15
    img[:, :, 1] += ring * 0.3
    img[:, :, 2] += ring * 0.9
    img[:, :, 0] += bg * 0.12
    img[:, :, 1] += bg * 0.12
    img[:, :, 2] += bg * 0.12

    ax.imshow(img, interpolation="nearest")
    n_core = int(core_2d.sum())
    n_ring = int(ring.sum())
    ax.set_title(f"{title}\n{n_core} core + {n_ring} ring", fontsize=10)
    ax.axis("off")


def visualize_soft_masks(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 30,
    exclude_sources: list = None,
    data_root: Path = None,
    min_r: int = 8,
    max_r: int = 16,
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

    decay_types = ["linear", "gaussian", "cosine"]

    print(f"Processing {n_samples} samples...")
    print(f"Dilation: r = max({min_r}, min({max_r}, 0.5*sqrt(area)))")
    print(f"Decay types: {decay_types}")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]
        atype = sample["anomaly_type"]

        orig_pct = mask.sum().item() / mask.numel() * 100
        area = int(mask.sum().item())
        r = adaptive_dilation_radius(mask, min_r=min_r, max_r=max_r)
        dilated = dilate_mask(mask, r)

        img_np = tensor_to_np01(image)
        mask_np = mask.squeeze(0).numpy()
        dilated_np = dilated.squeeze(0).numpy()

        # Downsample to 64x64
        core_64 = downsample_mask_maxpool(mask, 64).squeeze(0).numpy()
        dilated_64 = downsample_mask_maxpool(dilated, 64).squeeze(0).numpy()

        # Compute soft masks for each decay type
        soft_masks = {}
        for decay in decay_types:
            soft_masks[decay] = compute_soft_mask(core_64, dilated_64, decay=decay)

        # === Figure: 3 rows (one per decay) x 5 cols ===
        fig, axes = plt.subplots(len(decay_types), 5, figsize=(22, 4 * len(decay_types)))

        for row, decay in enumerate(decay_types):
            soft = soft_masks[decay]

            # Col 0: Original image + mask overlay
            overlay_dual_mask(
                axes[row, 0], img_np, mask_np, dilated_np,
                f"{atype} (r={r})\nred=core, blue=ring"
            )

            # Col 1: Binary core at 64x64
            draw_binary_grid(axes[row, 1], core_64, "Core mask 64x64")

            # Col 2: Dilated at 64x64 — red=core, blue=ring
            draw_dual_grid(axes[row, 2], core_64, dilated_64, "Dilated mask 64x64")

            # Col 3: Soft weight mask
            draw_soft_mask(axes[row, 3], soft,
                          f"Soft loss weight ({decay})", show_colorbar=True)

            # Col 4: Zoomed crop around anomaly
            # Find bounding box of dilated mask + padding
            ys, xs = np.where(dilated_64 > 0.5)
            if len(ys) > 0:
                pad = 4
                y0 = max(0, ys.min() - pad)
                y1 = min(64, ys.max() + pad + 1)
                x0 = max(0, xs.min() - pad)
                x1 = min(64, xs.max() + pad + 1)
                # Make square
                h, w = y1 - y0, x1 - x0
                side = max(h, w)
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                y0 = max(0, cy - side // 2)
                y1 = min(64, y0 + side)
                x0 = max(0, cx - side // 2)
                x1 = min(64, x0 + side)
                crop = soft[y0:y1, x0:x1]
            else:
                crop = soft

            draw_soft_mask(axes[row, 4], crop,
                          f"Zoomed ({decay})\ncore cells shown bright", show_colorbar=True)

            # Add row label
            axes[row, 0].set_ylabel(decay.upper(), fontsize=14, fontweight="bold",
                                     rotation=0, labelpad=60, va="center")

        plt.suptitle(
            f"Sample {i} — {atype} — {orig_pct:.2f}% coverage — r={r} — "
            f"Soft loss weight at 64x64\n"
            f"Core (bright) = weight 1.0 | Ring = decaying weight | Background (dark) = weight 0.0",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === 1D profile comparison plot ===
    # Show the decay curves
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.linspace(0, 1, 100)
    ax.plot(x, 1.0 - x, label="Linear", linewidth=2)
    sigma = 1.0 / np.sqrt(2 * np.log(20))
    ax.plot(x, np.exp(-0.5 * (x / sigma) ** 2), label="Gaussian", linewidth=2)
    ax.plot(x, 0.5 * (1 + np.cos(np.pi * x)), label="Cosine", linewidth=2)
    ax.set_xlabel("Normalized distance from core (0=core edge, 1=ring edge)", fontsize=11)
    ax.set_ylabel("Loss weight", fontsize=11)
    ax.set_title("Decay profiles for soft loss mask in dilation ring", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_dir / "decay_profiles.png", dpi=150)
    plt.close()

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize soft loss mask for UNet dilation ring")
    parser.add_argument("--splits-dir", type=str,
                        default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str,
                        default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/soft_loss_mask_viz")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    parser.add_argument("--min-r", type=int, default=8)
    parser.add_argument("--max-r", type=int, default=16)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_soft_masks(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
        min_r=args.min_r,
        max_r=args.max_r,
    )
