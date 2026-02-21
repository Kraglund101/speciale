"""
Visualize UNet mask downsampling: 512 → 64 → 32 → 16 → 8.

Shows how the core mask and dilated mask (r=8–16) change at each UNet
spatial resolution under our maxpool downsampling strategy.

For each sample:
  Row 0: Core mask (red) at each resolution, upscaled to 512 for visual comparison
  Row 1: Dilated mask with core (red) + ring (blue) at each resolution
  Row 2: Native-resolution grid view of dilated mask at each level

Dilation is computed once at 512x512, then both masks are maxpool-downsampled
together through UNet resolutions.
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
from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

SEED = 42
UNET_SIZES = [64, 32, 16, 8]


def downsample_mask_maxpool(mask: torch.Tensor, target_size: int) -> torch.Tensor:
    """Maxpool downsample mask to target_size x target_size."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    h, w = mask.shape[-2], mask.shape[-1]
    kernel_h = h // target_size
    kernel_w = w // target_size
    # Resize to exact multiple if needed
    exact_h = kernel_h * target_size
    exact_w = kernel_w * target_size
    if exact_h != h or exact_w != w:
        mask = F.interpolate(mask.float(), size=(exact_h, exact_w), mode="nearest")
    pooled = F.max_pool2d(mask.float(), kernel_size=(kernel_h, kernel_w))
    return (pooled > 0.5).float().squeeze(0).squeeze(0)


def tensor_to_np01(t: torch.Tensor) -> np.ndarray:
    """Convert [-1,1] image tensor to [0,1] numpy HWC."""
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def draw_mask_overlay(ax, img_np, core_np, ring_np, title, show_image=True):
    """Draw image with core (red) + ring (blue) overlay."""
    if show_image:
        ax.imshow(img_np)
    else:
        ax.imshow(np.ones_like(img_np) * 0.95)
    h, w = core_np.shape
    ov_core = np.zeros((h, w, 4), dtype=np.float32)
    ov_core[:, :, 0] = 1.0
    ov_core[:, :, 3] = (core_np > 0.5).astype(np.float32) * 0.5
    ax.imshow(ov_core, extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ov_ring = np.zeros((h, w, 4), dtype=np.float32)
    ov_ring[:, :, 2] = 1.0
    ov_ring[:, :, 3] = (ring_np > 0.5).astype(np.float32) * 0.4
    ax.imshow(ov_ring, extent=[0, img_np.shape[1], img_np.shape[0], 0])
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def draw_grid_view(ax, core_small, ring_small, title, grid_size):
    """Draw native-resolution grid with colored cells."""
    gs = grid_size
    canvas = np.ones((gs, gs, 3), dtype=np.float32) * 0.92
    c = core_small.numpy() if torch.is_tensor(core_small) else core_small
    r = ring_small if isinstance(ring_small, np.ndarray) else ring_small.numpy()

    for row in range(gs):
        for col in range(gs):
            if c[row, col] > 0.5:
                canvas[row, col] = [0.9, 0.15, 0.15]
            elif r[row, col] > 0.5:
                canvas[row, col] = [0.15, 0.3, 0.9]

    ax.imshow(canvas, interpolation="nearest")
    # Grid lines
    for j in range(gs + 1):
        ax.axhline(y=j - 0.5, color="white", linewidth=0.5, alpha=0.6)
        ax.axvline(x=j - 0.5, color="white", linewidth=0.5, alpha=0.6)

    n_core = int((c > 0.5).sum())
    n_ring = int((r > 0.5).sum())
    n_total = gs * gs
    ax.set_title(f"{title}\n{n_core}c + {n_ring}r / {n_total}", fontsize=8)
    ax.axis("off")


def visualize(
    splits_dir: Path, save_dir: Path, n_samples: int = 30,
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

    # Track stats per resolution
    stats = {s: {"core": [], "ring": [], "lost_core": 0} for s in UNET_SIZES}

    print(f"Processing {n_samples} samples...")
    print(f"Method: maxpool downsampling, dilation at 512 (r=8–16)")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]  # [1, 512, 512]
        atype = sample["anomaly_type"]
        pct = mask.sum().item() / mask.numel() * 100

        img_np = tensor_to_np01(image)
        mask_512 = mask.squeeze(0)  # [512, 512]

        # Dilate at 512 (UNet range: r=8–16)
        r_512 = adaptive_dilation_radius(mask, min_r=8, max_r=16)
        dilated_512 = dilate_mask(mask, r_512).squeeze(0)  # [512, 512]
        ring_512 = np.clip(dilated_512.numpy() - mask_512.numpy(), 0, 1)

        # Downsample through UNet resolutions
        cores = {}
        dilateds = {}
        rings = {}
        for s in UNET_SIZES:
            cores[s] = downsample_mask_maxpool(mask_512, s)
            dilateds[s] = downsample_mask_maxpool(dilated_512, s)
            rings[s] = np.clip(dilateds[s].numpy() - cores[s].numpy(), 0, 1)

            n_core = int((cores[s] > 0.5).sum())
            n_ring = int((rings[s] > 0.5).sum())
            stats[s]["core"].append(n_core)
            stats[s]["ring"].append(n_ring)
            if n_core == 0 and mask_512.sum() > 0:
                stats[s]["lost_core"] += 1

        # === Figure: 1 row x 5 cols (512 + 4 UNet sizes) ===
        n_cols = 1 + len(UNET_SIZES)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))

        # Core + ring overlay, upscaled to 512 for comparison
        draw_mask_overlay(axes[0], img_np, mask_512.numpy(), ring_512,
                          f"512\u00d7512 dilated r={r_512}\n{int(dilated_512.sum())} px")

        for j, s in enumerate(UNET_SIZES):
            core_up = F.interpolate(
                cores[s].unsqueeze(0).unsqueeze(0), size=(512, 512), mode="nearest"
            ).squeeze().numpy()
            ring_up = F.interpolate(
                torch.from_numpy(rings[s]).unsqueeze(0).unsqueeze(0),
                size=(512, 512), mode="nearest"
            ).squeeze().numpy()
            n_core = int((cores[s] > 0.5).sum())
            n_ring = int((rings[s] > 0.5).sum())
            draw_mask_overlay(axes[j+1], img_np, core_up, ring_up,
                              f"{s}\u00d7{s}\n{n_core}c + {n_ring}r / {s*s}")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core (anomaly)"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label=f"Dilation ring (r={r_512})"),
            mpatches.Patch(color=(0.92, 0.92, 0.92), label="Normal (no anomaly)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {pct:.2f}% coverage\n"
            f"Maxpool downsampling \u2022 Dilation at 512 (r=clamp(0.5\u00b7\u221aarea, 8, 16) = {r_512})",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0.07, 0.04, 1, 0.92])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # === Summary stats ===
    print(f"\n{'='*70}")
    print(f"SUMMARY \u2014 Maxpool downsampling through UNet resolutions")
    print(f"{'='*70}")
    print(f"\n{'Resolution':<12} {'Core mean':>10} {'Core med':>10} {'Ring mean':>10} {'Ring med':>10} {'Lost core':>10}")
    print("-" * 64)
    for s in UNET_SIZES:
        c = np.array(stats[s]["core"])
        r = np.array(stats[s]["ring"])
        lost = stats[s]["lost_core"]
        print(f"{s:>3}\u00d7{s:<8} {c.mean():>10.1f} {np.median(c):>10.0f} {r.mean():>10.1f} {np.median(r):>10.0f} {lost:>10}")

    # Summary figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for j, s in enumerate(UNET_SIZES):
        c = np.array(stats[s]["core"])
        r = np.array(stats[s]["ring"])
        axes[j].hist(c, bins=20, alpha=0.6, color="red", label="Core cells")
        axes[j].hist(r, bins=20, alpha=0.5, color="blue", label="Ring cells")
        axes[j].set_title(f"{s}\u00d7{s} (lost {stats[s]['lost_core']}/{n_samples})")
        axes[j].set_xlabel("Cell count")
        axes[j].legend(fontsize=8)
    plt.suptitle(f"Cell counts at each UNet resolution (N={n_samples})",
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
    parser.add_argument("--save-dir", default="results/unet/downsampling")
    parser.add_argument("--n-samples", type=int, default=30)
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
