"""
Visualize mask round-trip through UNet resolutions.

Shows the downsampling path (encoder) and upsampling path (decoder)
side by side so you can see the information loss and blocky artifacts
when going down to 8x8 and back up to 512x512.

Layout per sample (2 rows x 6 cols):
  Row 0 (Down): 512 → 64 → 32 → 16 → 8 → (blank)
  Row 1 (Up):   (blank) → 8 → 16 → 32 → 64 → 512 (reconstructed)

All masks shown at 512x512 via nearest upsample for visual comparison.
Both nearest and maxpool downsampling shown.
"""
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
RESOLUTIONS = [64, 32, 16, 8]


def tensor_to_np01(t: torch.Tensor) -> np.ndarray:
    return ((t.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)


def downsample_maxpool(mask: torch.Tensor, size: int) -> torch.Tensor:
    _, h, w = mask.shape
    kernel = h // size
    target = size * kernel
    if h != target:
        mask = F.interpolate(mask.unsqueeze(0), size=(target, target), mode="nearest").squeeze(0)
    pooled = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel).squeeze(0)
    return (pooled > 0.5).float()


def downsample_nearest(mask: torch.Tensor, size: int) -> torch.Tensor:
    return (F.interpolate(
        mask.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0) > 0.5).float()


def upsample_nearest(mask: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        mask.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)


def show_mask(ax, mask_512: np.ndarray, title: str, img_np: np.ndarray = None):
    """Show mask at 512x512 resolution, optionally overlaid on image."""
    if img_np is not None:
        ax.imshow(img_np)
        overlay = np.zeros((512, 512, 4), dtype=np.float32)
        overlay[:, :, 0] = 1.0
        overlay[:, :, 3] = mask_512 * 0.5
        ax.imshow(overlay)
    else:
        # Red=anomaly, dark=background
        rgb = np.zeros((512, 512, 3), dtype=np.float32)
        rgb[:, :, 0] = mask_512 * 0.9
        rgb[:, :, 1] = mask_512 * 0.15
        rgb[:, :, 2] = mask_512 * 0.15
        bg = (1 - mask_512)
        rgb[:, :, 0] += bg * 0.12
        rgb[:, :, 1] += bg * 0.12
        rgb[:, :, 2] += bg * 0.12
        ax.imshow(rgb)
    n = int(mask_512.sum())
    pct = n / (512 * 512) * 100
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def visualize_roundtrip(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 50,
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

    print(f"Processing {n_samples} samples...")

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]      # [1, 512, 512]
        atype = sample["anomaly_type"]

        img_np = tensor_to_np01(image)
        orig_mask = mask.squeeze(0).numpy()
        orig_pct = orig_mask.sum() / orig_mask.size * 100

        # 4 rows x 6 cols:
        # Row 0: Nearest DOWN path (512 → 64 → 32 → 16 → 8) shown at 512
        # Row 1: UP path from Nearest-downsampled 8x8 (always nearest upsample)
        # Row 2: Maxpool DOWN path
        # Row 3: UP path from Maxpool-downsampled 8x8 (always nearest upsample)
        fig, axes = plt.subplots(4, 6, figsize=(24, 16))

        for method_idx, (method_name, downsample_fn) in enumerate([
            ("Nearest", downsample_nearest),
            ("Maxpool", downsample_maxpool),
        ]):
            row_down = method_idx * 2
            row_up = method_idx * 2 + 1

            # === DOWN path ===
            # Col 0: Original at 512
            show_mask(axes[row_down, 0], orig_mask,
                      f"Original 512x512\n{orig_pct:.2f}%", img_np)

            # Progressively downsample and show each at 512
            down_masks = {}
            current = mask
            for j, res in enumerate(RESOLUTIONS):
                col = j + 1
                down = downsample_fn(current, res)
                down_masks[res] = down
                # Show upsampled to 512 for visual comparison
                up_to_512 = upsample_nearest(down, 512).squeeze(0).numpy()
                n_cells = int(down.sum().item())
                total = res * res
                show_mask(axes[row_down, col], up_to_512,
                          f"{method_name} DOWN → {res}x{res}\n{n_cells}/{total} cells")
                current = down  # progressive: each step downsamples from previous

            # Col 5: blank for down row
            axes[row_down, 5].axis("off")
            axes[row_down, 5].text(0.5, 0.5, f"{method_name}\nDOWN ↓",
                                    transform=axes[row_down, 5].transAxes,
                                    ha="center", va="center", fontsize=14,
                                    fontweight="bold", color="gray")

            # === UP path (starting from 8x8, upsample back) ===
            # Col 0: blank for up row
            axes[row_up, 0].axis("off")
            axes[row_up, 0].text(0.5, 0.5, f"UP ↑\n(from {method_name})",
                                  transform=axes[row_up, 0].transAxes,
                                  ha="center", va="center", fontsize=14,
                                  fontweight="bold", color="gray")

            # Start from 8x8 and upsample back through each resolution
            up_resolutions = list(reversed(RESOLUTIONS))  # [8, 16, 32, 64]
            current_up = down_masks[8]  # start from the smallest

            for j, res in enumerate(up_resolutions):
                col = j + 1
                if j == 0:
                    # First: just show the 8x8 mask
                    up_to_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
                    n_cells = int(current_up.sum().item())
                    show_mask(axes[row_up, col], up_to_512,
                              f"Start: {res}x{res}\n{n_cells}/{res*res} cells")
                else:
                    # Upsample from current to next resolution
                    current_up = upsample_nearest(current_up, res)
                    current_up = (current_up > 0.5).float()
                    up_to_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
                    n_cells = int(current_up.sum().item())
                    show_mask(axes[row_up, col], up_to_512,
                              f"UP → {res}x{res}\n{n_cells}/{res*res} cells")

            # Col 5: final upsample to 512 overlaid on image for comparison
            final_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
            show_mask(axes[row_up, 5], final_512,
                      f"Reconstructed 512x512\nvs original", img_np)

        # Legend
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.65, label="Anomaly mask")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=10,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {orig_pct:.2f}% coverage \u2014 Mask Round-Trip Through UNet",
            fontsize=16, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n_samples} done")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize mask round-trip through UNet resolutions")
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/unet_mask_roundtrip")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_roundtrip(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
    )
