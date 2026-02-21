"""
Visualize maxpool mask round-trip: 64 → 32 → 16 → 8 → 16 → 32 → 64.

Shows the mask at UNet internal resolutions (starting from 64x64 latent space),
downsampled via maxpool and upsampled via nearest. All shown at 512x512 for visual
comparison. Single row per sample: down path then up path.
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
RESOLUTIONS_DOWN = [64, 32, 16, 8]
RESOLUTIONS_UP = [16, 32, 64]


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


def upsample_nearest(mask: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(
        mask.unsqueeze(0), size=(size, size), mode="nearest"
    ).squeeze(0)


def show_mask(ax, mask_512: np.ndarray, title: str, img_np: np.ndarray = None):
    if img_np is not None:
        ax.imshow(img_np)
        overlay = np.zeros((512, 512, 4), dtype=np.float32)
        overlay[:, :, 0] = 1.0
        overlay[:, :, 3] = mask_512 * 0.5
        ax.imshow(overlay)
    else:
        rgb = np.zeros((512, 512, 3), dtype=np.float32)
        rgb[:, :, 0] = mask_512 * 0.9
        rgb[:, :, 1] = mask_512 * 0.15
        rgb[:, :, 2] = mask_512 * 0.15
        bg = (1 - mask_512)
        rgb[:, :, 0] += bg * 0.12
        rgb[:, :, 1] += bg * 0.12
        rgb[:, :, 2] += bg * 0.12
        ax.imshow(rgb)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


def visualize_maxpool_roundtrip(
    splits_dir: Path,
    save_dir: Path,
    n_samples: int = 10,
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

        # Layout: 2 rows x 5 cols
        # Row 0 (DOWN): image+mask | 64 | 32 | 16 | 8
        # Row 1 (UP):   8 | 16 | 32 | 64 | 64 vs original
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))

        # === DOWN path: maxpool 64 → 32 → 16 → 8 ===
        # Col 0: original image + mask
        show_mask(axes[0, 0], orig_mask, f"Original 512x512\n{orig_pct:.2f}%", img_np)

        # First downsample 512 → 64 via maxpool
        down_64 = downsample_maxpool(mask, 64)
        down_masks = {64: down_64}
        current = down_64

        up_512 = upsample_nearest(current, 512).squeeze(0).numpy()
        n_cells = int(current.sum().item())
        show_mask(axes[0, 1], up_512, f"64x64\n{n_cells}/{64*64} cells")

        # Progressive maxpool: 64 → 32 → 16 → 8
        for j, res in enumerate([32, 16, 8]):
            current = downsample_maxpool(current, res)
            down_masks[res] = current
            up_512 = upsample_nearest(current, 512).squeeze(0).numpy()
            n_cells = int(current.sum().item())
            show_mask(axes[0, j + 2], up_512, f"DOWN {res}x{res}\n{n_cells}/{res*res} cells")

        # === UP path: nearest 8 → 16 → 32 → 64 ===
        current_up = down_masks[8]

        # Col 0: start at 8x8
        up_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
        n_cells = int(current_up.sum().item())
        show_mask(axes[1, 0], up_512, f"Start 8x8\n{n_cells}/{8*8} cells")

        for j, res in enumerate([16, 32, 64]):
            current_up = upsample_nearest(current_up, res)
            current_up = (current_up > 0.5).float()
            up_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
            n_cells = int(current_up.sum().item())
            show_mask(axes[1, j + 1], up_512, f"UP {res}x{res}\n{n_cells}/{res*res} cells")

        # Col 4: final 64x64 overlaid on image for comparison with original
        final_512 = upsample_nearest(current_up, 512).squeeze(0).numpy()
        show_mask(axes[1, 4], final_512, f"Reconstructed 64x64\nvs original", img_np)

        # Legend
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.65, label="Anomaly mask")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=10,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {orig_pct:.2f}% \u2014 Maxpool Down / Nearest Up",
            fontsize=16, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

    print(f"Saved {n_samples} samples to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Maxpool mask round-trip: 64→8→64")
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/maxpool_roundtrip")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--exclude-sources", type=str, nargs="*", default=[])
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = Path(args.data_root) if args.data_root else None
    if data_root and not data_root.is_absolute():
        data_root = project_root / data_root

    visualize_maxpool_roundtrip(
        splits_dir=project_root / args.splits_dir,
        save_dir=project_root / args.save_dir,
        n_samples=args.n_samples,
        exclude_sources=args.exclude_sources,
        data_root=data_root,
    )
