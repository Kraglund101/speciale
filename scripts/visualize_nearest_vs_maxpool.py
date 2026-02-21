"""
Simple illustration: nearest vs maxpool downsampling on anomaly masks.

Shows a synthetic 16x16 grid with small anomalies to demonstrate
how nearest interpolation loses small features while maxpool preserves them.
Also shows a real example from the dataset.
"""
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_grid_mask(ax, mask_2d, title, grid_size, show_grid=True):
    """Draw mask as colored grid cells."""
    gs = grid_size
    img = np.zeros((gs, gs, 3), dtype=np.float32)
    # Anomaly = red, background = light gray
    img[:, :, 0] = mask_2d * 0.9 + (1 - mask_2d) * 0.9
    img[:, :, 1] = mask_2d * 0.2 + (1 - mask_2d) * 0.9
    img[:, :, 2] = mask_2d * 0.2 + (1 - mask_2d) * 0.9

    ax.imshow(img, interpolation="nearest")

    if show_grid:
        for j in range(gs + 1):
            ax.axhline(y=j - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(x=j - 0.5, color='black', linewidth=0.5, alpha=0.3)

    n = int(mask_2d.sum())
    ax.set_title(f"{title}\n{n}/{gs*gs} cells active", fontsize=13)
    ax.axis("off")


def example_synthetic():
    """Create synthetic examples showing nearest vs maxpool."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    # === Example 1: Single small anomaly (3x3 pixels in 64x64) ===
    mask1 = torch.zeros(1, 64, 64)
    mask1[0, 30:33, 30:33] = 1.0  # 3x3 anomaly

    nearest1 = F.interpolate(mask1.unsqueeze(0), size=(8, 8), mode="nearest").squeeze()
    kernel1 = 64 // 8
    maxpool1 = F.max_pool2d(mask1.unsqueeze(0), kernel_size=kernel1).squeeze()
    maxpool1 = (maxpool1 > 0.5).float()

    draw_grid_mask(axes[0, 0], mask1.squeeze().numpy(), "Original 64x64\n(3x3 anomaly)", 64, show_grid=False)
    draw_grid_mask(axes[0, 1], (nearest1 > 0.5).float().numpy(), "Nearest -> 8x8", 8)
    draw_grid_mask(axes[0, 2], maxpool1.numpy(), "Maxpool -> 8x8", 8)

    # === Example 2: Thin scratch (1px wide, 20px long in 64x64) ===
    mask2 = torch.zeros(1, 64, 64)
    mask2[0, 20:40, 32] = 1.0  # 1px wide vertical scratch

    nearest2 = F.interpolate(mask2.unsqueeze(0), size=(8, 8), mode="nearest").squeeze()
    maxpool2 = F.max_pool2d(mask2.unsqueeze(0), kernel_size=kernel1).squeeze()
    maxpool2 = (maxpool2 > 0.5).float()

    draw_grid_mask(axes[1, 0], mask2.squeeze().numpy(), "Original 64x64\n(1px scratch)", 64, show_grid=False)
    draw_grid_mask(axes[1, 1], (nearest2 > 0.5).float().numpy(), "Nearest -> 8x8", 8)
    draw_grid_mask(axes[1, 2], maxpool2.numpy(), "Maxpool -> 8x8", 8)

    # === Example 3: Multiple tiny spots (2x2 each in 64x64) ===
    mask3 = torch.zeros(1, 64, 64)
    mask3[0, 5:7, 5:7] = 1.0
    mask3[0, 50:52, 10:12] = 1.0
    mask3[0, 15:17, 55:57] = 1.0

    nearest3 = F.interpolate(mask3.unsqueeze(0), size=(8, 8), mode="nearest").squeeze()
    maxpool3 = F.max_pool2d(mask3.unsqueeze(0), kernel_size=kernel1).squeeze()
    maxpool3 = (maxpool3 > 0.5).float()

    draw_grid_mask(axes[2, 0], mask3.squeeze().numpy(), "Original 64x64\n(3 tiny spots, 2x2 each)", 64, show_grid=False)
    draw_grid_mask(axes[2, 1], (nearest3 > 0.5).float().numpy(), "Nearest -> 8x8", 8)
    draw_grid_mask(axes[2, 2], maxpool3.numpy(), "Maxpool -> 8x8", 8)

    # Add row labels
    for row, label in enumerate(["Small blob", "Thin scratch", "Scattered spots"]):
        axes[row, 0].set_ylabel(label, fontsize=16, fontweight="bold", rotation=0,
                                labelpad=80, va="center")

    plt.suptitle("Nearest Interpolation vs Max Pooling for Mask Downsampling\n"
                 "Red = anomaly. Nearest can lose small features entirely.",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def example_real(splits_dir, data_root, n_samples=50, save_dir=None):
    """Show nearest vs maxpool on real dataset samples. One image per sample."""
    from src.data.anomaly_dataset import AnomalyDataset

    dataset = AnomalyDataset(
        splits_dir, image_size=512,
        data_root=data_root, return_reference=False,
    )

    random.seed(42)
    n_samples = min(n_samples, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)

    target_size = 16  # CLIP patch grid
    kernel = 512 // target_size  # 32

    stats_nearest = []
    stats_maxpool = []
    n_lost = 0

    for i, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"]
        mask = sample["mask"]
        atype = sample["anomaly_type"]
        pct = mask.sum().item() / mask.numel() * 100

        img_np = ((image.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        mask_np = mask.squeeze(0).numpy()

        # Nearest
        nearest = F.interpolate(mask.unsqueeze(0), size=(target_size, target_size), mode="nearest")
        nearest = (nearest.squeeze() > 0.5).float().numpy()
        n_nearest = int(nearest.sum())

        # Maxpool
        maxpool = F.max_pool2d(mask.unsqueeze(0), kernel_size=kernel)
        maxpool = (maxpool.squeeze() > 0.5).float().numpy()
        n_maxpool = int(maxpool.sum())

        stats_nearest.append(n_nearest)
        stats_maxpool.append(n_maxpool)
        if n_nearest == 0:
            n_lost += 1

        # 1x4 figure per sample
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Col 0: image + mask overlay
        canvas = img_np.copy()
        canvas[mask_np == 0] *= 0.7
        fg = mask_np > 0.5
        for c in range(3):
            canvas[fg, c] = canvas[fg, c] * 0.55 + np.array([1.0, 0.2, 0.2])[c] * 0.45
        axes[0].imshow(canvas)
        axes[0].set_title(f"{atype} ({pct:.2f}%)\n{int(mask_np.sum())} pixels", fontsize=13)
        axes[0].axis("off")

        # Col 1: original mask
        axes[1].imshow(mask_np, cmap="Reds", interpolation="nearest", vmin=0, vmax=1)
        axes[1].set_title(f"Mask 512x512", fontsize=13)
        axes[1].axis("off")

        # Col 2: nearest 16x16
        draw_grid_mask(axes[2], nearest, f"Nearest -> 16x16", target_size)
        if n_nearest == 0:
            axes[2].text(0.5, 0.5, "LOST", transform=axes[2].transAxes,
                        ha="center", va="center", fontsize=24, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round", fc="red", alpha=0.8))

        # Col 3: maxpool 16x16
        draw_grid_mask(axes[3], maxpool, f"Maxpool -> 16x16", target_size)

        # Legend
        legend_handles = [mpatches.Patch(color=(1.0, 0.2, 0.2), alpha=0.65, label="Anomaly mask")]
        fig.legend(handles=legend_handles, loc="lower center", ncol=1, fontsize=13,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(f"Sample {i} \u2014 Nearest ({n_nearest} patches) vs Maxpool ({n_maxpool} patches)",
                    fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0.07, 1, 0.94])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # Summary
    sn = np.array(stats_nearest)
    sm = np.array(stats_maxpool)
    print(f"\n{'='*60}")
    print("SUMMARY — Nearest vs Maxpool at 16x16")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Mean':>8} {'Median':>8} {'Lost':>10} {'<=2':>8}")
    print("-" * 55)
    print(f"{'Nearest':<15} {sn.mean():>8.1f} {np.median(sn):>8.1f} {(sn==0).sum():>10}/{n_samples} {(sn<=2).sum():>8}")
    print(f"{'Maxpool':<15} {sm.mean():>8.1f} {np.median(sm):>8.1f} {(sm==0).sum():>10}/{n_samples} {(sm<=2).sum():>8}")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(sn, bins=30, alpha=0.6, label="Nearest", color="blue")
    axes[0].hist(sm, bins=30, alpha=0.6, label="Maxpool", color="orange")
    axes[0].set_xlabel("Active patches (of 256)")
    axes[0].set_title("Patch count distribution at 16x16")
    axes[0].legend()

    axes[1].scatter(sn, sm, alpha=0.4, s=15)
    axes[1].plot([0, max(sm.max(), sn.max())], [0, max(sm.max(), sn.max())], 'r--', alpha=0.5)
    axes[1].set_xlabel("Nearest patches")
    axes[1].set_ylabel("Maxpool patches")
    axes[1].set_title(f"Nearest vs Maxpool\n(lost with nearest: {(sn==0).sum()}/{n_samples})")

    plt.suptitle("Nearest vs Maxpool Downsampling at CLIP 16x16 Resolution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/unet/nearest_vs_maxpool")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    save_dir = project_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    splits_dir = project_root / args.splits_dir
    data_root = project_root / args.data_root

    fig = example_real(splits_dir, data_root, n_samples=args.n_samples, save_dir=save_dir)
    print(f"Saved to {save_dir}/")
