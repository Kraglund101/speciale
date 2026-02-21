"""
Visualize mask downsampling from 1024x1024 (RealIAD native) to 512x512 (SD 1.5).

1x3 per sample:
  Col 0: Original mask at 1024x1024 with image overlay
  Col 1: Nearest interpolation to 512x512
  Col 2: Maxpool to 512x512

Shows area stats and whether any anomaly pixels are lost.
"""
import sys
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms

SEED = 42

# Consistent colors
MASK_COLOR = np.array([1.0, 0.2, 0.2])  # Red for anomaly mask
MASK_ALPHA = 0.45


def overlay_mask(ax, img_np: np.ndarray, mask_np: np.ndarray, darken_bg: bool = True):
    """Overlay mask on image with colored highlight and optional background darkening."""
    h, w = img_np.shape[:2]
    canvas = img_np.copy()

    if darken_bg:
        # Slightly darken non-anomaly regions for contrast
        bg = (mask_np == 0)
        canvas[bg] *= 0.7

    # Blend mask color
    fg = mask_np > 0.5
    for c in range(3):
        canvas[fg, c] = canvas[fg, c] * (1 - MASK_ALPHA) + MASK_COLOR[c] * MASK_ALPHA

    ax.imshow(canvas)


def visualize(data_root: Path, splits_dir: Path, save_dir: Path, n_samples: int = 200):
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Collect all samples
    samples = []
    for jf in sorted(splits_dir.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        atype = jf.stem
        for s in data["samples"]:
            img_path = s.get("image_path")
            mask_path = s.get("mask_path")
            if img_path and mask_path:
                img_full = data_root / img_path if not Path(img_path).is_absolute() else Path(img_path)
                mask_full = data_root / mask_path if not Path(mask_path).is_absolute() else Path(mask_path)
                samples.append({"image": str(img_full), "mask": str(mask_full), "type": atype})

    n_samples = min(n_samples, len(samples))
    chosen = random.sample(samples, n_samples)

    to_tensor = transforms.ToTensor()
    stats_nearest_area = []
    stats_maxpool_area = []
    stats_orig_area = []
    stats_nearest_lost = 0
    stats_maxpool_lost = 0

    print(f"Processing {n_samples} samples...")

    for i, s in enumerate(chosen):
        try:
            img_pil = Image.open(s["image"]).convert("RGB")
            mask_pil = Image.open(s["mask"]).convert("L")
        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue

        atype = s["type"]

        # Original at 1024
        mask_1024 = to_tensor(mask_pil)  # [1, 1024, 1024]
        mask_1024 = (mask_1024 > 0.5).float()
        area_1024 = int(mask_1024.sum().item())
        stats_orig_area.append(area_1024)

        if area_1024 == 0:
            continue

        # Nearest to 512
        mask_nearest = F.interpolate(mask_1024.unsqueeze(0), size=(512, 512), mode="nearest").squeeze(0)
        mask_nearest = (mask_nearest > 0.5).float()
        area_nearest = int(mask_nearest.sum().item())
        stats_nearest_area.append(area_nearest)
        if area_nearest == 0:
            stats_nearest_lost += 1

        # Maxpool to 512
        mask_maxpool = F.max_pool2d(mask_1024.unsqueeze(0), kernel_size=2).squeeze(0)
        mask_maxpool = (mask_maxpool > 0.5).float()
        area_maxpool = int(mask_maxpool.sum().item())
        stats_maxpool_area.append(area_maxpool)
        if area_maxpool == 0:
            stats_maxpool_lost += 1

        # Downscale image to 512 for display
        img_512 = img_pil.resize((512, 512), Image.LANCZOS)
        img_512_np = np.array(img_512).astype(np.float32) / 255.0

        mask_nearest_np = mask_nearest.squeeze().numpy()
        mask_maxpool_np = mask_maxpool.squeeze().numpy()

        # Upscale 1024 mask to display size for overlay
        mask_1024_display = np.array(mask_pil.resize((512, 512), Image.NEAREST)).astype(np.float32) / 255.0
        mask_1024_display = (mask_1024_display > 0.5).astype(np.float32)

        pct = area_1024 / (1024 * 1024) * 100
        expected = area_1024 / 4

        fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

        # Col 0: Original 1024 (displayed at 512 with mask overlay)
        overlay_mask(axes[0], img_512_np, mask_1024_display)
        axes[0].set_title(f"Original 1024x1024\n{atype} — {area_1024} px ({pct:.3f}%)", fontsize=13)
        axes[0].axis("off")

        # Col 1: Nearest 512
        overlay_mask(axes[1], img_512_np, mask_nearest_np)
        diff_n_pct = (area_nearest - expected) / expected * 100 if expected > 0 else 0
        axes[1].set_title(f"Nearest \u2192 512x512\n{area_nearest} px ({diff_n_pct:+.1f}% vs expected)", fontsize=13)
        axes[1].axis("off")
        if area_nearest == 0:
            axes[1].text(0.5, 0.5, "LOST", transform=axes[1].transAxes,
                        ha="center", va="center", fontsize=24, color="white", fontweight="bold",
                        bbox=dict(boxstyle="round", fc="red", alpha=0.8))

        # Col 2: Maxpool 512
        overlay_mask(axes[2], img_512_np, mask_maxpool_np)
        diff_m_pct = (area_maxpool - expected) / expected * 100 if expected > 0 else 0
        axes[2].set_title(f"Maxpool \u2192 512x512\n{area_maxpool} px ({diff_m_pct:+.1f}% vs expected)", fontsize=13)
        axes[2].axis("off")

        # Legend
        legend_patch = mpatches.Patch(color=MASK_COLOR, alpha=MASK_ALPHA + 0.2, label="Anomaly mask")
        fig.legend(handles=[legend_patch], loc="lower center", ncol=1, fontsize=13,
                   frameon=True, fancybox=True, shadow=False, borderpad=0.6)

        plt.suptitle(f"Sample {i} — 1024 \u2192 512 Downsampling Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0.06, 1, 0.95])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # Summary
    n = len(stats_nearest_area)
    sn = np.array(stats_nearest_area)
    sm = np.array(stats_maxpool_area)
    so = np.array(stats_orig_area[:n])

    print(f"\n{'='*60}")
    print("SUMMARY — 1024x1024 -> 512x512 Mask Downsampling")
    print(f"{'='*60}")
    print(f"Samples: {n}")
    print(f"Original area (1024): mean {so.mean():.0f} px  median {np.median(so):.0f} px")
    print(f"")
    print(f"{'Method':<12} {'Mean area':>10} {'Lost':>10} {'Area diff':>12}")
    print("-" * 50)
    expected = so / 4  # expected area at 512 if perfect scaling
    print(f"{'Nearest':<12} {sn.mean():>10.0f} {stats_nearest_lost:>10}/{n} {(sn - expected).mean():>+12.1f} px")
    print(f"{'Maxpool':<12} {sm.mean():>10.0f} {stats_maxpool_lost:>10}/{n} {(sm - expected).mean():>+12.1f} px")

    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(expected, sn, alpha=0.3, s=10, label="Nearest", color="#4477AA")
    axes[0].scatter(expected, sm, alpha=0.3, s=10, label="Maxpool", color="#EE7733")
    max_val = max(expected.max(), sn.max(), sm.max())
    axes[0].plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1, label="Perfect (y=x)")
    axes[0].set_xlabel("Expected area at 512 (orig / 4)", fontsize=13)
    axes[0].set_ylabel("Actual area at 512", fontsize=13)
    axes[0].set_title("Area Preservation", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9, frameon=True)

    diff_n = sn - expected
    diff_m = sm - expected
    axes[1].hist(diff_n, bins=40, alpha=0.6, label="Nearest", color="#4477AA")
    axes[1].hist(diff_m, bins=40, alpha=0.6, label="Maxpool", color="#EE7733")
    axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    axes[1].set_xlabel("Area difference (actual \u2212 expected) [px]", fontsize=13)
    axes[1].set_ylabel("Count", fontsize=13)
    axes[1].set_title("Area Error Distribution", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9, frameon=True)

    plt.suptitle(f"1024 \u2192 512 Downsampling: Nearest vs Maxpool  ({n} samples)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--splits-dir", type=str, default="anomverse_extension/datasets/realiad_1024/concepts_10k")
    parser.add_argument("--data-root", type=str, default="anomverse_extension/datasets/realiad_1024")
    parser.add_argument("--save-dir", type=str, default="results/1024_to_512_comparison")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = project_root / args.data_root
    splits_dir = project_root / args.splits_dir
    save_dir = project_root / args.save_dir

    visualize(data_root, splits_dir, save_dir, n_samples=args.n_samples)
