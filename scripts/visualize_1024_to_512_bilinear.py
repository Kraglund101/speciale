"""
Visualize mask downsampling from 1024x1024 to 512x512.

1x5 per sample:
  Col 0: Original 1024x1024
  Col 1: Nearest
  Col 2: Bilinear (threshold > 0.5)
  Col 3: Bilinear (threshold > 0.75)
  Col 4: Maxpool
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

MASK_COLOR = np.array([1.0, 0.2, 0.2])
MASK_ALPHA = 0.45

METHODS = [
    {"key": "nearest",      "label": "Nearest",            "color": "#4477AA"},
    {"key": "bilinear_050", "label": "Bilinear (t>=0.50)",  "color": "#66CCEE"},
    {"key": "bilinear_075", "label": "Bilinear (t>0.75)",  "color": "#228833"},
    {"key": "maxpool",      "label": "Maxpool",            "color": "#EE7733"},
]


def overlay_mask(ax, img_np: np.ndarray, mask_np: np.ndarray, darken_bg: bool = True):
    """Overlay mask on image with colored highlight and optional background darkening."""
    canvas = img_np.copy()
    if darken_bg:
        canvas[mask_np == 0] *= 0.7
    fg = mask_np > 0.5
    for c in range(3):
        canvas[fg, c] = canvas[fg, c] * (1 - MASK_ALPHA) + MASK_COLOR[c] * MASK_ALPHA
    ax.imshow(canvas)


def downsample_all(mask_1024: torch.Tensor) -> dict[str, torch.Tensor]:
    """Downsample a [1, 1024, 1024] binary mask to 512x512 with all methods."""
    m = mask_1024.unsqueeze(0)  # [1, 1, 1024, 1024]

    nearest = F.interpolate(m, size=(512, 512), mode="nearest").squeeze(0)
    nearest = (nearest > 0.5).float()

    bilinear = F.interpolate(m, size=(512, 512), mode="bilinear", align_corners=False).squeeze(0)
    bilinear_050 = (bilinear >= 0.50).float()
    bilinear_075 = (bilinear > 0.75).float()

    maxpool = F.max_pool2d(m, kernel_size=2).squeeze(0)
    maxpool = (maxpool > 0.5).float()

    return {
        "nearest": nearest,
        "bilinear_050": bilinear_050,
        "bilinear_075": bilinear_075,
        "maxpool": maxpool,
    }


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

    # Stats per method
    stats = {m["key"]: {"areas": [], "lost": 0} for m in METHODS}
    stats_orig = []

    print(f"Processing {n_samples} samples...")

    for i, s in enumerate(chosen):
        try:
            img_pil = Image.open(s["image"]).convert("RGB")
            mask_pil = Image.open(s["mask"]).convert("L")
        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue

        atype = s["type"]

        mask_1024 = to_tensor(mask_pil)  # [1, 1024, 1024]
        mask_1024 = (mask_1024 > 0.5).float()
        area_1024 = int(mask_1024.sum().item())
        if area_1024 == 0:
            continue

        stats_orig.append(area_1024)
        results = downsample_all(mask_1024)

        for m in METHODS:
            area = int(results[m["key"]].sum().item())
            stats[m["key"]]["areas"].append(area)
            if area == 0:
                stats[m["key"]]["lost"] += 1

        # --- Per-sample figure: 1x5 ---
        img_512 = img_pil.resize((512, 512), Image.LANCZOS)
        img_512_np = np.array(img_512).astype(np.float32) / 255.0

        mask_1024_display = np.array(mask_pil.resize((512, 512), Image.NEAREST)).astype(np.float32) / 255.0
        mask_1024_display = (mask_1024_display > 0.5).astype(np.float32)

        pct = area_1024 / (1024 * 1024) * 100
        expected = area_1024 / 4

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

        # Col 0: Original
        overlay_mask(axes[0], img_512_np, mask_1024_display)
        axes[0].set_title(f"Original 1024\u00d71024\n{atype} \u2014 {area_1024} px ({pct:.3f}%)", fontsize=13)
        axes[0].axis("off")

        # Cols 1-4: methods
        for col, m in enumerate(METHODS, start=1):
            mask_np = results[m["key"]].squeeze().numpy()
            area = int(results[m["key"]].sum().item())
            diff_pct = (area - expected) / expected * 100 if expected > 0 else 0

            overlay_mask(axes[col], img_512_np, mask_np)
            axes[col].set_title(f"{m['label']} \u2192 512\n{area} px ({diff_pct:+.1f}%)", fontsize=13)
            axes[col].axis("off")

            if area == 0:
                axes[col].text(0.5, 0.5, "LOST", transform=axes[col].transAxes,
                              ha="center", va="center", fontsize=20, color="white", fontweight="bold",
                              bbox=dict(boxstyle="round", fc="red", alpha=0.8))

        # Legend
        legend_patch = mpatches.Patch(color=MASK_COLOR, alpha=MASK_ALPHA + 0.2, label="Anomaly mask")
        fig.legend(handles=[legend_patch], loc="lower center", ncol=1, fontsize=13,
                   frameon=True, fancybox=True, borderpad=0.5)

        plt.suptitle(f"Sample {i} \u2014 1024 \u2192 512 Downsampling Comparison", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0.06, 1, 0.94])
        plt.savefig(save_dir / f"{i:04d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_samples} done")

    # --- Summary stats ---
    n = len(stats_orig)
    so = np.array(stats_orig, dtype=np.float64)
    expected_arr = so / 4

    print(f"\n{'='*75}")
    print(f"SUMMARY - 1024x1024 -> 512x512 Mask Downsampling ({n} samples)")
    print(f"{'='*75}")
    print(f"Original area: mean {so.mean():.0f} px, median {np.median(so):.0f} px\n")
    print(f"{'Method':<22} {'Mean area':>10} {'Lost':>10} {'Mean diff':>12} {'Mean diff %':>14}")
    print("-" * 70)

    method_arrays = {}
    for m in METHODS:
        arr = np.array(stats[m["key"]]["areas"], dtype=np.float64)
        method_arrays[m["key"]] = arr
        diff = arr - expected_arr
        diff_pct = diff / so * 100
        lost = stats[m["key"]]["lost"]
        print(f"{m['label']:<22} {arr.mean():>10.0f} {lost:>6}/{n:<3} {diff.mean():>+10.1f} px {diff_pct.mean():>+12.4f}%")

    # --- Summary plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for m in METHODS:
        arr = method_arrays[m["key"]]
        axes[0].scatter(expected_arr, arr, alpha=0.25, s=8, label=m["label"], color=m["color"])

    max_val = max(expected_arr.max(), max(a.max() for a in method_arrays.values()))
    axes[0].plot([0, max_val], [0, max_val], "k--", alpha=0.4, linewidth=1, label="Perfect (y=x)")
    axes[0].set_xlabel("Expected area at 512 (orig / 4)", fontsize=10)
    axes[0].set_ylabel("Actual area at 512", fontsize=10)
    axes[0].set_title("Area Preservation", fontsize=16, fontweight="bold")
    axes[0].legend(fontsize=8, frameon=True)

    for m in METHODS:
        diff = method_arrays[m["key"]] - expected_arr
        axes[1].hist(diff, bins=40, alpha=0.5, label=m["label"], color=m["color"])

    axes[1].axvline(x=0, color="black", linestyle="--", alpha=0.4, linewidth=1)
    axes[1].set_xlabel("Area difference (actual \u2212 expected) [px]", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Area Error Distribution", fontsize=16, fontweight="bold")
    axes[1].legend(fontsize=8, frameon=True)

    plt.suptitle(f"1024 \u2192 512 Downsampling: All Methods ({n} samples)",
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
    parser.add_argument("--save-dir", type=str, default="results/1024_to_512_bilinear")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_root = project_root / args.data_root
    splits_dir = project_root / args.splits_dir
    save_dir = project_root / args.save_dir

    visualize(data_root, splits_dir, save_dir, n_samples=args.n_samples)
