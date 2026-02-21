"""Visualize the new CLIP patch-space dilation strategy.

For each sample shows:
  Col 0: Original image + mask overlay
  Col 1: 16x16 grid — maxpool only (before dilation)
  Col 2: 16x16 grid — maxpool + 1px dilation (current approach)
  Col 3: Diff — patches added by dilation (yellow = new)

Picks diverse samples: tiny masks, medium, large, multi-group.
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
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_ROOT = Path(r"C:\Users\frede\desktop\kandidat\speciale\anomverse_extension\datasets\full_training_dataset")
SPLITS_DIR = DATA_ROOT / "splits_by_type"
OUT_PATH = Path(__file__).parent / "temp_clip_patch_dilation.png"

# Target resolution for CLIP input (ViT-H/14 = 14x14 patches on 224, but our images
# may be 512x512 or 1024x1024; the 16x16 grid comes from the actual spatial dims
# of CLIP hidden states which is input_size/patch_size)
GRID = 16


def load_diverse_samples(n=12):
    """Load samples with diverse mask sizes."""
    samples = []
    for jf in sorted(SPLITS_DIR.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        for entry in data.get("images", data if isinstance(data, list) else []):
            img_path = DATA_ROOT / entry["image_path"]
            mask_path = DATA_ROOT / entry["mask_path"]
            if img_path.exists() and mask_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "type": jf.stem,
                })
    random.shuffle(samples)

    # Bin by mask coverage: tiny (<0.5%), small (0.5-2%), medium (2-10%), large (>10%)
    bins = {"tiny": [], "small": [], "medium": [], "large": []}
    for s in samples:
        if all(len(b) >= 3 for b in bins.values()):
            break
        mask = np.array(Image.open(s["mask_path"]).convert("L"))
        coverage = (mask > 128).sum() / mask.size
        s["coverage"] = coverage
        if coverage < 0.005 and len(bins["tiny"]) < 3:
            bins["tiny"].append(s)
        elif 0.005 <= coverage < 0.02 and len(bins["small"]) < 3:
            bins["small"].append(s)
        elif 0.02 <= coverage < 0.10 and len(bins["medium"]) < 3:
            bins["medium"].append(s)
        elif coverage >= 0.10 and len(bins["large"]) < 3:
            bins["large"].append(s)

    result = []
    for cat in ["tiny", "small", "medium", "large"]:
        result.extend(bins[cat])
    return result[:n]


def mask_to_grid(mask_np, grid_size=GRID):
    """Downsample mask to grid via maxpool, return before and after 1px dilation."""
    h, w = mask_np.shape
    mask_t = torch.from_numpy((mask_np > 128).astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Maxpool downsample to grid
    kernel_h = h // grid_size
    kernel_w = w // grid_size
    # Crop to exact multiple
    crop_h = kernel_h * grid_size
    crop_w = kernel_w * grid_size
    mask_cropped = mask_t[:, :, :crop_h, :crop_w]
    grid_before = F.max_pool2d(mask_cropped, kernel_size=(kernel_h, kernel_w))  # [1,1,16,16]

    # 1px dilation in patch space
    grid_after = F.max_pool2d(grid_before, kernel_size=3, stride=1, padding=1)

    return (grid_before[0, 0].numpy() > 0.5).astype(np.uint8), \
           (grid_after[0, 0].numpy() > 0.5).astype(np.uint8)


def draw_grid(ax, grid, title, cmap="Blues", highlight_diff=None):
    """Draw a 16x16 grid with cell borders."""
    display = np.zeros((GRID, GRID, 3))
    for r in range(GRID):
        for c in range(GRID):
            if highlight_diff is not None and highlight_diff[r, c]:
                display[r, c] = [1.0, 0.9, 0.0]  # yellow = new from dilation
            elif grid[r, c]:
                display[r, c] = [0.2, 0.5, 1.0]  # blue = anomaly patch
            else:
                display[r, c] = [0.95, 0.95, 0.95]  # light gray = background

    ax.imshow(display, interpolation="nearest")
    # Grid lines
    for i in range(GRID + 1):
        ax.axhline(i - 0.5, color="gray", linewidth=0.3)
        ax.axvline(i - 0.5, color="gray", linewidth=0.3)

    n_patches = int(grid.sum())
    ax.set_title(f"{title}\n{n_patches} patches", fontsize=8)
    ax.axis("off")


def main():
    samples = load_diverse_samples(12)
    if not samples:
        print("No samples found!")
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for row, s in enumerate(samples):
        img = np.array(Image.open(s["image_path"]).convert("RGB").resize((512, 512)))
        mask_np = np.array(Image.open(s["mask_path"]).convert("L").resize((512, 512), Image.NEAREST))

        grid_before, grid_after = mask_to_grid(mask_np)
        diff = (grid_after > 0) & (grid_before == 0)

        # Col 0: image + mask overlay
        overlay = img.copy().astype(np.float32)
        mask_bool = mask_np > 128
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0] * 0.5 + 128, 0, 255)
        overlay[mask_bool, 1] *= 0.5
        overlay[mask_bool, 2] *= 0.5
        axes[row, 0].imshow(overlay.astype(np.uint8))
        coverage = s.get("coverage", mask_bool.sum() / mask_bool.size)
        axes[row, 0].set_title(f"{s['type']}\n{coverage:.2%} coverage", fontsize=8)
        axes[row, 0].axis("off")

        # Col 1: grid before dilation
        draw_grid(axes[row, 1], grid_before, "Maxpool only")

        # Col 2: grid after 1px dilation
        draw_grid(axes[row, 2], grid_after, "Maxpool + 1px dilate")

        # Col 3: diff
        draw_grid(axes[row, 3], grid_after, f"+{int(diff.sum())} patches from dilation",
                  highlight_diff=diff)

    fig.suptitle("CLIP Patch-Space Dilation: maxpool → 16x16 → 1px dilate", fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
