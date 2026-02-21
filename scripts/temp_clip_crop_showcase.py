"""
Showcase CLIP cropping strategy across ~50 product classes.
10 random samples per class, showing:
- Original image with mask overlay + group bboxes
- The selected group's 224×224 CLIP crop with mask overlay (NO dilation)
- 16×16 CLIP patch grid showing which patches contain anomaly

One PNG per product class (2 rows × 10 cols).
"""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.crop_utils import get_component_groups, clip_crop_from_groups

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

COLORS = [
    (1, 0.2, 0.2), (0.2, 0.6, 1), (0.2, 0.9, 0.3), (1, 0.7, 0.1),
    (0.8, 0.2, 0.9), (0.1, 0.9, 0.9), (1, 0.4, 0.7), (0.6, 0.6, 0.2),
]


def mp16(m: torch.Tensor) -> torch.Tensor:
    """Maxpool mask to 16×16 CLIP patch grid."""
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[-1] != 224:
        m = F.interpolate(m.float(), size=(224, 224), mode="nearest")
    return (F.max_pool2d(m.float(), kernel_size=14).squeeze() > 0.5).float()


def main():
    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_clip_crop_showcase"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / "master_training.json", encoding="utf-8") as f:
        samples = json.load(f)

    # Group by product
    by_product = defaultdict(list)
    for s in samples:
        by_product[s.get("product", "?")].append(s)

    # All products with at least 1 sample, sorted by count descending
    eligible = [(p, ss) for p, ss in by_product.items() if len(ss) >= 1]
    eligible.sort(key=lambda x: -len(x[1]))
    selected = eligible

    print(f"Visualizing {len(selected)} products, 10 samples each")

    for pi, (product, prod_samples) in enumerate(selected):
        # Pick 10 random samples
        chosen = random.sample(prod_samples, min(10, len(prod_samples)))
        n = len(chosen)

        fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 7))
        if n == 1:
            axes = axes.reshape(2, 1)

        for si, s in enumerate(chosen):
            img_path = data_root / s["image_path"]
            mask_path = data_root / s["mask_path"]

            if not img_path.exists() or not mask_path.exists():
                for r in range(2):
                    axes[r, si].set_facecolor("black")
                    axes[r, si].axis("off")
                continue

            img_pil = Image.open(str(img_path)).convert("RGB")
            mask_pil = Image.open(str(mask_path)).convert("L")
            iw, ih = img_pil.size

            img_np = np.array(img_pil).astype(np.float32) / 255.0
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)

            if mask_np.sum() == 0:
                for r in range(2):
                    axes[r, si].imshow(img_np)
                    axes[r, si].set_title("no mask", fontsize=6)
                    axes[r, si].axis("off")
                continue

            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # [3, H, W]
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)     # [1, H, W]

            # Get groups
            groups = get_component_groups(mask_np, ih, iw)
            ng = len(groups)

            # Row 0: original with mask overlay + group bboxes
            axes[0, si].imshow(img_np)
            # Color-code each group's mask
            for gi, g in enumerate(groups):
                col = COLORS[gi % len(COLORS)]
                ov = np.zeros((ih, iw, 4), dtype=np.float32)
                ov[:, :, 0], ov[:, :, 1], ov[:, :, 2] = col
                ov[:, :, 3] = (g["mask"] > 0.5).astype(np.float32) * 0.5
                axes[0, si].imshow(ov)
                # Draw group bbox
                y0, y1, x0, x1 = g["bbox"]
                axes[0, si].add_patch(plt.Rectangle(
                    (x0, y0), x1 - x0 + 1, y1 - y0 + 1,
                    linewidth=1.5, edgecolor=col, facecolor="none", linestyle="--"))

            defect = s.get("defect_type", "?")
            axes[0, si].set_title(f"{ng}g | {defect}\n{ih}×{iw}", fontsize=5)
            axes[0, si].axis("off")

            # Row 1: CLIP crop (random group, 224×224, NO dilation)
            img_crop, mask_crop = clip_crop_from_groups(img_t, mask_t, groups, crop_size=224)
            crop_np = img_crop.permute(1, 2, 0).numpy()
            cmask_np = mask_crop.squeeze().numpy()

            axes[1, si].imshow(crop_np)
            # Mask overlay (red)
            ov_c = np.zeros((224, 224, 4), dtype=np.float32)
            ov_c[:, :, 0] = 1.0
            ov_c[:, :, 3] = (cmask_np > 0.5).astype(np.float32) * 0.5
            axes[1, si].imshow(ov_c)

            # 16×16 grid
            c16 = mp16(mask_crop)
            cell = 224 / 16
            nc16 = int(c16.sum())
            for rr in range(16):
                for cc in range(16):
                    if c16[rr, cc] > 0.5:
                        axes[1, si].add_patch(plt.Rectangle(
                            (cc * cell, rr * cell), cell, cell,
                            lw=0.4, ec="red", fc="none"))
            for j in range(17):
                axes[1, si].axhline(y=j * cell, color="white", lw=0.15, alpha=0.25)
                axes[1, si].axvline(x=j * cell, color="white", lw=0.15, alpha=0.25)

            # Find which group was selected (check which group's mask overlaps with crop)
            # Just show stats
            area_pct = 100 * cmask_np.sum() / (224 * 224)
            axes[1, si].set_title(f"{nc16}/256 patches\n{area_pct:.1f}% area", fontsize=5)
            axes[1, si].axis("off")

        plt.suptitle(f"[{pi+1}/{len(selected)}] {product} ({len(prod_samples)} samples)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0.01, 0.02, 1, 0.93])
        plt.savefig(save_dir / f"{pi:02d}_{product}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{pi+1}/{len(selected)}] {product} — {n} samples, {len(prod_samples)} total")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
