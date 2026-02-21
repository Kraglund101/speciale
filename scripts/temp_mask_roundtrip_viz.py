"""
Mask roundtrip visualization: maxpool to 64×64 → upscale back to 512×512.

Shows exactly which pixels in the 512×512 image the UNet will modify.

For 30 samples:
  Col 0: Original 512×512 image with GT mask (red)
  Col 1: GT mask → maxpool to 64×64 → nearest upscale to 512 (red = expanded area, green = GT boundary)
  Col 2: Stats overlay

Red  = area UNet will modify (from 64×64 roundtrip)
Green = original GT mask boundary
The red area OUTSIDE the green boundary = pixels modified beyond the GT mask.
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).parent.parent))
random.seed(42)


def main():
    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "unet" / "mask_roundtrip_512"

    with open(data_root / "master_training.json", encoding="utf-8") as f:
        samples = json.load(f)

    # Pick 30 random samples with non-empty masks
    random.shuffle(samples)
    chosen = []
    for s in samples:
        mask_path = data_root / s["mask_path"]
        if mask_path.exists():
            m = np.array(Image.open(str(mask_path)).convert("L"))
            if m.max() > 127:
                chosen.append(s)
                if len(chosen) == 30:
                    break

    for si, s in enumerate(chosen):
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]

        img_pil = Image.open(str(img_path)).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(mask_path)).convert("L").resize((512, 512), Image.NEAREST)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

        # Maxpool to 64×64
        mask_64 = F.max_pool2d(mask_t, kernel_size=8)  # [1,1,64,64]
        # Upscale back to 512×512 (nearest = blocky, shows exact cells)
        mask_roundtrip = F.interpolate(mask_64, size=(512, 512), mode="nearest")
        mask_rt_np = mask_roundtrip.squeeze().numpy()

        gt_area = mask_np.sum()
        rt_area = (mask_rt_np > 0.5).sum()
        expansion = rt_area / max(gt_area, 1)
        overflow_area = ((mask_rt_np > 0.5) & (mask_np < 0.5)).sum()
        overflow_pct = 100 * overflow_area / max(rt_area, 1)

        active_cells = int((mask_64.squeeze() > 0.5).sum().item())
        coverage_pct = 100 * gt_area / (512 * 512)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Col 0: Original + GT mask
        axes[0].imshow(img_np)
        ov = np.zeros((512, 512, 4), dtype=np.float32)
        ov[:, :, 0] = 1.0
        ov[:, :, 3] = mask_np * 0.5
        axes[0].imshow(ov)
        axes[0].set_title(f"GT mask @ 512×512\n{int(gt_area)} px ({coverage_pct:.2f}%)", fontsize=9)
        axes[0].axis("off")

        # Col 1: Roundtrip overlay
        axes[1].imshow(img_np)

        # Red = roundtrip expanded area
        ov_rt = np.zeros((512, 512, 4), dtype=np.float32)
        ov_rt[:, :, 0] = 1.0
        ov_rt[:, :, 3] = (mask_rt_np > 0.5).astype(np.float32) * 0.35

        # Green = GT mask (so you see what's original vs expanded)
        ov_gt = np.zeros((512, 512, 4), dtype=np.float32)
        ov_gt[:, :, 1] = 1.0
        ov_gt[:, :, 3] = mask_np * 0.4

        axes[1].imshow(ov_rt)
        axes[1].imshow(ov_gt)

        # Draw 8×8 grid in the region of interest
        ys, xs = np.where(mask_rt_np > 0.5)
        if len(ys) > 0:
            margin = 32
            y_min, y_max = max(0, ys.min() - margin), min(511, ys.max() + margin)
            x_min, x_max = max(0, xs.min() - margin), min(511, xs.max() + margin)
            for j in range(0, 513, 8):
                if y_min <= j <= y_max:
                    axes[1].axhline(y=j, xmin=x_min/512, xmax=x_max/512,
                                    color="white", lw=0.3, alpha=0.4)
                if x_min <= j <= x_max:
                    axes[1].axvline(x=j, ymin=1-y_max/512, ymax=1-y_min/512,
                                    color="white", lw=0.3, alpha=0.4)

        axes[1].set_title(
            f"64×64 roundtrip → 512×512\n{int(rt_area)} px ({active_cells} latent cells)",
            fontsize=9,
        )
        axes[1].axis("off")

        # Col 2: Difference — pixels OUTSIDE GT but INSIDE roundtrip
        axes[2].imshow(img_np)

        # Show three zones:
        # Yellow = overflow (modified but not in GT mask)
        # Green = correctly covered (in both GT and roundtrip)
        # Nothing = not modified
        both = (mask_rt_np > 0.5) & (mask_np > 0.5)  # correctly covered
        overflow = (mask_rt_np > 0.5) & (mask_np < 0.5)  # expanded beyond GT

        ov_both = np.zeros((512, 512, 4), dtype=np.float32)
        ov_both[:, :, 1] = 1.0  # green
        ov_both[:, :, 3] = both.astype(np.float32) * 0.5

        ov_over = np.zeros((512, 512, 4), dtype=np.float32)
        ov_over[:, :, 0] = 1.0  # red/yellow
        ov_over[:, :, 1] = 0.7
        ov_over[:, :, 3] = overflow.astype(np.float32) * 0.5

        axes[2].imshow(ov_both)
        axes[2].imshow(ov_over)
        axes[2].set_title(
            f"Expansion: {expansion:.1f}×\n"
            f"Overflow: {int(overflow_area)} px ({overflow_pct:.0f}% of modified area)",
            fontsize=9,
        )
        axes[2].legend(
            handles=[
                Patch(facecolor=(0, 1, 0, 0.5), label="GT anomaly (correctly covered)"),
                Patch(facecolor=(1, 0.7, 0, 0.5), label="Overflow (modified beyond GT)"),
            ],
            fontsize=6, loc="lower right",
        )
        axes[2].axis("off")

        defect = s.get("defect_type", "?")
        product = s.get("product", "?")
        plt.suptitle(
            f"Sample {si} — {product} / {defect}\n"
            f"64×64 maxpool roundtrip: what the UNet actually modifies vs GT mask",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.savefig(save_dir / f"{si:04d}_{defect}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{si+1}/30] {product}/{defect} — {expansion:.1f}× expansion, {overflow_pct:.0f}% overflow")

    print(f"\nSaved 30 samples to {save_dir}/")


if __name__ == "__main__":
    main()
