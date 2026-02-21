"""
Recreate mask downsampling visualization.

5 columns per sample:
  Col 0: 512×512 with dilated mask (red=core, blue=ring)
  Col 1: 64×64 maxpooled (shown at 512 scale)
  Col 2: 32×32 maxpooled
  Col 3: 16×16 maxpooled
  Col 4: 8×8 maxpooled

Red = core anomaly, Blue = dilation ring, Gray = normal.
Dilation at 512 with r = clamp(0.5·sqrt(area), 8, 16).
"""
import sys
import json
import random
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import ndimage

random.seed(42)
np.random.seed(42)


def adaptive_dilation_radius(mask_np, min_r=8, max_r=16):
    """r = clamp(0.5 * sqrt(area), min_r, max_r)"""
    area = mask_np.sum()
    r = 0.5 * math.sqrt(area)
    return int(max(min_r, min(max_r, round(r))))


def dilate_mask(mask_np, radius):
    """Binary dilation with disk structuring element."""
    from scipy.ndimage import binary_dilation
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    struct = (x**2 + y**2) <= radius**2
    return binary_dilation(mask_np > 0.5, structure=struct).astype(np.float32)


def main():
    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "unet" / "downsampling"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / "master_training.json", encoding="utf-8") as f:
        samples = json.load(f)

    # Pick 30 random samples with non-empty masks (same seed as original)
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

    resolutions = [64, 32, 16, 8]

    for si, s in enumerate(chosen):
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]

        img_pil = Image.open(str(img_path)).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(mask_path)).convert("L").resize((512, 512), Image.NEAREST)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

        coverage_pct = 100 * mask_np.sum() / (512 * 512)
        r = adaptive_dilation_radius(mask_np, min_r=8, max_r=16)

        # Dilated mask at 512
        dilated_np = dilate_mask(mask_np, r)
        ring_np = (dilated_np > 0.5) & (mask_np < 0.5)  # ring = dilated minus core
        dilated_area = int(dilated_np.sum())

        # Dilated mask tensor for downsampling
        dilated_t = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(0)

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        defect = s.get("defect_type", "?")

        # Roundtrip CORE mask: maxpool 512→64→512
        core_64 = F.max_pool2d(mask_t, kernel_size=8)
        core_rt = F.interpolate(core_64, size=(512, 512), mode="nearest").squeeze().numpy()
        gt_area = mask_np.sum()
        core_rt_area = (core_rt > 0.5).sum()
        core_expansion = core_rt_area / max(gt_area, 1)
        core_overflow = (core_rt > 0.5) & (mask_np < 0.5)

        # Roundtrip DILATED mask: maxpool 512→64→512
        dil_64 = F.max_pool2d(dilated_t, kernel_size=8)
        dil_rt = F.interpolate(dil_64, size=(512, 512), mode="nearest").squeeze().numpy()
        dil_area = dilated_np.sum()
        dil_rt_area = (dil_rt > 0.5).sum()
        dil_expansion = dil_rt_area / max(dil_area, 1)
        dil_overflow = (dil_rt > 0.5) & (dilated_np < 0.5)  # beyond dilated mask

        # Col 0: flat segmentation map — each pixel gets ONE color
        # Priority (inner to outer):
        #   1. Red    = GT anomaly
        #   2. Yellow = core roundtrip overflow (8×8 rounding of core, within dilation)
        #   3. Blue   = dilation ring (not touched by core roundtrip)
        #   4. Green  = dilated roundtrip overflow (8×8 rounding of dilated, beyond dilation)
        #   5. Image  = untouched
        zone = np.zeros((512, 512), dtype=np.int32)
        zone[dil_rt > 0.5] = 4       # green: dilated RT overflow
        zone[dilated_np > 0.5] = 3    # blue: dilation ring + core
        zone[core_rt > 0.5] = 2       # yellow: core RT overflow (overwrites some blue)
        zone[mask_np > 0.5] = 1       # red: GT core (overwrites yellow)

        # Build colored overlay
        colors = {
            1: (1.0, 0.0, 0.0),   # red
            2: (1.0, 0.8, 0.0),   # yellow
            3: (0.2, 0.4, 1.0),   # blue
            4: (0.0, 0.9, 0.0),   # green
        }
        overlay = img_np.copy()
        for z, col in colors.items():
            mask_z = (zone == z)
            for c in range(3):
                overlay[:, :, c] = np.where(mask_z, img_np[:, :, c] * 0.4 + col[c] * 0.6, overlay[:, :, c])

        axes[0].imshow(overlay)
        axes[0].set_title(
            f"512×512 | dilation r={r}\n"
            f"Core RT: {core_expansion:.1f}× | Dilated RT: {dil_expansion:.1f}×",
            fontsize=7,
        )
        axes[0].axis("off")

        # Cols 1-4: maxpooled at each resolution
        for ri, res in enumerate(resolutions):
            pool_k = 512 // res
            total_cells = res * res

            # Maxpool core and dilated separately
            core_down = F.max_pool2d(mask_t, kernel_size=pool_k)
            dilated_down = F.max_pool2d(dilated_t, kernel_size=pool_k)

            core_cells = int((core_down.squeeze() > 0.5).sum().item())
            dilated_cells = int((dilated_down.squeeze() > 0.5).sum().item())
            ring_cells = dilated_cells - core_cells

            # Upscale to 512 for display
            core_up = F.interpolate(core_down, size=(512, 512), mode="nearest").squeeze().numpy()
            dilated_up = F.interpolate(dilated_down, size=(512, 512), mode="nearest").squeeze().numpy()
            ring_up = (dilated_up > 0.5) & (core_up < 0.5)

            axes[ri + 1].imshow(img_np)
            # Core (red)
            ov_c = np.zeros((512, 512, 4), dtype=np.float32)
            ov_c[:, :, 0] = 1.0
            ov_c[:, :, 3] = (core_up > 0.5).astype(np.float32) * 0.6
            axes[ri + 1].imshow(ov_c)
            # Ring (blue)
            ov_r = np.zeros((512, 512, 4), dtype=np.float32)
            ov_r[:, :, 2] = 1.0
            ov_r[:, :, 3] = ring_up.astype(np.float32) * 0.6
            axes[ri + 1].imshow(ov_r)

            axes[ri + 1].set_title(
                f"{res}×{res}\n{core_cells}c + {ring_cells}r / {total_cells}",
                fontsize=9,
            )
            axes[ri + 1].axis("off")

        # Legend
        legend_elements = [
            Patch(facecolor=(1, 0, 0), label="GT anomaly"),
            Patch(facecolor=(1, 0.8, 0), label="Core RT overflow (8×8 rounding beyond GT)"),
            Patch(facecolor=(0.2, 0.4, 1), label="Dilation ring"),
            Patch(facecolor=(0, 0.9, 0), label="Dilated RT overflow (8×8 rounding beyond dilation)"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=6.5,
                   bbox_to_anchor=(0.5, -0.02))

        plt.suptitle(
            f"Sample {si} — {defect} — {coverage_pct:.2f}% coverage\n"
            f"Maxpool downsampling \u2022 Dilation at 512 (r=clamp(0.5\u00b7\u221aarea, 8, 16) = {r})",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.88])
        plt.savefig(save_dir / f"roundtrip_{si:04d}_{defect}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{si+1}/30] {defect} — {coverage_pct:.2f}% coverage, r={r}")

    # Summary image: expansion stats
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ri, res in enumerate(resolutions):
        expansions_core = []
        expansions_dilated = []
        for s in chosen:
            mask_path = data_root / s["mask_path"]
            mask_pil = Image.open(str(mask_path)).convert("L").resize((512, 512), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

            r = adaptive_dilation_radius(mask_np)
            dilated_np = dilate_mask(mask_np, r)
            dilated_t = torch.from_numpy(dilated_np).unsqueeze(0).unsqueeze(0)

            pool_k = 512 // res
            core_down = F.max_pool2d(mask_t, kernel_size=pool_k)
            dilated_down = F.max_pool2d(dilated_t, kernel_size=pool_k)

            gt_area = mask_np.sum()
            if gt_area > 0:
                core_up = F.interpolate(core_down, size=(512, 512), mode="nearest").squeeze().numpy()
                dilated_up = F.interpolate(dilated_down, size=(512, 512), mode="nearest").squeeze().numpy()
                expansions_core.append((core_up > 0.5).sum() / gt_area)
                expansions_dilated.append((dilated_up > 0.5).sum() / gt_area)

        axes[ri].bar(["Core", "Dilated"], [np.median(expansions_core), np.median(expansions_dilated)],
                     color=["red", "blue"], alpha=0.6)
        axes[ri].set_title(f"{res}×{res}", fontsize=10)
        axes[ri].set_ylabel("Median expansion ×")
        axes[ri].axhline(y=1.0, color="gray", ls="--", lw=0.8)

    plt.suptitle("Median expansion factor (roundtrip area / GT area) across 30 samples",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_dir / "summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved 30 + summary to {save_dir}/")


if __name__ == "__main__":
    main()
