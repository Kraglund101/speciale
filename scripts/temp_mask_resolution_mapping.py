"""
Visualize mask at each UNet resolution mapped back to 512×512.

Shows for 20 random samples:
  Row 0: Original image (512×512) with GT mask overlay
  Row 1: Mask @ 64×64 → upscaled to 512 (1 cell = 8×8 px)
  Row 2: Mask @ 32×32 → upscaled to 512 (1 cell = 16×16 px)
  Row 3: Mask @ 16×16 → upscaled to 512 (1 cell = 32×32 px)
  Row 4: Mask @ 8×8  → upscaled to 512 (1 cell = 64×64 px)

Each row shows the blocky upscaled mask overlaid on the image,
so you can see exactly what area each resolution "covers."
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

random.seed(42)

def main():
    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_mask_resolution_mapping"
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / "master_training.json", encoding="utf-8") as f:
        samples = json.load(f)

    # Pick 20 random samples with non-empty masks
    random.shuffle(samples)
    chosen = []
    for s in samples:
        mask_path = data_root / s["mask_path"]
        if mask_path.exists():
            m = np.array(Image.open(str(mask_path)).convert("L"))
            if m.max() > 127:
                chosen.append(s)
                if len(chosen) == 20:
                    break

    resolutions = [64, 32, 16, 8]
    res_labels = [
        "64×64\n1 cell = 8×8 px\n(16×16 @ 1024)",
        "32×32\n1 cell = 16×16 px\n(32×32 @ 1024)",
        "16×16\n1 cell = 32×32 px\n(64×64 @ 1024)",
        "8×8\n1 cell = 64×64 px\n(128×128 @ 1024)",
    ]

    n = len(chosen)
    fig, axes = plt.subplots(5, n, figsize=(2.5 * n, 12))

    for si, s in enumerate(chosen):
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]

        img_pil = Image.open(str(img_path)).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(mask_path)).convert("L").resize((512, 512), Image.NEAREST)

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)

        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1,1,512,512]

        # Row 0: original + GT mask
        axes[0, si].imshow(img_np)
        ov = np.zeros((512, 512, 4), dtype=np.float32)
        ov[:, :, 0] = 1.0
        ov[:, :, 3] = mask_np * 0.5
        axes[0, si].imshow(ov)
        area_pct = 100 * mask_np.sum() / (512 * 512)
        defect = s.get("defect_type", "?")
        axes[0, si].set_title(f"{defect}\n{area_pct:.2f}%", fontsize=5)
        axes[0, si].axis("off")

        # Rows 1-4: downsampled via maxpool, upscaled back to 512
        for ri, res in enumerate(resolutions):
            pool_k = 512 // res
            mask_down = F.max_pool2d(mask_t, kernel_size=pool_k)  # [1,1,res,res]

            # Upscale back to 512 with nearest (shows blocky cells)
            mask_up = F.interpolate(mask_down, size=(512, 512), mode="nearest")
            mask_up_np = mask_up.squeeze().numpy()

            active_cells = int((mask_down.squeeze() > 0.5).sum().item())
            total_cells = res * res

            axes[ri + 1, si].imshow(img_np)
            ov = np.zeros((512, 512, 4), dtype=np.float32)
            ov[:, :, 0] = 1.0
            ov[:, :, 3] = (mask_up_np > 0.5).astype(np.float32) * 0.4

            # Also show cell grid
            cell_px = 512 // res
            # Draw GT mask boundary in green
            ov_gt = np.zeros((512, 512, 4), dtype=np.float32)
            ov_gt[:, :, 1] = 1.0
            ov_gt[:, :, 3] = mask_np * 0.3
            axes[ri + 1, si].imshow(ov)
            axes[ri + 1, si].imshow(ov_gt)

            # Grid lines
            for j in range(0, 513, cell_px):
                axes[ri + 1, si].axhline(y=j, color="white", lw=0.15, alpha=0.3)
                axes[ri + 1, si].axvline(x=j, color="white", lw=0.15, alpha=0.3)

            # Expansion ratio
            gt_area = mask_np.sum()
            up_area = (mask_up_np > 0.5).sum()
            expansion = up_area / max(gt_area, 1)

            axes[ri + 1, si].set_title(
                f"{active_cells}/{total_cells} cells\n{expansion:.1f}× expansion",
                fontsize=5,
            )
            axes[ri + 1, si].axis("off")

    # Row labels
    row_labels = ["GT mask (512)"] + res_labels
    for ri, label in enumerate(row_labels):
        axes[ri, 0].set_ylabel(label, fontsize=6, rotation=0, labelpad=60, va="center")

    plt.suptitle("Mask resolution mapping: what area does each cell cover?", fontsize=10, fontweight="bold")
    plt.tight_layout(rect=[0.08, 0.01, 1, 0.95])
    plt.savefig(save_dir / "resolution_mapping_20samples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved to {save_dir}/resolution_mapping_20samples.png")

    # Also print summary stats across all 20 samples
    print("\nExpansion stats (maxpool roundtrip area / GT area):")
    for res in resolutions:
        expansions = []
        for s in chosen:
            mask_path = data_root / s["mask_path"]
            mask_pil = Image.open(str(mask_path)).convert("L").resize((512, 512), Image.NEAREST)
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            pool_k = 512 // res
            mask_down = F.max_pool2d(mask_t, kernel_size=pool_k)
            mask_up = F.interpolate(mask_down, size=(512, 512), mode="nearest")
            gt_area = mask_np.sum()
            up_area = (mask_up.squeeze().numpy() > 0.5).sum()
            if gt_area > 0:
                expansions.append(up_area / gt_area)
        print(f"  {res}×{res}: mean {np.mean(expansions):.1f}×, "
              f"median {np.median(expansions):.1f}×, "
              f"max {np.max(expansions):.1f}×")


if __name__ == "__main__":
    main()
