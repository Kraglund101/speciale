"""Quick viz: latent-dilated masks downsampled through UNet resolutions."""
import json, random, math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).parent.parent
DATA_BASE = ROOT / "anomverse_extension" / "datasets" / "full_training_dataset"

def dilate_latent(mask_64, radius):
    if radius <= 0:
        return mask_64
    dilated = mask_64
    for _ in range(radius):
        dilated = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
    return (dilated > 0.5).float()

def stratified_sample(samples, n_target, rng):
    by_product = defaultdict(list)
    for i, s in enumerate(samples):
        by_product[s["product"]].append(i)
    products = sorted(by_product.keys())
    base = n_target // len(products)
    remainder = n_target - base * len(products)
    shuffled = list(products); rng.shuffle(shuffled)
    selected = []
    for j, prod in enumerate(shuffled):
        count = base + (1 if j < remainder else 0)
        pool = by_product[prod]
        selected.extend(rng.sample(pool, min(count, len(pool))))
    if len(selected) < n_target:
        used = set(selected)
        rem = [i for i in range(len(samples)) if i not in used]
        rng.shuffle(rem)
        selected.extend(rem[:n_target - len(selected)])
    rng.shuffle(selected)
    return selected[:n_target]

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--save-dir", default="results/unet/downsample_viz")
    args = parser.parse_args()

    save_dir = ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(123)

    with open(DATA_BASE / "master_training.json", encoding="utf-8") as f:
        all_samples = json.load(f)
    all_samples = [s for s in all_samples if s["dataset"] == "realiad"]
    valid = [s for s in all_samples if (DATA_BASE / s["mask_path"]).exists()]
    indices = stratified_sample(valid, args.n_samples, rng)

    resolutions = [64, 32, 16, 8]
    dilation_variants = [
        (0, "Core only"),
        (1, "1px dilation"),
        (2, "2px dilation"),
    ]

    for si, idx in enumerate(indices):
        s = valid[idx]
        product, defect = s["product"], s.get("defect_type", "?")

        img_pil = Image.open(str(DATA_BASE / s["image_path"])).convert("RGB").resize((512, 512), Image.BILINEAR)
        mask_pil = Image.open(str(DATA_BASE / s["mask_path"])).convert("L").resize((512, 512), Image.NEAREST)
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        area_px = int(mask_np.sum())
        if area_px == 0:
            continue
        coverage = 100 * area_px / (512 * 512)

        # Core at 64x64
        core_64 = F.max_pool2d(mask_t, kernel_size=8)

        # 3 rows (variants) x 4 cols (resolutions)
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))

        for ri, (radius, label) in enumerate(dilation_variants):
            dilated_64 = dilate_latent(core_64, radius)
            core_np_64 = core_64.squeeze().numpy()
            dil_np_64 = dilated_64.squeeze().numpy()

            for ci, res in enumerate(resolutions):
                ax = axes[ri, ci]

                if res == 64:
                    core_at_res = core_np_64
                    dil_at_res = dil_np_64
                else:
                    # Maxpool from 64 to lower res
                    pool_k = 64 // res
                    core_down = F.max_pool2d(core_64, kernel_size=pool_k)
                    dil_down = F.max_pool2d(dilated_64, kernel_size=pool_k)
                    core_at_res = core_down.squeeze().numpy()
                    dil_at_res = dil_down.squeeze().numpy()

                ring_at_res = ((dil_at_res > 0.5) & (core_at_res < 0.5)).astype(np.float32)
                core_cells = int((core_at_res > 0.5).sum())
                ring_cells = int((ring_at_res > 0.5).sum())

                # Build cell image at native res
                img_small = np.array(img_pil.resize((res, res), Image.BILINEAR)).astype(np.float32) / 255.0
                cell_img = img_small.copy()

                core_bool = core_at_res > 0.5
                ring_bool = ring_at_res > 0.5
                for c, val in enumerate([1.0, 0.0, 0.0]):
                    cell_img[:, :, c] = np.where(core_bool, img_small[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])
                for c, val in enumerate([0.2, 0.4, 1.0]):
                    cell_img[:, :, c] = np.where(ring_bool, img_small[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])

                ax.imshow(cell_img, interpolation="nearest", extent=[0, 512, 512, 0])

                # Grid
                cell_px = 512.0 / res
                for g in range(res + 1):
                    ax.axhline(y=g * cell_px, color="white", linewidth=0.2, alpha=0.35)
                    ax.axvline(x=g * cell_px, color="white", linewidth=0.2, alpha=0.35)

                ax.set_xlim(0, 512); ax.set_ylim(512, 0)
                ax.set_title(f"{res}x{res}\n{core_cells}c + {ring_cells}r", fontsize=8)
                ax.axis("off")

                # Row label on first column
                if ci == 0:
                    ax.set_ylabel(label, fontsize=10, fontweight="bold")

        legend_elements = [
            Patch(facecolor=(1, 0, 0), label="Core (anomaly)"),
            Patch(facecolor=(0.2, 0.4, 1), label="Dilation ring"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=8,
                   bbox_to_anchor=(0.5, -0.01))

        plt.suptitle(
            f"Sample {si} — {product} / {defect} — {coverage:.2f}% coverage\n"
            f"Maxpool downsampling through UNet resolutions (64→32→16→8)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        plt.savefig(save_dir / f"ds_{si:04d}_{product}_{defect}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  {si}: {product}/{defect} — core={int((core_np_64>0.5).sum())}c, cov={coverage:.2f}%")

    print(f"\nSaved to {save_dir}/")

if __name__ == "__main__":
    main()
