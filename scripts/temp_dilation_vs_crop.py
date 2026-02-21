"""
Test: dilation before vs after random scale crop for CLIP pathway.

Shows at different crop sizes S:
  Row 0: Dilate at 1024 (r=2-10) THEN crop S→224 (ring may vanish at large S)
  Row 1: Crop S→224 THEN dilate at 224 (r=2-10) (ring always visible)

Also shows the 16x16 CLIP patch grid to see actual token coverage.
"""
import sys
import json
import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

SEED = 42


def load_samples(splits_dir, data_root, n=6):
    samples = []
    for json_path in sorted(splits_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        atype = json_path.stem
        for entry in data.get("samples", []):
            img_path = data_root / entry["image_path"]
            mask_path = data_root / entry["mask_path"]
            if img_path.exists() and mask_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "anomaly_type": atype,
                })
    random.shuffle(samples)
    return samples[:n]


def get_anomaly_center(mask_np):
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0:
        return mask_np.shape[0] // 2, mask_np.shape[1] // 2
    return int(ys.mean()), int(xs.mean())


def crop_and_resize(img_t, mask_t, crop_s, target, center):
    """Crop crop_s centered on center, resize to target. Tensors [C,H,W] or [1,H,W]."""
    _, h, w = img_t.shape
    cy, cx = center
    crop_s = max(target, min(crop_s, min(h, w)))

    y0 = max(0, cy - crop_s // 2)
    x0 = max(0, cx - crop_s // 2)
    if y0 + crop_s > h:
        y0 = max(0, h - crop_s)
    if x0 + crop_s > w:
        x0 = max(0, w - crop_s)

    img_c = img_t[:, y0:y0+crop_s, x0:x0+crop_s]
    mask_c = mask_t[:, y0:y0+crop_s, x0:x0+crop_s]

    img_out = F.interpolate(img_c.unsqueeze(0), size=(target, target),
                            mode="bilinear", align_corners=False).squeeze(0)
    mask_out = F.interpolate(mask_c.unsqueeze(0).float(), size=(target, target),
                             mode="nearest").squeeze(0)
    return img_out, mask_out


def downsample_mask_maxpool_16(mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
    # Ensure 224x224
    if mask.shape[-1] != 224:
        mask = F.interpolate(mask.float(), size=(224, 224), mode="nearest")
    pooled = F.max_pool2d(mask.float(), kernel_size=14).squeeze()
    return (pooled > 0.5).float()


def draw_with_overlay(ax, img_np, core_np, ring_np, title):
    """Draw image with core (red) + ring (blue) overlay."""
    ax.imshow(img_np)
    h, w = core_np.shape
    ov_c = np.zeros((h, w, 4), dtype=np.float32)
    ov_c[:, :, 0] = 1.0
    ov_c[:, :, 3] = (core_np > 0.5).astype(np.float32) * 0.5
    ax.imshow(ov_c)
    ov_r = np.zeros((h, w, 4), dtype=np.float32)
    ov_r[:, :, 2] = 1.0
    ov_r[:, :, 3] = (ring_np > 0.5).astype(np.float32) * 0.4
    ax.imshow(ov_r)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def draw_patches(ax, img_np, core_16, dil_16, title):
    """Draw 16x16 patch grid with core (red) + ring (blue)."""
    ax.imshow(img_np)
    gs = 16
    cell = img_np.shape[0] / gs  # assume square
    c = core_16.numpy() if torch.is_tensor(core_16) else core_16
    d = dil_16.numpy() if torch.is_tensor(dil_16) else dil_16
    r = np.clip(d - c, 0, 1)

    for row in range(gs):
        for col in range(gs):
            if c[row, col] > 0.5:
                ax.add_patch(plt.Rectangle((col*cell, row*cell), cell, cell,
                             linewidth=0, facecolor='red', alpha=0.35))
            elif r[row, col] > 0.5:
                ax.add_patch(plt.Rectangle((col*cell, row*cell), cell, cell,
                             linewidth=0, facecolor='blue', alpha=0.3))

    for j in range(gs + 1):
        ax.axhline(y=j*cell, color='white', linewidth=0.3, alpha=0.4)
        ax.axvline(x=j*cell, color='white', linewidth=0.3, alpha=0.4)

    n_c = int(c.sum())
    n_r = int(r.sum())
    ax.set_title(f"{title}\n{n_c}c+{n_r}r={n_c+n_r}/256", fontsize=8)
    ax.axis("off")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "temp_dilation_vs_crop"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(splits_dir, data_root, n=6)
    crop_sizes = [224, 400, 600, 800, 1024]

    for i, sample in enumerate(samples):
        img_pil = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask_pil = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np_raw = np.array(mask_pil)
        mask_pil = Image.fromarray(((mask_np_raw > 127).astype(np.uint8) * 255), mode="L")
        atype = sample["anomaly_type"]

        # Convert to tensors
        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255).unsqueeze(0)

        center = get_anomaly_center(mask_t.squeeze(0).numpy())

        # Dilate at 1024
        r_1024 = adaptive_dilation_radius(mask_t, min_r=2, max_r=10)
        dilated_1024 = dilate_mask(mask_t, r_1024)

        n_cols = len(crop_sizes)
        fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 16))

        for j, cs in enumerate(crop_sizes):
            # === Approach A: Dilate at 1024, then crop ===
            img_224_a, core_224_a = crop_and_resize(img_t, mask_t, cs, 224, center)
            _, dil_224_a = crop_and_resize(img_t, dilated_1024, cs, 224, center)
            ring_224_a = torch.clamp(dil_224_a - core_224_a, 0, 1)

            img_np_a = img_224_a.permute(1, 2, 0).numpy()
            core_np_a = core_224_a.squeeze(0).numpy()
            ring_np_a = ring_224_a.squeeze(0).numpy()

            core_16_a = downsample_mask_maxpool_16(core_224_a)
            dil_16_a = downsample_mask_maxpool_16(dil_224_a)

            # === Approach B: Crop first, then dilate at 224 ===
            img_224_b, core_224_b = crop_and_resize(img_t, mask_t, cs, 224, center)
            r_224 = adaptive_dilation_radius(core_224_b, min_r=2, max_r=10)
            dil_224_b = dilate_mask(core_224_b, r_224)
            ring_224_b = torch.clamp(dil_224_b - core_224_b, 0, 1)

            img_np_b = img_224_b.permute(1, 2, 0).numpy()
            core_np_b = core_224_b.squeeze(0).numpy()
            ring_np_b = ring_224_b.squeeze(0).numpy()

            core_16_b = downsample_mask_maxpool_16(core_224_b)
            dil_16_b = downsample_mask_maxpool_16(dil_224_b)

            zoom = 1024 / cs

            # Row 0: Approach A — pixel overlay
            draw_with_overlay(axes[0, j], img_np_a, core_np_a, ring_np_a,
                              f"S={cs} ({zoom:.1f}x)\nDilate@1024 r={r_1024}, then crop")

            # Row 1: Approach A — 16x16 patches
            draw_patches(axes[1, j], img_np_a, core_16_a, dil_16_a,
                         f"Patches (dilate@1024)")

            # Row 2: Approach B — pixel overlay
            draw_with_overlay(axes[2, j], img_np_b, core_np_b, ring_np_b,
                              f"S={cs} ({zoom:.1f}x)\nCrop, then dilate@224 r={r_224}")

            # Row 3: Approach B — 16x16 patches
            draw_patches(axes[3, j], img_np_b, core_16_b, dil_16_b,
                         f"Patches (dilate@224)")

        axes[0, 0].set_ylabel("A: Dilate@1024\nthen crop\n(pixel)", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[1, 0].set_ylabel("A: Dilate@1024\n16x16 patches", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[2, 0].set_ylabel("B: Crop first\nthen dilate@224\n(pixel)", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[3, 0].set_ylabel("B: Crop first\n16x16 patches", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=80, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 Dilation order with random scale crop\n"
            f"A: dilate r={r_1024} at 1024 then crop  vs  B: crop then dilate at 224",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0.1, 0.03, 1, 0.93])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype} (r@1024={r_1024})")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
