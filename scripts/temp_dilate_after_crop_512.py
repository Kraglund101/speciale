"""
Dilate AFTER crop+resize to 224 with random scale S — starting from 512x512.

Same as temp_dilate_after_crop.py but images are first resized:
  - Image: 1024 → 512 via bilinear
  - Mask:  1024 → 512 via maxpool (preserves small anomalies)

Then the exact same bbox-centered crop + dilate-after pipeline runs from 512.

S_min = max(224, bbox_side_at_512)
S_max = min(512, ...)  — same logic, just from 512 source

Row 0: 224x224 image + core (red) + ring (blue) overlay
Row 1: 16x16 CLIP patch grid
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


def load_samples(master_json, data_root, n=200, product_filter=None):
    with open(master_json, encoding="utf-8") as f:
        all_samples = json.load(f)
    samples = []
    for entry in all_samples:
        if product_filter and entry.get("product") != product_filter:
            continue
        img_path = data_root / entry["image_path"]
        mask_path = data_root / entry["mask_path"]
        if img_path.exists() and mask_path.exists():
            samples.append({
                "image_path": str(img_path),
                "mask_path": str(mask_path),
                "anomaly_type": entry.get("defect_type", "unknown"),
            })
    random.shuffle(samples)
    return samples[:n]


def resize_to_512(img_pil, mask_pil):
    """Resize image with bilinear and mask with maxpool to 512x512."""
    # Image: bilinear
    img_512 = img_pil.resize((512, 512), Image.BILINEAR)

    # Mask: maxpool (preserves small anomalies)
    mask_np = np.array(mask_pil).astype(np.float32) / 255.0
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1, 1, 1024, 1024]
    mask_512 = F.max_pool2d(mask_t, kernel_size=2)  # 1024 → 512
    mask_512 = (mask_512.squeeze() > 0.5).float()
    mask_512_np = (mask_512.numpy() * 255).astype(np.uint8)
    mask_512_pil = Image.fromarray(mask_512_np, mode="L")

    return img_512, mask_512_pil


def get_bbox_info(mask_np):
    """Return bbox center, bbox side (with 10% pad), and anomaly area."""
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0:
        h, w = mask_np.shape
        return (h // 2, w // 2), 0, 0
    cy = int((ys.min() + ys.max()) / 2)
    cx = int((xs.min() + xs.max()) / 2)
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    pad = max(1, int(side * 0.1))
    bbox_side = side + 2 * pad
    area = int((mask_np > 0.5).sum())
    return (cy, cx), bbox_side, area


def crop_and_resize(img_t, mask_t, crop_s, target, center):
    _, h, w = img_t.shape
    cy, cx = center
    crop_s = max(target, min(crop_s, min(h, w)))
    y0 = max(0, cy - crop_s // 2)
    x0 = max(0, cx - crop_s // 2)
    if y0 + crop_s > h: y0 = max(0, h - crop_s)
    if x0 + crop_s > w: x0 = max(0, w - crop_s)
    img_c = img_t[:, y0:y0+crop_s, x0:x0+crop_s]
    mask_c = mask_t[:, y0:y0+crop_s, x0:x0+crop_s]
    img_out = F.interpolate(img_c.unsqueeze(0), size=(target, target),
                            mode="bilinear", align_corners=False).squeeze(0)
    mask_out = F.interpolate(mask_c.unsqueeze(0).float(), size=(target, target),
                             mode="nearest").squeeze(0)
    return img_out, mask_out


def downsample_mask_maxpool_16(mask):
    if mask.dim() == 2: mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3: mask = mask.unsqueeze(0)
    if mask.shape[-1] != 224:
        mask = F.interpolate(mask.float(), size=(224, 224), mode="nearest")
    return (F.max_pool2d(mask.float(), kernel_size=14).squeeze() > 0.5).float()


def draw_overlay(ax, img_np, core_np, ring_np, title):
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
    ax.imshow(img_np)
    gs = 16
    cell = img_np.shape[0] / gs
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
    n_c, n_r = int(c.sum()), int(r.sum())
    ax.set_title(f"{title}\n{n_c}c+{n_r}r={n_c+n_r}/256", fontsize=8)
    ax.axis("off")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    master_json = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_dilate_after_crop_512_mint"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(master_json, data_root, n=50, product_filter="mint")

    for i, sample in enumerate(samples):
        # Load at 1024
        img_pil_1024 = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask_pil_1024 = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_raw_1024 = np.array(mask_pil_1024)
        mask_pil_1024 = Image.fromarray(((mask_raw_1024 > 127).astype(np.uint8) * 255), mode="L")

        # Resize to 512: bilinear for image, maxpool for mask
        img_pil, mask_pil = resize_to_512(img_pil_1024, mask_pil_1024)

        atype = sample["anomaly_type"]

        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255).unsqueeze(0)
        center, bbox, area_512 = get_bbox_info(mask_t.squeeze(0).numpy())

        # Also get 1024 stats for reference
        mask_1024_np = np.array(mask_pil_1024).astype(np.float32) / 255
        _, bbox_1024, area_1024 = get_bbox_info(mask_1024_np)

        # S_min = max(224, bbox_side_at_512)
        s_min = max(224, bbox)
        # S_max = 512 (image is 512, can't crop larger)
        s_max = 512

        if s_min > s_max:
            s_max = s_min  # bbox > 512 at 512 scale (unlikely but handle it)

        # Pick evenly spaced valid crop sizes
        if s_min == s_max:
            crop_sizes = [s_min]
        else:
            n_steps = min(6, s_max - s_min + 1)
            crop_sizes = [int(s_min + (s_max - s_min) * k / (n_steps - 1)) for k in range(n_steps)]

        # +1 column for original image
        n_cols = 1 + len(crop_sizes)
        fig, axes = plt.subplots(2, n_cols, figsize=(3.8 * n_cols, 10))
        if n_cols == 1:
            axes = axes.reshape(2, 1)
        fig.subplots_adjust(hspace=0.45)

        # Column 0: 512 image with mask overlay
        orig_np = img_t.permute(1, 2, 0).numpy()
        orig_mask_np = mask_t.squeeze(0).numpy()
        pct_512 = area_512 / (512 * 512) * 100
        draw_overlay(axes[0, 0], orig_np, orig_mask_np,
                     np.zeros_like(orig_mask_np),
                     f"512x512 (from 1024)\n"
                     f"area={area_512}px ({pct_512:.2f}%)\n"
                     f"bbox={bbox}px (was {bbox_1024}@1024)")
        # Row 1, col 0: mask overlay
        axes[1, 0].imshow(orig_np)
        ov_m = np.zeros((*orig_mask_np.shape, 4), dtype=np.float32)
        ov_m[:, :, 0] = 1.0
        ov_m[:, :, 3] = (orig_mask_np > 0.5).astype(np.float32) * 0.5
        axes[1, 0].imshow(ov_m)
        axes[1, 0].set_title(f"Mask @512\n{area_512}px ({pct_512:.2f}%)", fontsize=8)
        axes[1, 0].axis("off")

        for j, cs in enumerate(crop_sizes):
            col = j + 1
            img_224, core_224 = crop_and_resize(img_t, mask_t, cs, 224, center)

            # Dilate AFTER at 224
            r_224 = adaptive_dilation_radius(core_224, min_r=1, max_r=10)
            dil_224 = dilate_mask(core_224, r_224)
            ring_224 = torch.clamp(dil_224 - core_224, 0, 1)

            img_np = img_224.permute(1, 2, 0).numpy()
            core_np = core_224.squeeze(0).numpy()
            ring_np = ring_224.squeeze(0).numpy()

            core_16 = downsample_mask_maxpool_16(core_224)
            dil_16 = downsample_mask_maxpool_16(dil_224)

            zoom = 512 / cs
            area_224 = int((core_np > 0.5).sum())
            pct = area_224 / core_np.size * 100
            raw_r = 0.5 * (area_224 ** 0.5)

            draw_overlay(axes[0, col], img_np, core_np, ring_np,
                         f"S={cs} ({zoom:.1f}x zoom)\n"
                         f"area={area_224}px  0.5\u00b7\u221a{area_224}={raw_r:.1f} \u2192 r={r_224}\n"
                         f"{pct:.2f}% coverage")
            draw_patches(axes[1, col], img_np, core_16, dil_16,
                         f"S={cs} \u2022 r={r_224} @224")

        axes[0, 0].set_ylabel("512 source /\n224\u00d7224 overlay", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[1, 0].set_ylabel("512 source /\n16\u00d716 patches", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=80, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring (r=1\u201310 @224)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 512 source (bilinear img, maxpool mask from 1024)\n"
            f"bbox={bbox}px @512 (was {bbox_1024}@1024), area={area_512}px @512 ({area_1024}@1024)\n"
            f"S in [{s_min}, {s_max}]",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0.08, 0.04, 1, 0.88])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype} (bbox={bbox}@512 was {bbox_1024}@1024, S in [{s_min},{s_max}])")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
