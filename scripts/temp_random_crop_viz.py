"""
Visualize the new independent CLIP + UNet cropping strategy.

CLIP crop (→ 224x224):
  - If bbox+10%pad <= 224: crop=224, random feasible position
  - Else: log-uniform S in [bbox+10%pad, img_dim], random feasible position, resize to 224

UNet crop (→ 512x512):
  - If img <= 512: native (no crop)
  - Else: log-uniform S in [512, img_dim], random feasible position, resize to 512

"Random feasible position" = top-left sampled uniformly over ALL positions
that keep the padded bbox fully inside the crop. No center bias.

Row 0: CLIP crops at 224 with core (red) + dilation ring (blue)
Row 1: CLIP 16x16 patch grid
Row 2: UNet crops at 512 with mask overlay
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
N_REALIZATIONS = 6  # number of random crops to show per sample


def load_samples(master_json, data_root, n=50, product_filter=None):
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


def get_bbox(mask_np):
    """Return (y_min, y_max, x_min, x_max), bbox_side, area. All in pixel coords."""
    ys, xs = np.where(mask_np > 0.5)
    if len(ys) == 0:
        h, w = mask_np.shape
        return (h // 2, h // 2, w // 2, w // 2), 0, 0
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    area = int((mask_np > 0.5).sum())
    return (y_min, y_max, x_min, x_max), side, area


def pad_bbox(bbox, pad_frac, img_h, img_w):
    """Add fractional padding to bbox, clamp to image bounds. Returns padded (y0, y1, x0, x1)."""
    y_min, y_max, x_min, x_max = bbox
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    pad = max(1, int(side * pad_frac))
    return (
        max(0, y_min - pad),
        min(img_h - 1, y_max + pad),
        max(0, x_min - pad),
        min(img_w - 1, x_max + pad),
    )


def padded_bbox_size(pbbox):
    """Square side needed to contain the padded bbox."""
    y0, y1, x0, x1 = pbbox
    return max(y1 - y0 + 1, x1 - x0 + 1)


def sample_log_uniform(lo, hi):
    """Sample from log-uniform distribution in [lo, hi]."""
    if lo >= hi:
        return lo
    log_lo, log_hi = math.log(lo), math.log(hi)
    return int(round(math.exp(random.uniform(log_lo, log_hi))))


def sample_feasible_position(crop_s, pbbox, img_h, img_w):
    """
    Sample crop top-left uniformly over ALL feasible positions that keep
    the padded bbox fully inside the square crop.

    Returns (y0, x0).
    """
    py0, py1, px0, px1 = pbbox
    crop_s = min(crop_s, min(img_h, img_w))

    # y0 must satisfy: y0 <= py0  AND  y0 + crop_s > py1  AND  0 <= y0  AND  y0 + crop_s <= img_h
    y0_max = min(py0, img_h - crop_s)
    y0_min = max(0, py1 - crop_s + 1)

    x0_max = min(px0, img_w - crop_s)
    x0_min = max(0, px1 - crop_s + 1)

    if y0_min > y0_max or x0_min > x0_max:
        # Fallback: center on padded bbox
        cy = (py0 + py1) // 2
        cx = (px0 + px1) // 2
        y0 = max(0, min(cy - crop_s // 2, img_h - crop_s))
        x0 = max(0, min(cx - crop_s // 2, img_w - crop_s))
        return y0, x0

    y0 = random.randint(y0_min, y0_max)
    x0 = random.randint(x0_min, x0_max)
    return y0, x0


def do_crop_resize(img_t, mask_t, crop_s, target, pbbox, img_h, img_w):
    """Crop square of size crop_s at random feasible position, resize to target."""
    y0, x0 = sample_feasible_position(crop_s, pbbox, img_h, img_w)
    crop_s = min(crop_s, min(img_h, img_w))

    img_c = img_t[:, y0:y0+crop_s, x0:x0+crop_s]
    mask_c = mask_t[:, y0:y0+crop_s, x0:x0+crop_s]

    img_out = F.interpolate(img_c.unsqueeze(0), size=(target, target),
                            mode="bilinear", align_corners=False).squeeze(0)
    mask_out = F.interpolate(mask_c.unsqueeze(0).float(), size=(target, target),
                             mode="nearest").squeeze(0)
    return img_out, mask_out, y0, x0, crop_s


def downsample_mask_maxpool_16(mask):
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(0)
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
    ax.set_title(title, fontsize=7)
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
    ax.set_title(f"{title}\n{n_c}c+{n_r}r={n_c+n_r}/256", fontsize=7)
    ax.axis("off")


def draw_unet_overlay(ax, img_np, mask_np, title):
    ax.imshow(img_np)
    h, w = mask_np.shape
    ov = np.zeros((h, w, 4), dtype=np.float32)
    ov[:, :, 0] = 1.0
    ov[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.5
    ax.imshow(ov)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    master_json = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_random_crop_viz"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(master_json, data_root, n=50, product_filter="mint")
    print(f"Loaded {len(samples)} mint samples")

    for i, sample in enumerate(samples):
        img_pil = Image.open(sample["image_path"]).convert("RGB")
        mask_pil = Image.open(sample["mask_path"]).convert("L")

        # Work at native resolution (1024x1024 for RealIAD)
        img_w, img_h = img_pil.size
        mask_np_raw = np.array(mask_pil)
        mask_pil = Image.fromarray(((mask_np_raw > 127).astype(np.uint8) * 255), mode="L")

        atype = sample["anomaly_type"]
        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)
        mask_t = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255).unsqueeze(0)

        bbox_raw, bbox_side, area = get_bbox(mask_t.squeeze(0).numpy())
        pbbox = pad_bbox(bbox_raw, 0.10, img_h, img_w)
        pb_size = padded_bbox_size(pbbox)

        # --- CLIP crop parameters ---
        clip_target = 224
        if pb_size <= clip_target:
            clip_fixed = True
            clip_s_min = clip_target
            clip_s_max = clip_target
        else:
            clip_fixed = False
            clip_s_min = pb_size
            clip_s_max = min(img_h, img_w)

        # --- UNet crop parameters ---
        unet_target = 512
        if min(img_h, img_w) <= unet_target:
            unet_fixed = True
            unet_s_min = min(img_h, img_w)
            unet_s_max = unet_s_min
        else:
            unet_fixed = False
            unet_s_min = unet_target
            unet_s_max = min(img_h, img_w)

        # --- Generate realizations ---
        n_cols = 1 + N_REALIZATIONS  # original + N crops
        fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 12.5))
        fig.subplots_adjust(hspace=0.5)

        # Column 0: Original image
        orig_np = img_t.permute(1, 2, 0).numpy()
        orig_mask_np = mask_t.squeeze(0).numpy()
        pct = area / (img_h * img_w) * 100

        # Draw original with bbox rectangle
        axes[0, 0].imshow(orig_np)
        py0, py1, px0, px1 = pbbox
        rect = plt.Rectangle((px0, py0), px1-px0+1, py1-py0+1,
                              linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title(f"Original {img_w}x{img_h}\n"
                              f"bbox={bbox_side}px, pad={pb_size}px\n"
                              f"area={area} ({pct:.2f}%)", fontsize=7)
        axes[0, 0].axis("off")

        # Row 1 col 0: mask
        draw_unet_overlay(axes[1, 0], orig_np, orig_mask_np,
                          f"Mask @{img_h}\nCLIP: S in [{clip_s_min},{clip_s_max}]"
                          + (" FIXED" if clip_fixed else " log-U"))
        # Row 2 col 0: mask again for UNet reference
        draw_unet_overlay(axes[2, 0], orig_np, orig_mask_np,
                          f"Mask @{img_h}\nUNet: S in [{unet_s_min},{unet_s_max}]"
                          + (" NATIVE" if unet_fixed else " log-U"))

        for j in range(N_REALIZATIONS):
            col = j + 1

            # --- CLIP crop ---
            if clip_fixed:
                cs_clip = clip_target
            else:
                cs_clip = sample_log_uniform(clip_s_min, clip_s_max)

            img_224, core_224, cy0, cx0, cs_actual = do_crop_resize(
                img_t, mask_t, cs_clip, clip_target, pbbox, img_h, img_w)

            # Dilate after crop+resize
            r_224 = adaptive_dilation_radius(core_224, min_r=1, max_r=10)
            dil_224 = dilate_mask(core_224, r_224)
            ring_224 = torch.clamp(dil_224 - core_224, 0, 1)

            img_clip_np = img_224.permute(1, 2, 0).numpy()
            core_np = core_224.squeeze(0).numpy()
            ring_np = ring_224.squeeze(0).numpy()

            core_16 = downsample_mask_maxpool_16(core_224)
            dil_16 = downsample_mask_maxpool_16(dil_224)

            zoom = img_h / cs_actual
            area_224 = int((core_np > 0.5).sum())

            draw_overlay(axes[0, col], img_clip_np, core_np, ring_np,
                         f"CLIP S={cs_actual} ({zoom:.1f}x)\n"
                         f"pos=({cy0},{cx0}) r={r_224}\n"
                         f"area={area_224}px")
            draw_patches(axes[1, col], img_clip_np, core_16, dil_16,
                         f"CLIP S={cs_actual} r={r_224}")

            # --- UNet crop ---
            if unet_fixed:
                cs_unet = unet_s_min
                # Just resize the whole image
                img_512 = F.interpolate(img_t.unsqueeze(0), size=(unet_target, unet_target),
                                        mode="bilinear", align_corners=False).squeeze(0)
                mask_512 = F.interpolate(mask_t.unsqueeze(0).float(), size=(unet_target, unet_target),
                                         mode="nearest").squeeze(0)
                uy0, ux0 = 0, 0
                cs_unet_actual = min(img_h, img_w)
            else:
                cs_unet = sample_log_uniform(unet_s_min, unet_s_max)
                img_512, mask_512, uy0, ux0, cs_unet_actual = do_crop_resize(
                    img_t, mask_t, cs_unet, unet_target, pbbox, img_h, img_w)

            img_unet_np = img_512.permute(1, 2, 0).numpy()
            mask_unet_np = mask_512.squeeze(0).numpy()
            unet_zoom = img_h / cs_unet_actual
            mask_area_512 = int((mask_unet_np > 0.5).sum())

            draw_unet_overlay(axes[2, col], img_unet_np, mask_unet_np,
                              f"UNet S={cs_unet_actual} ({unet_zoom:.1f}x)\n"
                              f"pos=({uy0},{ux0})\n"
                              f"mask={mask_area_512}px @512")

        # Labels
        axes[0, 0].set_ylabel("CLIP 224\noverlay", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=65, va="center")
        axes[1, 0].set_ylabel("CLIP 224\n16x16 grid", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=65, va="center")
        axes[2, 0].set_ylabel("UNet 512\nmask", fontsize=9, fontweight="bold",
                               rotation=0, labelpad=65, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core anomaly"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring (r=1-10 @224)"),
            mpatches.Patch(edgecolor='lime', facecolor='none', linewidth=1.5,
                           linestyle='--', label="Bbox + 10% pad"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=8, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 bbox={bbox_side}px, padded={pb_size}px, "
            f"area={area}px @{img_h}\n"
            f"CLIP: {'fixed 224' if clip_fixed else f'log-U [{clip_s_min},{clip_s_max}]'}"
            f" | UNet: {'native' if unet_fixed else f'log-U [{unet_s_min},{unet_s_max}]'}"
            f" | Uniform position (no center bias)",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout(rect=[0.07, 0.04, 1, 0.90])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype} "
              f"(bbox={bbox_side}, pad={pb_size}, "
              f"CLIP:[{clip_s_min},{clip_s_max}], "
              f"UNet:[{unet_s_min},{unet_s_max}])")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
