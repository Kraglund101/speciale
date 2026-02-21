"""
Visualize per-component CLIP cropping for disjoint anomaly masks.

Strategy:
  - Connected components via scipy.ndimage.label
  - If all components fit in a single 224 crop (padded bbox <= 224): single crop
  - Else: separate 224 crop per component
    - Per-component: if comp_bbox+10%pad <= 224: crop=224, uniform random position
    - Else: log-uniform S in [comp_bbox+10%pad, img_dim], uniform random position, resize to 224

Shows:
  Row 0: Original image with all components color-coded + overall bbox
  Row 1+: One 224x224 crop per component (with dilation overlay + 16x16 grid)
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
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

SEED = 42
COMP_COLORS = [
    (1.0, 0.2, 0.2),   # red
    (0.2, 0.6, 1.0),   # blue
    (0.2, 0.9, 0.3),   # green
    (1.0, 0.7, 0.1),   # orange
    (0.8, 0.2, 0.9),   # purple
    (0.1, 0.9, 0.9),   # cyan
    (1.0, 0.4, 0.7),   # pink
    (0.6, 0.6, 0.2),   # olive
]


def load_samples(master_json, data_root, product_filter=None):
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
    return samples


def get_comp_bbox(comp_mask):
    """Return (y_min, y_max, x_min, x_max), side, area for a single component mask."""
    ys, xs = np.where(comp_mask > 0.5)
    if len(ys) == 0:
        return None, 0, 0
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    area = int(comp_mask.sum())
    return (y_min, y_max, x_min, x_max), side, area


def pad_bbox(bbox, pad_frac, img_h, img_w):
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
    y0, y1, x0, x1 = pbbox
    return max(y1 - y0 + 1, x1 - x0 + 1)


def sample_log_uniform(lo, hi):
    if lo >= hi:
        return lo
    return int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))


def sample_feasible_position(crop_s, pbbox, img_h, img_w):
    py0, py1, px0, px1 = pbbox
    crop_s = min(crop_s, min(img_h, img_w))
    y0_max = min(py0, img_h - crop_s)
    y0_min = max(0, py1 - crop_s + 1)
    x0_max = min(px0, img_w - crop_s)
    x0_min = max(0, px1 - crop_s + 1)
    if y0_min > y0_max or x0_min > x0_max:
        cy = (py0 + py1) // 2
        cx = (px0 + px1) // 2
        y0 = max(0, min(cy - crop_s // 2, img_h - crop_s))
        x0 = max(0, min(cx - crop_s // 2, img_w - crop_s))
        return y0, x0
    return random.randint(y0_min, y0_max), random.randint(x0_min, x0_max)


def crop_and_resize(img_t, mask_t, crop_s, target, pbbox, img_h, img_w):
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


def draw_patches_overlay(ax, img_np, core_224, dil_224, title):
    """Draw 224x224 image with core/ring overlay AND 16x16 grid."""
    ax.imshow(img_np)
    core_np = core_224.squeeze().numpy() if torch.is_tensor(core_224) else core_224.squeeze()
    dil_np = dil_224.squeeze().numpy() if torch.is_tensor(dil_224) else dil_224.squeeze()
    ring_np = np.clip(dil_np - core_np, 0, 1)
    h, w = core_np.shape

    # Overlay core + ring
    ov_c = np.zeros((h, w, 4), dtype=np.float32)
    ov_c[:, :, 0] = 1.0
    ov_c[:, :, 3] = (core_np > 0.5).astype(np.float32) * 0.45
    ax.imshow(ov_c)
    ov_r = np.zeros((h, w, 4), dtype=np.float32)
    ov_r[:, :, 2] = 1.0
    ov_r[:, :, 3] = (ring_np > 0.5).astype(np.float32) * 0.35
    ax.imshow(ov_r)

    # 16x16 grid
    core_16 = downsample_mask_maxpool_16(core_224)
    dil_16 = downsample_mask_maxpool_16(dil_224)
    gs = 16
    cell = h / gs
    c16 = core_16.numpy()
    d16 = dil_16.numpy()
    r16 = np.clip(d16 - c16, 0, 1)
    for row in range(gs):
        for col in range(gs):
            if c16[row, col] > 0.5:
                ax.add_patch(plt.Rectangle((col*cell, row*cell), cell, cell,
                             linewidth=0.5, edgecolor='red', facecolor='none'))
            elif r16[row, col] > 0.5:
                ax.add_patch(plt.Rectangle((col*cell, row*cell), cell, cell,
                             linewidth=0.5, edgecolor='blue', facecolor='none'))
    for j in range(gs + 1):
        ax.axhline(y=j*cell, color='white', linewidth=0.2, alpha=0.3)
        ax.axvline(x=j*cell, color='white', linewidth=0.2, alpha=0.3)

    n_c, n_r = int(c16.sum()), int(r16.sum())
    ax.set_title(f"{title}\n{n_c}c+{n_r}r={n_c+n_r}/256", fontsize=7)
    ax.axis("off")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    master_json = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_disjoint_crop_viz"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(master_json, data_root, product_filter="mint")

    # Filter to disjoint samples that need splitting
    disjoint_samples = []
    for s in all_samples:
        mask = np.array(Image.open(s["mask_path"]).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue
        labeled, n_comp = ndimage.label(mask_bin)
        if n_comp < 2:
            continue
        # Check if overall bbox fits 224
        ys, xs = np.where(mask_bin > 0)
        bbox_side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
        padded = bbox_side + max(1, int(bbox_side * 0.1)) * 2
        if padded <= 224:
            continue  # fits in single crop, not interesting
        disjoint_samples.append(s)

    random.shuffle(disjoint_samples)
    disjoint_samples = disjoint_samples[:50]
    print(f"Found {len(disjoint_samples)} disjoint mint samples that need splitting")

    for i, sample in enumerate(disjoint_samples):
        img_pil = Image.open(sample["image_path"]).convert("RGB")
        mask_pil = Image.open(sample["mask_path"]).convert("L")
        img_w, img_h = img_pil.size
        atype = sample["anomaly_type"]

        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        labeled, n_comp = ndimage.label(mask_np)

        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)

        # Get per-component info
        components = []
        for c in range(1, n_comp + 1):
            comp_mask = (labeled == c).astype(np.float32)
            bbox, side, area = get_comp_bbox(comp_mask)
            if bbox is None:
                continue
            pbbox = pad_bbox(bbox, 0.10, img_h, img_w)
            pb_size = padded_bbox_size(pbbox)
            components.append({
                "id": c,
                "mask": comp_mask,
                "bbox": bbox,
                "side": side,
                "area": area,
                "pbbox": pbbox,
                "pb_size": pb_size,
            })

        # Sort by area descending
        components.sort(key=lambda x: x["area"], reverse=True)

        # Layout: 2 rows x (1 + n_comp) columns
        # Row 0: original + per-component crops (overlay + grid combined)
        # Row 1: original mask colored + per-component individual masks at 224
        max_cols = min(1 + len(components), 9)  # cap at 8 components shown
        n_show = max_cols - 1
        fig, axes = plt.subplots(2, max_cols, figsize=(3.2 * max_cols, 8))
        if max_cols == 1:
            axes = axes.reshape(2, 1)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        # Column 0: Original with colored components + bbox rectangles
        orig_np = img_t.permute(1, 2, 0).numpy()
        axes[0, 0].imshow(orig_np)

        # Overlay each component with distinct color
        for ci, comp in enumerate(components[:n_show]):
            color = COMP_COLORS[ci % len(COMP_COLORS)]
            ov = np.zeros((img_h, img_w, 4), dtype=np.float32)
            ov[:, :, 0] = color[0]
            ov[:, :, 1] = color[1]
            ov[:, :, 2] = color[2]
            ov[:, :, 3] = (comp["mask"] > 0.5).astype(np.float32) * 0.5
            axes[0, 0].imshow(ov)
            # Draw padded bbox rectangle
            py0, py1, px0, px1 = comp["pbbox"]
            rect = plt.Rectangle((px0, py0), px1-px0+1, py1-py0+1,
                                  linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--')
            axes[0, 0].add_patch(rect)

        overall_ys, overall_xs = np.where(mask_np > 0.5)
        overall_bbox = max(overall_ys.max()-overall_ys.min()+1, overall_xs.max()-overall_xs.min()+1)
        overall_area = int(mask_np.sum())
        axes[0, 0].set_title(f"Original {img_w}x{img_h}\n"
                              f"{n_comp} components, bbox={overall_bbox}\n"
                              f"total area={overall_area}", fontsize=7)
        axes[0, 0].axis("off")

        # Row 1, col 0: summary stats
        axes[1, 0].imshow(orig_np)
        ov_all = np.zeros((img_h, img_w, 4), dtype=np.float32)
        ov_all[:, :, 0] = 1.0
        ov_all[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.5
        axes[1, 0].imshow(ov_all)
        comp_summary = ", ".join(f"{c['side']}px" for c in components[:n_show])
        axes[1, 0].set_title(f"Full mask\ncomp sizes: [{comp_summary}]", fontsize=7)
        axes[1, 0].axis("off")

        # Per-component crops
        for ci, comp in enumerate(components[:n_show]):
            col = ci + 1
            comp_mask_t = torch.from_numpy(comp["mask"]).unsqueeze(0)
            pb_size = comp["pb_size"]

            # CLIP crop strategy
            clip_target = 224
            if pb_size <= clip_target:
                cs = clip_target
                mode_str = "fixed 224"
            else:
                cs = sample_log_uniform(pb_size, min(img_h, img_w))
                mode_str = f"log-U [{pb_size},{min(img_h,img_w)}]"

            img_224, mask_224, y0, x0, cs_actual = crop_and_resize(
                img_t, comp_mask_t, cs, clip_target, comp["pbbox"], img_h, img_w)

            # Dilate after
            r_224 = adaptive_dilation_radius(mask_224, min_r=1, max_r=10)
            dil_224 = dilate_mask(mask_224, r_224)

            img_clip_np = img_224.permute(1, 2, 0).numpy()
            area_224 = int((mask_224.squeeze().numpy() > 0.5).sum())
            zoom = img_h / cs_actual
            color = COMP_COLORS[ci % len(COMP_COLORS)]

            draw_patches_overlay(axes[0, col], img_clip_np, mask_224, dil_224,
                                 f"C{comp['id']} S={cs_actual} ({zoom:.1f}x)\n"
                                 f"pos=({y0},{x0}) r={r_224}")

            # Row 1: same crop but show full mask (all components) at this crop location
            # to see what else is visible
            full_mask_t = torch.from_numpy(mask_np).unsqueeze(0)
            _, full_mask_crop, _, _, _ = crop_and_resize(
                img_t, full_mask_t, cs_actual, clip_target,
                comp["pbbox"], img_h, img_w)
            # Hack: reuse exact position by temporarily replacing sample_feasible_position
            # Instead just show the component mask info
            axes[1, col].imshow(img_clip_np)
            m_np = mask_224.squeeze().numpy()
            ov = np.zeros((clip_target, clip_target, 4), dtype=np.float32)
            ov[:, :, 0] = color[0]
            ov[:, :, 1] = color[1]
            ov[:, :, 2] = color[2]
            ov[:, :, 3] = (m_np > 0.5).astype(np.float32) * 0.6
            axes[1, col].imshow(ov)
            axes[1, col].set_title(f"C{comp['id']}: {comp['side']}px, "
                                    f"area={comp['area']}\n"
                                    f"padded={pb_size} | {mode_str}",
                                    fontsize=7)
            axes[1, col].axis("off")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring"),
        ]
        for ci in range(min(n_show, len(COMP_COLORS))):
            legend_handles.append(
                mpatches.Patch(color=COMP_COLORS[ci], alpha=0.6,
                               label=f"Component {ci+1}"))
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 6), fontsize=7,
                   frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {n_comp} disjoint components, "
            f"overall bbox={overall_bbox}px (>{224} \u2192 split)\n"
            f"Each component gets its own 224x224 CLIP crop",
            fontsize=9, fontweight="bold",
        )
        plt.tight_layout(rect=[0.02, 0.05, 1, 0.90])
        plt.savefig(save_dir / f"{i:02d}_{atype}_n{n_comp}.png", dpi=120, bbox_inches="tight")
        plt.close()
        comp_str = "+".join(str(c["side"]) for c in components[:n_show])
        print(f"  [{i+1}/{len(disjoint_samples)}] {atype} "
              f"n={n_comp} bbox={overall_bbox} comps=[{comp_str}]")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
