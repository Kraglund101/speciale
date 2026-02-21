"""
Visualize grouped CLIP crops for disjoint anomaly masks.

For each sample:
  1. Find connected components
  2. Greedily group components into 224x224 crops:
     - Sort by y-center, then try adding each to an existing group
     - A component joins a group if the merged padded bbox <= 224
     - Otherwise start a new group
  3. Components whose individual padded bbox > 224 get their own log-uniform crop

Shows per sample:
  Row 0: Original with color-coded components + group bbox rectangles
  Row 1: One 224x224 crop per group (with dilation + 16x16 grid)

Prints distribution stats at the end.
"""
import sys
import json
import random
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask
from src.utils.crop_utils import square_bbox

SEED = 42
GROUP_COLORS = [
    (1.0, 0.2, 0.2),
    (0.2, 0.6, 1.0),
    (0.2, 0.9, 0.3),
    (1.0, 0.7, 0.1),
    (0.8, 0.2, 0.9),
    (0.1, 0.9, 0.9),
    (1.0, 0.4, 0.7),
    (0.6, 0.6, 0.2),
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
    ys, xs = np.where(comp_mask > 0.5)
    if len(ys) == 0:
        return None, 0, 0
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    area = int(comp_mask.sum())
    return (y_min, y_max, x_min, x_max), side, area


def merge_bboxes(bbox_a, bbox_b):
    return (
        min(bbox_a[0], bbox_b[0]),
        max(bbox_a[1], bbox_b[1]),
        min(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    )


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


def _crop_region_viz(bbox, pad_frac, max_crop, img_h, img_w):
    """Actual square crop region: max(padded_size, max_crop), centered."""
    padded = pad_bbox(bbox, pad_frac, img_h, img_w)
    crop_s = max(padded_bbox_size(padded), max_crop)
    cy = (padded[0] + padded[1]) / 2.0
    cx = (padded[2] + padded[3]) / 2.0
    ry0 = int(round(cy - (crop_s - 1) / 2.0))
    ry1 = ry0 + crop_s - 1
    rx0 = int(round(cx - (crop_s - 1) / 2.0))
    rx1 = rx0 + crop_s - 1
    if ry0 < 0:
        ry1 -= ry0; ry0 = 0
    if ry1 >= img_h:
        ry0 -= (ry1 - img_h + 1); ry1 = img_h - 1
    if rx0 < 0:
        rx1 -= rx0; rx0 = 0
    if rx1 >= img_w:
        rx0 -= (rx1 - img_w + 1); rx1 = img_w - 1
    return (max(0, ry0), ry1, max(0, rx0), rx1)


def group_components(components, img_h, img_w, max_crop=224, pad_frac=0.10):
    """
    Group components using rectangular bboxes. Squarification only at crop time.

    1. Greedy merge: combine if padded union max-side ≤ max_crop
    2. Containment: if group B's bbox is inside group A's crop region → absorb
       (crop region = square of max(padded_size, max_crop))
    """
    sorted_idxs = sorted(range(len(components)), key=lambda i: (
        (components[i]["bbox"][0] + components[i]["bbox"][1]) / 2
    ))

    groups = []

    for idx in sorted_idxs:
        comp = components[idx]
        placed = False
        for g in groups:
            candidate = merge_bboxes(g["merged_bbox"], comp["bbox"])
            padded = pad_bbox(candidate, pad_frac, img_h, img_w)
            if padded_bbox_size(padded) <= max_crop:
                g["comp_idxs"].append(idx)
                g["merged_bbox"] = candidate
                placed = True
                break
        if not placed:
            groups.append({
                "comp_idxs": [idx],
                "merged_bbox": comp["bbox"],
            })

    # Pass 1: merge groups whose combined padded bbox fits in max_crop
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(len(groups) - 1, i, -1):
                candidate = merge_bboxes(
                    groups[i]["merged_bbox"], groups[j]["merged_bbox"],
                )
                padded = pad_bbox(candidate, pad_frac, img_h, img_w)
                if padded_bbox_size(padded) <= max_crop:
                    groups[i]["comp_idxs"].extend(groups[j]["comp_idxs"])
                    groups[i]["merged_bbox"] = candidate
                    groups.pop(j)
                    changed = True
            if changed:
                break

    # Pass 2: containment — crop region = square of max(padded_size, max_crop)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            ri = _crop_region_viz(
                groups[i]["merged_bbox"], pad_frac, max_crop, img_h, img_w)
            for j in range(i + 1, len(groups)):
                rj = _crop_region_viz(
                    groups[j]["merged_bbox"], pad_frac, max_crop, img_h, img_w)
                bb_i = groups[i]["merged_bbox"]
                bb_j = groups[j]["merged_bbox"]
                if (bb_j[0] >= ri[0] and bb_j[1] <= ri[1]
                        and bb_j[2] >= ri[2] and bb_j[3] <= ri[3]):
                    groups[i]["comp_idxs"].extend(groups[j]["comp_idxs"])
                    groups[i]["merged_bbox"] = merge_bboxes(bb_i, bb_j)
                    groups.pop(j)
                    changed = True
                    break
                elif (bb_i[0] >= rj[0] and bb_i[1] <= rj[1]
                        and bb_i[2] >= rj[2] and bb_i[3] <= rj[3]):
                    groups[j]["comp_idxs"].extend(groups[i]["comp_idxs"])
                    groups[j]["merged_bbox"] = merge_bboxes(bb_i, bb_j)
                    groups.pop(i)
                    changed = True
                    break
            if changed:
                break

    # Pass 3: overlap — merge groups whose crop regions share any pixel
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            ri = _crop_region_viz(
                groups[i]["merged_bbox"], pad_frac, max_crop, img_h, img_w)
            for j in range(i + 1, len(groups)):
                rj = _crop_region_viz(
                    groups[j]["merged_bbox"], pad_frac, max_crop, img_h, img_w)
                if (ri[0] <= rj[1] and ri[1] >= rj[0]
                        and ri[2] <= rj[3] and ri[3] >= rj[2]):
                    groups[i]["comp_idxs"].extend(groups[j]["comp_idxs"])
                    groups[i]["merged_bbox"] = merge_bboxes(
                        groups[i]["merged_bbox"], groups[j]["merged_bbox"])
                    groups.pop(j)
                    changed = True
                    break
            if changed:
                break

    return groups


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
    ax.imshow(img_np)
    core_np = core_224.squeeze().numpy() if torch.is_tensor(core_224) else core_224.squeeze()
    dil_np = dil_224.squeeze().numpy() if torch.is_tensor(dil_224) else dil_224.squeeze()
    ring_np = np.clip(dil_np - core_np, 0, 1)
    h, w = core_np.shape

    ov_c = np.zeros((h, w, 4), dtype=np.float32)
    ov_c[:, :, 0] = 1.0
    ov_c[:, :, 3] = (core_np > 0.5).astype(np.float32) * 0.45
    ax.imshow(ov_c)
    ov_r = np.zeros((h, w, 4), dtype=np.float32)
    ov_r[:, :, 2] = 1.0
    ov_r[:, :, 3] = (ring_np > 0.5).astype(np.float32) * 0.35
    ax.imshow(ov_r)

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
    save_dir = root / "results" / "temp_disjoint_grouped_viz"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_samples = load_samples(master_json, data_root, product_filter=None)

    # Filter to multi-component samples that need grouping
    print("Filtering disjoint samples...")
    disjoint_samples = []
    for s in all_samples:
        mask = np.array(Image.open(s["mask_path"]).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue
        labeled, n_comp = ndimage.label(mask_bin)
        if n_comp < 2:
            continue
        ys, xs = np.where(mask_bin > 0)
        bbox_side = max(ys.max()-ys.min()+1, xs.max()-xs.min()+1)
        padded = bbox_side + max(1, int(bbox_side * 0.1)) * 2
        if padded <= 224:
            continue
        disjoint_samples.append(s)

    random.shuffle(disjoint_samples)
    disjoint_samples = disjoint_samples[:50]

    for i, sample in enumerate(disjoint_samples):
        img_pil = Image.open(sample["image_path"]).convert("RGB")
        mask_pil = Image.open(sample["mask_path"]).convert("L")
        img_w, img_h = img_pil.size
        atype = sample["anomaly_type"]

        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        labeled, n_comp = ndimage.label(mask_np)

        img_t = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)

        # Build components
        components = []
        for c in range(1, n_comp + 1):
            comp_mask = (labeled == c).astype(np.float32)
            bbox, side, area = get_comp_bbox(comp_mask)
            if bbox is None:
                continue
            pbbox = pad_bbox(bbox, 0.10, img_h, img_w)
            pb_size = padded_bbox_size(pbbox)
            components.append({
                "id": c, "mask": comp_mask, "bbox": bbox,
                "side": side, "area": area, "pbbox": pbbox, "pb_size": pb_size,
            })

        # Group components
        groups = group_components(components, img_h, img_w, max_crop=224, pad_frac=0.10)

        # Build group masks and padded bboxes
        group_info = []
        for gi, g in enumerate(groups):
            comp_idxs = g["comp_idxs"]
            # Merge masks
            group_mask = np.zeros_like(mask_np)
            for idx in comp_idxs:
                group_mask = np.maximum(group_mask, components[idx]["mask"])
            # Merged bbox — squarify + pad for actual crop region
            merged_bbox = components[comp_idxs[0]]["bbox"]
            for idx in comp_idxs[1:]:
                merged_bbox = merge_bboxes(merged_bbox, components[idx]["bbox"])
            sq_bbox = square_bbox(merged_bbox, img_h, img_w)
            merged_pbbox = pad_bbox(sq_bbox, 0.10, img_h, img_w)
            merged_pbbox = square_bbox(merged_pbbox, img_h, img_w)
            merged_pb_size = padded_bbox_size(merged_pbbox)
            total_area = sum(components[idx]["area"] for idx in comp_idxs)
            group_info.append({
                "comp_idxs": comp_idxs,
                "n_comps": len(comp_idxs),
                "mask": group_mask,
                "bbox": merged_bbox,
                "pbbox": merged_pbbox,
                "pb_size": merged_pb_size,
                "area": total_area,
            })

        # Layout: 2 rows x (1 + n_groups) columns
        n_groups = len(group_info)
        max_cols = min(1 + n_groups, 9)
        n_show = max_cols - 1
        fig, axes = plt.subplots(2, max_cols, figsize=(3.5 * max_cols, 8.5))
        if max_cols == 1:
            axes = axes.reshape(2, 1)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)

        # Column 0: Original with group rectangles
        orig_np = img_t.permute(1, 2, 0).numpy()
        axes[0, 0].imshow(orig_np)

        # Color each component by its group
        comp_to_group = {}
        for gi, g in enumerate(group_info[:n_show]):
            for idx in g["comp_idxs"]:
                comp_to_group[idx] = gi

        for idx, comp in enumerate(components):
            gi = comp_to_group.get(idx, 0)
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            ov = np.zeros((img_h, img_w, 4), dtype=np.float32)
            ov[:, :, 0] = color[0]
            ov[:, :, 1] = color[1]
            ov[:, :, 2] = color[2]
            ov[:, :, 3] = (comp["mask"] > 0.5).astype(np.float32) * 0.5
            axes[0, 0].imshow(ov)

        # Draw group bbox rectangles
        for gi, g in enumerate(group_info[:n_show]):
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]
            py0, py1, px0, px1 = g["pbbox"]
            rect = plt.Rectangle((px0, py0), px1-px0+1, py1-py0+1,
                                  linewidth=2, edgecolor=color, facecolor='none',
                                  linestyle='--')
            axes[0, 0].add_patch(rect)

        overall_area = int(mask_np.sum())
        axes[0, 0].set_title(f"Original {img_w}x{img_h}\n"
                              f"{n_comp} comps -> {n_groups} groups\n"
                              f"area={overall_area}", fontsize=7)
        axes[0, 0].axis("off")

        # Row 1 col 0: full mask
        axes[1, 0].imshow(orig_np)
        ov_all = np.zeros((img_h, img_w, 4), dtype=np.float32)
        ov_all[:, :, 0] = 1.0
        ov_all[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.5
        axes[1, 0].imshow(ov_all)
        grp_summary = ", ".join(f"{g['n_comps']}c" for g in group_info[:n_show])
        axes[1, 0].set_title(f"Full mask\ngroups: [{grp_summary}]", fontsize=7)
        axes[1, 0].axis("off")

        # Per-group crops
        for gi, g in enumerate(group_info[:n_show]):
            col = gi + 1
            group_mask_t = torch.from_numpy(g["mask"]).unsqueeze(0)
            pb_size = g["pb_size"]
            clip_target = 224

            if pb_size <= clip_target:
                cs = clip_target
                mode_str = f"fixed 224"
            else:
                cs = sample_log_uniform(pb_size, min(img_h, img_w))
                mode_str = f"log-U [{pb_size},{min(img_h,img_w)}]"

            img_224, mask_224, y0, x0, cs_actual = crop_and_resize(
                img_t, group_mask_t, cs, clip_target, g["pbbox"], img_h, img_w)

            r_224 = 0
            dil_224 = mask_224

            img_clip_np = img_224.permute(1, 2, 0).numpy()
            zoom = img_h / cs_actual
            color = GROUP_COLORS[gi % len(GROUP_COLORS)]

            draw_patches_overlay(axes[0, col], img_clip_np, mask_224, dil_224,
                                 f"G{gi+1} ({g['n_comps']}c) S={cs_actual} ({zoom:.1f}x)\n"
                                 f"pos=({y0},{x0}) r={r_224}")

            # Row 1: show group mask colored
            axes[1, col].imshow(img_clip_np)
            m_np = mask_224.squeeze().numpy()
            ov = np.zeros((clip_target, clip_target, 4), dtype=np.float32)
            ov[:, :, 0] = color[0]
            ov[:, :, 1] = color[1]
            ov[:, :, 2] = color[2]
            ov[:, :, 3] = (m_np > 0.5).astype(np.float32) * 0.6
            axes[1, col].imshow(ov)
            comp_ids = [components[idx]["id"] for idx in g["comp_idxs"]]
            axes[1, col].set_title(f"G{gi+1}: C{comp_ids}\n"
                                    f"padded={pb_size} | {mode_str}\n"
                                    f"area={g['area']}", fontsize=7)
            axes[1, col].axis("off")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring"),
        ]
        for gi in range(min(n_show, len(GROUP_COLORS))):
            n_c = group_info[gi]["n_comps"] if gi < len(group_info) else "?"
            legend_handles.append(
                mpatches.Patch(color=GROUP_COLORS[gi], alpha=0.6,
                               label=f"Group {gi+1} ({n_c}c)"))
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(len(legend_handles), 6), fontsize=7,
                   frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 {n_comp} components \u2192 "
            f"{n_groups} groups\n"
            f"Greedy grouping into 224x224 crops (pad=10%)",
            fontsize=9, fontweight="bold",
        )
        plt.tight_layout(rect=[0.02, 0.05, 1, 0.90])
        plt.savefig(save_dir / f"{i:02d}_{atype}_n{n_comp}_g{n_groups}.png",
                    dpi=120, bbox_inches="tight")
        plt.close()
        grp_str = "+".join(f"{g['n_comps']}c" for g in group_info)
        print(f"  [{i+1}/{len(disjoint_samples)}] {atype} "
              f"{n_comp}comps -> {n_groups}grps [{grp_str}]")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
