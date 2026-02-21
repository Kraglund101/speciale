"""
Augmentation + adaptive crop from 1024 source.

Loads at 1024, applies augmentations at 1024, then adaptive crop to 224 for CLIP.
This enables lossless scale augmentation up to 2x.

Row 0: 1024x1024 augmented image + mask overlay
Row 1: 224x224 CLIP input (adaptive crop or downscale)
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42


def load_samples(splits_dir: Path, data_root: Path, n: int = 10):
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


def get_border_median_color(img: Image.Image):
    arr = np.array(img)
    if arr.ndim == 2:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        return int(np.median(border))
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    return tuple(int(x) for x in np.median(border, axis=0))


def apply_rotation(img, mask, angle):
    fill = get_border_median_color(img)
    img_rot = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)
    mask_rot = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    mask_np = np.array(mask_rot)
    mask_rot = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")
    return img_rot, mask_rot


def get_square_bbox(mask: Image.Image):
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    side = max(y_max - y_min + 1, x_max - x_min + 1)
    pad = max(1, int(side * 0.1))
    side += 2 * pad
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    return cy, cx, side


def adaptive_crop_224(img: Image.Image, mask: Image.Image, target: int = 224):
    """Adaptive crop from current resolution to 224. Crop if bbox <= 224, else downscale."""
    bbox = get_square_bbox(mask)
    h, w = img.size[1], img.size[0]

    if bbox is not None and bbox[2] <= target:
        cy, cx, side = bbox
        y0 = max(0, cy - target // 2)
        x0 = max(0, cx - target // 2)
        if y0 + target > h:
            y0 = max(0, h - target)
        if x0 + target > w:
            x0 = max(0, w - target)
        img_crop = img.crop((x0, y0, x0 + target, y0 + target))
        mask_crop = mask.crop((x0, y0, x0 + target, y0 + target))
        return img_crop, mask_crop, "crop", {"side": side, "y0": y0, "x0": x0}
    else:
        img_out = img.resize((target, target), Image.BILINEAR)
        mask_out = mask.resize((target, target), Image.NEAREST)
        side = bbox[2] if bbox else 0
        return img_out, mask_out, "full", {"side": side}


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "clip" / "augmentation_crop_viz_1024"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(splits_dir, data_root, n=10)

    angles = [random.uniform(0, 360) for _ in range(4)]
    brightness_val = random.uniform(0.8, 1.3)
    contrast_val = random.uniform(0.8, 1.3)

    augmentations = [
        ("Original", lambda img, mask: (img, mask)),
        (f"Rot {angles[0]:.0f}\u00b0", lambda img, mask, a=angles[0]: apply_rotation(img, mask, a)),
        (f"Rot {angles[1]:.0f}\u00b0", lambda img, mask, a=angles[1]: apply_rotation(img, mask, a)),
        ("H-Flip", lambda img, mask: (ImageOps.mirror(img), ImageOps.mirror(mask))),
        ("V-Flip", lambda img, mask: (ImageOps.flip(img), ImageOps.flip(mask))),
        ("Bright/Cont", lambda img, mask: (
            ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(brightness_val)).enhance(contrast_val),
            mask
        )),
        ("Blur r=1.2", lambda img, mask: (img.filter(ImageFilter.GaussianBlur(radius=1.2)), mask)),
    ]

    for i, sample in enumerate(samples):
        # Load at 1024
        img = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np = np.array(mask)
        mask = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")
        atype = sample["anomaly_type"]

        n_aug = len(augmentations)
        fig, axes = plt.subplots(2, n_aug, figsize=(3 * n_aug, 7))

        for j, (aug_name, aug_fn) in enumerate(augmentations):
            aug_img, aug_mask = aug_fn(img, mask)

            # Row 0: 1024x1024 with mask overlay + bbox
            img_np = np.array(aug_img).astype(np.float32) / 255
            mask_np = np.array(aug_mask).astype(np.float32) / 255

            # Downsample for display (1024 is too large for matplotlib)
            display_img = aug_img.resize((512, 512), Image.BILINEAR)
            display_mask = aug_mask.resize((512, 512), Image.NEAREST)
            disp_np = np.array(display_img).astype(np.float32) / 255
            disp_mask_np = np.array(display_mask).astype(np.float32) / 255

            axes[0, j].imshow(disp_np)
            ov = np.zeros((*disp_mask_np.shape, 4), dtype=np.float32)
            ov[:, :, 0] = 1.0
            ov[:, :, 3] = (disp_mask_np > 0.5).astype(np.float32) * 0.45
            axes[0, j].imshow(ov)

            bbox = get_square_bbox(aug_mask)
            if bbox is not None:
                cy, cx, side = bbox
                # Scale bbox to 512 display
                s = 512 / 1024
                rect_y0 = max(0, cy - side // 2) * s
                rect_x0 = max(0, cx - side // 2) * s
                color = "lime" if side <= 224 else "orange"
                rect = plt.Rectangle(
                    (rect_x0, rect_y0), side * s, side * s,
                    linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
                )
                axes[0, j].add_patch(rect)
                axes[0, j].set_title(f"{aug_name}\nbbox={side}px @1024", fontsize=8)
            else:
                axes[0, j].set_title(f"{aug_name}\nNo anomaly", fontsize=8)
            axes[0, j].axis("off")

            # Row 1: 224x224 CLIP input (adaptive crop from 1024)
            clip_img, clip_mask, method, info = adaptive_crop_224(aug_img, aug_mask, target=224)
            clip_np = np.array(clip_img).astype(np.float32) / 255
            clip_mask_np = np.array(clip_mask).astype(np.float32) / 255

            axes[1, j].imshow(clip_np)
            ov2 = np.zeros((*clip_mask_np.shape, 4), dtype=np.float32)
            ov2[:, :, 0] = 1.0
            ov2[:, :, 3] = (clip_mask_np > 0.5).astype(np.float32) * 0.45
            axes[1, j].imshow(ov2)

            border_color = "lime" if method == "crop" else "orange"
            for spine in axes[1, j].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)
            axes[1, j].set_title(f"224\u00d7224 ({method})", fontsize=8)
            axes[1, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        axes[0, 0].set_ylabel("1024\u00d7\u200b1024\n(display 512)", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=70, va="center")
        axes[1, 0].set_ylabel("224\u00d7224\nCLIP", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=70, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Anomaly mask"),
            mpatches.Patch(facecolor="none", edgecolor="lime", linewidth=2, linestyle="--",
                           label="Crop (bbox \u2264 224)"),
            mpatches.Patch(facecolor="none", edgecolor="orange", linewidth=2, linestyle="--",
                           label="Full downscale (bbox > 224)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 Augmentation at 1024 \u2192 adaptive crop 224 for CLIP",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0.08, 0.04, 1, 0.93])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype}")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
