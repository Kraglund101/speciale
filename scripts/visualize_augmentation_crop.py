"""
Visualize augmentation + adaptive crop pipeline for CLIP input.

For each sample, applies augmentations (rotation, flip, brightness/contrast, blur)
to the full 512x512 image+mask, then decides:
  - If anomaly square bbox <= 224: take 224x224 crop centered on anomaly
  - Else: downsample whole image to 224x224

Rotation uses median-border fill to avoid black borders.

Shows 10 samples from RealIAD and 10 from MVTec (data/concepts with source_dataset=mvtec).
"""
import math
import sys
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

SEED = 42


def load_realiad_samples(splits_dir: Path, data_root: Path, n: int = 10):
    """Load n random samples from RealIAD."""
    samples = []
    for json_path in sorted(splits_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        entries = data.get("samples", [])
        atype = json_path.stem
        for entry in entries:
            img_path = data_root / entry["image_path"]
            mask_path = data_root / entry["mask_path"]
            if img_path.exists() and mask_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "anomaly_type": atype,
                    "source": "RealIAD",
                })
    random.shuffle(samples)
    return samples[:n]


def load_mvtec_samples(splits_dir: Path, n: int = 10):
    """Load n random samples from MVTec (via data/concepts JSONs)."""
    samples = []
    for json_path in sorted(splits_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        entries = data.get("images", [])
        atype = json_path.stem
        for entry in entries:
            if entry.get("source_dataset") != "mvtec":
                continue
            img_full = entry.get("image_path_full", "")
            mask_full = entry.get("mask_path_full", "")
            if not img_full or not mask_full:
                continue
            img_path = Path(img_full)
            mask_path = Path(mask_full)
            if not img_path.is_absolute():
                img_path = splits_dir.parent.parent / img_path
            if not mask_path.is_absolute():
                mask_path = splits_dir.parent.parent / mask_path
            if img_path.exists() and mask_path.exists():
                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "anomaly_type": atype,
                    "source": "MVTec",
                })
    random.shuffle(samples)
    return samples[:n]


def load_image_and_mask(sample: dict, image_size: int = 512):
    """Load and resize image + mask to image_size."""
    img = Image.open(sample["image_path"]).convert("RGB").resize(
        (image_size, image_size), Image.BILINEAR
    )
    mask = Image.open(sample["mask_path"]).convert("L").resize(
        (image_size, image_size), Image.NEAREST
    )
    mask_np = np.array(mask)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_np, mode="L")
    return img, mask


def reflect_pad_rotate(pil_img: Image.Image, angle: float, resample, fillcolor=0):
    """Rotate with reflection padding to avoid black borders.

    1. Pad image with reflection by enough to cover rotation corners
    2. Rotate the padded image (any residual fill is outside reflected area)
    3. Crop back to original size from center
    """
    w, h = pil_img.size
    # Pad by half the image size — enough for any rotation angle
    pad = max(w, h) // 2
    # Convert to numpy for reflection padding
    arr = np.array(pil_img)
    if arr.ndim == 2:
        padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    else:
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    padded_pil = Image.fromarray(padded)
    # Rotate padded image
    rotated = padded_pil.rotate(angle, resample=resample, expand=False, fillcolor=fillcolor)
    # Crop back to original size from center
    pw, ph = rotated.size
    cx, cy = pw // 2, ph // 2
    left = cx - w // 2
    top = cy - h // 2
    return rotated.crop((left, top, left + w, top + h))


def get_border_median_color(img: Image.Image):
    """Get median color from image border pixels."""
    arr = np.array(img)
    if arr.ndim == 2:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
        return int(np.median(border))
    else:
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
        return tuple(int(x) for x in np.median(border, axis=0))


def apply_rotation(img: Image.Image, mask: Image.Image, angle: float):
    """Rotate image and mask with median-border fill color."""
    fill = get_border_median_color(img)
    img_rot = img.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=fill)
    mask_rot = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)
    # Re-binarize mask after rotation
    mask_np = np.array(mask_rot)
    mask_np = (mask_np > 127).astype(np.uint8) * 255
    mask_rot = Image.fromarray(mask_np, mode="L")
    return img_rot, mask_rot


def apply_hflip(img: Image.Image, mask: Image.Image):
    return ImageOps.mirror(img), ImageOps.mirror(mask)


def apply_vflip(img: Image.Image, mask: Image.Image):
    return ImageOps.flip(img), ImageOps.flip(mask)


def apply_brightness_contrast(img: Image.Image, brightness: float, contrast: float):
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    return img


def apply_blur(img: Image.Image, radius: float = 1.0):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def get_square_bbox(mask: Image.Image):
    """Get square bounding box of mask. Returns (cy, cx, side) or None if empty."""
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    side = max(h, w)
    # Add 10% padding
    pad = max(1, int(side * 0.1))
    side = side + 2 * pad
    cy = (y_min + y_max) // 2
    cx = (x_min + x_max) // 2
    return cy, cx, side


def adaptive_crop_or_downscale(img: Image.Image, mask: Image.Image, target: int = 224):
    """If anomaly square bbox <= target, crop 224x224. Else downsample whole image."""
    bbox = get_square_bbox(mask)
    if bbox is None:
        return img.resize((target, target), Image.BILINEAR), "full", {"side": 0}

    cy, cx, side = bbox
    h, w = img.size[1], img.size[0]

    if side <= target:
        crop_side = target
        y0 = max(0, cy - crop_side // 2)
        x0 = max(0, cx - crop_side // 2)
        if y0 + crop_side > h:
            y0 = max(0, h - crop_side)
        if x0 + crop_side > w:
            x0 = max(0, w - crop_side)
        y1 = y0 + crop_side
        x1 = x0 + crop_side
        crop = img.crop((x0, y0, x1, y1))
        return crop, "crop", {"side": side, "y0": y0, "x0": x0, "crop_side": crop_side}
    else:
        return img.resize((target, target), Image.BILINEAR), "full", {"side": side}


def visualize(save_dir: Path, realiad_dir: Path, realiad_root: Path, mvtec_dir: Path):
    random.seed(SEED)
    np.random.seed(SEED)

    save_dir.mkdir(parents=True, exist_ok=True)

    realiad_samples = load_realiad_samples(realiad_dir, realiad_root, n=10)
    mvtec_samples = load_mvtec_samples(mvtec_dir, n=10)

    print(f"Loaded {len(realiad_samples)} RealIAD, {len(mvtec_samples)} MVTec samples")

    all_samples = realiad_samples + mvtec_samples

    angles = [random.uniform(0, 360) for _ in range(4)]
    brightness_val = random.uniform(0.8, 1.3)
    contrast_val = random.uniform(0.8, 1.3)
    blur_radius = 1.2

    augmentations = [
        ("Original", lambda img, mask: (img, mask)),
        (f"Rot {angles[0]:.0f}\u00b0", lambda img, mask, a=angles[0]: apply_rotation(img, mask, a)),
        (f"Rot {angles[1]:.0f}\u00b0", lambda img, mask, a=angles[1]: apply_rotation(img, mask, a)),
        (f"Rot {angles[2]:.0f}\u00b0", lambda img, mask, a=angles[2]: apply_rotation(img, mask, a)),
        (f"Rot {angles[3]:.0f}\u00b0", lambda img, mask, a=angles[3]: apply_rotation(img, mask, a)),
        ("H-Flip", lambda img, mask: apply_hflip(img, mask)),
        ("V-Flip", lambda img, mask: apply_vflip(img, mask)),
        ("Bright/Cont", lambda img, mask: (apply_brightness_contrast(img, brightness_val, contrast_val), mask)),
        (f"Blur r={blur_radius}", lambda img, mask: (apply_blur(img, blur_radius), mask)),
    ]

    for i, sample in enumerate(all_samples):
        img, mask = load_image_and_mask(sample, image_size=512)
        source = sample["source"]
        atype = sample["anomaly_type"]

        n_aug = len(augmentations)
        # 2 rows: image+overlay+bbox, CLIP+overlay
        fig, axes = plt.subplots(2, n_aug, figsize=(3 * n_aug, 7))

        for j, (aug_name, aug_fn) in enumerate(augmentations):
            aug_img, aug_mask = aug_fn(img, mask)

            # Compute dilation for this augmented mask
            mask_t = torch.from_numpy(np.array(aug_mask).astype(np.float32) / 255).unsqueeze(0)  # [1, H, W]
            dil_r = adaptive_dilation_radius(mask_t, min_r=2, max_r=10)
            dilated_t = dilate_mask(mask_t, dil_r)
            mask_np = mask_t.squeeze(0).numpy()
            dilated_np = dilated_t.squeeze(0).numpy()
            ring_np = np.clip(dilated_np - mask_np, 0, 1)
            h, w = mask_np.shape
            n_core = int((mask_np > 0.5).sum())
            n_ring = int((ring_np > 0.5).sum())

            # Row 0: 512x512 image + core (red) + ring (blue) overlay + bbox
            img_np = np.array(aug_img).astype(np.float32) / 255
            axes[0, j].imshow(img_np)
            ov_core = np.zeros((h, w, 4), dtype=np.float32)
            ov_core[:, :, 0] = 1.0
            ov_core[:, :, 3] = mask_np * 0.45
            axes[0, j].imshow(ov_core)
            ov_ring = np.zeros((h, w, 4), dtype=np.float32)
            ov_ring[:, :, 2] = 1.0
            ov_ring[:, :, 3] = ring_np * 0.4
            axes[0, j].imshow(ov_ring)

            bbox = get_square_bbox(aug_mask)
            if bbox is not None:
                cy, cx, side = bbox
                rect_y0 = max(0, cy - side // 2)
                rect_x0 = max(0, cx - side // 2)
                color = "lime" if side <= 224 else "orange"
                rect = plt.Rectangle(
                    (rect_x0, rect_y0),
                    min(side, 512 - rect_x0), min(side, 512 - rect_y0),
                    linewidth=2, edgecolor=color, facecolor="none", linestyle="--"
                )
                axes[0, j].add_patch(rect)
            title_line2 = f"r={dil_r} bbox={side}px" if bbox else "No anomaly"
            axes[0, j].set_title(f"{aug_name}\n{title_line2}", fontsize=8)
            axes[0, j].axis("off")

            # Row 1: CLIP input (224x224) + core/ring overlay
            clip_input, method, info = adaptive_crop_or_downscale(aug_img, aug_mask, target=224)
            clip_np = np.array(clip_input).astype(np.float32) / 255
            axes[1, j].imshow(clip_np)

            # Also crop/downscale the masks to match CLIP input
            aug_mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode="L")
            dil_mask_pil = Image.fromarray((dilated_np * 255).astype(np.uint8), mode="L")
            if method == "crop":
                y0, x0, cs = info["y0"], info["x0"], info["crop_side"]
                clip_core = mask_np[y0:y0+cs, x0:x0+cs]
                clip_dil = dilated_np[y0:y0+cs, x0:x0+cs]
            else:
                clip_core = np.array(aug_mask_pil.resize((224, 224), Image.NEAREST)).astype(np.float32) / 255
                clip_dil = np.array(dil_mask_pil.resize((224, 224), Image.NEAREST)).astype(np.float32) / 255
            clip_ring = np.clip(clip_dil - clip_core, 0, 1)
            ch, cw = clip_core.shape
            ov_c = np.zeros((ch, cw, 4), dtype=np.float32)
            ov_c[:, :, 0] = 1.0
            ov_c[:, :, 3] = (clip_core > 0.5).astype(np.float32) * 0.45
            axes[1, j].imshow(ov_c)
            ov_r = np.zeros((ch, cw, 4), dtype=np.float32)
            ov_r[:, :, 2] = 1.0
            ov_r[:, :, 3] = (clip_ring > 0.5).astype(np.float32) * 0.4
            axes[1, j].imshow(ov_r)

            border_color = "lime" if method == "crop" else "orange"
            for spine in axes[1, j].spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)
            axes[1, j].set_title(f"224\u00d7224 ({method})", fontsize=8)
            axes[1, j].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Row labels
        axes[0, 0].set_ylabel("512\u00d7512\nOverlay", fontsize=11, fontweight="bold",
                               rotation=0, labelpad=70, va="center")
        axes[1, 0].set_ylabel("224\u00d7224\nCLIP", fontsize=11, fontweight="bold",
                               rotation=0, labelpad=70, va="center")

        legend_handles = [
            mpatches.Patch(color=(0.9, 0.15, 0.15), label="Core (anomaly)"),
            mpatches.Patch(color=(0.15, 0.3, 0.9), label="Dilation ring (r=2\u201310)"),
            mpatches.Patch(facecolor="none", edgecolor="lime", linewidth=2, linestyle="--",
                           label="Crop (bbox \u2264 224)"),
            mpatches.Patch(facecolor="none", edgecolor="orange", linewidth=2, linestyle="--",
                           label="Full downscale (bbox > 224)"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=4,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {source} \u2014 {atype}\n"
            f"Median-fill rotation \u2022 Adaptive dilation (r=clamp(0.5\u00b7\u221aarea, 2, 10)) \u2022 Adaptive crop/downscale",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0.08, 0.04, 1, 0.93])
        plt.savefig(save_dir / f"{i:02d}_{source}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()

        print(f"  [{i+1}/{len(all_samples)}] {source}/{atype}")

    print(f"\nSaved {len(all_samples)} samples to {save_dir}/")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    visualize(
        save_dir=project_root / "results" / "clip" / "augmentation_crop_viz_512",
        realiad_dir=project_root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k",
        realiad_root=project_root / "anomverse_extension" / "datasets" / "realiad_1024",
        mvtec_dir=project_root / "data" / "concepts",
    )
