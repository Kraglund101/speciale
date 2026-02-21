"""
Side-by-side: scale from 512 (interpolated) vs scale from 1024 (lossless).
Shows adaptive crop to 224 after scaling.
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

SEED = 42


def load_sample(splits_dir, data_root):
    random.seed(SEED)
    for json_path in sorted(splits_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data.get("samples", []):
            img_path = data_root / entry["image_path"]
            mask_path = data_root / entry["mask_path"]
            if img_path.exists() and mask_path.exists():
                return str(img_path), str(mask_path), json_path.stem
    return None


def get_square_bbox(mask):
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


def adaptive_crop(img, mask, target=224):
    bbox = get_square_bbox(mask)
    h, w = img.size[1], img.size[0]
    if bbox is not None and bbox[2] <= target:
        cy, cx, side = bbox
        y0 = max(0, cy - target // 2)
        x0 = max(0, cx - target // 2)
        if y0 + target > h: y0 = max(0, h - target)
        if x0 + target > w: x0 = max(0, w - target)
        return img.crop((x0, y0, x0 + target, y0 + target)), "crop"
    return img.resize((target, target), Image.BILINEAR), "full"


def scale_from_512(img_512, mask_512, scale, target=224):
    """Scale from 512 source — upscaling involves interpolation."""
    w, h = 512, 512
    new_size = int(512 * scale)
    img_s = img_512.resize((new_size, new_size), Image.BILINEAR)
    mask_s = mask_512.resize((new_size, new_size), Image.NEAREST)
    # Center crop to 512
    if new_size >= 512:
        x0 = (new_size - 512) // 2
        y0 = (new_size - 512) // 2
        img_s = img_s.crop((x0, y0, x0 + 512, y0 + 512))
        mask_s = mask_s.crop((x0, y0, x0 + 512, y0 + 512))
    else:
        # Pad
        border = np.array(img_s)
        b = np.concatenate([border[0, :], border[-1, :], border[:, 0], border[:, -1]], axis=0)
        fill = tuple(int(x) for x in np.median(b, axis=0))
        out = Image.new("RGB", (512, 512), fill)
        mout = Image.new("L", (512, 512), 0)
        off = (512 - new_size) // 2
        out.paste(img_s, (off, off))
        mout.paste(mask_s, (off, off))
        img_s, mask_s = out, mout
    clip, method = adaptive_crop(img_s, mask_s, target)
    return clip, method


def scale_from_1024(img_1024, mask_1024, scale, target=224):
    """Scale from 1024 source — lossless up to 2x."""
    crop_size = int(1024 / scale)
    if crop_size <= 1024:
        cx, cy = 512, 512
        x0 = max(0, cx - crop_size // 2)
        y0 = max(0, cy - crop_size // 2)
        img_c = img_1024.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        mask_c = mask_1024.crop((x0, y0, x0 + crop_size, y0 + crop_size))
        # Now we have the "zoomed" view, adaptive crop to 224
        clip, method = adaptive_crop(img_c, mask_c, target)
    else:
        scaled = int(target * scale)
        img_s = img_1024.resize((scaled, scaled), Image.BILINEAR)
        mask_s = mask_1024.resize((scaled, scaled), Image.NEAREST)
        fill = tuple(int(x) for x in np.median(
            np.concatenate([np.array(img_s)[0, :], np.array(img_s)[-1, :]], axis=0), axis=0))
        out = Image.new("RGB", (target, target), fill)
        off = (target - scaled) // 2
        out.paste(img_s, (off, off))
        clip = out
        method = "full+pad"
    return clip, method


def main():
    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "temp_scale_512_vs_1024"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load a few samples
    random.seed(SEED)
    samples = []
    for json_path in sorted(splits_dir.glob("*.json")):
        with open(json_path) as f:
            data = json.load(f)
        for entry in data.get("samples", []):
            img_path = data_root / entry["image_path"]
            mask_path = data_root / entry["mask_path"]
            if img_path.exists() and mask_path.exists():
                samples.append((str(img_path), str(mask_path), json_path.stem))
    random.shuffle(samples)
    samples = samples[:5]

    scales = [1.0, 1.25, 1.5, 2.0]

    for si, (img_path, mask_path, atype) in enumerate(samples):
        img_1024 = Image.open(img_path).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask_1024 = Image.open(mask_path).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np = np.array(mask_1024)
        mask_1024 = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")

        img_512 = img_1024.resize((512, 512), Image.BILINEAR)
        mask_512 = mask_1024.resize((512, 512), Image.NEAREST)

        fig, axes = plt.subplots(2, len(scales), figsize=(4 * len(scales), 8.5))

        for j, s in enumerate(scales):
            clip_512, m512 = scale_from_512(img_512, mask_512, s)
            clip_1024, m1024 = scale_from_1024(img_1024, mask_1024, s)

            axes[0, j].imshow(np.array(clip_512))
            axes[0, j].set_title(f"From 512 \u2192 scale {s}x\n({m512})", fontsize=9)
            axes[0, j].axis("off")

            axes[1, j].imshow(np.array(clip_1024))
            axes[1, j].set_title(f"From 1024 \u2192 scale {s}x\n({m1024})", fontsize=9)
            axes[1, j].axis("off")

        axes[0, 0].set_ylabel("512 source\n(interpolated)", fontsize=11, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[1, 0].set_ylabel("1024 source\n(lossless)", fontsize=11, fontweight="bold",
                               rotation=0, labelpad=80, va="center")

        plt.suptitle(f"Sample {si} \u2014 {atype} \u2014 Scale augmentation: 512 vs 1024 source \u2192 224 CLIP crop",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0.1, 0, 1, 0.93])
        plt.savefig(save_dir / f"{si:02d}_{atype}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [{si+1}/{len(samples)}] {atype}")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
