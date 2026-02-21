"""
Visualize random scale crop from 1024 for CLIP input.

For each sample, shows different crop sizes S in [224, 1024] centered on anomaly,
resized to 224x224. All downsampling — zero quality loss.

This achieves scale decoupling: CLIP sees the anomaly at varying scales,
independent of the UNet mask scale.
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SEED = 42


def load_samples(splits_dir, data_root, n=10):
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


def get_anomaly_center(mask):
    """Get center of anomaly mass."""
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return mask.size[1] // 2, mask.size[0] // 2
    return int(ys.mean()), int(xs.mean())


def get_square_bbox_side(mask):
    """Get square bbox side length."""
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return 0
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    pad = max(1, int(side * 0.1))
    return side + 2 * pad


def random_scale_crop(img, mask, crop_s, target=224):
    """Crop a crop_s x crop_s region centered on anomaly, resize to target.

    Args:
        img: PIL image at native resolution
        mask: PIL mask at native resolution
        crop_s: Size of crop window (>= target, <= native resolution)
        target: Output size (224)

    Returns:
        (img_224, mask_224)
    """
    h, w = img.size[1], img.size[0]
    cy, cx = get_anomaly_center(mask)

    # Clamp crop_s to valid range
    crop_s = max(target, min(crop_s, min(h, w)))

    # Center crop on anomaly, keep within bounds
    y0 = max(0, cy - crop_s // 2)
    x0 = max(0, cx - crop_s // 2)
    if y0 + crop_s > h:
        y0 = max(0, h - crop_s)
    if x0 + crop_s > w:
        x0 = max(0, w - crop_s)

    img_crop = img.crop((x0, y0, x0 + crop_s, y0 + crop_s))
    mask_crop = mask.crop((x0, y0, x0 + crop_s, y0 + crop_s))

    img_out = img_crop.resize((target, target), Image.BILINEAR)
    mask_out = mask_crop.resize((target, target), Image.NEAREST)

    return img_out, mask_out


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "clip" / "random_scale_crop_1024"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(splits_dir, data_root, n=10)

    # Fixed crop sizes to show the full range
    crop_sizes = [224, 336, 448, 600, 800, 1024]

    for i, sample in enumerate(samples):
        img = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np = np.array(mask)
        mask = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")
        atype = sample["anomaly_type"]
        bbox_side = get_square_bbox_side(mask)

        n_cols = len(crop_sizes)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))

        for j, cs in enumerate(crop_sizes):
            clip_img, clip_mask = random_scale_crop(img, mask, cs, target=224)
            clip_np = np.array(clip_img).astype(np.float32) / 255
            mask_np = np.array(clip_mask).astype(np.float32) / 255

            axes[j].imshow(clip_np)
            ov = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            ov[:, :, 0] = 1.0
            ov[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.45
            axes[j].imshow(ov)

            pct = (mask_np > 0.5).sum() / mask_np.size * 100
            # Effective scale relative to "full image" view
            effective_scale = 1024 / cs
            lossless = "native" if cs == 224 else f"{cs}\u2192224"
            axes[j].set_title(
                f"S={cs} ({lossless})\n"
                f"~{effective_scale:.1f}x zoom \u2022 {pct:.2f}%",
                fontsize=8,
            )
            axes[j].axis("off")

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 bbox={bbox_side}px @1024\n"
            f"Random crop S\u2208[224, 1024] from 1024 \u2192 resize 224 (always downsampling)",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype} (bbox={bbox_side}px)")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
