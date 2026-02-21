"""Quick viz: scale augmentation on anomaly images."""
import sys
import random
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

SEED = 42


def load_samples(splits_dir: Path, data_root: Path, n: int = 5):
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


def scale_transform(img_1024: Image.Image, mask_1024: Image.Image, scale: float, target: int = 512):
    """Scale from 1024 source, then crop/pad to target (512).

    scale=1.0 → standard 1024→512 downsample (baseline)
    scale=2.0 → center crop 512 from 1024 (native resolution, zero quality loss)
    scale=0.5 → downsample 1024→256, center-pad to 512

    Effective source crop size = 1024 / scale, then resize to target.
    """
    src = 1024
    # Size of the crop from the 1024 source
    crop_size = int(src / scale)

    if crop_size <= src:
        # Crop from center of 1024
        cx, cy = src // 2, src // 2
        x0 = max(0, cx - crop_size // 2)
        y0 = max(0, cy - crop_size // 2)
        x1 = min(src, x0 + crop_size)
        y1 = min(src, y0 + crop_size)
        img_crop = img_1024.crop((x0, y0, x1, y1))
        mask_crop = mask_1024.crop((x0, y0, x1, y1))
        # Resize to target
        img_out = img_crop.resize((target, target), Image.BILINEAR)
        mask_out = mask_crop.resize((target, target), Image.NEAREST)
    else:
        # Object appears smaller — downsample beyond 512, then pad
        scaled_size = int(target * scale)  # e.g. scale=0.5 → 256
        img_scaled = img_1024.resize((scaled_size, scaled_size), Image.BILINEAR)
        mask_scaled = mask_1024.resize((scaled_size, scaled_size), Image.NEAREST)

        arr = np.array(img_scaled)
        border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
        fill = tuple(int(x) for x in np.median(border, axis=0))

        img_out = Image.new("RGB", (target, target), fill)
        mask_out = Image.new("L", (target, target), 0)
        x0 = (target - scaled_size) // 2
        y0 = (target - scaled_size) // 2
        img_out.paste(img_scaled, (x0, y0))
        mask_out.paste(mask_scaled, (x0, y0))

    return img_out, mask_out


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "temp_scale_viz"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(splits_dir, data_root, n=8)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    for i, sample in enumerate(samples):
        img = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np = np.array(mask)
        mask = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")

        atype = sample["anomaly_type"]

        fig, axes = plt.subplots(1, len(scales), figsize=(3.5 * len(scales), 4))

        for j, s in enumerate(scales):
            img_s, mask_s = scale_transform(img, mask, s, target=512)
            img_np = np.array(img_s).astype(np.float32) / 255
            mask_np = np.array(mask_s).astype(np.float32) / 255

            axes[j].imshow(img_np)
            ov = np.zeros((*mask_np.shape, 4), dtype=np.float32)
            ov[:, :, 0] = 1.0
            ov[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.45
            axes[j].imshow(ov)

            pct = (mask_np > 0.5).sum() / mask_np.size * 100
            bold = " (baseline)" if s == 1.0 else ""
            axes[j].set_title(f"Scale {s}x{bold}\n{pct:.2f}% coverage", fontsize=9)
            axes[j].axis("off")

        plt.suptitle(f"Sample {i} — {atype} — Scale augmentation from 1024 source (lossless up to 2x)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout(rect=[0.06, 0, 1, 0.94])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype}")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
