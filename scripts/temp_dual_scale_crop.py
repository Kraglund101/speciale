"""
Visualize independent random scale crops for UNet and CLIP pathways.

From 1024 source:
  UNet: random S in [512, 1024] centered on anomaly → resize 512
  CLIP: random S in [224, 1024] centered on anomaly → resize 224

Both sampled independently — full scale decoupling between pathways.
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


def load_samples(splits_dir, data_root, n=8):
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
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return mask.size[1] // 2, mask.size[0] // 2
    return int(ys.mean()), int(xs.mean())


def get_square_bbox_side(mask):
    mask_np = np.array(mask)
    ys, xs = np.where(mask_np > 127)
    if len(ys) == 0:
        return 0
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    pad = max(1, int(side * 0.1))
    return side + 2 * pad


def scale_crop(img, mask, crop_s, target):
    """Crop crop_s x crop_s centered on anomaly, resize to target."""
    h, w = img.size[1], img.size[0]
    cy, cx = get_anomaly_center(mask)
    crop_s = max(target, min(crop_s, min(h, w)))

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
    return img_out, mask_out, x0, y0, crop_s


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    root = Path(__file__).parent.parent
    splits_dir = root / "anomverse_extension" / "datasets" / "realiad_1024" / "concepts_10k"
    data_root = root / "anomverse_extension" / "datasets" / "realiad_1024"
    save_dir = root / "results" / "temp_dual_scale_crop"
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(splits_dir, data_root, n=8)

    # Show 5 independent samples per image (simulating 5 training steps)
    n_pairs = 5

    for i, sample in enumerate(samples):
        img = Image.open(sample["image_path"]).convert("RGB").resize((1024, 1024), Image.BILINEAR)
        mask = Image.open(sample["mask_path"]).convert("L").resize((1024, 1024), Image.NEAREST)
        mask_np = np.array(mask)
        mask = Image.fromarray(((mask_np > 127).astype(np.uint8) * 255), mode="L")
        atype = sample["anomaly_type"]
        bbox_side = get_square_bbox_side(mask)

        # 3 rows: 1024 overview, UNet 512, CLIP 224
        fig, axes = plt.subplots(3, n_pairs, figsize=(4 * n_pairs, 12))

        for j in range(n_pairs):
            # Independent random crops
            unet_s = random.randint(512, 1024)
            clip_s = random.randint(224, 1024)

            unet_img, unet_mask, ux0, uy0, unet_s_actual = scale_crop(img, mask, unet_s, target=512)
            clip_img, clip_mask, cx0, cy0, clip_s_actual = scale_crop(img, mask, clip_s, target=224)

            # Row 0: 1024 overview showing both crop regions
            disp = img.resize((512, 512), Image.BILINEAR)
            disp_mask = mask.resize((512, 512), Image.NEAREST)
            disp_np = np.array(disp).astype(np.float32) / 255
            disp_mask_np = np.array(disp_mask).astype(np.float32) / 255

            axes[0, j].imshow(disp_np)
            ov = np.zeros((*disp_mask_np.shape, 4), dtype=np.float32)
            ov[:, :, 0] = 1.0
            ov[:, :, 3] = (disp_mask_np > 0.5).astype(np.float32) * 0.4
            axes[0, j].imshow(ov)

            # Draw UNet crop box (green)
            s = 512 / 1024
            rect_u = plt.Rectangle(
                (ux0 * s, uy0 * s), unet_s_actual * s, unet_s_actual * s,
                linewidth=2.5, edgecolor="lime", facecolor="none", linestyle="-"
            )
            axes[0, j].add_patch(rect_u)
            # Draw CLIP crop box (cyan)
            rect_c = plt.Rectangle(
                (cx0 * s, cy0 * s), clip_s_actual * s, clip_s_actual * s,
                linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--"
            )
            axes[0, j].add_patch(rect_c)
            axes[0, j].set_title(f"Step {j}\nUNet S={unet_s_actual}, CLIP S={clip_s_actual}", fontsize=9)
            axes[0, j].axis("off")

            # Row 1: UNet 512x512
            unet_np = np.array(unet_img).astype(np.float32) / 255
            unet_mask_np = np.array(unet_mask).astype(np.float32) / 255
            axes[1, j].imshow(unet_np)
            ov2 = np.zeros((*unet_mask_np.shape, 4), dtype=np.float32)
            ov2[:, :, 0] = 1.0
            ov2[:, :, 3] = (unet_mask_np > 0.5).astype(np.float32) * 0.4
            axes[1, j].imshow(ov2)
            unet_zoom = 1024 / unet_s_actual
            unet_pct = (unet_mask_np > 0.5).sum() / unet_mask_np.size * 100
            axes[1, j].set_title(f"S={unet_s_actual}\u2192512 ({unet_zoom:.1f}x)\n{unet_pct:.2f}%", fontsize=9)
            axes[1, j].axis("off")

            # Row 2: CLIP 224x224
            clip_np = np.array(clip_img).astype(np.float32) / 255
            clip_mask_np = np.array(clip_mask).astype(np.float32) / 255
            axes[2, j].imshow(clip_np)
            ov3 = np.zeros((*clip_mask_np.shape, 4), dtype=np.float32)
            ov3[:, :, 0] = 1.0
            ov3[:, :, 3] = (clip_mask_np > 0.5).astype(np.float32) * 0.4
            axes[2, j].imshow(ov3)
            clip_zoom = 1024 / clip_s_actual
            clip_pct = (clip_mask_np > 0.5).sum() / clip_mask_np.size * 100
            axes[2, j].set_title(f"S={clip_s_actual}\u2192224 ({clip_zoom:.1f}x)\n{clip_pct:.2f}%", fontsize=9)
            axes[2, j].axis("off")

        axes[0, 0].set_ylabel("1024 overview\n(crop regions)", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[1, 0].set_ylabel("UNet 512\nS\u2208[512,1024]", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=80, va="center")
        axes[2, 0].set_ylabel("CLIP 224\nS\u2208[224,1024]", fontsize=10, fontweight="bold",
                               rotation=0, labelpad=80, va="center")

        legend_handles = [
            mpatches.Patch(facecolor="none", edgecolor="lime", linewidth=2.5,
                           label="UNet crop S\u2208[512,1024]"),
            mpatches.Patch(facecolor="none", edgecolor="cyan", linewidth=2.5, linestyle="--",
                           label="CLIP crop S\u2208[224,1024]"),
            mpatches.Patch(color=(0.9, 0.15, 0.15, 0.4), label="Anomaly mask"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3,
                   fontsize=9, frameon=True, borderpad=0.5)

        plt.suptitle(
            f"Sample {i} \u2014 {atype} \u2014 bbox={bbox_side}px @1024\n"
            f"Independent scale crops: UNet and CLIP see different scales each step",
            fontsize=12, fontweight="bold",
        )
        plt.tight_layout(rect=[0.09, 0.04, 1, 0.92])
        plt.savefig(save_dir / f"{i:02d}_{atype}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [{i+1}/{len(samples)}] {atype} (bbox={bbox_side}px)")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
