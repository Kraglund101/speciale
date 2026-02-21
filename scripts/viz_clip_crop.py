"""Visualize CLIP crop pipeline on 100 random samples.

Shows: full image | CLIP 224x224 crop with 16x16 grid overlay (active tokens highlighted)
"""
import sys, random, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.crop_utils import clip_crop
from torchvision import transforms

# Load dataset
splits_dir = Path("anomverse_extension/datasets/full_training_dataset/splits_by_type")
data_root = Path("anomverse_extension/datasets/full_training_dataset")
exclude = {"MulSen_AD", "mvtec3d", "MANTA_TINY_256"}

samples = []
for f in sorted(splits_dir.glob("*.json")):
    data = json.loads(f.read_text())
    atype = f.stem
    for s in data["images"]:
        source = s.get("source_dataset", "")
        if source not in exclude:
            s["anomaly_type"] = atype
            samples.append(s)

random.seed(42)
chosen = random.sample(samples, min(100, len(samples)))

out_dir = Path("results/clip_crop_examples")
out_dir.mkdir(exist_ok=True)
# Clean old files
for old in out_dir.glob("*.png"):
    old.unlink()

to_tensor = transforms.ToTensor()
GRID = 16
CELL = 224 // GRID  # 14px per token

for i, s in enumerate(chosen):
    img_path = data_root / s["image_path"]
    mask_path = data_root / s["mask_path"]

    img_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")
    orig_w, orig_h = img_pil.size

    img_t = to_tensor(img_pil)       # [3, H, W] in [0, 1]
    mask_t = to_tensor(mask_pil)     # [1, H, W]
    mask_t = (mask_t > 0.5).float()

    # CLIP crop
    crop_img, crop_mask = clip_crop(img_t, mask_t, crop_size=224, pad_frac=0.10)

    # 16x16 maxpool token mask
    mask_16 = F.max_pool2d(crop_mask.unsqueeze(0), kernel_size=CELL)  # [1,1,16,16]
    mask_16_np = mask_16[0, 0].numpy()
    n_active = (mask_16_np > 0).sum()

    # Convert for display
    img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    crop_np = (crop_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    mask_full_np = mask_t[0].numpy()

    # Full image: red overlay
    full_overlay = img_np.astype(float).copy()
    full_overlay[..., 0] = np.clip(full_overlay[..., 0] + mask_full_np * 80, 0, 255)
    full_overlay[..., 1] *= (1 - mask_full_np * 0.3)
    full_overlay[..., 2] *= (1 - mask_full_np * 0.3)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5.5))

    # Left: full image with anomaly overlay
    ax0.imshow(full_overlay.astype(np.uint8))
    ax0.set_title(f"Full {orig_h}x{orig_w}", fontsize=9)
    ax0.axis("off")

    # Right: CLIP crop with 16x16 grid overlay
    ax1.imshow(crop_np)

    # Draw grid + highlight active tokens
    for gy in range(GRID):
        for gx in range(GRID):
            x0 = gx * CELL
            y0 = gy * CELL
            if mask_16_np[gy, gx] > 0:
                # Active token: red semi-transparent fill
                rect = Rectangle((x0, y0), CELL, CELL,
                                 linewidth=0.8, edgecolor='red',
                                 facecolor='red', alpha=0.3)
                ax1.add_patch(rect)
            else:
                # Inactive token: faint grid line
                rect = Rectangle((x0, y0), CELL, CELL,
                                 linewidth=0.3, edgecolor='white',
                                 facecolor='none', alpha=0.3)
                ax1.add_patch(rect)

    ax1.set_title(f"CLIP 224x224 — {n_active}/256 tokens active", fontsize=9)
    ax1.axis("off")

    atype = s.get("anomaly_type", "?")
    product = s.get("product", "")
    fig.suptitle(f"#{i} — {product} / {atype}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{i:03d}.png", dpi=150, bbox_inches="tight")
    plt.close()

print(f"Saved 100 images to {out_dir}/")
