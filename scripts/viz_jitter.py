"""Visualize brightness/contrast jitter on 100 random samples."""
import sys, random, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load dataset
splits_dir = Path("anomverse_extension/datasets/full_training_dataset/splits_by_type")
data_root = Path("anomverse_extension/datasets/full_training_dataset")
exclude = {"MulSen_AD", "mvtec3d", "MANTA_TINY_256"}

samples = []
for f in sorted(splits_dir.glob("*.json")):
    data = json.loads(f.read_text())
    for s in data["images"]:
        source = s.get("source_dataset", "")
        if source not in exclude:
            samples.append(s)

random.seed(42)
chosen = random.sample(samples, min(100, len(samples)))

out_dir = Path("results/jitter_examples")
out_dir.mkdir(exist_ok=True)

brightness_range = (0.85, 1.15)
contrast_range = (0.85, 1.15)

for i, s in enumerate(chosen):
    img_path = data_root / s["image_path"]
    mask_path = data_root / s["mask_path"]

    img = Image.open(img_path).convert("RGB").resize((512, 512))
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.NEAREST)

    b_factor = random.uniform(*brightness_range)
    c_factor = random.uniform(*contrast_range)

    jittered = ImageEnhance.Brightness(img).enhance(b_factor)
    jittered = ImageEnhance.Contrast(jittered).enhance(c_factor)

    img_np = np.array(img).astype(float)
    jit_np = np.array(jittered).astype(float)
    mask_np = (np.array(mask) > 127).astype(float)

    overlay_orig = img_np.copy()
    overlay_orig[..., 0] = np.clip(overlay_orig[..., 0] + mask_np * 80, 0, 255)
    overlay_orig[..., 1] = overlay_orig[..., 1] * (1 - mask_np * 0.3)
    overlay_orig[..., 2] = overlay_orig[..., 2] * (1 - mask_np * 0.3)

    overlay_jit = jit_np.copy()
    overlay_jit[..., 0] = np.clip(overlay_jit[..., 0] + mask_np * 80, 0, 255)
    overlay_jit[..., 1] = overlay_jit[..., 1] * (1 - mask_np * 0.3)
    overlay_jit[..., 2] = overlay_jit[..., 2] * (1 - mask_np * 0.3)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    ax0.imshow(overlay_orig.astype(np.uint8))
    ax0.set_title("Original", fontsize=10)
    ax0.axis("off")
    ax1.imshow(overlay_jit.astype(np.uint8))
    ax1.set_title(f"Jittered (b={b_factor:.2f}, c={c_factor:.2f})", fontsize=10)
    ax1.axis("off")
    atype = s.get("anomaly_type", Path(img_path).parent.name)
    product = s.get("product", "")
    fig.suptitle(f"#{i} — {product} / {atype}", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / f"{i:03d}.png", dpi=100, bbox_inches="tight")
    plt.close()

print(f"Saved 100 images to {out_dir}/")
