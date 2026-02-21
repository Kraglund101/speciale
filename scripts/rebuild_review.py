"""Rebuild caption_review folder with fresh samples, overlays, and analysis."""
import json
import random
import shutil
import os

import numpy as np
from PIL import Image

random.seed(123)

BASE = "C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets"
OUT = "C:/Users/frede/Desktop/caption_review"

# Clean old files
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))

json_path = f"{BASE}/combined_datasets.json"
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Sample: 5 realiad, 5 mvtec, 4 mvtec_ad2, 3 mvtec3d = 17
chosen = []
for ds, n in [("realiad", 5), ("mvtec", 5), ("mvtec_ad2", 4), ("mvtec3d", 3)]:
    pool = [s for s in data["samples"] if s["dataset"] == ds]
    by_product: dict[str, list] = {}
    for s in pool:
        by_product.setdefault(s["product"], []).append(s)
    products = list(by_product.keys())
    random.shuffle(products)
    for prod in products[:n]:
        chosen.append(random.choice(by_product[prod]))

# Build review
results = []
for i, s in enumerate(chosen, 1):
    name = f"{i:02d}_{s['dataset']}_{s['product']}_{s['defect_type']}"

    img_path = os.path.join(BASE, s["image_path"])
    mask_path = os.path.join(BASE, s["mask_path"])

    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"SKIP {name}: file missing")
        continue

    ext_img = os.path.splitext(img_path)[1]

    # Copy image and mask
    shutil.copy2(img_path, f"{OUT}/{name}_img{ext_img}")
    shutil.copy2(mask_path, f"{OUT}/{name}_mask.png")

    # Create overlay
    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.NEAREST)

    img_arr = np.array(img, dtype=np.float32)
    mask_arr = np.array(mask, dtype=np.float32)
    binary = (mask_arr > 0).astype(np.float32)

    overlay = img_arr.copy()
    overlay[binary > 0, 0] = overlay[binary > 0, 0] * 0.5 + 255 * 0.5
    overlay[binary > 0, 1] = overlay[binary > 0, 1] * 0.5
    overlay[binary > 0, 2] = overlay[binary > 0, 2] * 0.5

    Image.fromarray(overlay.astype(np.uint8)).save(f"{OUT}/{name}_overlay.png")

    pct = binary.sum() / binary.size * 100
    results.append((name, s, pct))
    print(f"{name} | mask={pct:.2f}% | caption: {s['caption'][:100]}...")

# Write summary
with open(f"{OUT}/SUMMARY.txt", "w", encoding="utf-8") as f:
    f.write("CAPTION QUALITY REVIEW (v2 captions, fresh samples)\n")
    f.write("=" * 55 + "\n\n")
    for name, s, pct in results:
        f.write(f"{name}\n")
        f.write(f"  DATASET:     {s['dataset']}\n")
        f.write(f"  PRODUCT:     {s['product']}\n")
        f.write(f"  DEFECT TYPE: {s['defect_type']}\n")
        f.write(f"  MASK COV:    {pct:.2f}%\n")
        f.write(f"  CAPTION:     {s['caption']}\n")
        f.write(f"  CAPTION_77:  {s.get('caption_77', '')}\n")
        f.write(f"  SHORT:       {s.get('caption_short', '')}\n")
        f.write(f"\n")

print(f"\nDone! {len(results)} samples in {OUT}")
