"""
Resize all images > 1024x1024 down to 1024x1024 in-place.

Image: PIL bilinear
Mask:  torch adaptive_max_pool2d (preserves small anomalies)
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from collections import Counter

def main():
    root = Path(__file__).parent.parent
    master = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"

    with open(master, encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Total samples: {len(samples)}")

    # Track unique file paths (some samples may share images)
    seen_images = set()
    seen_masks = set()
    resized_images = 0
    resized_masks = 0
    size_counter = Counter()
    skipped = 0

    for i, s in enumerate(samples):
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]

        # --- Image ---
        if str(img_path) not in seen_images and img_path.exists():
            seen_images.add(str(img_path))
            img = Image.open(str(img_path))
            w, h = img.size
            if w > 1024 or h > 1024:
                size_counter[f"{w}x{h}"] += 1
                img_resized = img.convert("RGB").resize((1024, 1024), Image.BILINEAR)
                img_resized.save(str(img_path))
                img_resized.close()
                resized_images += 1
            img.close()

        # --- Mask ---
        if str(mask_path) not in seen_masks and mask_path.exists():
            seen_masks.add(str(mask_path))
            mask = Image.open(str(mask_path)).convert("L")
            w, h = mask.size
            if w > 1024 or h > 1024:
                mask_np = np.array(mask).astype(np.float32) / 255.0
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                mask_1024 = F.adaptive_max_pool2d(mask_t, output_size=(1024, 1024))
                mask_1024_np = (mask_1024.squeeze().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_1024_np, mode="L")
                mask_pil.save(str(mask_path))
                mask_pil.close()
                resized_masks += 1
            mask.close()

        if (i + 1) % 5000 == 0:
            print(f"  scanned {i+1}/{len(samples)} "
                  f"(resized {resized_images} imgs, {resized_masks} masks so far)")

    print(f"\nDone!")
    print(f"Resized {resized_images} images, {resized_masks} masks")
    print(f"Size distribution of resized files:")
    for key, count in sorted(size_counter.items(), key=lambda x: -x[1]):
        print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
