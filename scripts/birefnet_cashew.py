"""
BiRefNet foreground segmentation on VisA cashew normal images.
Saves binary + soft masks into a 'birefnet_masks' folder alongside the images.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────
CASHEW_DIR = Path(
    r"C:\Users\frede\desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy\cashew"
)
IMAGE_DIR = CASHEW_DIR / "Data" / "Images" / "Normal"
OUTPUT_DIR = CASHEW_DIR / "birefnet_masks" / "Normal"

MODEL_NAME = "ZhengPeng7/BiRefNet"
INPUT_SIZE = (1024, 1024)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main() -> None:
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

    # Load model
    from transformers import AutoModelForImageSegmentation

    print(f"Loading {MODEL_NAME} ...")
    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    ).to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Collect images
    images = sorted(f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in EXTS)
    print(f"Found {len(images)} normal cashew images")
    if not images:
        print("No images found — check IMAGE_DIR path.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    success, failed = 0, 0
    for img_path in tqdm(images, desc="BiRefNet"):
        try:
            image = Image.open(img_path).convert("RGB")
            original_size = image.size

            inp = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(inp)
                pred = preds[-1] if isinstance(preds, (list, tuple)) else preds
                pred = pred.sigmoid()

            mask = pred[0].squeeze().cpu().numpy()

            # Binary mask (threshold 0.5)
            binary = (mask > 0.5).astype(np.float32)
            binary_pil = Image.fromarray((binary * 255).astype(np.uint8))
            binary_pil = binary_pil.resize(original_size, Image.BILINEAR)
            binary_pil.save(OUTPUT_DIR / f"{img_path.stem}_binary.png")

            # Soft probability mask
            soft_pil = Image.fromarray((mask * 255).astype(np.uint8))
            soft_pil = soft_pil.resize(original_size, Image.BILINEAR)
            soft_pil.save(OUTPUT_DIR / f"{img_path.stem}_soft.png")

            success += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"FAILED {img_path.name}: {e}")

    print(f"\nDone — {success} ok, {failed} failed")
    print(f"Masks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
