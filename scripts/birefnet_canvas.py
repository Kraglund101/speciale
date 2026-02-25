"""
BiRefNet foreground segmentation on images.
Saves binary masks to an output directory.

Usage:
    python scripts/birefnet_canvas.py --input <image_dir> --output <mask_dir>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

MODEL_NAME = "ZhengPeng7/BiRefNet"
INPUT_SIZE = (1024, 1024)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".JPG"}


def main() -> None:
    parser = argparse.ArgumentParser(description="BiRefNet foreground segmentation")
    parser.add_argument("--input", required=True, type=str, help="Directory of input images")
    parser.add_argument("--output", required=True, type=str, help="Directory to save masks")
    args = parser.parse_args()

    IMAGE_DIR = Path(args.input)
    OUTPUT_DIR = Path(args.output)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))

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

    images = sorted(f for f in IMAGE_DIR.iterdir() if f.suffix.lower() in EXTS)
    print(f"Found {len(images)} canvas normal images")

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
            binary = (mask > 0.5).astype(np.float32)

            binary_pil = Image.fromarray((binary * 255).astype(np.uint8))
            binary_pil = binary_pil.resize(original_size, Image.BILINEAR)
            binary_pil.save(OUTPUT_DIR / f"{img_path.stem}.png")

            success += 1
        except Exception as e:
            failed += 1
            tqdm.write(f"FAILED {img_path.name}: {e}")

    print(f"\nDone — {success} ok, {failed} failed")
    print(f"Masks saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
