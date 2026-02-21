"""Visualize how flips + jitter change anomaly appearance."""
from PIL import Image, ImageEnhance, ImageDraw
import torchvision.transforms.functional as TF
from pathlib import Path
import glob

# Find the fold image
data_root = Path(r"C:\Users\frede\desktop\kandidat\speciale\anomverse_extension\datasets\full_training_dataset")
fold_candidates = list(data_root.glob("**/fold/*016*"))
print(f"Found: {[str(p) for p in fold_candidates[:5]]}")

# Use first match or fallback
img_path = fold_candidates[0] if fold_candidates else None
if img_path is None:
    print("Could not find fold/016 image")
    exit(1)

print(f"Using: {img_path}")
img = Image.open(img_path).convert("RGB").resize((512, 512))

# Create variants
original = img.copy()
h_flip = TF.hflip(img)
v_flip = TF.vflip(img)
both_flip = TF.vflip(TF.hflip(img))

dark_hi = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(0.85)).enhance(1.15)
bright_lo = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1.15)).enhance(0.85)
v_flip_dark = ImageEnhance.Contrast(ImageEnhance.Brightness(v_flip).enhance(0.85)).enhance(1.15)

def label(im, text):
    im = im.copy()
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 512, 28], fill="black")
    draw.text((10, 5), text, fill="white")
    return im

imgs = [
    label(original, "Original"),
    label(h_flip, "H-Flip"),
    label(v_flip, "V-Flip"),
    label(both_flip, "H+V Flip"),
    label(dark_hi, "Dark + High Contrast"),
    label(bright_lo, "Bright + Low Contrast"),
    label(v_flip_dark, "V-Flip + Dark + High Contrast"),
    label(Image.new("RGB", (512, 512), (40, 40, 40)), ""),  # padding
]

# 2x4 grid
grid = Image.new("RGB", (512 * 4, 512 * 2))
for i, im in enumerate(imgs[:4]):
    grid.paste(im, (512 * i, 0))
for i, im in enumerate(imgs[4:]):
    grid.paste(im, (512 * i, 512))

out = Path(r"C:\Users\frede\desktop\kandidat\speciale\results\flip_demo.png")
out.parent.mkdir(parents=True, exist_ok=True)
grid.save(out)
print(f"Saved: {out}")
