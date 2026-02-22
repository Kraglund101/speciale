"""
Visualize the full transform pipeline for ONE easy example:
  1) Original 1024x1024 real anomaly + mask
  2) Tight-crop of mask (bbox extraction)
  3) After flip(s)
  4) After rotation
  5) After scale
  6) Placed on canvas (full 1024x1024)
  7) CLIP crop (224x224 from placed mask)
  8) CLIP 16x16 token grid

Shows exactly what transforms happen and in what order.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.crop_utils import clip_crop
from src.utils.mask_utils import downsample_mask_maxpool

SEED = 123  # Different seed for variety
random.seed(SEED)
np.random.seed(SEED)

EXP = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\VisA_validation_dataset\datasets\easy_test\cashew\experiment_UniNet"
)
VIZ_SIZE = 512
SCALE_RANGE = (0.8, 1.2)


def load_binary_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def to_viz(img, size=VIZ_SIZE):
    if isinstance(img, np.ndarray):
        img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8))
    return img.resize((size, size), Image.LANCZOS)


def add_label(img, text, font_size=14):
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 4), text, fill=(255, 255, 255), font=font)
    return img


def overlay_mask_rgba(base_img, mask, color, alpha):
    base = base_img.convert("RGBA") if isinstance(base_img, Image.Image) else Image.fromarray(base_img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    if mask.shape[:2] != (base.size[1], base.size[0]):
        mask = np.array(Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (base.size[0], base.size[1]), Image.NEAREST)).astype(np.float32) / 255.0
    mask_bool = mask > 0.5
    overlay_arr[mask_bool] = [color[0], color[1], color[2], int(alpha * 255)]
    return Image.alpha_composite(base, Image.fromarray(overlay_arr)).convert("RGB")


# Pick one easy example
ref_id = "036"  # A medium-sized easy anomaly
ref_img_path = EXP / "source" / "anomalies" / "easy" / "imgs" / f"{ref_id}.JPG"
ref_mask_path = EXP / "source" / "anomalies" / "easy" / "masks" / f"{ref_id}.png"
canvas_path = EXP / "source" / "normals" / "canvas" / "imgs" / "031.JPG"
canvas_fg_path = EXP / "source" / "normals" / "canvas" / "fg_masks" / "031.png"

ref_img = Image.open(ref_img_path).convert("RGB")
ref_mask = load_binary_mask(ref_mask_path)
canvas_img = Image.open(canvas_path).convert("RGB")
canvas_fg = load_binary_mask(canvas_fg_path)

img_h, img_w = ref_mask.shape
print(f"Original image: {img_w}x{img_h}")
print(f"Mask pixels: {ref_mask.sum():.0f} ({ref_mask.sum()/(img_h*img_w)*100:.2f}%)")

# === Step 1: Original 1024x1024 with mask ===
c1 = add_label(to_viz(ref_img), f"1. Original ({ref_id}) {img_w}x{img_h}")
c2 = add_label(to_viz(overlay_mask_rgba(ref_img, ref_mask, (255, 30, 30), 0.5)),
               f"2. Original + mask ({ref_mask.sum():.0f}px)")

# === Step 2: Tight crop (bbox extraction) ===
ys, xs = np.where(ref_mask > 0.5)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1
crop = ref_mask[y0:y1, x0:x1]
crop_img = np.array(ref_img)[y0:y1, x0:x1]
print(f"Tight crop: {x1-x0}x{y1-y0} (from bbox [{x0},{y0}]-[{x1},{y1}])")

# Show crop with mask overlay
crop_overlay = overlay_mask_rgba(Image.fromarray(crop_img), crop, (255, 30, 30), 0.5)
c3 = add_label(to_viz(crop_overlay), f"3. Tight crop {x1-x0}x{y1-y0}")

# === Step 3: After flips ===
flip_h = random.random() < 0.5
flip_v = random.random() < 0.5
flipped = crop.copy()
flipped_img = crop_img.copy()
flip_label = ""
if flip_v:
    flipped = flipped[::-1, :].copy()
    flipped_img = flipped_img[::-1, :].copy()
    flip_label += "V"
if flip_h:
    flipped = flipped[:, ::-1].copy()
    flipped_img = flipped_img[:, ::-1].copy()
    flip_label += "H"
if not flip_label:
    flip_label = "none"
print(f"Flips: {flip_label}")

flipped_overlay = overlay_mask_rgba(Image.fromarray(flipped_img), flipped, (255, 30, 30), 0.5)
c4 = add_label(to_viz(flipped_overlay), f"4. Flip: {flip_label}")

# === Step 4: After rotation ===
angle = random.uniform(0, 360)
crop_pil = Image.fromarray((flipped * 255).astype(np.uint8))
rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)
rotated_arr = (np.array(rotated).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
rh, rw = rotated_arr.shape
print(f"Rotation: {angle:.1f}° -> {rw}x{rh}")

# Show rotated mask on gray background
rot_vis = np.full((rh, rw, 3), 40, dtype=np.uint8)
rot_vis[rotated_arr > 0.5] = [255, 30, 30]
c5 = add_label(to_viz(Image.fromarray(rot_vis)), f"5. Rotate {angle:.1f}° -> {rw}x{rh}")

# === Step 5: After scale ===
scale = random.uniform(*SCALE_RANGE)
new_w, new_h = max(1, int(rw * scale)), max(1, int(rh * scale))
scaled = rotated.resize((new_w, new_h), Image.NEAREST)
scaled_arr = (np.array(scaled).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
print(f"Scale: {scale:.2f}x -> {new_w}x{new_h}")

# Show scaled mask on gray background
sc_vis = np.full((new_h, new_w, 3), 40, dtype=np.uint8)
sc_vis[scaled_arr > 0.5] = [255, 30, 30]
c6 = add_label(to_viz(Image.fromarray(sc_vis)), f"6. Scale {scale:.2f}x -> {new_w}x{new_h}")

# === Step 6: Placed on canvas ===
# Try to place it
H, W = canvas_fg.shape
fg_ys, fg_xs = np.where(canvas_fg > 0.5)
sh, sw = scaled_arr.shape
placed = None
for _ in range(2000):
    idx = random.randint(0, len(fg_ys) - 1)
    cy, cx = fg_ys[idx], fg_xs[idx]
    top, left = cy - sh // 2, cx - sw // 2
    if top < 0 or left < 0 or top + sh > H or left + sw > W:
        continue
    candidate = np.zeros((H, W), dtype=np.float32)
    candidate[top:top + sh, left:left + sw] = scaled_arr
    if candidate.sum() == (candidate * canvas_fg).sum():
        placed = candidate
        break

if placed is not None:
    cov = placed.sum() / (H * W) * 100
    print(f"Placed: {placed.sum():.0f}px ({cov:.2f}%) on canvas 031")
    placed_vis = overlay_mask_rgba(canvas_img, placed, (255, 30, 30), 0.5)
    c7 = add_label(to_viz(placed_vis), f"7. Placed on canvas ({cov:.2f}%)")
else:
    print("WARNING: Failed to place, showing empty canvas")
    c7 = add_label(to_viz(canvas_img), "7. FAILED TO PLACE")
    placed = np.zeros((H, W), dtype=np.float32)

# === Step 7: CLIP crop (224x224) ===
# clip_crop expects [C,H,W] float tensor for image, [1,H,W] for mask
canvas_arr = np.array(canvas_img).astype(np.float32) / 255.0
canvas_t = torch.from_numpy(canvas_arr).permute(2, 0, 1)  # [3, H, W]
placed_t = torch.from_numpy(placed).unsqueeze(0)  # [1, H, W]
clip_img_t, clip_mask_t = clip_crop(canvas_t, placed_t, crop_size=224)
# Convert back to numpy/PIL
clip_img_224 = Image.fromarray((clip_img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
clip_mask_arr = clip_mask_t[0].numpy()
print(f"CLIP crop: 224x224, mask pixels: {(clip_mask_arr > 0.5).sum():.0f}")

clip_overlay = overlay_mask_rgba(clip_img_224, clip_mask_arr, (255, 30, 30), 0.5)
c8 = add_label(to_viz(clip_overlay), f"8. CLIP crop 224x224 ({(clip_mask_arr > 0.5).sum():.0f}px)")

# === Step 8: CLIP 16x16 token grid ===
mask_t = torch.from_numpy(clip_mask_arr).unsqueeze(0).unsqueeze(0).float()
clip_mask_16 = downsample_mask_maxpool(mask_t, 16)[0, 0].numpy()
n_active = int((clip_mask_16 > 0.5).sum())

base_clip = clip_img_224.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
cell_sz = VIZ_SIZE // 16
overlay_clip = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
for r in range(16):
    for c in range(16):
        if clip_mask_16[r, c] > 0.5:
            for y in range(r * cell_sz, (r + 1) * cell_sz):
                for x in range(c * cell_sz, (c + 1) * cell_sz):
                    overlay_clip.putpixel((x, y), (255, 30, 30, 100))
base_clip = Image.alpha_composite(base_clip, overlay_clip)
draw = ImageDraw.Draw(base_clip)
for k in range(17):
    pos = k * cell_sz
    draw.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 60), width=1)
    draw.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 60), width=1)
c9 = add_label(base_clip.convert("RGB"), f"9. CLIP 16x16 tokens ({n_active}/256)")

# === Assemble panel ===
# Two rows: 5 columns each (last cell of row 2 is padding)
cols_per_row = 5
row_h = c1.height
panel_w = VIZ_SIZE * cols_per_row
panel_h = row_h * 2
panel = Image.new("RGB", (panel_w, panel_h), (20, 20, 20))

row1 = [c1, c2, c3, c4, c5]
row2 = [c6, c7, c8, c9]

for j, col in enumerate(row1):
    panel.paste(col, (j * VIZ_SIZE, 0))
for j, col in enumerate(row2):
    panel.paste(col, (j * VIZ_SIZE, row_h))

out_path = EXP.parent / "viz" / "transform_pipeline.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
panel.save(out_path, quality=95)
print(f"\nSaved: {out_path}")
