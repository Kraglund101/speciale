"""
Finds mint anomaly samples where the mask has >= 3 connected components.
Loads each mask, binarizes, counts connected components, and creates a
visualization grid for those with >= 3 separate anomaly regions.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

# ── Config ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\frede\desktop\kandidat\speciale")
DATA_ROOT = PROJECT_ROOT / "anomverse_extension" / "datasets" / "full_training_dataset"
MASTER_JSON = DATA_ROOT / "master_training.json"
OUTPUT_PNG = PROJECT_ROOT / "scripts" / "temp_mint_groups.png"
THRESHOLD = 128  # binarize mask: pixel > THRESHOLD → anomaly
MIN_COMPONENTS = 3
MIN_COMPONENT_SIZE = 5  # ignore tiny noise components (< 5 pixels)

# ── Load master training data ──────────────────────────────────────────
print(f"Loading master training JSON from: {MASTER_JSON}")
with open(MASTER_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

mint_samples = [item for item in data if item.get("product", "").lower() == "mint"]
print(f"Found {len(mint_samples)} mint samples in master training JSON")
print(f"Defect types: {set(item['defect_type'] for item in mint_samples)}")

# ── Analyze each mint sample's mask ────────────────────────────────────
results = []

for i, sample in enumerate(mint_samples):
    mask_rel = sample["mask_path"]
    mask_path = DATA_ROOT / mask_rel

    if not mask_path.exists():
        continue

    # Load and binarize mask
    mask_img = Image.open(mask_path).convert("L")
    mask_arr = np.array(mask_img)
    binary = (mask_arr > THRESHOLD).astype(np.uint8)

    if binary.sum() == 0:
        continue  # skip empty masks

    # Count connected components (scipy labels background as 0)
    labeled, num_features = ndimage.label(binary)

    # Filter out tiny noise components
    valid_components = 0
    for comp_id in range(1, num_features + 1):
        comp_size = np.sum(labeled == comp_id)
        if comp_size >= MIN_COMPONENT_SIZE:
            valid_components += 1

    if valid_components >= MIN_COMPONENTS:
        image_rel = sample["image_path"]
        image_path = DATA_ROOT / image_rel

        results.append({
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "num_components": valid_components,
            "defect_type": sample["defect_type"],
            "product": sample["product"],
            "total_anomaly_pixels": int(binary.sum()),
            "mask_rel": mask_rel,
            "image_rel": image_rel,
        })

    if (i + 1) % 200 == 0:
        print(f"  Processed {i + 1}/{len(mint_samples)} masks... ({len(results)} with >= {MIN_COMPONENTS} components so far)")

print(f"\nDone. Found {len(results)} mint samples with >= {MIN_COMPONENTS} connected components "
      f"(min component size: {MIN_COMPONENT_SIZE} px)")

# ── Print results ──────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'#':<4} {'Components':<12} {'Defect Type':<20} {'Anomaly Px':<12} {'Image Path'}")
print("-" * 100)
for idx, r in enumerate(sorted(results, key=lambda x: -x["num_components"])):
    print(f"{idx+1:<4} {r['num_components']:<12} {r['defect_type']:<20} {r['total_anomaly_pixels']:<12} {r['image_rel']}")
print("=" * 100)

# ── Create visualization grid ──────────────────────────────────────────
if not results:
    print("No samples found with >= 3 components. Exiting.")
    sys.exit(0)

# Sort by number of components (descending)
results_sorted = sorted(results, key=lambda x: -x["num_components"])

# Limit to max 20 for visualization
max_show = min(20, len(results_sorted))
results_show = results_sorted[:max_show]

THUMB_SIZE = 256
COLS = 3  # image | mask | labeled
ROWS = max_show
PADDING = 4
TEXT_HEIGHT = 24

cell_w = THUMB_SIZE + PADDING
cell_h = THUMB_SIZE + TEXT_HEIGHT + PADDING
grid_w = COLS * cell_w + PADDING
grid_h = ROWS * cell_h + PADDING

grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
draw = ImageDraw.Draw(grid)

# Try to get a monospace font
try:
    font = ImageFont.truetype("consola.ttf", 14)
except Exception:
    try:
        font = ImageFont.truetype("cour.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

# Color map for labeled components
def label_to_color_image(labeled_arr: np.ndarray) -> Image.Image:
    """Convert a labeled array to an RGB image with distinct colors per component."""
    np.random.seed(42)
    max_label = labeled_arr.max()
    colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
    # Background stays black
    # Generate distinct colors for each component
    for lbl in range(1, max_label + 1):
        hue = (lbl * 137) % 360  # golden angle spacing for distinct hues
        # Convert HSV to RGB (simple approach)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.9, 0.95)
        colors[lbl] = [int(r * 255), int(g * 255), int(b * 255)]

    rgb = colors[labeled_arr]
    return Image.fromarray(rgb.astype(np.uint8))


for row_idx, r in enumerate(results_show):
    y_offset = row_idx * cell_h + PADDING

    # Load image
    if os.path.exists(r["image_path"]):
        img = Image.open(r["image_path"]).convert("RGB").resize(
            (THUMB_SIZE, THUMB_SIZE), Image.LANCZOS
        )
    else:
        img = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE), (128, 128, 128))

    # Load mask
    mask_img = Image.open(r["mask_path"]).convert("L")
    mask_arr = np.array(mask_img)
    binary = (mask_arr > THRESHOLD).astype(np.uint8)
    labeled, _ = ndimage.label(binary)

    # Create mask visualization (white on black)
    mask_vis = Image.fromarray((binary * 255).astype(np.uint8)).convert("RGB").resize(
        (THUMB_SIZE, THUMB_SIZE), Image.NEAREST
    )

    # Create labeled visualization
    labeled_vis = label_to_color_image(labeled).resize(
        (THUMB_SIZE, THUMB_SIZE), Image.NEAREST
    )

    # Paste into grid
    for col_idx, tile in enumerate([img, mask_vis, labeled_vis]):
        x_offset = col_idx * cell_w + PADDING
        grid.paste(tile, (x_offset, y_offset))

    # Draw text info
    text = f"#{row_idx+1}  {r['num_components']} groups | {r['defect_type']} | {os.path.basename(r['image_rel'])}"
    text_x = PADDING
    text_y = y_offset + THUMB_SIZE + 2
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

# Add column headers at the top by shifting everything down
HEADER_H = 28
final_grid = Image.new("RGB", (grid_w, grid_h + HEADER_H), (255, 255, 255))
final_grid.paste(grid, (0, HEADER_H))
draw2 = ImageDraw.Draw(final_grid)
headers = ["Anomaly Image", "Binary Mask", "Labeled Components"]
for col_idx, header in enumerate(headers):
    x = col_idx * cell_w + PADDING + THUMB_SIZE // 2 - len(header) * 4
    draw2.text((x, 6), header, fill=(0, 0, 0), font=font)

final_grid.save(str(OUTPUT_PNG), "PNG")
print(f"\nVisualization saved to: {OUTPUT_PNG}")
print(f"Grid dimensions: {final_grid.size[0]} x {final_grid.size[1]} pixels")
print(f"Showing {max_show} of {len(results)} total samples with >= {MIN_COMPONENTS} components")
