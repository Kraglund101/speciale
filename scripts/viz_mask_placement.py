"""
Generate placed anomaly masks for all generation anomalies (47 easy + 3 hard).
- Uniform rotation in [0, 360]
- Scale in [0.8, 1.2]
- Place within BiRefNet foreground region (ALL pixels must be inside FG)
- Dilate with band_mode=2 at 64x64
- Save viz panels + placed masks
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────
EXP = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy_test\cashew\experiment_UniNet"
)
CANVAS_IMGS = EXP / "source" / "normals" / "canvas" / "imgs"
CANVAS_MASKS = EXP / "source" / "normals" / "canvas" / "fg_masks"

VIZ_SIZE = 512
SCALE_RANGE = (0.8, 1.2)

import json

# Load splits to get correct ref pool (test_anomalies only)
with open(EXP / "splits.json") as f:
    splits = json.load(f)


def load_binary_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def place_anomaly_mask(
    anomaly_mask: np.ndarray,
    foreground_mask: np.ndarray,
    max_position_attempts: int = 500,
) -> np.ndarray | None:
    """Place entire anomaly mask so ALL pixels are within foreground."""
    H, W = foreground_mask.shape
    ys, xs = np.where(anomaly_mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = anomaly_mask[y0:y1, x0:x1]

    fg_ys, fg_xs = np.where(foreground_mask > 0.5)
    if len(fg_ys) == 0:
        return None

    # Random flips (independent, p=0.5 each)
    if random.random() < 0.5:
        crop = crop[::-1, :].copy()   # vertical flip
    if random.random() < 0.5:
        crop = crop[:, ::-1].copy()   # horizontal flip

    # Uniform rotation [0, 360]
    angle = random.uniform(0, 360)
    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
    rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)

    # Scale [0.8, 1.2]
    scale = random.uniform(*SCALE_RANGE)
    rw, rh = rotated.size
    new_w, new_h = max(1, int(rw * scale)), max(1, int(rh * scale))
    scaled = rotated.resize((new_w, new_h), Image.NEAREST)
    scaled_arr = (np.array(scaled).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    if scaled_arr.sum() == 0:
        return None

    sh, sw = scaled_arr.shape
    for _ in range(max_position_attempts):
        idx = random.randint(0, len(fg_ys) - 1)
        cy, cx = fg_ys[idx], fg_xs[idx]
        top, left = cy - sh // 2, cx - sw // 2
        if top < 0 or left < 0 or top + sh > H or left + sw > W:
            continue
        placed = np.zeros((H, W), dtype=np.float32)
        placed[top:top + sh, left:left + sw] = scaled_arr
        # ALL anomaly pixels must be within foreground
        if placed.sum() == (placed * foreground_mask).sum():
            return placed
    return None


def overlay_mask_rgba(img: Image.Image, mask: np.ndarray, color: tuple, alpha: float) -> Image.Image:
    base = img.convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    overlay[:, :, 3] = (mask * alpha * 255).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(overlay)).convert("RGB")


def make_latent_viz(tensor_64: torch.Tensor) -> Image.Image:
    arr = tensor_64[0, 0].numpy()
    return Image.fromarray((arr * 255).astype(np.uint8)).resize(
        (VIZ_SIZE, VIZ_SIZE), Image.NEAREST
    ).convert("RGB")


def add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    bar_h = font_size + 10
    out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 3), text, fill=(255, 255, 200), font=font)
    return out


def to_viz(img: Image.Image) -> Image.Image:
    return img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS)


# ── Collect canvas normals ────────────────────────────────────────────────
canvas_imgs = sorted(CANVAS_IMGS.glob("*.JPG"))
print(f"Canvas normals: {len(canvas_imgs)}")

# Clean stale placed masks from previous runs
easy_placed_dir = EXP / "synthetic" / "placed_masks" / "easy"
easy_placed_dir.mkdir(parents=True, exist_ok=True)
for old in easy_placed_dir.glob("*.png"):
    old.unlink()
easy_viz_dir = EXP.parent / "viz" / "easy"
easy_viz_dir.mkdir(parents=True, exist_ok=True)
for old in easy_viz_dir.glob("*.png"):
    old.unlink()

# Only use test_anomalies.easy as refs (hard handled by redo_hard_placement.py)
easy_ref_ids = splits["experiment1"]["test_anomalies"]["easy"]  # 47 filenames
hard_ref_ids = splits["experiment1"]["test_anomalies"]["hard"]  # 3 filenames
# Reserve 3 canvases for hard (pick last 3 alphabetically from unused)
hard_canvas_count = len(hard_ref_ids)

ref_dir = EXP / "source" / "anomalies" / "easy" / "masks"
all_ref_masks = []
for fname in easy_ref_ids:
    mask_name = fname.replace(".JPG", ".png")
    mask_path = ref_dir / mask_name
    if mask_path.exists() and load_binary_mask(mask_path).sum() > 50:
        all_ref_masks.append((mask_path, "easy"))
print(f"  easy refs (from test_anomalies): {len(all_ref_masks)}/{len(easy_ref_ids)}")

# Shuffle refs and pair with first N canvases (leave last 3 for hard)
random.shuffle(all_ref_masks)
easy_canvases = canvas_imgs[:-hard_canvas_count] if hard_canvas_count > 0 else canvas_imgs
n_total = min(len(easy_canvases), len(all_ref_masks))
pairs = list(zip(easy_canvases[:n_total], all_ref_masks[:n_total]))
print(f"Generating {n_total} placements...\n")

from src.utils.crop_utils import clip_crop

total_success, total_failed = 0, 0
manifest = []
for i, (canvas_path, (initial_ref_path, diff)) in enumerate(pairs):
    canvas_img = Image.open(canvas_path).convert("RGB")
    canvas_fg = load_binary_mask(CANVAS_MASKS / f"{canvas_path.stem}.png")
    img_w, img_h = canvas_img.size

    # Resolve paths for this difficulty
    ref_imgs_dir = EXP / "source" / "anomalies" / diff / "imgs"
    viz_dir = EXP.parent / "viz" / diff
    viz_dir.mkdir(parents=True, exist_ok=True)
    placed_masks_dir = EXP / "synthetic" / "placed_masks" / diff
    placed_masks_dir.mkdir(parents=True, exist_ok=True)

    # Try initial ref 5 times, then try other refs from same difficulty
    placed = None
    ref_path = initial_ref_path
    same_diff_refs = [p for p, d in all_ref_masks if d == diff]
    candidates = [initial_ref_path] + [r for r in same_diff_refs if r != initial_ref_path]
    for cand in candidates:
        ref_mask = load_binary_mask(cand)
        for _attempt in range(5):
            placed = place_anomaly_mask(ref_mask, canvas_fg)
            if placed is not None:
                ref_path = cand
                break
        if placed is not None:
            break
    if ref_path != initial_ref_path:
        print(f"    (canvas={canvas_path.stem}: swapped ref {initial_ref_path.stem} → {ref_path.stem})")

    if placed is None:
        print(f"  [{i+1:2d}/{n_total}] [{diff}] canvas={canvas_path.stem} — FAILED ALL REFS")
        total_failed += 1
        continue
    ref_mask = load_binary_mask(ref_path)

    # Downsample to 64x64
    placed_512 = np.array(
        Image.fromarray((placed * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    ).astype(np.float32) / 255.0
    placed_512 = (placed_512 > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(placed_512).unsqueeze(0).unsqueeze(0).float()
    core_64 = downsample_mask_maxpool(mask_t, 64)

    # Band dilation on core (matches training: no extra shadow dilation)
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=2, core_ratio=0.8
    )

    # Roundtrip for col 7
    core_up = F.interpolate(core_64, size=(img_h, img_w), mode="nearest")[0, 0].numpy()
    band_up = F.interpolate(band_mask, size=(img_h, img_w), mode="nearest")[0, 0].numpy()

    coverage = placed.sum() / (img_h * img_w) * 100
    print(f"  [{i+1:2d}/{n_total}] [{diff}] canvas={canvas_path.stem} ref={ref_path.stem} "
          f"— {placed.sum():.0f}px ({coverage:.2f}%) core={core_64.sum().item():.0f} "
          f"dilated={dilated_binary.sum().item():.0f}")

    # ── Save placed mask (image resolution) ───────────────────────────
    placed_pil = Image.fromarray((placed * 255).astype(np.uint8))
    placed_pil.save(placed_masks_dir / f"{canvas_path.stem}.png")

    # ── Load real anomaly image + CLIP crop ─────────────────────────
    ref_img_path = ref_imgs_dir / f"{ref_path.stem}.JPG"
    ref_img = Image.open(ref_img_path).convert("RGB") if ref_img_path.exists() else None

    # CLIP crop of the real anomaly (what conditions the IP-Adapter)
    clip_crop_img = None
    if ref_img is not None:
        ref_tensor = torch.from_numpy(
            np.array(ref_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]
        ref_mask_tensor = torch.from_numpy(ref_mask).unsqueeze(0).float()  # [1, H, W]
        clip_img_t, clip_mask_t = clip_crop(ref_tensor, ref_mask_tensor, crop_size=224)
        # Convert back to PIL
        clip_np = (clip_img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        clip_crop_img = Image.fromarray(clip_np)
        clip_mask_np = clip_mask_t.squeeze(0).numpy()
        clip_crop_with_mask = overlay_mask_rgba(clip_crop_img, clip_mask_np, (255, 30, 30), 0.45)

    # ── CLIP mask at 16x16 (IP-Adapter token grid) + 1px dilation ──
    clip_mask_16 = None
    clip_dilation_only = None
    if clip_crop_img is not None:
        clip_mask_16_t = downsample_mask_maxpool(
            clip_mask_t.unsqueeze(0), 16  # [1, 1, 16, 16]
        )
        clip_mask_16 = clip_mask_16_t[0, 0].numpy()  # [16, 16]
        # +1px shadow dilation at 16x16
        clip_dilated_16_t = F.max_pool2d(clip_mask_16_t, kernel_size=3, stride=1, padding=1)
        clip_dilated_16_t = (clip_dilated_16_t > 0.5).float()
        clip_dilated_16 = clip_dilated_16_t[0, 0].numpy()
        clip_dilation_only = (
            (clip_dilated_16 > 0.5) & (clip_mask_16 < 0.5)
        ).astype(np.float32)

    # ── Build viz panel (1 row x 6 cols) ──────────────────────────────
    if ref_img is not None:
        c1 = add_label(to_viz(ref_img), f"Real anomaly ({ref_path.stem}) [{diff}]")
        c2 = add_label(to_viz(overlay_mask_rgba(ref_img, ref_mask, (255, 30, 30), 0.5)),
                        "Real anomaly + mask")
    else:
        c1 = add_label(Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)), "N/A")
        c2 = add_label(Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)), "N/A")

    if clip_crop_img is not None:
        c3 = add_label(clip_crop_with_mask.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS),
                        "CLIP crop 224x224 (IP-Adapter input)")
    else:
        c3 = add_label(Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)), "N/A")

    # Col 4: 16x16 CLIP mask — RED=original, CYAN=+1px dilation (hard only)
    if clip_mask_16 is not None:
        n_orig = int(clip_mask_16.sum())
        base_clip = clip_crop_img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
        cell_sz = VIZ_SIZE // 16
        overlay = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
        for r in range(16):
            for c_idx in range(16):
                if clip_mask_16[r, c_idx] > 0.5:
                    color = (255, 30, 30, 100)  # RED = original
                elif diff == "hard" and clip_dilation_only[r, c_idx] > 0.5:
                    color = (0, 220, 255, 100)  # CYAN = dilation (hard only)
                else:
                    continue
                for y in range(r * cell_sz, (r + 1) * cell_sz):
                    for x in range(c_idx * cell_sz, (c_idx + 1) * cell_sz):
                        overlay.putpixel((x, y), color)
        base_clip = Image.alpha_composite(base_clip, overlay)
        draw = ImageDraw.Draw(base_clip)
        for k in range(17):
            pos = k * cell_sz
            draw.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 80), width=1)
            draw.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 80), width=1)
        if diff == "hard":
            n_dilated = int(clip_dilated_16.sum())
            n_new = n_dilated - n_orig
            c4_label = f"CLIP 16x16: {n_orig}(red) +{n_new}(cyan) = {n_dilated}"
        else:
            c4_label = f"CLIP 16x16 tokens ({n_orig}/256 active)"
        c4 = add_label(base_clip.convert("RGB"), c4_label)
    else:
        c4 = add_label(Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)), "N/A")

    # Col 5: Placed mask on canvas (transformed: rotated + scaled)
    placed_on_fg = overlay_mask_rgba(canvas_img, canvas_fg, (0, 200, 0), 0.2)
    placed_on_fg = overlay_mask_rgba(placed_on_fg, placed, (255, 30, 30), 0.5)
    c5 = add_label(to_viz(placed_on_fg),
                    f"Placed on canvas {canvas_path.stem} ({coverage:.2f}%)")

    # Col 6: 64x64 with band=2 — same style as CLIP grid (overlay + grid lines)
    core_64_vis = core_64[0, 0].numpy()
    band_64_vis = band_mask[0, 0].numpy()
    base_64 = canvas_img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
    cell_sz_64 = VIZ_SIZE // 64
    overlay_64 = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
    for r in range(64):
        for c_idx in range(64):
            if core_64_vis[r, c_idx] > 0.5:
                color = (255, 0, 0, 150)
            elif band_64_vis[r, c_idx] > 0.5:
                color = (255, 220, 0, 150)
            else:
                continue
            for y in range(r * cell_sz_64, (r + 1) * cell_sz_64):
                for x in range(c_idx * cell_sz_64, (c_idx + 1) * cell_sz_64):
                    overlay_64.putpixel((x, y), color)
    base_64 = Image.alpha_composite(base_64, overlay_64)
    draw_64 = ImageDraw.Draw(base_64)
    for k in range(65):
        pos = k * cell_sz_64
        draw_64.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 40), width=1)
        draw_64.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 40), width=1)
    c6 = add_label(base_64.convert("RGB"),
                    f"64x64 core(red)+band(yellow) [c={core_64.sum().item():.0f} b={band_mask.sum().item():.0f}]")

    # Col 7: Roundtrip — dilated 64x64 mapped back to image res on canvas
    rt = overlay_mask_rgba(canvas_img, band_up, (255, 220, 0), 0.45)
    rt = overlay_mask_rgba(rt, core_up, (255, 0, 0), 0.55)
    c7 = add_label(to_viz(rt),
                    f"Roundtrip on canvas [c={core_64.sum().item():.0f} d={dilated_binary.sum().item():.0f}]")

    # Assemble single row (7 cols)
    row_h = c1.height
    n_cols = 7
    panel = Image.new("RGB", (VIZ_SIZE * n_cols, row_h), (20, 20, 20))
    for j, col in enumerate([c1, c2, c3, c4, c5, c6, c7]):
        panel.paste(col, (j * VIZ_SIZE, 0))

    panel.save(viz_dir / f"{canvas_path.stem}.png", quality=95)
    manifest.append({
        "canvas_id": canvas_path.stem,
        "ref_id": ref_path.stem,
        "difficulty": diff,
    })
    total_success += 1

# Save manifest
manifest_path = easy_placed_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved: {manifest_path} ({len(manifest)} entries)")

print(f"\nDone: {total_success} placed, {total_failed} failed")
print(f"Viz panels: {EXP.parent / 'viz'}")
print(f"Placed masks: {EXP / 'synthetic' / 'placed_masks'}")
