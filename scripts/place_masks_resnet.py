"""
Place anomaly masks on canvas normals for experiment_ResNet.

47 easy (mask inside FG) + 3 hard (FG inside mask, 100% coverage).
Saves placed masks + manifest.json + viz panels.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.crop_utils import clip_crop
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────
EXP = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy_test\cashew\experiment_ResNet"
)
CASHEW_BASE = EXP.parent  # .../cashew/

CANVAS_IMGS = EXP / "source" / "canvas_normals" / "imgs"
CANVAS_MASKS = EXP / "source" / "canvas_normals" / "masks"

REF_EASY_IMGS = EXP / "source" / "reference_anomalies" / "easy" / "imgs"
REF_EASY_MASKS = EXP / "source" / "reference_anomalies" / "easy" / "masks"
REF_HARD_IMGS = EXP / "source" / "reference_anomalies" / "hard" / "imgs"
REF_HARD_MASKS = EXP / "source" / "reference_anomalies" / "hard" / "masks"

PLACED_EASY = EXP / "anomaly" / "masks" / "easy"
PLACED_HARD = EXP / "anomaly" / "masks" / "hard"
VIZ_DIR = CASHEW_BASE / "viz_resnet"

VIZ_SIZE = 512
SCALE_RANGE = (0.8, 1.2)


# ── Helpers ───────────────────────────────────────────────────────────────
def load_binary_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def overlay_mask_rgba(img, mask, color, alpha):
    base = img.convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[:, :, 0] = color[0]
    overlay[:, :, 1] = color[1]
    overlay[:, :, 2] = color[2]
    overlay[:, :, 3] = (mask * alpha * 255).astype(np.uint8)
    return Image.alpha_composite(base, Image.fromarray(overlay)).convert("RGB")


def add_label(img, text, font_size=18):
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


def to_viz(img):
    return img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS)


# ── Easy placement: mask inside FG ────────────────────────────────────────
def place_easy_mask(anomaly_mask, fg_mask, max_attempts=500):
    """Place anomaly mask so ALL its pixels are inside FG."""
    H, W = fg_mask.shape
    ys, xs = np.where(anomaly_mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = anomaly_mask[y0:y1, x0:x1]

    fg_ys, fg_xs = np.where(fg_mask > 0.5)
    if len(fg_ys) == 0:
        return None

    # Random flips (independent, p=0.5 each)
    if random.random() < 0.5:
        crop = crop[::-1, :].copy()
    if random.random() < 0.5:
        crop = crop[:, ::-1].copy()

    # Uniform rotation + scale
    angle = random.uniform(0, 360)
    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
    rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)

    scale = random.uniform(*SCALE_RANGE)
    rw, rh = rotated.size
    new_w, new_h = max(1, int(rw * scale)), max(1, int(rh * scale))
    scaled = rotated.resize((new_w, new_h), Image.NEAREST)
    scaled_arr = (np.array(scaled).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    if scaled_arr.sum() == 0:
        return None

    sh, sw = scaled_arr.shape
    for _ in range(max_attempts):
        idx = random.randint(0, len(fg_ys) - 1)
        cy, cx = fg_ys[idx], fg_xs[idx]
        top, left = cy - sh // 2, cx - sw // 2
        if top < 0 or left < 0 or top + sh > H or left + sw > W:
            continue
        placed = np.zeros((H, W), dtype=np.float32)
        placed[top : top + sh, left : left + sw] = scaled_arr
        if placed.sum() == (placed * fg_mask).sum():
            return placed
    return None


# ── Hard placement: FG inside mask (100% coverage) ───────────────────────
def place_hard_mask(ref_mask, canvas_fgs_dict, scale_range, used):
    """Exhaustive search: 4 flips × 72 angles × progressive scale widening."""
    ys, xs = np.where(ref_mask > 0.5)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_raw = ref_mask[y0:y1, x0:x1]

    fg_info = {}
    for cid, fg in canvas_fgs_dict.items():
        if cid in used:
            continue
        fg_ys, fg_xs = np.where(fg > 0.5)
        if len(fg_ys) == 0:
            continue
        fg_info[cid] = (fg, fg.shape, fg_ys.mean(), fg_xs.mean(), fg.sum())

    angles = list(np.linspace(0, 360, 72, endpoint=False))
    flips = [(False, False), (True, False), (False, True), (True, True)]

    max_scale_bump = 5  # up to +0.5 above original max
    for bump in range(max_scale_bump + 1):
        lo = scale_range[0]
        hi = scale_range[1] + bump * 0.1
        scales = list(np.linspace(lo, hi, max(5, int((hi - lo) / 0.05) + 1)))
        random.shuffle(angles)
        random.shuffle(scales)

        for fv, fh in flips:
            crop = crop_raw.copy()
            if fv:
                crop = crop[::-1, :].copy()
            if fh:
                crop = crop[:, ::-1].copy()

            for angle in angles:
                crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
                rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)
                rw, rh = rotated.size

                for scale_try in scales:
                    new_w = max(1, int(rw * scale_try))
                    new_h = max(1, int(rh * scale_try))
                    scaled = rotated.resize((new_w, new_h), Image.NEAREST)
                    scaled_arr = (
                        np.array(scaled).astype(np.float32) / 255.0 > 0.5
                    ).astype(np.float32)
                    if scaled_arr.sum() == 0:
                        continue
                    sh, sw = scaled_arr.shape
                    m_ys, m_xs = np.where(scaled_arr > 0.5)
                    m_cy, m_cx = m_ys.mean(), m_xs.mean()

                    for cid, (fg, (H, W), fg_cy, fg_cx, fg_sum) in fg_info.items():
                        if sh > H or sw > W:
                            continue
                        top = int(fg_cy - m_cy)
                        left = int(fg_cx - m_cx)
                        if top < 0 or left < 0 or top + sh > H or left + sw > W:
                            continue

                        placed = np.zeros((H, W), dtype=np.float32)
                        placed[top : top + sh, left : left + sw] = scaled_arr

                        fg_inside = (fg * placed).sum()
                        if fg_inside >= fg_sum:
                            cov = placed.sum() / (H * W) * 100
                            flip_str = ("V" if fv else "") + ("H" if fh else "") or "none"
                            scale_note = f" (bumped +{bump*0.1:.1f})" if bump > 0 else ""
                            print(
                                f"  -> canvas={cid}, angle={angle:.1f}, "
                                f"scale={scale_try:.2f}{scale_note}, "
                                f"flip={flip_str}, coverage={cov:.1f}%"
                            )
                            return (cid, placed, angle, scale_try, cov)

        if bump < max_scale_bump:
            print(f"  No match at scale [{lo:.1f}, {hi:.1f}], widening...")

    return None


# ── Viz panel for one placement ──────────────────────────────────────────
def make_viz_panel(
    canvas_img, canvas_fg, placed, ref_img, ref_mask,
    canvas_id, ref_id, difficulty, canvas_img_path,
):
    """Build 7-col viz panel (same layout as UniNet)."""
    img_w, img_h = canvas_img.size

    # 64x64 processing
    placed_512 = np.array(
        Image.fromarray((placed * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    ).astype(np.float32) / 255.0
    placed_512 = (placed_512 > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(placed_512).unsqueeze(0).unsqueeze(0).float()
    core_64 = downsample_mask_maxpool(mask_t, 64)
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=2, core_ratio=0.8
    )
    core_up = F.interpolate(core_64, size=(img_h, img_w), mode="nearest")[0, 0].numpy()
    band_up = F.interpolate(band_mask, size=(img_h, img_w), mode="nearest")[0, 0].numpy()

    # CLIP crop
    ref_tensor = torch.from_numpy(
        np.array(ref_img).astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    ref_mask_t = torch.from_numpy(ref_mask).unsqueeze(0).float()
    clip_img_t, clip_mask_t = clip_crop(ref_tensor, ref_mask_t, crop_size=224)
    clip_np = (clip_img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    clip_crop_img = Image.fromarray(clip_np)
    clip_mask_np = clip_mask_t.squeeze(0).numpy()

    # CLIP 16x16
    clip_mask_16_t = downsample_mask_maxpool(clip_mask_t.unsqueeze(0), 16)
    clip_mask_16 = clip_mask_16_t[0, 0].numpy()
    clip_dilated_16_t = F.max_pool2d(clip_mask_16_t, kernel_size=3, stride=1, padding=1)
    clip_dilated_16 = (clip_dilated_16_t > 0.5).float()[0, 0].numpy()
    clip_dilation_only = ((clip_dilated_16 > 0.5) & (clip_mask_16 < 0.5)).astype(np.float32)
    n_orig = int(clip_mask_16.sum())

    coverage = placed.sum() / (img_h * img_w) * 100

    # Col 1: Reference anomaly
    c1 = add_label(to_viz(ref_img), f"Real anomaly ({ref_id}) [{difficulty}]")

    # Col 2: Reference + mask
    c2 = add_label(
        to_viz(overlay_mask_rgba(ref_img, ref_mask, (255, 30, 30), 0.5)),
        "Real anomaly + mask",
    )

    # Col 3: CLIP crop
    c3 = add_label(
        overlay_mask_rgba(clip_crop_img, clip_mask_np, (255, 30, 30), 0.45).resize(
            (VIZ_SIZE, VIZ_SIZE), Image.LANCZOS
        ),
        "CLIP crop 224x224",
    )

    # Col 4: CLIP 16x16 grid
    base_clip = clip_crop_img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
    cell_sz = VIZ_SIZE // 16
    overlay_img = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
    for r in range(16):
        for ci in range(16):
            if clip_mask_16[r, ci] > 0.5:
                color = (255, 30, 30, 100)
            elif clip_dilation_only[r, ci] > 0.5:
                color = (0, 220, 255, 100)
            else:
                continue
            for y in range(r * cell_sz, (r + 1) * cell_sz):
                for x in range(ci * cell_sz, (ci + 1) * cell_sz):
                    overlay_img.putpixel((x, y), color)
    base_clip = Image.alpha_composite(base_clip, overlay_img)
    draw = ImageDraw.Draw(base_clip)
    for k in range(17):
        pos = k * cell_sz
        draw.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 80), width=1)
        draw.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 80), width=1)
    c4 = add_label(base_clip.convert("RGB"), f"CLIP 16x16 ({n_orig}/256 active)")

    # Col 5: Placed on canvas
    placed_on_fg = overlay_mask_rgba(canvas_img, canvas_fg, (0, 200, 0), 0.2)
    placed_on_fg = overlay_mask_rgba(placed_on_fg, placed, (255, 30, 30), 0.5)
    c5 = add_label(to_viz(placed_on_fg), f"Placed on canvas {canvas_id} ({coverage:.2f}%)")

    # Col 6: 64x64 core + band grid
    core_64_vis = core_64[0, 0].numpy()
    band_64_vis = band_mask[0, 0].numpy()
    base_64 = canvas_img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
    cell_sz_64 = VIZ_SIZE // 64
    overlay_64 = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
    for r in range(64):
        for ci in range(64):
            if core_64_vis[r, ci] > 0.5:
                color = (255, 0, 0, 150)
            elif band_64_vis[r, ci] > 0.5:
                color = (255, 220, 0, 150)
            else:
                continue
            for y in range(r * cell_sz_64, (r + 1) * cell_sz_64):
                for x in range(ci * cell_sz_64, (ci + 1) * cell_sz_64):
                    overlay_64.putpixel((x, y), color)
    base_64 = Image.alpha_composite(base_64, overlay_64)
    draw_64 = ImageDraw.Draw(base_64)
    for k in range(65):
        pos = k * cell_sz_64
        draw_64.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 40), width=1)
        draw_64.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 40), width=1)
    c6 = add_label(
        base_64.convert("RGB"),
        f"64x64 [c={core_64.sum().item():.0f} b={band_mask.sum().item():.0f}]",
    )

    # Col 7: Roundtrip
    rt = overlay_mask_rgba(canvas_img, band_up, (255, 220, 0), 0.45)
    rt = overlay_mask_rgba(rt, core_up, (255, 0, 0), 0.55)
    c7 = add_label(
        to_viz(rt),
        f"Roundtrip [c={core_64.sum().item():.0f} d={dilated_binary.sum().item():.0f}]",
    )

    row_h = c1.height
    panel = Image.new("RGB", (VIZ_SIZE * 7, row_h), (20, 20, 20))
    for j, col in enumerate([c1, c2, c3, c4, c5, c6, c7]):
        panel.paste(col, (j * VIZ_SIZE, 0))
    return panel


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

# Collect canvases + refs
canvas_paths = sorted(CANVAS_IMGS.glob("*.JPG")) + sorted(CANVAS_IMGS.glob("*.png"))
print(f"Canvas normals: {len(canvas_paths)}")

easy_ref_masks = sorted(REF_EASY_MASKS.glob("*.png"))
hard_ref_masks = sorted(REF_HARD_MASKS.glob("*.png"))
print(f"Easy refs: {len(easy_ref_masks)}, Hard refs: {len(hard_ref_masks)}")

# Preload all canvas FG masks
canvas_fgs = {}
for cp in canvas_paths:
    fg_path = CANVAS_MASKS / f"{cp.stem}.png"
    if fg_path.exists():
        canvas_fgs[cp.stem] = load_binary_mask(fg_path)

# ── Clean stale output ────────────────────────────────────────────────────
for d in [PLACED_EASY, PLACED_HARD]:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    for old in d.glob("*.json"):
        old.unlink()

for d in [VIZ_DIR / "easy", VIZ_DIR / "hard"]:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()

# ── EASY placement ────────────────────────────────────────────────────────
print(f"\n=== Easy placement ({len(easy_ref_masks)} refs) ===")

# Shuffle refs, pair with first N canvases (leave last 3 for hard)
easy_ref_list = list(easy_ref_masks)
random.shuffle(easy_ref_list)

hard_count = len(hard_ref_masks)
easy_canvases = canvas_paths[:-hard_count] if hard_count > 0 else canvas_paths
n_easy = min(len(easy_canvases), len(easy_ref_list))

easy_manifest = []
used_canvases = set()

for i in range(n_easy):
    canvas_path = easy_canvases[i]
    initial_ref_path = easy_ref_list[i]
    canvas_fg = canvas_fgs[canvas_path.stem]
    canvas_img = Image.open(canvas_path).convert("RGB")
    img_h, img_w = canvas_img.size[1], canvas_img.size[0]

    # Try initial ref 5 times, then try other refs
    placed = None
    ref_path = initial_ref_path
    candidates = [initial_ref_path] + [r for r in easy_ref_list if r != initial_ref_path]
    for cand in candidates:
        ref_mask = load_binary_mask(cand)
        for _ in range(5):
            placed = place_easy_mask(ref_mask, canvas_fg)
            if placed is not None:
                ref_path = cand
                break
        if placed is not None:
            break

    if placed is None:
        print(f"  [{i+1:2d}/{n_easy}] canvas={canvas_path.stem} — FAILED")
        continue

    if ref_path != initial_ref_path:
        print(f"    (swapped ref {initial_ref_path.stem} → {ref_path.stem})")

    ref_mask = load_binary_mask(ref_path)
    coverage = placed.sum() / (img_h * img_w) * 100

    # Save placed mask
    Image.fromarray((placed * 255).astype(np.uint8)).save(
        PLACED_EASY / f"{canvas_path.stem}.png"
    )
    used_canvases.add(canvas_path.stem)

    # Manifest
    easy_manifest.append({
        "canvas_id": canvas_path.stem,
        "ref_id": ref_path.stem,
        "difficulty": "easy",
    })

    # Viz
    ref_img = Image.open(REF_EASY_IMGS / f"{ref_path.stem}.JPG").convert("RGB")
    panel = make_viz_panel(
        canvas_img, canvas_fg, placed, ref_img, ref_mask,
        canvas_path.stem, ref_path.stem, "easy", canvas_path,
    )
    panel.save(VIZ_DIR / "easy" / f"{canvas_path.stem}.png", quality=95)

    # Downsample stats for print
    placed_512 = np.array(
        Image.fromarray((placed * 255).astype(np.uint8)).resize((512, 512), Image.NEAREST)
    ).astype(np.float32) / 255.0
    mask_t = torch.from_numpy((placed_512 > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    core_64 = downsample_mask_maxpool(mask_t, 64)
    dilated, _, _, _ = create_latent_band_mask(core_64, band_mode=2, core_ratio=0.8)

    print(
        f"  [{i+1:2d}/{n_easy}] [easy] canvas={canvas_path.stem} ref={ref_path.stem} "
        f"— {placed.sum():.0f}px ({coverage:.2f}%) "
        f"core={core_64.sum().item():.0f} dilated={dilated.sum().item():.0f}"
    )

# Save easy manifest
with open(PLACED_EASY / "manifest.json", "w") as f:
    json.dump(easy_manifest, f, indent=2)
print(f"\nEasy manifest: {len(easy_manifest)} entries")

# ── HARD placement ────────────────────────────────────────────────────────
print(f"\n=== Hard placement ({len(hard_ref_masks)} refs) ===")

# Start with remaining standard canvases
# If 100% FG overlap fails, expand to extra normals
DATA_DIR = CASHEW_BASE / "Data" / "Images" / "Normal"
BIREFNET_DIR = CASHEW_BASE / "birefnet_masks" / "Normal"

# Add extra normals to canvas pool (for hard only)
all_canvas_fgs = dict(canvas_fgs)  # copy
extra_count = 0
if DATA_DIR.exists() and BIREFNET_DIR.exists():
    for img_path in sorted(DATA_DIR.glob("*.JPG")):
        if img_path.stem in all_canvas_fgs:
            continue
        fg_path = BIREFNET_DIR / f"{img_path.stem}_binary.png"
        if fg_path.exists():
            all_canvas_fgs[img_path.stem] = load_binary_mask(fg_path)
            extra_count += 1
    print(f"  Added {extra_count} extra normals (total pool: {len(all_canvas_fgs)})")

hard_manifest = []

for ref_mask_path in hard_ref_masks:
    ref_id = ref_mask_path.stem
    ref_mask = load_binary_mask(ref_mask_path)
    area_pct = ref_mask.sum() / (ref_mask.shape[0] * ref_mask.shape[1]) * 100
    print(f"\n  ref={ref_id} (mask area: {area_pct:.1f}%)")

    result = place_hard_mask(ref_mask, all_canvas_fgs, SCALE_RANGE, used_canvases)

    if result is None:
        print(f"  FAILED - no valid placement for ref={ref_id}")
        continue

    cid, placed, angle, scale, cov = result
    used_canvases.add(cid)

    # Save placed mask
    Image.fromarray((placed * 255).astype(np.uint8)).save(PLACED_HARD / f"{cid}.png")

    hard_manifest.append({
        "canvas_id": cid,
        "ref_id": ref_id,
        "difficulty": "hard",
    })

    # Viz — resolve canvas image path (standard or extra normals)
    canvas_img_path = CANVAS_IMGS / f"{cid}.JPG"
    if not canvas_img_path.exists():
        canvas_img_path = CANVAS_IMGS / f"{cid}.png"
    if not canvas_img_path.exists():
        canvas_img_path = DATA_DIR / f"{cid}.JPG"
    canvas_img = Image.open(canvas_img_path).convert("RGB")
    canvas_fg = all_canvas_fgs[cid]

    ref_img = Image.open(REF_HARD_IMGS / f"{ref_id}.JPG").convert("RGB")
    panel = make_viz_panel(
        canvas_img, canvas_fg, placed, ref_img, ref_mask,
        cid, ref_id, "hard", canvas_img_path,
    )
    panel.save(VIZ_DIR / "hard" / f"{ref_id}.png", quality=95)
    print(f"  Saved viz: {ref_id}.png -> canvas {cid}")

# Save hard manifest
with open(PLACED_HARD / "manifest.json", "w") as f:
    json.dump(hard_manifest, f, indent=2)
print(f"\nHard manifest: {len(hard_manifest)} entries")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Easy: {len(easy_manifest)} placed  |  Hard: {len(hard_manifest)} placed")
print(f"Placed masks: {PLACED_EASY.parent}")
print(f"Viz panels:   {VIZ_DIR}")
print(f"{'='*60}")
