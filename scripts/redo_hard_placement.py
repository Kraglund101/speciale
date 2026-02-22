"""Redo hard mask placement with (0.8, 1.2) scale range + viz with cyan CLIP dilation."""
import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import create_latent_band_mask, downsample_mask_maxpool
from src.utils.crop_utils import clip_crop

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

EXP = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\VisA_validation_dataset\datasets\easy_test\cashew\experiment_UniNet"
)
VIZ_SIZE = 512
SCALE_RANGE = (0.8, 1.2)

canvas_imgs_dir = EXP / "source" / "normals" / "canvas" / "imgs"
canvas_masks_dir = EXP / "source" / "normals" / "canvas" / "fg_masks"

# Extra normals pool: all normals NOT in train/test/canvas (for hard placement)
import json
DATA_DIR = EXP.parent / "Data" / "Images" / "Normal"
BIREFNET_DIR = EXP.parent / "birefnet_masks" / "Normal"
with open(EXP / "splits.json") as f:
    splits = json.load(f)
reserved_normals = set(
    splits["normals"]["train"] + splits["normals"]["test"] + splits["normals"]["canvas"]
)


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


# Preload all canvas FG masks (original 50 canvases)
canvas_paths = sorted(canvas_imgs_dir.glob("*.JPG"))
canvas_fgs = {}
for cp in canvas_paths:
    fg_path = canvas_masks_dir / f"{cp.stem}.png"
    if fg_path.exists():
        canvas_fgs[cp.stem] = load_binary_mask(fg_path)
print(f"Loaded {len(canvas_fgs)} canvas FG masks")

# Add extra normals (not in train/test/canvas) for hard placement
extra_count = 0
for img_path in sorted(DATA_DIR.glob("*.JPG")):
    if img_path.name in reserved_normals:
        continue
    if img_path.stem in canvas_fgs:
        continue
    # Need BiRefNet binary mask
    fg_path = BIREFNET_DIR / f"{img_path.stem}_binary.png"
    if fg_path.exists():
        canvas_fgs[img_path.stem] = load_binary_mask(fg_path)
        extra_count += 1
print(f"Added {extra_count} extra normals for hard placement (total: {len(canvas_fgs)})")

hard_refs = ["095", "096", "098"]


def place_hard_mask(ref_mask, canvas_fgs_dict, scale_range, used):
    """Inverse placement: position mask so FG is inside mask (100% coverage).

    Search grid: 4 flip combos × 72 angles × scales (progressive widening).
    Starts with scale_range, widens by +0.1 each retry until match found.
    """
    ys, xs = np.where(ref_mask > 0.5)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_raw = ref_mask[y0:y1, x0:x1]

    # Precompute canvas FG centers (skip used)
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

    # Progressive scale widening: start with scale_range, expand up by 0.1 each round
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
                    scaled_arr = (np.array(scaled).astype(np.float32) / 255.0 > 0.5).astype(
                        np.float32
                    )
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
                        if fg_inside >= fg_sum:  # 100% — T2I-adapter expects core = actual object
                            cov = placed.sum() / (H * W) * 100
                            flip_str = ("V" if fv else "") + ("H" if fh else "") or "none"
                            scale_note = f" (bumped +{bump*0.1:.1f})" if bump > 0 else ""
                            print(f"  -> canvas={cid}, angle={angle:.1f}, scale={scale_try:.2f}{scale_note}, flip={flip_str}, coverage={cov:.1f}%")
                            return (cid, placed, angle, scale_try, cov)

        if bump < max_scale_bump:
            print(f"  No match at scale [{lo:.1f}, {hi:.1f}], widening to [{lo:.1f}, {hi+0.1:.1f}]...")

    return None


# Exclude canvases already used by easy placements
easy_masks_dir = EXP / "synthetic" / "placed_masks" / "easy"
easy_used = set(p.stem for p in easy_masks_dir.glob("*.png"))
print(f"Easy used {len(easy_used)} canvases, excluding from hard search")

used_canvases = set(easy_used)
placements = {}

for ref_id in hard_refs:
    ref_mask = load_binary_mask(
        EXP / "source" / "anomalies" / "hard" / "masks" / f"{ref_id}.png"
    )
    area_pct = ref_mask.sum() / (ref_mask.shape[0] * ref_mask.shape[1]) * 100
    print(f"\n=== ref={ref_id} (mask area: {area_pct:.1f}%) ===")

    result = place_hard_mask(ref_mask, canvas_fgs, SCALE_RANGE, used_canvases)

    if result is None:
        print("  FAILED - no valid placement")
        continue

    cid, placed, angle, scale, cov = result
    used_canvases.add(cid)
    placements[ref_id] = (cid, placed, angle, scale, cov)
    print(f"  -> canvas={cid}, angle={angle:.1f}, scale={scale:.2f}, coverage={cov:.1f}%")


# Clean old placed masks
placed_dir = EXP / "synthetic" / "placed_masks" / "hard"
placed_dir.mkdir(parents=True, exist_ok=True)
for old in placed_dir.glob("*.png"):
    old.unlink()
    print(f"  Removed old mask: {old.name}")


# Save manifest
manifest = []
for ref_id, (cid, placed, angle, scale, cov) in placements.items():
    manifest.append({
        "canvas_id": cid,
        "ref_id": ref_id,
        "difficulty": "hard",
    })
manifest_path = placed_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved: {manifest_path} ({len(manifest)} entries)")

# Generate viz + save placed masks
print("\n=== Generating viz panels ===")
for ref_id, (cid, placed, angle, scale, cov) in placements.items():
    ref_img = Image.open(
        EXP / "source" / "anomalies" / "hard" / "imgs" / f"{ref_id}.JPG"
    ).convert("RGB")
    ref_mask = load_binary_mask(
        EXP / "source" / "anomalies" / "hard" / "masks" / f"{ref_id}.png"
    )
    # Load canvas image from original canvas dir or extra normals
    canvas_img_path = canvas_imgs_dir / f"{cid}.JPG"
    if not canvas_img_path.exists():
        canvas_img_path = DATA_DIR / f"{cid}.JPG"
    canvas_img = Image.open(canvas_img_path).convert("RGB")
    canvas_fg = canvas_fgs[cid]
    img_h, img_w = np.array(canvas_img).shape[:2]

    # Save placed mask
    Image.fromarray((placed * 255).astype(np.uint8)).save(placed_dir / f"{cid}.png")

    # CLIP crop
    ref_tensor = torch.from_numpy(
        np.array(ref_img).astype(np.float32) / 255.0
    ).permute(2, 0, 1)
    ref_mask_t = torch.from_numpy(ref_mask).unsqueeze(0).float()
    clip_img_t, clip_mask_t = clip_crop(ref_tensor, ref_mask_t, crop_size=224)
    clip_np = (clip_img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    clip_crop_img = Image.fromarray(clip_np)
    clip_mask_np = clip_mask_t.squeeze(0).numpy()
    clip_crop_with_mask = overlay_mask_rgba(
        clip_crop_img, clip_mask_np, (255, 30, 30), 0.45
    )

    # CLIP 16x16: original + dilated
    clip_mask_16_t = downsample_mask_maxpool(clip_mask_t.unsqueeze(0), 16)
    clip_mask_16 = clip_mask_16_t[0, 0].numpy()
    clip_dilated_16_t = F.max_pool2d(clip_mask_16_t, kernel_size=3, stride=1, padding=1)
    clip_dilated_16_t = (clip_dilated_16_t > 0.5).float()
    clip_dilated_16 = clip_dilated_16_t[0, 0].numpy()
    clip_dilation_only = (
        (clip_dilated_16 > 0.5) & (clip_mask_16 < 0.5)
    ).astype(np.float32)
    n_orig = int(clip_mask_16.sum())
    n_dilated = int(clip_dilated_16.sum())
    n_new = n_dilated - n_orig

    # UNet 64x64
    placed_512 = np.array(
        Image.fromarray((placed * 255).astype(np.uint8)).resize(
            (512, 512), Image.NEAREST
        )
    ).astype(np.float32) / 255.0
    placed_512 = (placed_512 > 0.5).astype(np.float32)
    mask_t = torch.from_numpy(placed_512).unsqueeze(0).unsqueeze(0).float()
    core_64 = downsample_mask_maxpool(mask_t, 64)

    # Band dilation on core (matches training: no extra shadow dilation)
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=2, core_ratio=0.8
    )
    core_up = F.interpolate(core_64, size=(img_h, img_w), mode="nearest")[
        0, 0
    ].numpy()
    band_up = F.interpolate(band_mask, size=(img_h, img_w), mode="nearest")[
        0, 0
    ].numpy()

    # === 6-column panel ===
    c1 = add_label(to_viz(ref_img), f"Real anomaly ({ref_id}) [hard]")
    c2 = add_label(
        to_viz(overlay_mask_rgba(ref_img, ref_mask, (255, 30, 30), 0.5)),
        "Real anomaly + mask",
    )
    c3 = add_label(
        clip_crop_with_mask.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS),
        "CLIP crop 224x224 (IP-Adapter input)",
    )

    # Col 4: CLIP 16x16 -- RED=original, CYAN=dilation
    base_clip = clip_crop_img.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert(
        "RGBA"
    )
    cell_sz = VIZ_SIZE // 16
    overlay_img = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))
    for r in range(16):
        for ci in range(16):
            if clip_mask_16[r, ci] > 0.5:
                color = (255, 30, 30, 100)  # RED = original
            elif clip_dilation_only[r, ci] > 0.5:
                color = (0, 220, 255, 100)  # CYAN = dilation
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
    c4 = add_label(
        base_clip.convert("RGB"),
        f"CLIP 16x16: {n_orig}(red) +{n_new}(cyan) = {n_dilated}",
    )

    # Col 5: Placed mask on canvas (transformed: rotated + scaled)
    placed_on_fg = overlay_mask_rgba(canvas_img, canvas_fg, (0, 200, 0), 0.2)
    placed_on_fg = overlay_mask_rgba(placed_on_fg, placed, (255, 30, 30), 0.5)
    c5 = add_label(
        to_viz(placed_on_fg),
        f"Placed on canvas {cid} ({cov:.1f}%)",
    )

    # Col 6: 64x64 with band=2 — same style as CLIP grid (overlay + grid lines)
    # Use dilated core (includes +1px shadow) — matches what training actually uses
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
    c6 = add_label(
        base_64.convert("RGB"),
        f"64x64 core(red)+band(yellow) [c={core_64.sum().item():.0f} b={band_mask.sum().item():.0f}]",
    )

    # Col 7: Roundtrip — dilated 64x64 mapped back to image res on canvas
    rt = overlay_mask_rgba(canvas_img, band_up, (255, 220, 0), 0.45)
    rt = overlay_mask_rgba(rt, core_up, (255, 0, 0), 0.55)
    c7 = add_label(
        to_viz(rt),
        f"Roundtrip on canvas [c={core_64.sum().item():.0f} d={dilated_binary.sum().item():.0f}]",
    )

    row_h = c1.height
    panel = Image.new("RGB", (VIZ_SIZE * 7, row_h), (20, 20, 20))
    for j, col in enumerate([c1, c2, c3, c4, c5, c6, c7]):
        panel.paste(col, (j * VIZ_SIZE, 0))

    viz_dir = EXP.parent / "viz" / "hard"
    viz_dir.mkdir(parents=True, exist_ok=True)
    panel.save(viz_dir / f"{ref_id}.png", quality=95)
    print(f"  Saved viz: {ref_id}.png -> canvas {cid}")

print("\nDone.")
