"""Visualize CLIP crop masks for cashew inference: raw vs roundtripped.

Shows what generate_cashew.py currently does (raw crop masks, no roundtrip)
vs what training does (unet_roundtrip_masks → separate dilated/core).

Layout per sample (2 rows × 4 cols):
  Row 0: Reference image | Raw crop 1 (16x16 grid) | Raw crop 2 (16x16 grid) | Reference + mask
  Row 1: (empty)         | Roundtripped crop 1      | Roundtripped crop 2      | Diff overlay
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.crop_utils import clip_crop_multi
from src.utils.mask_utils import downsample_mask_maxpool, unet_roundtrip_masks

SEED = 42
BAND_MODE = 2
VIZ_SIZE = 448  # 16x16 grid cells at 28px each

CASHEW_ROOT = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\VisA_validation_dataset\datasets\easy_test\cashew"
)
EXP = CASHEW_ROOT / "experiment_ResNet"


def load_binary_mask(path: Path) -> np.ndarray:
    m = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (m > 0.5).astype(np.float32)


def draw_16x16_grid(crop_pil: Image.Image, mask_16: np.ndarray,
                    core_16: np.ndarray = None, label: str = "") -> Image.Image:
    """Draw 16x16 grid on a crop with colored cells.

    Red = core (or all active if no core_16), Cyan = band (dilated - core).
    """
    img = crop_pil.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS).convert("RGBA")
    cell = VIZ_SIZE // 16
    overlay = Image.new("RGBA", (VIZ_SIZE, VIZ_SIZE), (0, 0, 0, 0))

    n_active = 0
    n_core = 0
    n_band = 0
    for r in range(16):
        for c in range(16):
            is_active = mask_16[r, c] > 0.5
            is_core = core_16[r, c] > 0.5 if core_16 is not None else is_active
            if is_core:
                color = (255, 30, 30, 100)  # Red = core
                n_core += 1
                n_active += 1
            elif is_active:
                color = (0, 220, 255, 100)  # Cyan = band (dilated - core)
                n_band += 1
                n_active += 1
            else:
                continue
            for y in range(r * cell, (r + 1) * cell):
                for x in range(c * cell, (c + 1) * cell):
                    overlay.putpixel((x, y), color)

    img = Image.alpha_composite(img, overlay)
    draw = ImageDraw.Draw(img)
    for k in range(17):
        pos = k * cell
        draw.line([(pos, 0), (pos, VIZ_SIZE - 1)], fill=(255, 255, 255, 80), width=1)
        draw.line([(0, pos), (VIZ_SIZE - 1, pos)], fill=(255, 255, 255, 80), width=1)

    # Label bar
    img = img.convert("RGB")
    bar_h = 24
    out = Image.new("RGB", (VIZ_SIZE, VIZ_SIZE + bar_h), (20, 20, 20))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    if core_16 is not None:
        text = f"{label} [core={n_core} band={n_band} total={n_active}]"
    else:
        text = f"{label} [{n_active} cells]"
    draw.text((4, 2), text, fill=(255, 255, 200), font=font)
    return out


def add_label(img: Image.Image, text: str) -> Image.Image:
    bar_h = 24
    out = Image.new("RGB", (img.width, img.height + bar_h), (20, 20, 20))
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 2), text, fill=(255, 255, 200), font=font)
    return out


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    # Load manifests
    easy_manifest = EXP / "anomaly" / "masks" / "easy" / "manifest.json"
    hard_manifest = EXP / "anomaly" / "masks" / "hard" / "manifest.json"

    entries = []
    if easy_manifest.exists():
        import json
        with open(easy_manifest) as f:
            entries += json.load(f)[:3]  # First 3 easy
    if hard_manifest.exists():
        import json
        with open(hard_manifest) as f:
            entries += json.load(f)[:2]  # First 2 hard

    if not entries:
        print("No manifest entries found")
        return

    out_dir = CASHEW_ROOT / "viz" / "clip_mask_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for entry in entries:
        ref_id = entry["ref_id"]
        difficulty = entry["difficulty"]

        ref_img_path = EXP / "source" / "reference_anomalies" / difficulty / "imgs" / f"{ref_id}.JPG"
        ref_mask_path = EXP / "source" / "reference_anomalies" / difficulty / "masks" / f"{ref_id}.png"

        if not ref_img_path.exists() or not ref_mask_path.exists():
            print(f"  Skipping {ref_id} — files not found")
            continue

        ref_pil = Image.open(ref_img_path).convert("RGB")
        ref_mask_np = load_binary_mask(ref_mask_path)
        ref_tensor = torch.from_numpy(
            np.array(ref_pil).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # [3, H, W]
        ref_mask_t = torch.from_numpy(ref_mask_np).unsqueeze(0).float()  # [1, H, W]

        random.seed(SEED)

        # --- Path A: Raw (what generate_cashew.py does) ---
        crops_raw, masks_raw, valid_raw = clip_crop_multi(
            ref_tensor, ref_mask_t, crop_size=224, n_groups=2,
        )

        # --- Path B: Roundtripped (what training does) ---
        random.seed(SEED)
        core_native, dil_native = unet_roundtrip_masks(ref_mask_t, BAND_MODE)
        crops_rt, masks_rt, valid_rt, extra_rt = clip_crop_multi(
            ref_tensor, ref_mask_t, crop_size=224, n_groups=2,
            clip_masks=[dil_native, core_native],
        )

        # Downsample masks to 16x16 for grid viz
        def to_16(m):
            return downsample_mask_maxpool(
                m.unsqueeze(0), 16
            )[0, 0].numpy()

        # Build panel: 2 rows × 4 cols
        # Row 0: Ref+mask | Raw crop 1 | Raw crop 2 | (validity info)
        # Row 1: (empty)  | RT crop 1  | RT crop 2  | (diff info)

        # Reference image with mask overlay
        ref_viz = ref_pil.resize((VIZ_SIZE, VIZ_SIZE), Image.LANCZOS)
        ref_viz = add_label(ref_viz, f"Reference {ref_id} [{difficulty}]")

        # Raw crops (no roundtrip, no dilation — single mask)
        for gi in range(2):
            if not valid_raw[gi]:
                continue

        raw_panels = []
        for gi in range(2):
            if valid_raw[gi]:
                crop_pil = Image.fromarray(
                    (crops_raw[gi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                mask_16 = to_16(masks_raw[gi])
                panel = draw_16x16_grid(crop_pil, mask_16, core_16=None,
                                        label=f"RAW crop {gi+1}")
            else:
                panel = add_label(
                    Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)),
                    f"RAW crop {gi+1} [NULL]"
                )
            raw_panels.append(panel)

        # Roundtripped crops (dilated + core from unet_roundtrip_masks)
        rt_panels = []
        for gi in range(2):
            if valid_rt[gi]:
                crop_pil = Image.fromarray(
                    (crops_rt[gi].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                dil_16 = to_16(extra_rt[gi][0])   # dilated
                core_16 = to_16(extra_rt[gi][1])   # core
                panel = draw_16x16_grid(crop_pil, dil_16, core_16=core_16,
                                        label=f"ROUNDTRIP crop {gi+1}")
            else:
                panel = add_label(
                    Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (40, 40, 40)),
                    f"ROUNDTRIP crop {gi+1} [NULL]"
                )
            rt_panels.append(panel)

        # Assemble: 2 rows × 3 cols (ref | crop1 | crop2)
        cell_h = VIZ_SIZE + 24  # image + label bar
        n_cols = 3
        n_rows = 2
        panel = Image.new("RGB", (VIZ_SIZE * n_cols, cell_h * n_rows), (20, 20, 20))

        # Row 0: ref + raw crops
        panel.paste(ref_viz, (0, 0))
        panel.paste(raw_panels[0], (VIZ_SIZE, 0))
        panel.paste(raw_panels[1], (VIZ_SIZE * 2, 0))

        # Row 1: empty + roundtripped crops
        info_panel = add_label(
            Image.new("RGB", (VIZ_SIZE, VIZ_SIZE), (30, 30, 30)),
            f"Red=core, Cyan=band"
        )
        panel.paste(info_panel, (0, cell_h))
        panel.paste(rt_panels[0], (VIZ_SIZE, cell_h))
        panel.paste(rt_panels[1], (VIZ_SIZE * 2, cell_h))

        out_path = out_dir / f"{difficulty}_{ref_id}.png"
        panel.save(out_path, quality=95)
        print(f"  Saved: {out_path.name}")

    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
