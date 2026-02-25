"""CLIP-vs-UNet mask alignment -- 4-column layout.

Shows the misalignment: what CLIP sees vs what UNet must reconstruct.

One row per sample:
  Col 0: Original image + mask outline
  Col 1: 64x64 core+band grid (core=red, band=blue)
  Col 2: CLIP crop -- raw mask (what CLIP currently sees)
  Col 3: CLIP crop -- roundtrip core+band (what UNet reconstructs)
         Blind zone: patches UNet needs but CLIP suppresses.
"""
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.crop_utils import get_component_groups, clip_crop_from_groups
from src.utils.mask_utils import downsample_mask_maxpool

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# -- helpers ----------------------------------------------------------------

def mp16(m: torch.Tensor) -> torch.Tensor:
    """Maxpool mask to 16x16 CLIP patch grid."""
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[-1] != 224:
        m = F.interpolate(m.float(), size=(224, 224), mode="nearest")
    return (F.max_pool2d(m.float(), kernel_size=14).squeeze() > 0.5).float()


def draw_crop_col(ax, crop_np, mask_np, title, mask_color, grid_color):
    """Draw 224x224 CLIP crop with mask overlay and 16x16 patch grid."""
    ax.imshow(crop_np)

    ov = np.zeros((224, 224, 4), dtype=np.float32)
    ov[:, :, 0], ov[:, :, 1], ov[:, :, 2] = mask_color
    ov[:, :, 3] = (mask_np > 0.5).astype(np.float32) * 0.5
    ax.imshow(ov)

    c16 = mp16(torch.from_numpy(mask_np).unsqueeze(0))
    cell = 224.0 / 16.0
    nc16 = int(c16.sum())
    area_pct = 100.0 * (mask_np > 0.5).sum() / (224 * 224)

    for rr in range(16):
        for cc in range(16):
            if c16[rr, cc] > 0.5:
                ax.add_patch(Rectangle(
                    (cc * cell, rr * cell), cell, cell,
                    lw=0.6, ec=grid_color, fc="none"))

    for j in range(17):
        ax.axhline(y=j * cell, color="white", lw=0.15, alpha=0.25)
        ax.axvline(x=j * cell, color="white", lw=0.15, alpha=0.25)

    ax.set_title(f"{title}\n{nc16}/256 patches | {area_pct:.1f}%", fontsize=7)
    ax.axis("off")


def draw_grid64(ax, img_64_np, core_np, band_np, title):
    """Draw 64x64 native grid with core (red) and band (blue) cells."""
    cell_img = img_64_np.copy()
    core_bool = core_np > 0.5
    band_bool = band_np > 0.5

    for c, val in enumerate([1.0, 0.0, 0.0]):
        cell_img[:, :, c] = np.where(
            core_bool, img_64_np[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])
    for c, val in enumerate([0.2, 0.4, 1.0]):
        cell_img[:, :, c] = np.where(
            band_bool, img_64_np[:, :, c] * 0.3 + val * 0.7, cell_img[:, :, c])

    ax.imshow(cell_img, interpolation="nearest", extent=[0, 64, 64, 0])

    for g in range(65):
        ax.axhline(y=g, color="white", linewidth=0.2, alpha=0.35)
        ax.axvline(x=g, color="white", linewidth=0.2, alpha=0.35)

    core_n = int(core_bool.sum())
    band_n = int(band_bool.sum())
    ax.set_title(f"{title}\n{core_n}c + {band_n}b", fontsize=7)
    ax.set_xlim(0, 64)
    ax.set_ylim(64, 0)
    ax.axis("off")


# -- main -------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--save-dir", default="results/clip/clip_vs_unet_alignment")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(data_root / "master_training.json", encoding="utf-8") as f:
        samples = json.load(f)

    valid = []
    random.shuffle(samples)
    for s in samples:
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]
        if not img_path.exists() or not mask_path.exists():
            continue
        mask_pil = Image.open(str(mask_path)).convert("L")
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)
        if mask_np.sum() < 10:
            continue
        iw, ih = mask_pil.size
        groups = get_component_groups(mask_np, ih, iw)
        if len(groups) >= 1:
            valid.append((s, groups))
            if len(valid) >= args.n_samples:
                break

    n = len(valid)
    print(f"Found {n} valid samples")
    if n == 0:
        return

    fig, axes = plt.subplots(n, 4, figsize=(22, 4.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_headers = [
        "Original + mask",
        "64x64 core + 2px band",
        "CLIP sees (raw mask)",
        "UNet reconstructs (RT core+band)",
    ]
    for j, hdr in enumerate(col_headers):
        axes[0, j].set_title(hdr, fontsize=9, fontweight="bold", pad=14)

    for si, (s, groups) in enumerate(valid):
        img_path = data_root / s["image_path"]
        mask_path = data_root / s["mask_path"]
        product = s.get("product", "?")
        defect = s.get("defect_type", "?")

        img_pil = Image.open(str(img_path)).convert("RGB")
        mask_pil = Image.open(str(mask_path)).convert("L")
        iw, ih = img_pil.size

        img_np = np.array(img_pil).astype(np.float32) / 255.0
        mask_np = (np.array(mask_pil) > 127).astype(np.float32)

        img_t = torch.from_numpy(img_np).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask_np).unsqueeze(0)

        # -- UNet roundtrip --
        mask_512 = downsample_mask_maxpool(mask_t, 512)
        core_64 = downsample_mask_maxpool(mask_512.unsqueeze(0), 64)
        dil1 = (F.max_pool2d(core_64, kernel_size=3, stride=1, padding=1) > 0.5).float()
        dil2 = (F.max_pool2d(dil1, kernel_size=3, stride=1, padding=1) > 0.5).float()
        band_64 = ((dil2 > 0.5) & (core_64 < 0.5)).float()

        core_orig = F.interpolate(
            F.interpolate(core_64, size=(512, 512), mode="nearest"),
            size=(ih, iw), mode="nearest").squeeze(0)
        dil2_orig = F.interpolate(
            F.interpolate(dil2, size=(512, 512), mode="nearest"),
            size=(ih, iw), mode="nearest").squeeze(0)

        # -- Pick group, crop --
        weights = [float(g["mask"].sum()) for g in groups]
        group = random.choices(groups, weights=weights, k=1)[0]

        state = random.getstate()
        crop_img, crop_raw = clip_crop_from_groups(img_t, mask_t, [group], crop_size=224)
        random.setstate(state)
        _, crop_core = clip_crop_from_groups(img_t, core_orig, [group], crop_size=224)
        random.setstate(state)
        _, crop_dil2 = clip_crop_from_groups(img_t, dil2_orig, [group], crop_size=224)

        crop_img_np = crop_img.permute(1, 2, 0).numpy()
        crop_raw_np = crop_raw.squeeze().numpy()
        crop_core_np = crop_core.squeeze().numpy()
        crop_dil2_np = crop_dil2.squeeze().numpy()
        crop_band_np = ((crop_dil2_np > 0.5) & (crop_core_np < 0.5)).astype(np.float32)

        cov_full = 100.0 * mask_np.sum() / (ih * iw)

        # -- Col 0: Original + mask outline --
        ax0 = axes[si, 0]
        ax0.imshow(img_np)
        outline = ndimage.binary_dilation(mask_np, iterations=2).astype(float) - mask_np
        ov_outline = np.stack([outline, np.zeros_like(outline),
                               np.zeros_like(outline),
                               outline * 0.8], axis=-1)
        ax0.imshow(ov_outline)
        ax0.set_ylabel(
            f"{product}\n{defect}\n{ih}x{iw}\n{cov_full:.2f}%",
            fontsize=7, rotation=0, labelpad=70, va="center")
        ax0.axis("off")

        # -- Col 1: 64x64 core+band grid --
        img_64 = F.interpolate(
            img_t.unsqueeze(0), size=(64, 64), mode="bilinear",
            align_corners=False).squeeze(0).permute(1, 2, 0).numpy()

        core_64_np = core_64.squeeze().numpy()
        band_64_np = band_64.squeeze().numpy()

        draw_grid64(axes[si, 1], img_64, core_64_np, band_64_np,
                    "Core + 2px band")

        # -- Col 2: CLIP crop raw mask (what CLIP sees) --
        draw_crop_col(axes[si, 2], crop_img_np, crop_raw_np, "CLIP mask (current)",
                      mask_color=(1.0, 0.15, 0.1), grid_color=(1.0, 0.2, 0.2))

        # -- Col 3: CLIP crop core+band RT (what UNet reconstructs) --
        ax3 = axes[si, 3]
        ax3.imshow(crop_img_np)

        # Core overlay (yellow)
        ov_core = np.zeros((224, 224, 4), dtype=np.float32)
        ov_core[:, :, 0], ov_core[:, :, 1], ov_core[:, :, 2] = 1.0, 0.85, 0.1
        ov_core[:, :, 3] = (crop_core_np > 0.5).astype(np.float32) * 0.45
        ax3.imshow(ov_core)

        # Band overlay (blue) -- blind zone
        ov_band = np.zeros((224, 224, 4), dtype=np.float32)
        ov_band[:, :, 0], ov_band[:, :, 1], ov_band[:, :, 2] = 0.15, 0.4, 1.0
        ov_band[:, :, 3] = (crop_band_np > 0.5).astype(np.float32) * 0.55
        ax3.imshow(ov_band)

        # Patch grid -- blind zone patches in red outline
        c16_raw = mp16(torch.from_numpy(crop_raw_np).unsqueeze(0))
        c16_dil = mp16(torch.from_numpy(crop_dil2_np).unsqueeze(0))
        c16_blind = ((c16_dil > 0.5) & (c16_raw < 0.5)).float()
        cell = 224.0 / 16.0

        n_dil = int(c16_dil.sum())
        n_raw = int(c16_raw.sum())
        n_blind = int(c16_blind.sum())

        for rr in range(16):
            for cc in range(16):
                if c16_dil[rr, cc] > 0.5:
                    if c16_blind[rr, cc] > 0.5:
                        ec, lw = (1.0, 0.0, 0.0), 1.4
                    else:
                        ec, lw = (0.9, 0.75, 0.1), 0.6
                    ax3.add_patch(Rectangle(
                        (cc * cell, rr * cell), cell, cell,
                        lw=lw, ec=ec, fc="none"))

        for j in range(17):
            ax3.axhline(y=j * cell, color="white", lw=0.15, alpha=0.25)
            ax3.axvline(x=j * cell, color="white", lw=0.15, alpha=0.25)

        area_pct = 100.0 * (crop_dil2_np > 0.5).sum() / (224 * 224)
        ax3.set_title(
            f"UNet region (RT)\n{n_dil}/256 patches | {area_pct:.1f}%\n"
            f"blind: {n_blind} patches CLIP misses",
            fontsize=7, color="red" if n_blind > 0 else "black")
        ax3.axis("off")

        print(f"  [{si:2d}] {product}/{defect} -- {ih}x{iw}, "
              f"cov={cov_full:.2f}%, "
              f"64x64: {int(core_64_np.sum())}c+{int(band_64_np.sum())}b, "
              f"CLIP: {n_raw}->{n_dil} (+{n_blind} blind)")

    # -- Legend --
    legend_elements = [
        Patch(fc=(1.0, 0.0, 0.0, 0.7), label="Core cells (64x64)"),
        Patch(fc=(0.2, 0.4, 1.0, 0.7), label="Band cells (2px dilation)"),
        Patch(fc=(1.0, 0.15, 0.1, 0.5), label="Raw CLIP mask (current)"),
        Patch(fc=(1.0, 0.85, 0.1, 0.45), label="UNet core (roundtrip)"),
        Patch(fc=(0.15, 0.4, 1.0, 0.55), label="UNet band (CLIP blind zone)"),
        Patch(ec=(1.0, 0.0, 0.0), fc="none", lw=1.4,
              label="Blind patches (UNet needs, CLIP ignores)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.005))

    plt.suptitle(
        "CLIP <-> UNet mask misalignment\n"
        "Col 2: what CLIP routes on (raw mask)  |  "
        "Col 3: what UNet must reconstruct (64x64 RT + 2px band)\n"
        "Red-outlined patches = blind zone: UNet needs them but CLIP suppresses them",
        fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0.07, 0.025, 1.0, 0.93])
    out_path = save_dir / "alignment_20.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
