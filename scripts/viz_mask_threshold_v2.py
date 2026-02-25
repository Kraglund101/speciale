"""Visualize mask threshold impact: 64×64 latent view + mapped back to 512×512.

Per sample, shows:
  Col 0: Original image with GT mask outline
  Col 1: bilinear>0.5 — core (white) + band (blue) at 64×64
  Col 2: bilinear>0.5 — affected region mapped back to 512×512 on image
  Col 3: bilinear>0   — core (white) + band (blue) at 64×64
  Col 4: bilinear>0   — affected region mapped back to 512×512 on image
  Col 5: DIFFERENCE   — extra pixels from >0 vs >0.5 (red) on image

Picks samples across resolutions, biased toward small anomalies where inflation matters most.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import downsample_mask_maxpool, create_latent_band_mask


def resize_mask_bilinear(mask_pil: Image.Image, size: int = 512,
                         threshold: float = 0.5) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > threshold).float()


def full_pipeline(mask_512: torch.Tensor, band_mode: int = 2):
    """Returns core_64, dilated_64, band_64, alpha_map_64."""
    mask_4d = mask_512.unsqueeze(0)  # [1, 1, 512, 512]
    core_64 = downsample_mask_maxpool(mask_4d, 64)  # [1, 1, 64, 64]
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=band_mode
    )
    return (core_64.squeeze(0).squeeze(0),
            dilated_binary.squeeze(0).squeeze(0),
            band_mask.squeeze(0).squeeze(0),
            alpha_map.squeeze(0).squeeze(0))


def latent_to_512(mask_64: torch.Tensor) -> np.ndarray:
    """Upsample 64×64 mask back to 512×512 via nearest (matches UNet latent→pixel mapping)."""
    m = mask_64.unsqueeze(0).unsqueeze(0)  # [1, 1, 64, 64]
    up = F.interpolate(m, size=(512, 512), mode="nearest")
    return up.squeeze().numpy()


def overlay_regions_on_image(img_512: np.ndarray, core_512: np.ndarray,
                             band_512: np.ndarray) -> np.ndarray:
    """Overlay core (yellow) and band (blue) on image."""
    out = img_512.copy().astype(np.float32) / 255.0
    # Darken background slightly
    bg = (core_512 < 0.5) & (band_512 < 0.5)
    out[bg] *= 0.4

    # Band = blue tint
    band_only = (band_512 > 0.5) & (core_512 < 0.5)
    out[band_only] = out[band_only] * 0.4 + np.array([0.3, 0.4, 0.9]) * 0.6

    # Core = yellow tint
    core_px = core_512 > 0.5
    out[core_px] = out[core_px] * 0.5 + np.array([1.0, 0.9, 0.2]) * 0.5

    return np.clip(out, 0, 1)


def render_64_map(core: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Render 64×64 core+band as RGB: core=white, band=blue, bg=dark gray."""
    viz = np.full((*core.shape, 3), 0.1)  # dark bg
    viz[band > 0.5] = [0.3, 0.4, 0.9]
    viz[core > 0.5] = [1.0, 0.95, 0.8]
    return viz


def main():
    root = Path("anomverse_extension/datasets/full_training_dataset")
    master_path = root / "master_training.json"

    with open(master_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("images") or data.get("samples", [])

    # Collect candidates by resolution, sort by mask area (prefer small anomalies)
    candidates = []
    seen = set()
    for e in entries:
        mp = e.get("mask_path_full") or e.get("mask_path_abs") or e.get("mask_path", "")
        ip = e.get("image_path_full") or e.get("image_path_abs") or e.get("image_path", "")
        if mp and not os.path.isabs(mp):
            mp = str(root / mp)
        if ip and not os.path.isabs(ip):
            ip = str(root / ip)
        if mp in seen or not os.path.exists(mp) or not os.path.exists(ip):
            continue
        seen.add(mp)
        try:
            m = Image.open(mp).convert("L")
            w, h = m.size
            if w == 512 and h == 512:
                continue  # skip 512 — no resize needed
            area = (np.array(m) > 127).sum()
            if area < 10:
                continue
            candidates.append({
                "ip": ip, "mp": mp, "res": f"{w}x{h}",
                "w": w, "h": h, "area": area,
            })
        except Exception:
            pass

    # Pick ~16 samples: mix of resolutions, biased toward small anomalies
    # Group by resolution, take small ones from each
    by_res = defaultdict(list)
    for c in candidates:
        by_res[c["res"]].append(c)
    for res in by_res:
        by_res[res].sort(key=lambda x: x["area"])

    # Target resolutions in priority order
    target_res = ["1024x1024", "800x800", "600x600", "900x900", "1000x1000", "700x700", "840x840"]
    selected = []
    # 2 small + 1 medium from each resolution until we have enough
    for res in target_res:
        if res not in by_res:
            continue
        pool = by_res[res]
        # Take smallest, a mid-small, and a medium
        indices = [0, min(len(pool)//4, len(pool)-1), min(len(pool)//2, len(pool)-1)]
        for idx in indices:
            if len(selected) >= 16:
                break
            selected.append(pool[idx])
        if len(selected) >= 16:
            break

    print(f"Selected {len(selected)} samples for visualization")

    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 6, figsize=(30, 4.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = [
        "Original + GT mask",
        "bilinear > 0.5\n@ 64×64",
        "bilinear > 0.5\naffected region @ 512",
        "bilinear > 0\n@ 64×64",
        "bilinear > 0\naffected region @ 512",
        "DIFFERENCE\nextra from > 0 (red)",
    ]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=11, fontweight="bold")

    for i, sample in enumerate(selected):
        img_pil = Image.open(sample["ip"]).convert("RGB")
        mask_pil = Image.open(sample["mp"]).convert("L")

        # Resize image to 512 for display
        img_512 = np.array(img_pil.resize((512, 512), Image.LANCZOS))

        # GT mask at 512 for outline
        gt_512 = np.array(mask_pil.resize((512, 512), Image.BILINEAR))
        gt_binary = (gt_512 > 127).astype(np.float32)

        # Two strategies
        m_05 = resize_mask_bilinear(mask_pil, threshold=0.5)
        m_00 = resize_mask_bilinear(mask_pil, threshold=0.0)

        core_05, dil_05, band_05, alpha_05 = full_pipeline(m_05)
        core_00, dil_00, band_00, alpha_00 = full_pipeline(m_00)

        # Map back to 512
        core_05_512 = latent_to_512(core_05)
        band_05_512 = latent_to_512(band_05)
        dil_05_512 = latent_to_512(dil_05)

        core_00_512 = latent_to_512(core_00)
        band_00_512 = latent_to_512(band_00)
        dil_00_512 = latent_to_512(dil_00)

        # --- Col 0: Original + GT mask outline ---
        img_display = img_512.copy().astype(np.float32) / 255.0
        # Draw mask outline
        from scipy import ndimage
        outline = ndimage.binary_dilation(gt_binary, iterations=1).astype(float) - gt_binary
        img_display[outline > 0.5] = [1, 0, 0]  # red outline
        axes[i, 0].imshow(img_display)

        # --- Col 1: bilinear>0.5 at 64×64 ---
        axes[i, 1].imshow(render_64_map(core_05.numpy(), band_05.numpy()),
                          interpolation="nearest")

        # --- Col 2: bilinear>0.5 affected @ 512 ---
        axes[i, 2].imshow(overlay_regions_on_image(img_512, core_05_512, band_05_512))

        # --- Col 3: bilinear>0 at 64×64 ---
        axes[i, 3].imshow(render_64_map(core_00.numpy(), band_00.numpy()),
                          interpolation="nearest")

        # --- Col 4: bilinear>0 affected @ 512 ---
        axes[i, 4].imshow(overlay_regions_on_image(img_512, core_00_512, band_00_512))

        # --- Col 5: Difference (extra pixels from >0) ---
        extra_core = ((core_00_512 > 0.5) & (core_05_512 < 0.5)).astype(float)
        extra_band = ((dil_00_512 > 0.5) & (dil_05_512 < 0.5)).astype(float)
        diff_img = img_512.copy().astype(np.float32) / 255.0
        # Show >0.5 region as base (semi-transparent)
        diff_img[dil_05_512 > 0.5] = diff_img[dil_05_512 > 0.5] * 0.5 + np.array([0.5, 0.5, 0.5]) * 0.5
        # Extra band = orange
        diff_img[extra_band > 0.5] = diff_img[extra_band > 0.5] * 0.3 + np.array([1.0, 0.5, 0.0]) * 0.7
        # Extra core = red
        diff_img[extra_core > 0.5] = diff_img[extra_core > 0.5] * 0.3 + np.array([1.0, 0.0, 0.0]) * 0.7
        diff_img = np.clip(diff_img, 0, 1)
        axes[i, 5].imshow(diff_img)

        # Stats for row label
        nc_05 = int(core_05.sum().item())
        nc_00 = int(core_00.sum().item())
        nt_05 = int(dil_05.sum().item())
        nt_00 = int(dil_00.sum().item())
        core_infl = ((nc_00 - nc_05) / nc_05 * 100) if nc_05 > 0 else 0
        total_infl = ((nt_00 - nt_05) / nt_05 * 100) if nt_05 > 0 else 0

        axes[i, 0].set_ylabel(
            f"{sample['res']}\n"
            f"mask area: {sample['area']}\n"
            f"core: {nc_05}→{nc_00} (+{core_infl:.0f}%)\n"
            f"total: {nt_05}→{nt_00} (+{total_infl:.0f}%)",
            fontsize=9, rotation=0, labelpad=100, va="center",
        )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.9, 0.2], label="Core region"),
        mpatches.Patch(facecolor=[0.3, 0.4, 0.9], label="Band region"),
        mpatches.Patch(facecolor=[1.0, 0.0, 0.0], label="Extra core (>0 vs >0.5)"),
        mpatches.Patch(facecolor=[1.0, 0.5, 0.0], label="Extra band (>0 vs >0.5)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        "Mask Threshold Impact Through Full Pipeline\n"
        "bilinear > 0.5  vs  bilinear > 0  →  maxpool 512→64  →  band dilation (mode 2)  →  mapped back to 512×512",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()

    out_path = "results/viz_mask_threshold_v2.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
