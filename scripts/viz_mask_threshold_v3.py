"""Visualize mask threshold impact: 64×64 latent view + mapped back to 512×512.

Fewer samples, larger panels. Picks cases where inflation is most visible.

Per sample, shows:
  Col 0: Original image with GT mask outline (red)
  Col 1: bilinear>0.5 — affected region overlaid on image (core=yellow, band=blue)
  Col 2: bilinear>0   — affected region overlaid on image (core=yellow, band=blue)
  Col 3: DIFFERENCE overlay — gray=shared, red=extra core, orange=extra band from >0
  Col 4: bilinear>0.5 at 64×64 (zoomed, core=white, band=blue)
  Col 5: bilinear>0   at 64×64 (zoomed, core=white, band=blue)
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
from scipy import ndimage

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
    mask_4d = mask_512.unsqueeze(0)
    core_64 = downsample_mask_maxpool(mask_4d, 64)
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=band_mode
    )
    return (core_64.squeeze(0).squeeze(0),
            dilated_binary.squeeze(0).squeeze(0),
            band_mask.squeeze(0).squeeze(0))


def latent_to_512(mask_64: torch.Tensor) -> np.ndarray:
    m = mask_64.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(m, size=(512, 512), mode="nearest")
    return up.squeeze().numpy()


def overlay_on_image(img_512: np.ndarray, core_512: np.ndarray,
                     band_512: np.ndarray) -> np.ndarray:
    out = img_512.copy().astype(np.float32) / 255.0
    bg = (core_512 < 0.5) & (band_512 < 0.5)
    out[bg] *= 0.35
    band_only = (band_512 > 0.5) & (core_512 < 0.5)
    out[band_only] = out[band_only] * 0.35 + np.array([0.25, 0.4, 0.95]) * 0.65
    core_px = core_512 > 0.5
    out[core_px] = out[core_px] * 0.4 + np.array([1.0, 0.85, 0.1]) * 0.6
    return np.clip(out, 0, 1)


def render_64_zoomed(core: np.ndarray, band: np.ndarray, pad: int = 4) -> np.ndarray:
    """Render 64×64 core+band as RGB, cropped to region of interest + padding."""
    all_mask = ((core > 0.5) | (band > 0.5)).astype(float)
    ys, xs = np.where(all_mask > 0.5)
    if len(ys) == 0:
        viz = np.full((20, 20, 3), 0.1)
        return viz
    y0 = max(0, ys.min() - pad)
    y1 = min(63, ys.max() + pad) + 1
    x0 = max(0, xs.min() - pad)
    x1 = min(63, xs.max() + pad) + 1

    # Make square
    h, w = y1 - y0, x1 - x0
    side = max(h, w)
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = max(0, cy - side // 2)
    y1 = min(64, y0 + side)
    x0 = max(0, cx - side // 2)
    x1 = min(64, x0 + side)

    c_crop = core[y0:y1, x0:x1]
    b_crop = band[y0:y1, x0:x1]

    viz = np.full((*c_crop.shape, 3), 0.12)
    # Grid lines
    for gy in range(c_crop.shape[0]):
        viz[gy, :] = np.maximum(viz[gy, :], 0.18)
    for gx in range(c_crop.shape[1]):
        viz[:, gx] = np.maximum(viz[:, gx], 0.18)
    viz[b_crop > 0.5] = [0.25, 0.4, 0.95]
    viz[c_crop > 0.5] = [1.0, 0.95, 0.85]
    return viz


def main():
    root = Path("anomverse_extension/datasets/full_training_dataset")
    master_path = root / "master_training.json"

    with open(master_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("images") or data.get("samples", [])

    # Collect non-512 candidates with their inflation stats
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
                continue
            area = (np.array(m) > 127).sum()
            if area < 10:
                continue

            # Compute inflation quickly
            m_05 = resize_mask_bilinear(m, threshold=0.5)
            m_00 = resize_mask_bilinear(m, threshold=0.0)
            c05, d05, b05 = full_pipeline(m_05)
            c00, d00, b00 = full_pipeline(m_00)
            nc05 = c05.sum().item()
            nc00 = c00.sum().item()
            nt05 = d05.sum().item()
            nt00 = d00.sum().item()
            core_infl = ((nc00 - nc05) / nc05 * 100) if nc05 > 0 else 0
            total_infl = ((nt00 - nt05) / nt05 * 100) if nt05 > 0 else 0

            candidates.append({
                "ip": ip, "mp": mp, "res": f"{w}x{h}", "area": area,
                "core_infl": core_infl, "total_infl": total_infl,
                "nc05": int(nc05), "nc00": int(nc00),
                "nt05": int(nt05), "nt00": int(nt00),
            })
        except Exception:
            pass

        if len(candidates) >= 3000:
            break

    print(f"Evaluated {len(candidates)} non-512 samples")

    # Sort by core inflation descending, pick top ~3 from each resolution
    by_res = defaultdict(list)
    for c in candidates:
        by_res[c["res"]].append(c)
    for res in by_res:
        by_res[res].sort(key=lambda x: -x["core_infl"])

    # Pick: 2 high-inflation + 1 moderate from the main resolutions
    target_res = ["1024x1024", "800x800", "900x900", "600x600"]
    selected = []
    for res in target_res:
        if res not in by_res:
            continue
        pool = by_res[res]
        # Highest inflation
        if len(pool) > 0:
            selected.append(pool[0])
        # ~p75 inflation
        if len(pool) > 3:
            selected.append(pool[len(pool) // 4])

    print(f"Selected {len(selected)} samples:")
    for s in selected:
        print(f"  {s['res']} area={s['area']} core_infl=+{s['core_infl']:.0f}% "
              f"total_infl=+{s['total_infl']:.0f}% core={s['nc05']}->{s['nc00']} "
              f"total={s['nt05']}->{s['nt00']}")

    # --- PLOT ---
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 6, figsize=(32, 5.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = [
        "Original + GT mask",
        "bilinear > 0.5\naffected @ 512",
        "bilinear > 0\naffected @ 512",
        "DIFFERENCE\n(red/orange = extra from > 0)",
        "bilinear > 0.5\n@ 64×64 (zoomed)",
        "bilinear > 0\n@ 64×64 (zoomed)",
    ]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=13, fontweight="bold", pad=10)

    for i, sample in enumerate(selected):
        img_pil = Image.open(sample["ip"]).convert("RGB")
        mask_pil = Image.open(sample["mp"]).convert("L")
        img_512 = np.array(img_pil.resize((512, 512), Image.LANCZOS))

        gt_512 = np.array(mask_pil.resize((512, 512), Image.BILINEAR))
        gt_binary = (gt_512 > 127).astype(np.float32)

        m_05 = resize_mask_bilinear(mask_pil, threshold=0.5)
        m_00 = resize_mask_bilinear(mask_pil, threshold=0.0)
        core_05, dil_05, band_05 = full_pipeline(m_05)
        core_00, dil_00, band_00 = full_pipeline(m_00)

        core_05_512 = latent_to_512(core_05)
        band_05_512 = latent_to_512(band_05)
        dil_05_512 = latent_to_512(dil_05)
        core_00_512 = latent_to_512(core_00)
        band_00_512 = latent_to_512(band_00)
        dil_00_512 = latent_to_512(dil_00)

        # Col 0: Original + mask outline
        img_disp = img_512.copy().astype(np.float32) / 255.0
        outline = ndimage.binary_dilation(gt_binary, iterations=2).astype(float) - gt_binary
        img_disp[outline > 0.5] = [1, 0.2, 0.1]
        axes[i, 0].imshow(img_disp)

        # Col 1: >0.5 affected @ 512
        axes[i, 1].imshow(overlay_on_image(img_512, core_05_512, band_05_512))

        # Col 2: >0 affected @ 512
        axes[i, 2].imshow(overlay_on_image(img_512, core_00_512, band_00_512))

        # Col 3: Difference
        extra_core = ((core_00_512 > 0.5) & (core_05_512 < 0.5)).astype(float)
        extra_band = ((dil_00_512 > 0.5) & (dil_05_512 < 0.5)).astype(float)
        shared_core = ((core_05_512 > 0.5) & (core_00_512 > 0.5)).astype(float)
        shared_band = ((band_05_512 > 0.5) & (core_05_512 < 0.5)).astype(float)

        diff_img = img_512.copy().astype(np.float32) / 255.0
        diff_img *= 0.3  # darken everything
        # Shared core = light gray
        diff_img[shared_core > 0.5] = [0.7, 0.7, 0.7]
        # Shared band = darker gray
        diff_img[shared_band > 0.5] = [0.4, 0.4, 0.5]
        # Extra band = orange
        diff_img[extra_band > 0.5] = [1.0, 0.55, 0.0]
        # Extra core = red
        diff_img[extra_core > 0.5] = [1.0, 0.1, 0.1]
        axes[i, 3].imshow(np.clip(diff_img, 0, 1))

        # Col 4-5: 64×64 zoomed
        axes[i, 4].imshow(render_64_zoomed(core_05.numpy(), band_05.numpy()),
                          interpolation="nearest", aspect="equal")
        axes[i, 5].imshow(render_64_zoomed(core_00.numpy(), band_00.numpy()),
                          interpolation="nearest", aspect="equal")

        # Row label
        axes[i, 0].set_ylabel(
            f"{sample['res']}\n"
            f"mask: {sample['area']} px\n"
            f"core@64: {sample['nc05']}->{sample['nc00']}  (+{sample['core_infl']:.0f}%)\n"
            f"total@64: {sample['nt05']}->{sample['nt00']}  (+{sample['total_infl']:.0f}%)",
            fontsize=10, rotation=0, labelpad=115, va="center",
        )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.85, 0.1], label="Core (generated region)"),
        mpatches.Patch(facecolor=[0.25, 0.4, 0.95], label="Band (transition zone)"),
        mpatches.Patch(facecolor=[0.7, 0.7, 0.7], label="Shared core (both thresholds)"),
        mpatches.Patch(facecolor=[1.0, 0.1, 0.1], label="Extra core from > 0"),
        mpatches.Patch(facecolor=[1.0, 0.55, 0.0], label="Extra band from > 0"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=11, bbox_to_anchor=(0.5, -0.015))

    plt.suptitle(
        "Mask Threshold Impact:  bilinear > 0.5  vs  bilinear > 0\n"
        "Full pipeline: resize->512 -> maxpool->64 -> band dilation (mode 2) -> mapped back to 512×512\n"
        "Samples selected by highest core inflation per resolution",
        fontsize=15, y=1.01,
    )
    plt.tight_layout()

    out_path = "results/viz_mask_threshold_v3.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
