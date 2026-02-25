"""Worst-case mask inflation visualization — v2 style with large panels.

Picks the top ~8 samples by core inflation across resolutions.
Shows the actual image with overlaid affected regions so you can
see exactly what image area gets modified.

Per sample row:
  Col 0: Original image + GT mask outline (red)
  Col 1: bilinear > 0.5 — core (yellow) + band (blue) on image @ 512
  Col 2: bilinear > 0   — core (yellow) + band (blue) on image @ 512
  Col 3: DIFFERENCE — shared (gray), extra core (red), extra band (orange)
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


def resize_mask_bilinear(mask_pil, size=512, threshold=0.5):
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > threshold).float()


def full_pipeline(mask_512, band_mode=2):
    mask_4d = mask_512.unsqueeze(0)
    core_64 = downsample_mask_maxpool(mask_4d, 64)
    dilated_binary, alpha_map, weight_map, band_mask = create_latent_band_mask(
        core_64, band_mode=band_mode
    )
    return (core_64.squeeze(0).squeeze(0),
            dilated_binary.squeeze(0).squeeze(0),
            band_mask.squeeze(0).squeeze(0))


def latent_to_512(mask_64):
    m = mask_64.unsqueeze(0).unsqueeze(0)
    up = F.interpolate(m, size=(512, 512), mode="nearest")
    return up.squeeze().numpy()


def overlay_on_image(img_512, core_512, band_512):
    out = img_512.copy().astype(np.float32) / 255.0
    bg = (core_512 < 0.5) & (band_512 < 0.5)
    out[bg] *= 0.35
    band_only = (band_512 > 0.5) & (core_512 < 0.5)
    out[band_only] = out[band_only] * 0.35 + np.array([0.25, 0.4, 0.95]) * 0.65
    core_px = core_512 > 0.5
    out[core_px] = out[core_px] * 0.4 + np.array([1.0, 0.85, 0.1]) * 0.6
    return np.clip(out, 0, 1)


def main():
    root = Path("anomverse_extension/datasets/full_training_dataset")
    master_path = root / "master_training.json"

    with open(master_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("images") or data.get("samples", [])

    # Evaluate all non-512 samples for inflation
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

        if len(candidates) >= 4000:
            break

    print(f"Evaluated {len(candidates)} non-512 samples")

    # Sort globally by total inflation, deduplicate by stats signature, take top 8
    candidates.sort(key=lambda x: -x["total_infl"])
    selected = []
    seen_sig = set()
    for c in candidates:
        sig = (c["nc05"], c["nc00"], c["nt05"], c["nt00"])
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        selected.append(c)
        if len(selected) >= 8:
            break

    print(f"\nTop 8 worst-case samples by total@64 inflation:")
    for s in selected:
        print(f"  {s['res']}  mask={s['area']}px  "
              f"core={s['nc05']}->{s['nc00']} (+{s['core_infl']:.0f}%)  "
              f"total={s['nt05']}->{s['nt00']} (+{s['total_infl']:.0f}%)")

    # --- PLOT ---
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 4, figsize=(28, 6.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = [
        "Original + GT mask (red outline)",
        "bilinear > 0.5\ncore (yellow) + band (blue) @ 512",
        "bilinear > 0\ncore (yellow) + band (blue) @ 512",
        "DIFFERENCE\ngray=shared  red=extra core  orange=extra band",
    ]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=14, fontweight="bold", pad=12)

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
        img_disp[outline > 0.5] = [1, 0.15, 0.1]
        axes[i, 0].imshow(img_disp)

        # Col 1: >0.5 overlay
        axes[i, 1].imshow(overlay_on_image(img_512, core_05_512, band_05_512))

        # Col 2: >0 overlay
        axes[i, 2].imshow(overlay_on_image(img_512, core_00_512, band_00_512))

        # Col 3: Difference
        extra_core = ((core_00_512 > 0.5) & (core_05_512 < 0.5)).astype(float)
        extra_band = ((dil_00_512 > 0.5) & (dil_05_512 < 0.5)).astype(float)
        shared_core = ((core_05_512 > 0.5) & (core_00_512 > 0.5)).astype(float)
        shared_band = ((band_05_512 > 0.5) & (core_05_512 < 0.5)).astype(float)

        diff_img = img_512.copy().astype(np.float32) / 255.0
        diff_img *= 0.25
        diff_img[shared_core > 0.5] = [0.7, 0.7, 0.7]
        diff_img[shared_band > 0.5] = [0.4, 0.4, 0.55]
        diff_img[extra_band > 0.5] = [1.0, 0.55, 0.0]
        diff_img[extra_core > 0.5] = [1.0, 0.1, 0.1]
        axes[i, 3].imshow(np.clip(diff_img, 0, 1))

        # Row label
        axes[i, 0].set_ylabel(
            f"{sample['res']}\n"
            f"mask: {sample['area']} px\n\n"
            f"core@64: {sample['nc05']} -> {sample['nc00']}\n"
            f"(+{sample['core_infl']:.0f}%)\n\n"
            f"total@64: {sample['nt05']} -> {sample['nt00']}\n"
            f"(+{sample['total_infl']:.0f}%)",
            fontsize=11, rotation=0, labelpad=120, va="center",
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
               fontsize=12, bbox_to_anchor=(0.5, -0.008))

    plt.suptitle(
        "WORST-CASE Mask Inflation:  bilinear > 0  vs  bilinear > 0.5\n"
        "Pipeline: resize->512  ->  maxpool->64  ->  band dilation (mode 2)  ->  mapped back to 512x512\n"
        "Top 8 samples globally by total@64 inflation",
        fontsize=16, y=1.005,
    )
    plt.tight_layout()

    out_path = "results/UNet/viz_mask_threshold_worst_cases.png"
    os.makedirs("results/UNet", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
