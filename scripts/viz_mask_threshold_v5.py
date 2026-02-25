"""Compare all 3 mask downsampling strategies through full pipeline to 64x64.

Columns:
  0: Original image + GT mask outline
  1: NEAREST > 0.5 (old bug) — affected region @ 512
  2: MAXPOOL (ceil fix) — affected region @ 512
  3: BILINEAR > 0.5 — affected region @ 512
  4: DIFF: maxpool vs nearest — green=recovered by maxpool, red=lost by nearest

Picks worst-case samples where the 3 strategies differ most.
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


def resize_mask_nearest(mask_pil, size=512):
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > 0.5).float()


def resize_mask_bilinear(mask_pil, size=512, threshold=0.5):
    t = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    m = t(mask_pil.convert("L"))
    return (m > threshold).float()


def resize_mask_maxpool(mask_pil, size=512):
    """Use downsample_mask_maxpool (ceil fix) for arbitrary resolution -> size."""
    m = transforms.ToTensor()(mask_pil.convert("L"))  # [1, H, W]
    m = (m > 0.5).float()
    return downsample_mask_maxpool(m, size)  # [1, size, size]


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

    # Evaluate all non-512 samples
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
            m_pil = Image.open(mp).convert("L")
            w, h = m_pil.size
            if w == 512 and h == 512:
                continue
            area = (np.array(m_pil) > 127).sum()
            if area < 10:
                continue

            m_near = resize_mask_nearest(m_pil)
            m_mpool = resize_mask_maxpool(m_pil)
            m_bilin = resize_mask_bilinear(m_pil, threshold=0.5)

            # Run each through full pipeline
            cn, dn, bn = full_pipeline(m_near)
            cm, dm, bm = full_pipeline(m_mpool)
            cb, db, bb = full_pipeline(m_bilin)

            ncn = cn.sum().item()
            ncm = cm.sum().item()
            ncb = cb.sum().item()
            ntn = dn.sum().item()
            ntm = dm.sum().item()
            ntb = db.sum().item()

            # How much does maxpool recover vs nearest?
            core_diff = ncm - ncn
            total_diff = ntm - ntn

            candidates.append({
                "ip": ip, "mp": mp, "res": f"{w}x{h}", "area": area,
                "ncn": int(ncn), "ncm": int(ncm), "ncb": int(ncb),
                "ntn": int(ntn), "ntm": int(ntm), "ntb": int(ntb),
                "core_diff": core_diff, "total_diff": total_diff,
            })
        except Exception:
            pass

        if len(candidates) >= 4000:
            break

    print(f"Evaluated {len(candidates)} non-512 samples")

    # Sort by total_diff descending (where maxpool helps most vs nearest)
    candidates.sort(key=lambda x: -x["total_diff"])
    selected = []
    seen_sig = set()
    for c in candidates:
        sig = (c["ncn"], c["ncm"], c["ntn"], c["ntm"])
        if sig in seen_sig:
            continue
        seen_sig.add(sig)
        selected.append(c)
        if len(selected) >= 8:
            break

    print(f"\nTop 8 samples where maxpool vs nearest differs most (total@64):")
    for s in selected:
        mp_vs_near_core = s['ncm'] - s['ncn']
        mp_vs_near_total = s['ntm'] - s['ntn']
        bi_vs_near_core = s['ncb'] - s['ncn']
        bi_vs_near_total = s['ntb'] - s['ntn']
        print(f"  {s['res']}  mask={s['area']}px  "
              f"core@64: near={s['ncn']} mpool={s['ncm']} bilin={s['ncb']}  "
              f"total@64: near={s['ntn']} mpool={s['ntm']} bilin={s['ntb']}")

    # --- PLOT ---
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 5, figsize=(35, 6.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_labels = [
        "Original + GT mask",
        "NEAREST > 0.5 (old)\ncore + band @ 512",
        "MAXPOOL (ceil fix)\ncore + band @ 512",
        "BILINEAR > 0.5\ncore + band @ 512",
        "DIFF: maxpool vs nearest\ngreen=recovered  red=lost by nearest",
    ]
    for j, label in enumerate(col_labels):
        axes[0, j].set_title(label, fontsize=13, fontweight="bold", pad=12)

    for i, sample in enumerate(selected):
        img_pil = Image.open(sample["ip"]).convert("RGB")
        mask_pil = Image.open(sample["mp"]).convert("L")
        img_512 = np.array(img_pil.resize((512, 512), Image.LANCZOS))

        gt_512 = np.array(mask_pil.resize((512, 512), Image.BILINEAR))
        gt_binary = (gt_512 > 127).astype(np.float32)

        m_near = resize_mask_nearest(mask_pil)
        m_mpool = resize_mask_maxpool(mask_pil)
        m_bilin = resize_mask_bilinear(mask_pil, threshold=0.5)

        cn, dn, bn = full_pipeline(m_near)
        cm, dm, bm = full_pipeline(m_mpool)
        cb, db, bb = full_pipeline(m_bilin)

        cn_512 = latent_to_512(cn); bn_512 = latent_to_512(bn); dn_512 = latent_to_512(dn)
        cm_512 = latent_to_512(cm); bm_512 = latent_to_512(bm); dm_512 = latent_to_512(dm)
        cb_512 = latent_to_512(cb); bb_512 = latent_to_512(bb); db_512 = latent_to_512(db)

        # Col 0: Original + mask outline
        img_disp = img_512.copy().astype(np.float32) / 255.0
        outline = ndimage.binary_dilation(gt_binary, iterations=2).astype(float) - gt_binary
        img_disp[outline > 0.5] = [1, 0.15, 0.1]
        axes[i, 0].imshow(img_disp)

        # Col 1: nearest
        axes[i, 1].imshow(overlay_on_image(img_512, cn_512, bn_512))

        # Col 2: maxpool
        axes[i, 2].imshow(overlay_on_image(img_512, cm_512, bm_512))

        # Col 3: bilinear > 0.5
        axes[i, 3].imshow(overlay_on_image(img_512, cb_512, bb_512))

        # Col 4: Difference maxpool vs nearest
        extra_core_mp = ((cm_512 > 0.5) & (cn_512 < 0.5)).astype(float)  # maxpool has, nearest doesn't
        extra_band_mp = ((dm_512 > 0.5) & (dn_512 < 0.5) & (cm_512 < 0.5)).astype(float)
        lost_core = ((cn_512 > 0.5) & (cm_512 < 0.5)).astype(float)  # nearest has, maxpool doesn't (shouldn't happen)
        shared_core = ((cn_512 > 0.5) & (cm_512 > 0.5)).astype(float)
        shared_band = ((bn_512 > 0.5) & (cn_512 < 0.5)).astype(float)

        diff_img = img_512.copy().astype(np.float32) / 255.0
        diff_img *= 0.25
        # Shared regions
        diff_img[shared_core > 0.5] = [0.6, 0.6, 0.6]
        diff_img[shared_band > 0.5] = [0.35, 0.35, 0.45]
        # Extra from maxpool = green (recovered pixels)
        diff_img[extra_band_mp > 0.5] = [0.1, 0.8, 0.3]
        diff_img[extra_core_mp > 0.5] = [0.0, 1.0, 0.2]
        # Lost by nearest but not maxpool = red (shouldn't happen much)
        diff_img[lost_core > 0.5] = [1.0, 0.1, 0.1]
        axes[i, 4].imshow(np.clip(diff_img, 0, 1))

        # Row label
        axes[i, 0].set_ylabel(
            f"{sample['res']}\n"
            f"mask: {sample['area']} px\n\n"
            f"core@64:\n"
            f"  near={sample['ncn']}\n"
            f"  mpool={sample['ncm']}\n"
            f"  bilin={sample['ncb']}\n\n"
            f"total@64:\n"
            f"  near={sample['ntn']}\n"
            f"  mpool={sample['ntm']}\n"
            f"  bilin={sample['ntb']}",
            fontsize=10, rotation=0, labelpad=120, va="center", family="monospace",
        )

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    legend_elements = [
        mpatches.Patch(facecolor=[1.0, 0.85, 0.1], label="Core (generated region)"),
        mpatches.Patch(facecolor=[0.25, 0.4, 0.95], label="Band (transition zone)"),
        mpatches.Patch(facecolor=[0.6, 0.6, 0.6], label="Shared (all methods)"),
        mpatches.Patch(facecolor=[0.0, 1.0, 0.2], label="Recovered by maxpool (vs nearest)"),
        mpatches.Patch(facecolor=[0.1, 0.8, 0.3], label="Extra band from maxpool"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=12, bbox_to_anchor=(0.5, -0.008))

    plt.suptitle(
        "3-Way Comparison:  NEAREST > 0.5 (old)  vs  MAXPOOL (ceil fix)  vs  BILINEAR > 0.5\n"
        "Full pipeline: resize->512 -> maxpool->64 -> band dilation (mode 2) -> mapped back to 512\n"
        "Sorted by where maxpool recovers most pixels vs nearest",
        fontsize=15, y=1.005,
    )
    plt.tight_layout()

    out_path = "results/UNet/viz_mask_3way_worst_cases.png"
    os.makedirs("results/UNet", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
