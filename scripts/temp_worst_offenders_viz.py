"""Visualize the worst offenders: samples needing 7+ groups (square bboxes)."""
import sys
import json
import random
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.mask_utils import adaptive_dilation_radius, dilate_mask

random.seed(42)
np.random.seed(42)

COLORS = [
    (1, 0.2, 0.2), (0.2, 0.6, 1), (0.2, 0.9, 0.3), (1, 0.7, 0.1),
    (0.8, 0.2, 0.9), (0.1, 0.9, 0.9), (1, 0.4, 0.7), (0.6, 0.6, 0.2),
    (0.4, 0.8, 0.6), (0.9, 0.5, 0.2), (0.5, 0.3, 0.8),
]

# From the square-bbox stats run — worst offenders (one per unique image)
SAMPLES = [
    ("realiad_1024/toothbrush/NG/CH/S0057/toothbrush_0057_NG_CH_C1_20230912152236.jpg",
     "realiad_1024/toothbrush/NG/CH/S0057/toothbrush_0057_NG_CH_C1_20230912152236.png",
     "toothbrush", "abrasion", "11grps 34comps"),
    ("AnomVerse_data_filtered/mvtec_ad_2/mvtec_ad_2/sheet_metal/test_public/bad/005_regular.png",
     "AnomVerse_data_filtered/mvtec_ad_2/mvtec_ad_2/sheet_metal/test_public/bad/005_regular_mask.png",
     "sheet_metal", "defect", "10grps 13comps"),
    ("realiad_1024/button_battery/NG/CH/S0032/button_battery_0032_NG_CH_C1_20230922092741.jpg",
     "realiad_1024/button_battery/NG/CH/S0032/button_battery_0032_NG_CH_C1_20230922092741.png",
     "button_battery", "abrasion", "9grps 28comps"),
    ("realiad_1024/toothbrush/NG/CH/S0027/toothbrush_0027_NG_CH_C1_20230912151459.jpg",
     "realiad_1024/toothbrush/NG/CH/S0027/toothbrush_0027_NG_CH_C1_20230912151459.png",
     "toothbrush", "abrasion", "9grps 93comps"),
    ("AnomVerse_data_filtered/BTech/BTech/02/test/ko/0075.png",
     "AnomVerse_data_filtered/BTech/BTech/02/test/ko/0075_mask.png",
     "fabric", "defect", "8grps 9comps"),
    ("realiad_1024/mint/NG/YW/S0136/mint_0136_NG_YW_C1_20230910130948.jpg",
     "realiad_1024/mint/NG/YW/S0136/mint_0136_NG_YW_C1_20230910130948.png",
     "mint", "foreign_objects", "8grps 13comps"),
    ("realiad_1024/toothbrush/NG/CH/S0010/toothbrush_0010_NG_CH_C1_20230912151148.jpg",
     "realiad_1024/toothbrush/NG/CH/S0010/toothbrush_0010_NG_CH_C1_20230912151148.png",
     "toothbrush", "abrasion", "8grps 24comps"),
    ("realiad_1024/fire_hood/NG/HS/S1268/fire_hood_1268_NG_HS_C1_20230928121737.jpg",
     "realiad_1024/fire_hood/NG/HS/S1268/fire_hood_1268_NG_HS_C1_20230928121737.png",
     "fire_hood", "scratch", "7grps 7comps"),
    ("AnomVerse_data_filtered/BTech/BTech/02/test/ko/0036.png",
     "AnomVerse_data_filtered/BTech/BTech/02/test/ko/0036_mask.png",
     "fabric", "defect", "7grps 11comps"),
]


def get_comp_bbox(m):
    ys, xs = np.where(m > 0.5)
    if len(ys) == 0:
        return None, 0, 0
    return ((int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())),
            max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), int(m.sum()))


def square_bb(bb, h, w):
    """Expand bbox to square centered on original center, clipped to image."""
    y0, y1, x0, x1 = bb
    bh = y1 - y0 + 1
    bw = x1 - x0 + 1
    s = max(bh, bw)
    cy = (y0 + y1) / 2.0
    cx = (x0 + x1) / 2.0
    ny0 = int(round(cy - (s - 1) / 2.0))
    ny1 = ny0 + s - 1
    nx0 = int(round(cx - (s - 1) / 2.0))
    nx1 = nx0 + s - 1
    if ny0 < 0:
        ny1 -= ny0; ny0 = 0
    if ny1 >= h:
        ny0 -= (ny1 - h + 1); ny1 = h - 1
    if nx0 < 0:
        nx1 -= nx0; nx0 = 0
    if nx1 >= w:
        nx0 -= (nx1 - w + 1); nx1 = w - 1
    ny0 = max(0, ny0)
    nx0 = max(0, nx0)
    return (ny0, ny1, nx0, nx1)


def merge_bb(a, b):
    return (min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3]))


def pad_bb(bb, f, h, w):
    y0, y1, x0, x1 = bb
    s = max(y1 - y0 + 1, x1 - x0 + 1)
    p = max(1, int(s * f))
    return (max(0, y0 - p), min(h - 1, y1 + p), max(0, x0 - p), min(w - 1, x1 + p))


def pbs(pb):
    return max(pb[1] - pb[0] + 1, pb[3] - pb[2] + 1)


def group_comps(comps, h, w):
    si = sorted(range(len(comps)),
                key=lambda i: (comps[i]["bbox"][0] + comps[i]["bbox"][1]) / 2)
    gs = []
    for idx in si:
        placed = False
        for g in gs:
            cand = merge_bb(g["mb"], comps[idx]["bbox"])
            cand = square_bb(cand, h, w)  # squarify after merge
            if pbs(pad_bb(cand, 0.1, h, w)) <= 224:
                g["idxs"].append(idx)
                g["mb"] = cand
                placed = True
                break
        if not placed:
            gs.append({"idxs": [idx], "mb": comps[idx]["bbox"]})
    return gs


def sample_lu(lo, hi):
    if lo >= hi:
        return lo
    return int(round(math.exp(random.uniform(math.log(lo), math.log(hi)))))


def feasible_pos(cs, pb, h, w):
    py0, py1, px0, px1 = pb
    cs = min(cs, min(h, w))
    y0x = min(py0, h - cs)
    y0n = max(0, py1 - cs + 1)
    x0x = min(px0, w - cs)
    x0n = max(0, px1 - cs + 1)
    if y0n > y0x or x0n > x0x:
        cy, cx = (py0 + py1) // 2, (px0 + px1) // 2
        return max(0, min(cy - cs // 2, h - cs)), max(0, min(cx - cs // 2, w - cs))
    return random.randint(y0n, y0x), random.randint(x0n, x0x)


def crop_resize(img_t, mask_t, cs, tgt, pb, h, w):
    y0, x0 = feasible_pos(cs, pb, h, w)
    cs = min(cs, min(h, w))
    ic = img_t[:, y0:y0 + cs, x0:x0 + cs]
    mc = mask_t[:, y0:y0 + cs, x0:x0 + cs]
    io = F.interpolate(ic.unsqueeze(0), size=(tgt, tgt),
                       mode="bilinear", align_corners=False).squeeze(0)
    mo = F.interpolate(mc.unsqueeze(0).float(), size=(tgt, tgt),
                       mode="nearest").squeeze(0)
    return io, mo, y0, x0, cs


def mp16(m):
    if m.dim() == 2:
        m = m.unsqueeze(0).unsqueeze(0)
    elif m.dim() == 3:
        m = m.unsqueeze(0)
    if m.shape[-1] != 224:
        m = F.interpolate(m.float(), size=(224, 224), mode="nearest")
    return (F.max_pool2d(m.float(), kernel_size=14).squeeze() > 0.5).float()


def find_mask_path(data_root, img_rel, master_samples):
    """Look up actual mask path from master JSON."""
    for s in master_samples:
        if s["image_path"] == img_rel:
            return data_root / s["mask_path"]
    return None


def main():
    root = Path(__file__).parent.parent
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"
    save_dir = root / "results" / "temp_worst_offenders"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load master for mask path lookup
    master = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    with open(master, encoding="utf-8") as f:
        master_samples = json.load(f)

    for si, (ip, mp_default, product, defect, desc) in enumerate(SAMPLES):
        img_path = data_root / ip
        if not img_path.exists():
            print(f"  SKIP {ip} — not found")
            continue

        # Try default mask path, fallback to master lookup
        mask_path = data_root / mp_default
        if not mask_path.exists():
            mask_path = find_mask_path(data_root, ip, master_samples)
            if mask_path is None or not mask_path.exists():
                print(f"  SKIP {ip} — mask not found")
                continue

        img_pil = Image.open(str(img_path)).convert("RGB")
        mask_pil = Image.open(str(mask_path)).convert("L")
        iw, ih = img_pil.size
        mnp = (np.array(mask_pil) > 127).astype(np.float32)
        labeled, nc = ndimage.label(mnp)
        img_t = torch.from_numpy(
            np.array(img_pil).astype(np.float32) / 255).permute(2, 0, 1)

        comps = []
        for c in range(1, nc + 1):
            cm = (labeled == c).astype(np.float32)
            bb, sd, ar = get_comp_bbox(cm)
            if bb:
                bb = square_bb(bb, ih, iw)  # squarify component bbox
                sd = bb[1] - bb[0] + 1  # side of square
                comps.append({
                    "id": c, "mask": cm, "bbox": bb, "side": sd, "area": ar,
                    "pbbox": pad_bb(bb, 0.1, ih, iw),
                    "pbs": pbs(pad_bb(bb, 0.1, ih, iw)),
                })

        groups = group_comps(comps, ih, iw)
        ng = len(groups)

        ginfo = []
        for g in groups:
            gmask = np.zeros_like(mnp)
            mbb = comps[g["idxs"][0]]["bbox"]
            for idx in g["idxs"]:
                gmask = np.maximum(gmask, comps[idx]["mask"])
                mbb = merge_bb(mbb, comps[idx]["bbox"])
            mbb = square_bb(mbb, ih, iw)  # squarify merged bbox
            mpb = pad_bb(mbb, 0.1, ih, iw)
            ginfo.append({
                "idxs": g["idxs"], "nc": len(g["idxs"]), "mask": gmask,
                "pbbox": mpb, "pbs": pbs(mpb),
                "area": sum(comps[idx]["area"] for idx in g["idxs"]),
            })

        # Cap displayed groups at 11
        n_show = min(ng, 11)
        ncols = 1 + n_show
        fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 9))
        if ncols == 1:
            axes = axes.reshape(2, 1)

        orig = img_t.permute(1, 2, 0).numpy()
        axes[0, 0].imshow(orig)

        # Color components by group
        c2g = {}
        for gi, g in enumerate(ginfo[:n_show]):
            for idx in g["idxs"]:
                c2g[idx] = gi
        for idx, comp in enumerate(comps):
            gi = c2g.get(idx, 0)
            col = COLORS[gi % len(COLORS)]
            ov = np.zeros((ih, iw, 4), dtype=np.float32)
            ov[:, :, 0], ov[:, :, 1], ov[:, :, 2] = col
            ov[:, :, 3] = (comp["mask"] > 0.5).astype(np.float32) * 0.5
            axes[0, 0].imshow(ov)

        for gi, g in enumerate(ginfo[:n_show]):
            col = COLORS[gi % len(COLORS)]
            py0, py1, px0, px1 = g["pbbox"]
            axes[0, 0].add_patch(plt.Rectangle(
                (px0, py0), px1 - px0 + 1, py1 - py0 + 1,
                linewidth=2, edgecolor=col, facecolor="none", linestyle="--"))

        axes[0, 0].set_title(
            f"Original {iw}x{ih}\n{nc} comps -> {ng} groups", fontsize=8)
        axes[0, 0].axis("off")

        axes[1, 0].imshow(orig)
        ov_a = np.zeros((ih, iw, 4), dtype=np.float32)
        ov_a[:, :, 0] = 1.0
        ov_a[:, :, 3] = (mnp > 0.5).astype(np.float32) * 0.5
        axes[1, 0].imshow(ov_a)
        axes[1, 0].set_title(f"Full mask\n{product} / {defect}", fontsize=7)
        axes[1, 0].axis("off")

        for gi, g in enumerate(ginfo[:n_show]):
            c = gi + 1
            gmt = torch.from_numpy(g["mask"]).unsqueeze(0)
            if g["pbs"] <= 224:
                cs = 224
                ms = "fixed"
            else:
                cs = sample_lu(g["pbs"], min(ih, iw))
                ms = f"logU[{g['pbs']},{min(ih, iw)}]"

            i224, m224, y0, x0, csa = crop_resize(
                img_t, gmt, cs, 224, g["pbbox"], ih, iw)
            r224 = adaptive_dilation_radius(m224, min_r=1, max_r=10)
            d224 = dilate_mask(m224, r224)
            inp = i224.permute(1, 2, 0).numpy()
            cnp = m224.squeeze().numpy()
            dnp = d224.squeeze().numpy()
            rnp = np.clip(dnp - cnp, 0, 1)

            axes[0, c].imshow(inp)
            ov_c = np.zeros((224, 224, 4), dtype=np.float32)
            ov_c[:, :, 0] = 1
            ov_c[:, :, 3] = (cnp > 0.5).astype(np.float32) * 0.45
            axes[0, c].imshow(ov_c)
            ov_r = np.zeros((224, 224, 4), dtype=np.float32)
            ov_r[:, :, 2] = 1
            ov_r[:, :, 3] = (rnp > 0.5).astype(np.float32) * 0.35
            axes[0, c].imshow(ov_r)

            c16 = mp16(m224)
            d16 = mp16(d224)
            cell = 224 / 16
            nc16 = int(c16.sum())
            nr16 = int(np.clip(d16.numpy() - c16.numpy(), 0, 1).sum())
            for rr in range(16):
                for cc in range(16):
                    if c16[rr, cc] > 0.5:
                        axes[0, c].add_patch(plt.Rectangle(
                            (cc * cell, rr * cell), cell, cell,
                            lw=0.5, ec="red", fc="none"))
                    elif d16[rr, cc] - c16[rr, cc] > 0.5:
                        axes[0, c].add_patch(plt.Rectangle(
                            (cc * cell, rr * cell), cell, cell,
                            lw=0.5, ec="blue", fc="none"))
            for j in range(17):
                axes[0, c].axhline(y=j * cell, color="white", lw=0.2, alpha=0.3)
                axes[0, c].axvline(x=j * cell, color="white", lw=0.2, alpha=0.3)

            zm = ih / csa
            axes[0, c].set_title(
                f"G{gi+1} ({g['nc']}c) S={csa} ({zm:.1f}x)\n"
                f"r={r224} {nc16}c+{nr16}r", fontsize=6)
            axes[0, c].axis("off")

            col = COLORS[gi % len(COLORS)]
            axes[1, c].imshow(inp)
            ov = np.zeros((224, 224, 4), dtype=np.float32)
            ov[:, :, 0], ov[:, :, 1], ov[:, :, 2] = col
            ov[:, :, 3] = (cnp > 0.5).astype(np.float32) * 0.6
            axes[1, c].imshow(ov)
            cids = [comps[idx]["id"] for idx in g["idxs"]]
            axes[1, c].set_title(
                f"G{gi+1}: {g['nc']}c\npad={g['pbs']} | {ms}", fontsize=6)
            axes[1, c].axis("off")

        plt.suptitle(
            f"#{si} {product} / {defect} \u2014 {desc}\n"
            f"{nc} comps \u2192 {ng} groups",
            fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0.01, 0.02, 1, 0.91])
        plt.savefig(
            save_dir / f"{si:02d}_{product}_{defect}_g{ng}.png",
            dpi=130, bbox_inches="tight")
        plt.close()
        gsz = "+".join(f"{g['nc']}c" for g in ginfo[:n_show])
        print(f"  [{si+1}/{len(SAMPLES)}] {product} {desc} [{gsz}]")

    print(f"\nSaved to {save_dir}/")


if __name__ == "__main__":
    main()
