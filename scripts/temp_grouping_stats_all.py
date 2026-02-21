"""
Grouping stats across ALL products. No visualization, just numbers.
Uses SQUARE bounding boxes throughout.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from scipy import ndimage


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
    # Shift to stay within image bounds
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


def pb_size(pb):
    return max(pb[1] - pb[0] + 1, pb[3] - pb[2] + 1)


def group_comps(comps, h, w):
    si = sorted(range(len(comps)),
                key=lambda i: (comps[i][0] + comps[i][1]) / 2)
    gs = []
    for idx in si:
        placed = False
        for g in gs:
            cand = merge_bb(g["mb"], comps[idx])
            cand = square_bb(cand, h, w)  # squarify after merge
            if pb_size(pad_bb(cand, 0.1, h, w)) <= 224:
                g["idxs"].append(idx)
                g["mb"] = cand
                placed = True
                break
        if not placed:
            gs.append({"idxs": [idx], "mb": comps[idx]})
    return gs


def main():
    from PIL import Image

    root = Path(__file__).parent.parent
    master = root / "anomverse_extension" / "datasets" / "full_training_dataset" / "master_training.json"
    data_root = root / "anomverse_extension" / "datasets" / "full_training_dataset"

    with open(master, encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Total samples: {len(samples)}")

    # Global counters
    total = 0
    single_comp = 0
    multi_fits_224 = 0
    multi_needs_groups = 0

    n_groups_dist = Counter()
    comps_per_group_dist = Counter()
    n_comps_dist = Counter()

    # Per-product counters
    product_max_groups = {}  # product -> max groups seen

    # Track worst offenders
    worst = []  # (n_groups, n_comps, product, defect, image_path)

    for i, s in enumerate(samples):
        mask_path = data_root / s["mask_path"]
        if not mask_path.exists():
            continue

        mask = np.array(Image.open(str(mask_path)).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue

        total += 1
        h, w = mask_bin.shape
        labeled, nc = ndimage.label(mask_bin)
        n_comps_dist[nc] += 1

        product = s.get("product", "?")

        if nc == 1:
            single_comp += 1
            n_groups_dist[1] += 1
            if product not in product_max_groups or 1 > product_max_groups[product]:
                product_max_groups[product] = max(product_max_groups.get(product, 0), 1)
            continue

        # Multi-component — check if all comps fit in one square 224 crop
        ys, xs = np.where(mask_bin > 0)
        overall_bb = (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()))
        overall_sq = square_bb(overall_bb, h, w)
        if pb_size(pad_bb(overall_sq, 0.1, h, w)) <= 224:
            multi_fits_224 += 1
            n_groups_dist[1] += 1
            comps_per_group_dist[nc] += 1
            product_max_groups[product] = max(product_max_groups.get(product, 0), 1)
            continue

        multi_needs_groups += 1
        comp_bboxes = []
        for c in range(1, nc + 1):
            cm = (labeled == c).astype(np.float32)
            bb, _, _ = get_comp_bbox(cm)
            if bb:
                comp_bboxes.append(square_bb(bb, h, w))

        groups = group_comps(comp_bboxes, h, w)
        ng = len(groups)
        n_groups_dist[ng] += 1
        for g in groups:
            comps_per_group_dist[len(g["idxs"])] += 1

        product_max_groups[product] = max(product_max_groups.get(product, 0), ng)

        if ng >= 5:
            worst.append((ng, nc, product, s.get("defect_type", "?"), s["image_path"]))

        if (i + 1) % 5000 == 0:
            print(f"  scanned {i+1}/{len(samples)}...")

    # Print results
    print(f"\n{'='*70}")
    print(f"GROUPING STATS — ALL PRODUCTS ({total} samples)")
    print(f"{'='*70}")
    print(f"Single component:       {single_comp:6d} ({100*single_comp/total:.1f}%)")
    print(f"Multi, fits single 224: {multi_fits_224:6d} ({100*multi_fits_224/total:.1f}%)")
    print(f"Multi, needs grouping:  {multi_needs_groups:6d} ({100*multi_needs_groups/total:.1f}%)")

    print(f"\n--- Components per sample ---")
    for k in sorted(n_comps_dist.keys()):
        print(f"  {k:3d} components: {n_comps_dist[k]:5d}")

    print(f"\n--- Groups (224-crops) per sample ---")
    for k in sorted(n_groups_dist.keys()):
        print(f"  {k:3d} groups: {n_groups_dist[k]:5d} ({100*n_groups_dist[k]/total:.2f}%)")

    print(f"\n--- Components per group ---")
    for k in sorted(comps_per_group_dist.keys()):
        print(f"  {k:3d} comps/group: {comps_per_group_dist[k]:5d}")

    print(f"\n--- Max groups per product (top 20) ---")
    for product, mg in sorted(product_max_groups.items(), key=lambda x: -x[1])[:20]:
        print(f"  {product:25s}: max {mg} groups")

    print(f"\n--- Samples with 5+ groups ({len(worst)}) ---")
    worst.sort(key=lambda x: -x[0])
    for ng, nc, product, defect, path in worst[:30]:
        print(f"  {ng}grps {nc}comps  {product:20s} {defect:20s} {path}")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
