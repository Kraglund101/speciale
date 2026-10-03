#!/usr/bin/env python3
"""Build an OOD texture-blend test dataset for the cashew anomaly detector.

For each cashew normal:
- Load its binary foreground mask (from birefnet)
- Place a random ellipse INSIDE the FG mask (so the patch lies fully on cashew)
- Blend a random DTD texture into the canvas at the ellipse, with one of {α=0.2, α=0.4, α=0.6}
- Save canvas.png + gt_mask.png + meta.json per sample

Also creates a clean cohort: same normals saved at full opacity (α=0.0, no patch).

Output structure (each opacity is a separate "shift" we can compute metrics on):
    ood_test_dir/
        opacity_000/  ← clean normals
            000/canvas.png, gt_mask.png (all zeros), meta.json
        opacity_020/
            000/canvas.png, gt_mask.png, meta.json
        opacity_040/
            ...
        opacity_060/
            ...

Notes:
- canvas is resized to 256x256 to match training resolution
- gt_mask is 256x256 binary (1 = OOD patch position, 0 = clean)
- texture is also resized to fit inside the ellipse bounding box
- All randomness is seed-controlled
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def load_binary_mask(path: Path, size: int = 256) -> np.ndarray:
    """Load a binary FG mask, resize to (size, size), threshold at 0.5."""
    m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m, dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)


def load_canvas(path: Path, size: int = 256) -> np.ndarray:
    """Load canvas image, resize to (size, size, 3), uint8."""
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def _make_shape_mask(shape_type: int, H: int, W: int, cy: int, cx: int,
                     dmin: int, dmax: int, rng: random.Random) -> np.ndarray | None:
    """Generate a binary shape mask of the requested type, centered at (cy, cx).

    shape_type:
        0 = ellipse (round defect)
        1 = irregular blob (organic/contamination)
        2 = multi-component (2-3 small touching ellipses; pitting/cluster)
        3 = scratch (very elongated ellipse, aspect ratio 4-8)
    """
    yy, xx = np.mgrid[0:H, 0:W]
    dy = yy - cy
    dx = xx - cx

    if shape_type == 0:
        # Ellipse
        a = rng.randint(max(1, dmin // 2), max(1, dmax // 2))
        b = rng.randint(max(1, dmin // 2), max(1, dmax // 2))
        theta = rng.uniform(0.0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated_y = dy * cos_t + dx * sin_t
        rotated_x = -dy * sin_t + dx * cos_t
        return ((rotated_x / a) ** 2 + (rotated_y / b) ** 2 <= 1.0).astype(np.uint8)

    elif shape_type == 1:
        # Irregular blob: polygon with N vertices on a circle, radii perturbed and smoothed
        N = 12
        r_base = rng.randint(max(1, dmin // 2), max(1, dmax // 2))
        ang_off = rng.uniform(0.0, 2 * np.pi)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + ang_off
        radii = np.array([r_base * rng.uniform(0.55, 1.35) for _ in range(N)])
        # Smooth: simple 3-tap moving average so the blob is not too jagged
        radii_smooth = (radii + np.roll(radii, 1) + np.roll(radii, -1)) / 3.0
        ys = cy + radii_smooth * np.sin(angles)
        xs = cx + radii_smooth * np.cos(angles)
        img = Image.new("L", (W, H), 0)
        ImageDraw.Draw(img).polygon(list(zip(xs.tolist(), ys.tolist())), fill=255)
        return (np.array(img) > 127).astype(np.uint8)

    elif shape_type == 2:
        # Multi-component: 2-3 small ellipses with random offsets (may overlap)
        n_comp = rng.randint(2, 3)
        mask = np.zeros((H, W), dtype=np.uint8)
        comp_max = max(2, dmax // 2)
        for _ in range(n_comp):
            off_y = rng.randint(-comp_max, comp_max)
            off_x = rng.randint(-comp_max, comp_max)
            ny, nx = cy + off_y, cx + off_x
            a = rng.randint(max(1, dmin // 3), max(2, comp_max))
            b = rng.randint(max(1, dmin // 3), max(2, comp_max))
            theta = rng.uniform(0.0, np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dy_c = yy - ny
            dx_c = xx - nx
            ry = dy_c * cos_t + dx_c * sin_t
            rx = -dy_c * sin_t + dx_c * cos_t
            e = ((rx / a) ** 2 + (ry / b) ** 2 <= 1.0)
            mask = mask | e.astype(np.uint8)
        return mask

    elif shape_type == 3:
        # Scratch: elongated ellipse, aspect ratio 4-8
        L = rng.randint(max(4, dmin), max(4, dmax))
        a = max(2, L // 2)
        aspect = rng.randint(4, 8)
        b = max(1, a // aspect)
        theta = rng.uniform(0.0, np.pi)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated_y = dy * cos_t + dx * sin_t
        rotated_x = -dy * sin_t + dx * cos_t
        return ((rotated_x / a) ** 2 + (rotated_y / b) ** 2 <= 1.0).astype(np.uint8)

    return None


def make_shape_inside_fg(
    fg_mask: np.ndarray,
    shape_type: int,
    size_range: tuple,
    max_attempts: int = 200,
    rng: random.Random = random,
) -> np.ndarray | None:
    """Place a shape of given type fully inside FG. Adaptively shrinks size if needed."""
    H, W = fg_mask.shape
    fg_pixels = np.argwhere(fg_mask > 0)
    if len(fg_pixels) == 0:
        return None

    cur_min, cur_max = size_range
    ABSOLUTE_MIN = 6
    while cur_max >= ABSOLUTE_MIN:
        eff_min = max(ABSOLUTE_MIN, cur_min if cur_max > cur_min else ABSOLUTE_MIN)
        for _ in range(max_attempts):
            cy, cx = fg_pixels[rng.randint(0, len(fg_pixels) - 1)]
            mask = _make_shape_mask(shape_type, H, W, cy, cx, eff_min, cur_max, rng)
            if mask is None or mask.sum() == 0:
                continue
            outside = ((mask == 1) & (fg_mask == 0)).sum()
            if outside == 0:
                return mask
        cur_max = int(cur_max * 0.75)
        cur_min = max(ABSOLUTE_MIN, int(cur_min * 0.5))
    return None


def feather_mask(mask: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Gaussian-blur the binary mask edge for soft alpha blend.

    Returns float32 mask in [0, 1].
    """
    pil = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred, dtype=np.float32) / 255.0


def blend_texture(canvas: np.ndarray, texture: np.ndarray, mask: np.ndarray,
                  alpha: float) -> np.ndarray:
    """Alpha-blend texture into canvas where mask > 0.

    Returns uint8 canvas with the texture blended in.
    """
    mask3 = mask[..., None] * alpha  # [H, W, 1]
    out = canvas.astype(np.float32) * (1.0 - mask3) + texture.astype(np.float32) * mask3
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cashew-base", default="C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew")
    p.add_argument("--dtd-dir", default="C:/Users/frede/Desktop/kandidat/speciale/datasets/dtd/images")
    p.add_argument("--output-dir", default="C:/Users/frede/Desktop/kandidat/speciale/results/ood_texture_test_subtle")
    p.add_argument("--n-ood-per-opacity", type=int, default=56,
                   help="Number of unseen normals to use for OOD construction per opacity level (matches cashew_100 test split: 56 anomalies)")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--opacities", nargs="+", type=float, default=[0.05, 0.1])
    p.add_argument("--patch-size-min", type=int, default=8,
                   help="Minimum patch diameter in pixels (8 = pinpoint defect)")
    p.add_argument("--patch-size-max", type=int, default=50,
                   help="Maximum patch diameter in pixels (50 = moderate defect)")
    p.add_argument("--size-tiers", type=int, default=5,
                   help="Stratify sample IDs across this many size buckets to guarantee size diversity")
    p.add_argument("--shape-types", type=int, default=4,
                   help="Number of shape types: 0=ellipse, 1=blob, 2=multi-component, 3=scratch")
    p.add_argument("--feather-sigma", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    cashew_base = Path(args.cashew_base)
    normal_dir = cashew_base / "Data" / "Images" / "Normal"
    fg_dir = cashew_base / "birefnet_masks" / "Normal"
    dtd_dir = Path(args.dtd_dir)
    out_dir = Path(args.output_dir)

    # Read splits.json to exclude any normal SEEN during training
    splits_path = cashew_base / "experiment_UniNet" / "splits.json"
    splits = json.load(open(splits_path))
    train_normals = {n.replace(".JPG", "") for n in splits["normals"]["train"]}
    canvas_normals = {n.replace(".JPG", "") for n in splits["normals"]["canvas"]}
    test_normals = [n.replace(".JPG", "") for n in splits["normals"]["test"]]  # preserve order
    seen_normals = train_normals | canvas_normals
    print(f"Splits: train={len(train_normals)}, canvas={len(canvas_normals)}, test={len(test_normals)}")

    # Pool 1: Clean cohort = the 50 test normals (held out from training, used for OOD-clean comparison)
    clean_pool = [n for n in test_normals if (fg_dir / f"{n}_binary.png").exists()]
    print(f"Clean cohort: {len(clean_pool)} test normals (from splits.json)")

    # Pool 2: OOD source = unseen pool (NOT in train/canvas/test). Used for OOD anomaly construction.
    all_on_disk = [p.stem for p in sorted(normal_dir.glob("*.JPG")) if (fg_dir / f"{p.stem}_binary.png").exists()]
    unseen_pool = [n for n in all_on_disk if n not in seen_normals and n not in test_normals]
    rng.shuffle(unseen_pool)
    ood_pool = unseen_pool[:args.n_ood_per_opacity]
    print(f"OOD source pool: {len(unseen_pool)} unseen normals, using {len(ood_pool)} for OOD construction.")
    print(f"Disjoint pools verified: clean ∩ ood = {set(clean_pool) & set(ood_pool)}")

    # Collect DTD textures (flat list across all categories)
    dtd_categories = sorted(d for d in dtd_dir.iterdir() if d.is_dir())
    all_textures = []
    for cat in dtd_categories:
        all_textures.extend(sorted(cat.glob("*.jpg")))
    print(f"DTD textures available: {len(all_textures)} across {len(dtd_categories)} categories.")

    # Generate clean (opacity_clean from CLEAN_POOL) + each opacity (from OOD_POOL)
    placement_failures = 0

    # 1) Clean cohort — use the 50 test normals UNTOUCHED
    clean_dir = out_dir / "opacity_clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Generating clean cohort (n={len(clean_pool)}) ===")
    for i, idx in enumerate(clean_pool):
        sample_dir = clean_dir / f"{i:04d}"
        sample_dir.mkdir(exist_ok=True)
        canvas = load_canvas(normal_dir / f"{idx}.JPG", args.image_size)
        Image.fromarray(canvas).save(sample_dir / "canvas.png")
        Image.fromarray(np.zeros((args.image_size, args.image_size), dtype=np.uint8)).save(sample_dir / "gt_mask.png")
        meta = {"source_normal": idx, "alpha": 0.0, "patch_size_px": 0, "is_anomaly": False, "pool": "test"}
        json.dump(meta, open(sample_dir / "meta.json", "w"), indent=2)

    # 2) OOD cohorts — use unseen pool, one OOD per opacity level
    for alpha in args.opacities:
        cohort_tag = f"a{int(round(alpha * 1000)):03d}"  # 3-digit milli-alpha (025=0.025, 800=0.800)
        cohort_dir = out_dir / f"opacity_{cohort_tag}"
        cohort_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Generating {cohort_tag} (α={alpha}, n={len(ood_pool)}) ===")
        for i, idx in enumerate(ood_pool):
            sample_dir = cohort_dir / f"{i:04d}"
            sample_dir.mkdir(exist_ok=True)

            canvas = load_canvas(normal_dir / f"{idx}.JPG", args.image_size)
            fg_mask = load_binary_mask(fg_dir / f"{idx}_binary.png", args.image_size)

            # OOD: deterministic shape + random texture + blend
            # FIX: seed depends ONLY on i (NOT on cohort_tag) → same shape across all opacities,
            # only α differs. This is a clean opacity-only ablation.
            sample_rng = random.Random(args.seed * 1000 + i * 9973)  # 9973 prime, no cohort
            # Stratified shape × size:
            #   shape_type = i % 4  (4 shape kinds: ellipse, blob, multi-component, scratch)
            #   tier_idx = (i // 4) % 5  (5 size tiers within [patch_size_min, patch_size_max])
            shape_type = i % args.shape_types
            if args.size_tiers > 1:
                tier_idx = (i // args.shape_types) % args.size_tiers
                tier_width = (args.patch_size_max - args.patch_size_min) / args.size_tiers
                tier_min = int(args.patch_size_min + tier_idx * tier_width)
                tier_max = int(args.patch_size_min + (tier_idx + 1) * tier_width)
                size_range = (tier_min, tier_max)
            else:
                size_range = (args.patch_size_min, args.patch_size_max)
            ellipse = make_shape_inside_fg(
                fg_mask,
                shape_type=shape_type,
                size_range=size_range,
                rng=sample_rng,
            )
            if ellipse is None:
                placement_failures += 1
                # Fallback: skip — write blank canvas (counted as clean)
                Image.fromarray(canvas).save(sample_dir / "canvas.png")
                Image.fromarray(np.zeros((args.image_size, args.image_size), dtype=np.uint8)).save(sample_dir / "gt_mask.png")
                json.dump({"source_normal": idx, "alpha": 0.0, "patch_size_px": 0,
                          "is_anomaly": False, "placement_failed": True},
                          open(sample_dir / "meta.json", "w"), indent=2)
                continue

            # Random DTD texture
            tex_path = all_textures[sample_rng.randint(0, len(all_textures) - 1)]
            tex = Image.open(tex_path).convert("RGB").resize(
                (args.image_size, args.image_size), Image.LANCZOS)
            texture_np = np.array(tex, dtype=np.uint8)

            # Soft mask via feather
            soft_mask = feather_mask(ellipse, sigma=args.feather_sigma)

            # Blend
            out_canvas = blend_texture(canvas, texture_np, soft_mask, alpha)

            # GT mask = binary (use the un-feathered ellipse for ground truth)
            Image.fromarray(out_canvas).save(sample_dir / "canvas.png")
            Image.fromarray((ellipse * 255).astype(np.uint8)).save(sample_dir / "gt_mask.png")
            shape_names = ["ellipse", "blob", "multi", "scratch"]
            meta = {
                "source_normal": idx,
                "alpha": alpha,
                "patch_size_px": int(ellipse.sum()),
                "texture": f"{tex_path.parent.name}/{tex_path.name}",
                "shape_type": int(shape_type),
                "shape_name": shape_names[shape_type] if shape_type < len(shape_names) else "unknown",
                "is_anomaly": True,
            }
            json.dump(meta, open(sample_dir / "meta.json", "w"), indent=2)

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(ood_pool)} done")

    print(f"\nDone. Placement failures: {placement_failures}.")
    print(f"Dataset at: {out_dir}")


if __name__ == "__main__":
    main()
