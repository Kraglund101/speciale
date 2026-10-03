#!/usr/bin/env python3
"""Probe: paint donor holes onto clean cashews at typical cashew-hole size, with a re-seeding visibility gate.

Donors (--mix, default an even split):
  hazelnut   MVTec hazelnut "hole" — the current cross-object partner. Median 12964 px, 128 px across,
             dL* +0.1 (a crater exposing pale flesh), so it must be shrunk 6-9x to reach cashew-hole size.
  fire_hood  RealIAD fire_hood "pit" — median 321 px, 20 px across, dL* -4.8. Already cashew-sized, so it is
             placed at (almost) native scale.
Target: the median cashew "small holes" mask, 415 px at 1274x1176 (~23 px across), dL* -21.6.

What is placed on the canvas is a REAL cashew "small holes" mask (--mask-source cashew, the default), carried
through the ordinary placement augmentation (random flip / rotation / scale 0.8-1.2x / position inside the
foreground). The generator is CONDITIONED on the donor anomaly at its native scale: the donor image and the
donor's own mask feed the CLIP crop. So the shape being filled is a genuine cashew defect footprint, while the
appearance comes from the other object -- exactly the transfer the extension experiment will use.
--mask-source donor reverts to placing the donor's own mask, rescaled to the median cashew hole area.

No extra placement constraints, so the method stays justifiable.

Acceptance: a generation counts as a real anomaly only if the REFINED mask is clearly smaller than the LATENT
(roundtripped) mask it was thresholded inside: refined / roundtrip <= --max-fill. The roundtripped mask is what
the UNet can actually paint (mask -> 512 -> 64x64 maxpool -> band dilation -> 512); on a no-op the refiner has no
gradient and returns that whole region (fill ~1.0), on a real defect it keeps ~0.3 of it. p90 CIELAB distance is
logged alongside, and only gates the result if --min-de is raised above 0.

Retry policy (--tries-per-position, --max-positions): try the seed, and if it fails try ONE more seed at the same
spot; if that fails too, sample a NEW POSITION for the mask and start over there. Attempts and positions used are
logged per case.

  python scripts/hole_donor_probe.py --n 10 --mix hazelnut,fire_hood
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import generate_cashew as gc  # noqa: E402
from src.utils.perceptual_mask_utils import compute_dilated_mask_512  # noqa: E402
from src.utils.placement_utils import place_easy_mask  # noqa: E402

AV = ROOT / "anomverse_extension/datasets/full_training_dataset"
HZ_IMG = AV / "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/hole"
HZ_MASK = AV / "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/ground_truth/hole"
CASHEW_MASKS = gc.CASHEW_ROOT / "Data" / "Masks" / "Anomaly"
PAIRING = ROOT / "results/masked_patch_knn/pairing_constrained.json"
OUT = ROOT / "results/hole_donor_probe"
CKPTS = {"core-50k @20k": ROOT / "results/hero_50k_server/checkpoint_20000",
         "core-50k @40k": ROOT / "results/hero_40k_download/checkpoint_40000",
         # clean generator (2026-09-28): master data only (no cashew_visa), constant LR 1e-4, 20k steps
         "clean-20k": ROOT / "results/generator_nocashew_20k_constlr/checkpoint_20000"}
RT_COL, RF_COL, PL_COL = (0, 110, 255), (255, 0, 255), (0, 255, 255)


def binary(path: Path) -> np.ndarray:
    """Any non-zero pixel is mask: VisA GT masks store 1/2, our placed masks store 255."""
    return (np.array(Image.open(path).convert("L")) > 0).astype(np.float32)


def target_area() -> float:
    pairs = json.loads(PAIRING.read_text(encoding="utf-8"))
    return float(np.median([(np.array(Image.open(CASHEW_MASKS / f"{r['id']}.png").convert("L")) > 0).sum()
                            for r in pairs if r["cls"] == "holes"]))


def donor_pool(name: str) -> list[dict]:
    """Reference anomalies of one donor: id, image, mask, area (compact single-component ones preferred)."""
    if name == "hazelnut":
        pairs = json.loads(PAIRING.read_text(encoding="utf-8"))
        ids = sorted({r["nn_id"] for r in pairs if r["cls"] == "holes"})
        pool = [{"src": "hazelnut", "ref": i, "img": HZ_IMG / f"{i}.png", "mask": HZ_MASK / f"{i}_mask.png"}
                for i in ids]
    elif name == "cashew":                      # within-domain control: conditioning is a real cashew hole
        pairs = json.loads(PAIRING.read_text(encoding="utf-8"))
        pool = []
        for i in sorted({r["id"] for r in pairs if r["cls"] == "holes"}):
            img = next((q for q in [gc.CASHEW_ROOT / "Data/Images/Anomaly" / f"{i}.JPG",
                                    gc.CASHEW_ROOT / "Data/Images/Anomaly/easy_imgs" / f"{i}.JPG"] if q.exists()),
                       None)
            if img is not None:
                pool.append({"src": "cashew", "ref": i, "img": img, "mask": CASHEW_MASKS / f"{i}.png"})
    elif name == "fire_hood":
        master = json.loads((AV / "master_training.json").read_text(encoding="utf-8"))
        pool = [{"src": "fire_hood", "ref": Path(x["image_path"]).stem.split("_")[2],
                 "img": AV / x["image_path"], "mask": AV / x["mask_path"]}
                for x in master if x["dataset"] == "realiad" and x["product"] == "fire_hood"
                and "pit" in x["defect_type"].lower()]
    else:
        raise ValueError(f"unknown donor {name}")
    out = []
    for p in pool:
        if not (p["img"].exists() and p["mask"].exists()):
            continue
        m = binary(p["mask"]) > 0.5
        if m.sum() < 5:
            continue
        lab, n = nd.label(m)
        ys, xs = np.nonzero(m)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        out.append({**p, "area": int(m.sum()), "components": int(n), "elong": float(max(h, w) / min(h, w))})
    return out


def normalise_mask(src: Path, name: str) -> Path:
    """Write a 0/255 copy of a donor mask (VisA stores 1/2/5; downstream loaders threshold at 0.5).

    `name` MUST be unique per donor IMAGE. RealIAD photographs one sample from several cameras (C1-C5), each a
    separate image+mask, so naming by sample number let two views overwrite each other's mask and conditioned a
    generation on the wrong region (2026-09-25). Refuses to overwrite an existing file with different content.
    """
    d = OUT / "ref_masks"
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{name}.png"
    arr = (np.array(Image.open(src).convert("L")) > 0).astype(np.uint8) * 255
    if arr.max() == 0:
        raise ValueError(f"donor mask {src} is empty")
    if out.exists():                                 # never rewrite: parallel processes read these files (race 2026-10-03)
        if not np.array_equal(np.array(Image.open(out).convert("L")), arr):
            raise RuntimeError(f"{out} already holds a DIFFERENT mask - two donors share the name '{name}'")
        return out
    tmp = d / f"{name}.{os.getpid()}.tmp.png"; Image.fromarray(arr).save(tmp); os.replace(tmp, out)
    return out


def cashew_hole_masks() -> list[dict]:
    """The 20 single-label cashew 'small holes' anomalies: mask (the footprint to place) and source image."""
    pairs = json.loads(PAIRING.read_text(encoding="utf-8"))
    out = []
    for i in sorted({r["id"] for r in pairs if r["cls"] == "holes"}):
        mp = CASHEW_MASKS / f"{i}.png"
        img = next((q for q in [gc.CASHEW_ROOT / "Data/Images/Anomaly" / f"{i}.JPG",
                                gc.CASHEW_ROOT / "Data/Images/Anomaly/easy_imgs" / f"{i}.JPG"] if q.exists()), None)
        area = int((np.array(Image.open(mp).convert("L")) > 0).sum())
        out.append({"id": i, "mask": mp, "img": img, "area": area})
    return sorted(out, key=lambda x: x["area"])


def build_cases(n: int, mix: list[str], seed: int) -> list[dict]:
    """Even split over donors; hazelnut spread across its size range, fire_hood the compact cashew-sized ones."""
    tgt = target_area()
    canvases = sorted(p.stem for p in gc.CANVAS_IMGS.glob("*.JPG"))
    random.Random(seed).shuffle(canvases)
    cases: list[dict] = []
    per = [n // len(mix) + (1 if k < n % len(mix) else 0) for k in range(len(mix))]
    for donor, k in zip(mix, per):
        pool = donor_pool(donor)
        if donor in ("hazelnut", "cashew"):
            pool.sort(key=lambda p: p["area"])
            picked = [pool[round(i * (len(pool) - 1) / max(k - 1, 1))] for i in range(k)]
        else:                                   # compact, single-component, nearest to the cashew median
            pool = [p for p in pool if p["components"] == 1 and p["elong"] < 1.6]
            pool.sort(key=lambda p: abs(p["area"] - tgt))
            picked = pool[:k]
        for p in picked:
            c = {**p, "canvas": canvases[len(cases)], "target": int(tgt)}
            c["mask"] = normalise_mask(c["mask"], f"{c['src']}_{Path(c['img']).stem}")   # unique per IMAGE   # donors may store 1/2/5, not 255
            c["scale"] = scale_for(c)
            cases.append(c)
    holes = cashew_hole_masks()                     # real cashew footprints, spread over their size range
    for k, c in enumerate(cases):
        h = holes[round(k * (len(holes) - 1) / max(len(cases) - 1, 1))]
        c["cashew_id"], c["cashew_mask"], c["cashew_img"], c["cashew_area"] = h["id"], h["mask"], h["img"], h["area"]
    # NOTE the in-distribution arm ends up conditioned on the same cashew whose hole supplies the footprint.
    # That is deliberate and matches generate_cashew.py: a reference anomaly supplies BOTH the conditioning
    # (its image + mask) and the placed mask, and the target is an unseen normal CANVAS. Nothing leaks - the
    # model never sees the canvas it must paint onto. The cross-object arm differs only in that the appearance
    # comes from a donor class while the footprint stays a real cashew hole.
    # every case must carry its OWN reference image + mask: same size, distinct files
    seen = {}
    for k, c in enumerate(cases):
        iw, ih = Image.open(c["img"]).size
        mw, mh = Image.open(c["mask"]).size
        if (iw, ih) != (mw, mh):
            raise RuntimeError(f"case {k}: reference image {iw}x{ih} vs mask {mw}x{mh} ({c['img']}, {c['mask']})")
        prev = seen.setdefault(str(c["mask"]), str(c["img"]))
        if prev != str(c["img"]):
            raise RuntimeError(f"case {k}: mask {c['mask']} is shared by two different reference images")
    return cases


def scale_for(c: dict) -> float:
    """Linear scale that brings the donor mask to the target area on the canvas pixel grid."""
    fg = binary(gc.CANVAS_IMGS.parent / "masks" / f"{c['canvas']}.png") > 0.5
    m = resize_mask(binary(c["mask"]) > 0.5, fg.shape)
    return float(np.sqrt(c["target"] / m.sum()))


def place_one(c: dict, seed: int, pos: int, out_dir: Path, mask_source: str = "cashew") -> Path:
    """Place the footprint on the canvas: a real cashew hole mask under the usual placement augmentation
    (mask_source='cashew'), or the donor's own mask rescaled to cashew-hole area (mask_source='donor')."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fg = binary(gc.CANVAS_IMGS.parent / "masks" / f"{c['canvas']}.png") > 0.5
    src = c["cashew_mask"] if mask_source == "cashew" else c["mask"]
    m = (np.array(Image.open(src).convert("L")) > 0).astype(np.float32)
    if m.shape != fg.shape:
        m = resize_mask(m > 0.5, fg.shape).astype(np.float32)
    rng = (0.8, 1.2) if mask_source == "cashew" else (c["scale"], c["scale"])   # thesis placement augmentation
    placed = place_easy_mask(m, fg.astype(np.float32), scale_range=rng, seed=seed)
    if placed is None:
        raise RuntimeError(f"placement failed for {c['src']} {c['ref']} on {c['canvas']}")
    c["placed_area"] = int(placed.sum())
    c["edge_dist"] = float(nd.distance_transform_edt(fg)[placed > 0.5].min())
    path = out_dir / f"{c['src']}_{c['ref']}_on_{c['canvas']}_pos{pos}.png"
    c["placed_source"] = mask_source
    Image.fromarray((placed * 255).astype(np.uint8)).save(path)
    return path


def roundtrip_mask(placed_path: Path) -> np.ndarray:
    """Exactly as compute_refined_masks.py builds it, from the NATIVE-resolution placed mask."""
    native = (np.array(Image.open(placed_path).convert("L")).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    return compute_dilated_mask_512(native, band_mode=2, target_size=512) > 0.5


def p90_de(gen: Path, canvas: Path, placed: Path) -> float:
    from skimage.color import rgb2lab
    m = to512(np.array(Image.open(placed).convert("L")) > 0)

    def lab(p: Path) -> np.ndarray:
        return rgb2lab(np.array(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)) / 255.0)

    return float(np.percentile(np.linalg.norm(lab(gen) - lab(canvas), axis=2)[m], 90))


def write_prep_case(c: dict, i: int, prep: Path) -> None:
    """(Re)write the refiner's per-case prep folder for the case's CURRENT placed mask."""
    d = prep / f"{i:03d}"
    d.mkdir(parents=True, exist_ok=True)
    json.dump({"idx": i, "normal": gc.resolve_canvas_image(c["canvas"]).name, "is_hard": False, "seed": 42},
              open(d / "meta.json", "w"), indent=1)
    shutil.copy(c["mask_path"], d / "placed_mask.png")
    Image.open(gc.resolve_canvas_image(c["canvas"])).convert("RGB").save(d / "canvas.png")
    Image.open(c["img"]).convert("RGB").save(d / "ref_image.png")
    Image.open(c["mask"]).convert("L").save(d / "ref_mask.png")


def refine_dir(gen_dir: Path, prep: Path, out_dir: Path) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts/compute_refined_masks.py"),
                    "--gen-dir", str(gen_dir), "--prep-dir", str(prep), "--output-dir", str(out_dir),
                    "--fg-dir", str(gc.CASHEW_ROOT / "birefnet_masks" / "Normal"), "--canvas-ref", "raw"],
                   check=True, cwd=str(ROOT), stdout=subprocess.DEVNULL)


def components_of(mask_path: Path, min_px512: int = 0) -> tuple[list[np.ndarray], list[int]]:
    """Connected components of a placed footprint (native resolution, largest first).

    Components smaller than `min_px512` pixels AT 512 (the generation resolution) are excluded from the
    footprint altogether: below roughly one latent cell the generator paints them only by luck, and a mask that
    marks an unpainted blob is label noise. Returns (kept, dropped_sizes_at_512).
    """
    m = np.array(Image.open(mask_path).convert("L")) > 127
    lab, n = nd.label(m)
    comps = sorted(((lab == k) for k in range(1, n + 1)), key=lambda c: -c.sum())
    kept, dropped = [], []
    for c in comps:
        px512 = int(to512(c).sum())
        if c.sum() < 3 or px512 < min_px512:
            dropped.append(px512)
        else:
            kept.append(c)
    return kept, dropped


def comp_de(gen: Path, canvas: Path, comp512: np.ndarray) -> float:
    """p90 CIELAB distance inside ONE component."""
    from skimage.color import rgb2lab

    def lab(q: Path) -> np.ndarray:
        return rgb2lab(np.array(Image.open(q).convert("RGB").resize((512, 512), Image.LANCZOS)) / 255.0)

    return float(np.percentile(np.linalg.norm(lab(gen) - lab(canvas), axis=2)[comp512], 90))


def resize_mask(m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to (h, w): MAX-POOL when shrinking (never lose a small component), NEAREST when
    enlarging (nothing can be lost)."""
    import torch
    from src.utils.mask_utils import downsample_mask_maxpool
    h, w = shape
    if h <= m.shape[0] and w <= m.shape[1]:
        t = torch.from_numpy(m.astype(np.float32))[None, None]
        return torch.nn.functional.adaptive_max_pool2d(t, (h, w))[0, 0].numpy() > 0.5
    return np.array(Image.fromarray((m.astype(np.uint8) * 255)).resize((w, h), Image.NEAREST)) > 127


def to512(m: np.ndarray) -> np.ndarray:
    """Native mask -> 512 by MAX-POOL (never NEAREST: subsampling drops pixels of small components, which made
    the size floor reject footprints that are in fact above it)."""
    import torch
    from src.utils.mask_utils import downsample_mask_maxpool
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return downsample_mask_maxpool(t, 512)[0, 0].numpy() > 0.5


def generate_checkpoint(cases: list[dict], name: str, ckpt: Path, a) -> tuple[dict, dict]:
    """One pass per disjoint component, each painting into the previous pass's output.

    Per component: try --tries-per-position seeds; a component counts as painted when its own p90 dE reaches
    --min-comp-de. Components that never paint are dropped from the footprint, so the refined mask (computed once
    at the end, on the union of the painted components) never marks untouched pixels. If NO component paints, the
    whole footprint is moved to a new position and the sample starts over.
    """
    tag = name.replace(" ", "").replace("@", "_at_")
    gc.EXP = OUT / tag
    prep = OUT / tag / "prep"
    print(f"\n=== {name}  ({ckpt})", flush=True)
    pipeline, ip_adapter, t2i_adapter = gc.load_models(ckpt)
    ids = {i: f"{c['src']}-{c['ref']}" for i, c in enumerate(cases)}
    defect_map = {v: "small holes" for v in ids.values()}
    local = [dict(c) for c in cases]
    picks, stats = {}, {}

    for i, c in enumerate(local):
        canvas0 = gc.resolve_canvas_image(c["canvas"])
        history: list[dict] = []
        for pos in range(a.max_positions):
            c["mask_path"] = place_one(c, a.seed + i + 7919 * pos, pos, OUT / tag / "masks", a.mask_source)
            comps, dropped = components_of(c["mask_path"], a.min_comp_px)
            if dropped:
                print(f"  [{name}] {ids[i]} pos {pos}: dropped {len(dropped)} component(s) below "
                      f"{a.min_comp_px} px at 512 {dropped}", flush=True)
            if not comps:
                print(f"  [{name}] {ids[i]} pos {pos}: every component too small -> NEW POSITION", flush=True)
                continue
            work = OUT / tag / f"case{i:03d}" / f"pos{pos}"
            work.mkdir(parents=True, exist_ok=True)
            # what gets painted in one go: each component separately (multi) or the whole footprint (single)
            passes = comps if a.edit_mode in ("multi", "latent") else [np.any(np.stack(comps), axis=0)]
            cur = canvas0
            ctx = None        # latent-carry state (edit-mode latent): previous pass's latent + pixels
            for k, region in enumerate(passes):
                rmask = work / f"pass{k}.png"
                Image.fromarray((region * 255).astype(np.uint8)).save(rmask)
                r512 = to512(region)
                best = None
                for t in range(a.tries_per_position):
                    gc.generate_one(
                        pipeline, ip_adapter, t2i_adapter,
                        canvas_id=c["canvas"], ref_id=ids[i], difficulty="easy", defect_map=defect_map,
                        noise_strength=a.noise_strength, num_steps=a.num_steps,
                        guidance_scale=a.guidance_scale, band_mode=a.band_mode,
                        seed=a.seed + i + 100 * k + 1000 * t + 10000 * pos, device="cuda", layout="A",
                        ref_img_override=c["img"], ref_mask_override=c["mask"],
                        placed_mask_override=rmask, save_raw=True,
                        canvas_override=None if a.edit_mode == "latent" else cur,
                        context=ctx if a.edit_mode == "latent" else None,
                    )
                    state = gc.LAST_STATE     # this try's state; becomes the next pass's context if accepted
                    out = work / f"pass{k}_try{t}.png"
                    shutil.copy(gc.EXP / "anomaly" / "imgs" / "easy" / f"{ids[i]}.png", out)
                    de = comp_de(out, canvas0, r512)
                    history.append({"pos": pos, "comp": k, "try": t + 1, "px": int(region.sum()), "de": de,
                                    "ok": de >= a.min_comp_de, "gen": str(out), "mask": str(rmask)})
                    print(f"  [{name}] {ids[i]} pos {pos} pass {k} ({int(region.sum())} px) try {t + 1}: "
                          f"dE {de:5.1f}  {'painted' if de >= a.min_comp_de else 'blank'}", flush=True)
                    if best is None or de > best[1]:
                        best = (out, de, state)
                    if de >= a.min_comp_de:
                        break
                if best[1] >= a.min_comp_de:
                    cur = best[0]
                    ctx = best[2]
            # measure EVERY original component on the final image, whichever mode produced it
            kept, comp_stats = [], []
            for k, comp in enumerate(comps):
                de = comp_de(Path(cur), canvas0, to512(comp))
                ok = de >= a.min_comp_de
                comp_stats.append({"comp": k, "px": int(comp.sum()), "de": de, "ok": bool(ok)})
                if ok:
                    kept.append(comp)
            if kept:
                keep_mask = np.any(np.stack(kept), axis=0)
                kept_path = work / "footprint_kept.png"      # per-component cleanup: only painted blobs
                Image.fromarray((keep_mask * 255).astype(np.uint8)).save(kept_path)
                c["mask_path"] = kept_path
                write_prep_case(c, i, prep)
                gen_dir, ref_dir = work / "final_gen", work / "final_refined"
                gen_dir.mkdir(exist_ok=True)
                shutil.copy(cur, gen_dir / f"{i:03d}.png")
                refine_dir(gen_dir, prep, ref_dir)
                rt = roundtrip_mask(kept_path)
                rm = np.array(Image.open(ref_dir / f"{i:03d}.png").convert("L")) > 127
                fill = float(rm.sum() / max(rt.sum(), 1))
                picks[i] = {"gen": str(cur), "refined": str(ref_dir / f"{i:03d}.png"),
                            "mask": str(kept_path), "pos": pos}
                stats[str(i)] = {"positions": pos + 1, "passes": len(comps), "kept": len(kept),
                                 "too_small": dropped, "dropped": len(comps) - len(kept), "fill": fill,
                                 "de": max(x["de"] for x in comp_stats), "ok": True,
                                 "attempts": len(history), "components": comp_stats,
                                 "history": history, "mask_path": str(kept_path)}
                print(f"  [{name}] {ids[i]} [{a.edit_mode}]: kept {len(kept)}/{len(comps)} components at "
                      f"position {pos}, "
                      f"fill {fill:.2f}", flush=True)
                break
            print(f"  [{name}] {ids[i]}: nothing painted at position {pos} -> NEW POSITION", flush=True)
        else:
            # no position produced a usable sample: either nothing painted, or every component was too small
            too_small_only = not history
            picks[i] = {"gen": str(canvas0), "refined": None, "mask": str(c["mask_path"]),
                        "pos": a.max_positions - 1}
            stats[str(i)] = {"positions": a.max_positions, "passes": 0, "kept": 0, "dropped": 0,
                             "too_small": [], "fill": 1.0, "de": 0.0, "ok": False,
                             "unusable": "footprint below the paintable size" if too_small_only else "no-op",
                             "attempts": len(history), "components": [], "history": history,
                             "mask_path": str(c["mask_path"])}
            print(f"  [{name}] {ids[i]}: UNUSABLE — "
                  f"{'every component below the size floor' if too_small_only else 'nothing painted anywhere'}",
                  flush=True)
    del pipeline, ip_adapter, t2i_adapter
    import torch
    torch.cuda.empty_cache()
    return picks, stats


def zoom(img: Image.Image, overlays: list[tuple[np.ndarray, tuple]], box: int = 128, out: int = 320) -> Image.Image:
    ms = [m if m.shape == (512, 512) else to512(m) for m, _ in overlays]
    ys, xs = np.nonzero(ms[-1])
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (256, 256)
    x0, y0 = int(np.clip(cx - box // 2, 0, 512 - box)), int(np.clip(cy - box // 2, 0, 512 - box))
    crop = img.resize((512, 512), Image.LANCZOS).crop((x0, y0, x0 + box, y0 + box)).resize((out, out), Image.NEAREST)
    a = np.array(crop).copy()
    for m, colour in zip(ms, [c for _, c in overlays]):
        sub = m[y0:y0 + box, x0:x0 + box]
        edge = (sub & ~np.pad(sub, 1)[2:, 1:-1]) | (sub & ~np.pad(sub, 1)[:-2, 1:-1]) \
            | (sub & ~np.pad(sub, 1)[1:-1, 2:]) | (sub & ~np.pad(sub, 1)[1:-1, :-2])
        e = resize_mask(edge, (out, out))          # max-pool when shrinking - never NEAREST down
        a[e] = (0.5 * np.array(colour) + 0.5 * a[e]).astype(np.uint8)
    return Image.fromarray(a)


def sheet(cases: list[dict], picks: dict, stats: dict, path: Path) -> None:
    cell, pad, head = 320, 8, 96
    cols = ["donor reference = conditioning (cyan = its mask)", "real cashew hole that donates the footprint"]
    for name in CKPTS:
        cols += [f"{name}: canvas + placed mask", f"{name}: generated", f"{name}: latent BLUE + refined MAGENTA"]
    sh = Image.new("RGB", (pad + len(cols) * (cell + pad), head + len(cases) * (cell + 58)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    dr.text((pad, 8), "Cross-object transfer: REAL cashew hole masks as the footprint (placement augmentation), "
                      "donor anomaly as the conditioning", fill=(255, 255, 255))
    dr.text((pad, 26), "BLUE = latent (roundtripped) mask, what the UNet can paint | MAGENTA = refined mask | "
                       "no-op => magenta fills blue", fill=(150, 190, 255))
    dr.text((pad, 44), "rule: refined/latent > gate => nothing happened; re-seed once, then move the mask to a "
                       "new position", fill=(170, 170, 170))
    for j, cname in enumerate(cols):
        dr.text((pad + j * (cell + pad), 78), cname, fill=(210, 210, 210))
    for i, c in enumerate(cases):
        y = head + i * (cell + 58)
        ref_m = binary(c["mask"]) > 0.5
        tiles = [zoom(Image.open(c["img"]).convert("RGB"), [(ref_m, PL_COL)], box=384)]
        cm = (np.array(Image.open(c["cashew_mask"]).convert("L")) > 0)
        tiles.append(zoom(Image.open(c["cashew_img"]).convert("RGB"), [(cm, PL_COL)], box=200))
        note = []
        for name in CKPTS:
            tag = name.replace(" ", "").replace("@", "_at_")
            s = stats[name][str(i)]
            placed = binary(Path(s["mask_path"])) > 0.5
            rt = roundtrip_mask(Path(s["mask_path"]))
            pk = picks[name][i]
            g = Image.open(pk["gen"]).convert("RGB")
            rm = binary(Path(pk["refined"])) > 0.5 if pk["refined"] else np.zeros((512, 512), bool)
            tiles += [zoom(Image.open(gc.resolve_canvas_image(c["canvas"])).convert("RGB"), [(placed, PL_COL)]),
                      zoom(g, [(placed, PL_COL)]), zoom(g, [(rt, RT_COL), (rm, RF_COL)])]
            note.append(f"{name}: {s['passes']} passes, kept {s['kept']}/{s['passes']} comps, "
                        f"{s['attempts']} gens / {s['positions']} pos, fill {s['fill']:.2f}"
                        + ("" if s["ok"] else "  FAILED"))
        for j, t in enumerate(tiles):
            sh.paste(t, (pad + j * (cell + pad), y))
        dr.text((pad, y + cell + 6), f"conditioning: {c['src']} {c['ref']} ({c['area']} px) | footprint: real "
                                     f"cashew {c['cashew_id']} ({c['cashew_area']} px) | canvas {c['canvas']}",
                fill=(200, 200, 200))
        dr.text((pad, y + cell + 24), " | ".join(note),
                fill=(255, 190, 190) if "FAILED" in " ".join(note) else (170, 220, 170))
    sh.save(path)
    print("sheet:", path)


def audit_sheet(cases: list[dict], stats: dict, name: str, path: Path) -> None:
    """Every attempt of every case: what was generated, and why it was rejected or accepted."""
    cell, pad, head = 300, 6, 74
    ncol = max(len(stats[str(i)]["history"]) for i in range(len(cases)))
    sh = Image.new("RGB", (pad + ncol * (cell + pad), head + len(cases) * (cell + 44)), (16, 16, 16))
    dr = ImageDraw.Draw(sh)
    dr.text((pad, 8), f"Attempt trail — {name}. Every generation, in order, including the rejected ones.",
            fill=(255, 255, 255))
    dr.text((pad, 26), "BLUE = latent mask, MAGENTA = refined mask. Rejected = refined fills the latent region "
                       "(nothing was generated).", fill=(150, 190, 255))
    dr.text((pad, 44), "a new position is sampled after 2 failed tries at the same spot", fill=(170, 170, 170))
    for i, c in enumerate(cases):
        y = head + i * (cell + 44)
        for k, h in enumerate(stats[str(i)]["history"]):
            comp = binary(Path(h["mask"])) > 0.5
            g = Image.open(h["gen"]).convert("RGB")
            tile = zoom(g, [(comp, PL_COL)])
            dr_t = ImageDraw.Draw(tile)
            dr_t.rectangle([0, 0, cell - 1, cell - 1],
                           outline=(60, 220, 60) if h["ok"] else (230, 60, 60), width=4)
            sh.paste(tile, (pad + k * (cell + pad), y))
            dr.text((pad + k * (cell + pad), y + cell + 4),
                    f"pos {h['pos']} comp {h['comp']} ({h['px']} px) try {h['try']} | dE {h['de']:.0f} | "
                    f"{'painted' if h['ok'] else 'blank'}",
                    fill=(170, 220, 170) if h["ok"] else (255, 170, 170))
        dr.text((pad, y + cell + 22), f"{c['src']} {c['ref']} on canvas {c['canvas']} | footprint cashew "
                                      f"{c['cashew_id']} ({c['cashew_area']} px)", fill=(200, 200, 200))
    sh.save(path)
    print("audit sheet:", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mix", default="fire_hood")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default results/hole_donor_probe)")
    ap.add_argument("--ckpts", default=None, help="comma-separated subset of checkpoints, e.g. 'core-50k @20k'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--noise-strength", type=float, default=0.7)
    ap.add_argument("--band-mode", type=int, default=2)
    ap.add_argument("--mask-source", choices=["cashew", "donor"], default="cashew",
                    help="cashew: place a real cashew hole mask (default); donor: place the donor mask rescaled")
    ap.add_argument("--max-fill", type=float, default=0.5,
                    help="accept only if refined/latent mask <= this (higher means nothing was generated)")
    ap.add_argument("--min-de", type=float, default=0.0, help="optional extra gate on p90 dE (0 = off)")
    ap.add_argument("--min-comp-de", type=float, default=15.0,
                    help="a component counts as painted when its own p90 dE reaches this")
    ap.add_argument("--min-comp-px", type=int, default=24,
                    help="footprint components smaller than this many pixels AT 512 are excluded from the mask "
                         "(one latent cell is 8x8=64 px at 512; below ~24 px the generator rarely paints them)")
    ap.add_argument("--edit-mode", choices=["multi", "single", "latent"], default="multi",
                    help="multi: one pass per disjoint component, each RE-ENCODING the previous output (background "
                         "compressed once per pass). latent: one pass per component with latent carry - each pass "
                         "starts from and blends against the previous pass's latent, background decoded once. "
                         "single: one pass over the whole footprint.")
    ap.add_argument("--tries-per-position", type=int, default=2)
    ap.add_argument("--max-positions", type=int, default=4)
    a = ap.parse_args()

    global OUT, CKPTS
    if a.out:
        OUT = a.out
    if a.ckpts:
        want = [x.strip() for x in a.ckpts.split(",")]
        CKPTS = {k: v for k, v in CKPTS.items() if k in want}
        if not CKPTS:
            raise SystemExit(f"--ckpts matched nothing; available: {list(CKPTS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    gc.setup_experiment("ResNet")
    cases = build_cases(a.n, a.mix.split(","), a.seed)
    for c in cases:
        print(f"  cond {c['src']:10s} {c['ref']:>6} ({c['area']:6d} px) | footprint cashew {c['cashew_id']} "
              f"({c['cashew_area']:5d} px) | canvas {c['canvas']}")

    picks, stats = {}, {}
    for name, ckpt in CKPTS.items():
        picks[name], stats[name] = generate_checkpoint(cases, name, ckpt, a)
    sheet(cases, picks, stats, OUT / "hole_donor_probe.png")
    for name in CKPTS:
        audit_sheet(cases, stats[name], name,
                    OUT / f"attempts_{name.replace(' ', '').replace('@', '_at_')}.png")
    json.dump({"cases": [{k: str(v) for k, v in c.items()} for c in cases], "stats": stats},
              open(OUT / "cases.json", "w"), indent=1)

    print("\n| case | footprint | " + " | ".join(f"{n}: comps kept | gens | pos" for n in CKPTS) + " |")
    for i, c in enumerate(cases):
        row = []
        for n in CKPTS:
            s = stats[n][str(i)]
            row += [f"{s['kept']}/{s['passes']}{'' if s['ok'] else ' (failed)'}", str(s["attempts"]),
                    str(s["positions"])]
        print(f"| {c['src']}-{c['ref']} | cashew {c['cashew_id']} | " + " | ".join(row) + " |")
    for n in CKPTS:
        st = list(stats[n].values())
        comps = sum(x["passes"] for x in st)
        kept = sum(x["kept"] for x in st)
        small = sum(len(x.get("too_small", [])) for x in st)
        print(f"{n}: samples accepted {sum(x['ok'] for x in st)}/{len(st)}; components painted {kept}/{comps} "
              f"(+{small} excluded as too small); generations {sum(x['attempts'] for x in st)}; "
              f"mean positions {np.mean([x['positions'] for x in st]):.1f}")


if __name__ == "__main__":
    main()
