#!/usr/bin/env python3
"""Check the T=17 rule on the REMAINING cashew anomaly classes (everything except holes, which calibrated it).

For each single-label cashew anomaly of results/masked_patch_knn/pairing_constrained.json (cls != holes):
  footprint = that cashew's own VisA mask, placed on a normal canvas (placement_utils.place_easy_mask: CLIP-procedure
              transforms, scale 0.8-1.2), pieces < 8 px dropped (hole_donor_probe.components_of)
  three one-shot single-edit generations on the SAME footprint + canvas + seed:
    in-dist  : conditioning = the cashew anomaly itself (image + its own mask), caption = its VisA label
    cross    : conditioning = its paired donor (pipe_fryum / fire_hood image + that image's own mask), same caption
    in_nocap / cross_nocap : the same two WITHOUT a caption (blank prompt " "; an empty string would silently fall
               back to "a photo of a {type} defect" in generate_anomagic_single) - caption sensitivity
    blank    : NO anomaly information (IP-Adapter zeroed, caption "a photo of a cashew", T2I off)
  label = refine_inprocess.refine(T=17, blobs < 16 px dropped); accepted = label non-empty.
Blank repaints must all come out empty; generations are judged by eye on the per-class sheets.

  python scripts/class_calibration.py --out results/class_calib
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import generate_cashew as gc  # noqa: E402
import hole_donor_probe as h  # noqa: E402
from refine_inprocess import refine  # noqa: E402
from src.utils.placement_utils import place_easy_mask  # noqa: E402

PAIRS = ROOT / "results/masked_patch_knn/pairing_constrained.json"
ARMS = ("in", "cross", "in_nocap", "cross_nocap", "blank")
CASHEW_MASKS = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/Data/Masks/Anomaly"
PIPE_MASKS = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/restVisA/VisA_20220922/pipe_fryum/Data/Masks/Anomaly"
DATA = ROOT / "anomverse_extension/datasets/full_training_dataset"


def donor_mask(row: dict, master: dict) -> Path:
    """The paired donor image's OWN mask (source of truth: the dataset files)."""
    if row["nn_src"] == "pipe_fryum":
        return PIPE_MASKS / f"{row['nn_id']}.png"
    rel = str(Path(row["nn_img"]).relative_to(DATA)).replace("\\", "/")
    return DATA / master[rel]["mask_path"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results/class_calib")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=17.0)
    ap.add_argument("--min-blob-px", type=int, default=16)
    ap.add_argument("--classes", default=None, help="comma list of classes to run (default: every class except holes)")
    ap.add_argument("--skip-nocap", action="store_true", help="skip the no-caption arms")
    a = ap.parse_args()
    gc.setup_experiment("ResNet")
    h.OUT = a.out
    a.out.mkdir(parents=True, exist_ok=True)
    want = set(a.classes.split(",")) if a.classes else None
    rows = [r for r in json.loads(PAIRS.read_text(encoding="utf-8")) if (r["cls"] in want if want else r["cls"] != "holes")]
    arms = [x for x in ARMS if not (a.skip_nocap and x.endswith("nocap"))]
    master = {e["image_path"].replace("\\", "/"): e for e in json.loads((DATA / "master_training.json").read_text(encoding="utf-8"))
              if e.get("dataset") == "realiad"}
    canvases = sorted(p.stem for p in gc.CANVAS_IMGS.glob("*.JPG"))
    random.Random(a.seed).shuffle(canvases)

    # ---- build cases: footprint placement + references -------------------------------------------------------
    cases = []
    for k, r in enumerate(rows):
        if "hard_imgs" in r["img"]:
            raise RuntimeError(f"{r['id']} is a hard (full-object) case - the rule does not apply; handle separately")
        canvas = canvases[k % len(canvases)]
        fg = h.binary(gc.CANVAS_IMGS.parent / "masks" / f"{canvas}.png") > 0.5
        m = (np.array(Image.open(CASHEW_MASKS / f"{r['id']}.png").convert("L")) > 0)
        if m.shape != fg.shape:
            m = h.resize_mask(m, fg.shape)
        placed = place_easy_mask(m.astype(np.float32), fg.astype(np.float32), scale_range=(0.6, 1.0), seed=a.seed + k)
        rec = {"k": k, "id": r["id"], "cls": r["cls"], "raw": r["raw"], "canvas": canvas, "donor": r["nn_src"], "donor_id": r["nn_id"]}
        if placed is None:
            rec["skipped"] = "placement failed (mask does not fit inside the canvas foreground)"
            cases.append(rec); continue
        work = a.out / r["cls"] / f"{r['id']}"; work.mkdir(parents=True, exist_ok=True)
        raw_fp = work / "footprint_raw.png"
        Image.fromarray((placed > 0.5).astype(np.uint8) * 255).save(raw_fp)
        comps, dropped = h.components_of(raw_fp, 8)
        if not comps:
            rec["skipped"] = "every piece below the 8 px floor"; cases.append(rec); continue
        fp = work / "footprint.png"
        Image.fromarray((np.any(np.stack(comps), 0) * 255).astype(np.uint8)).save(fp)
        rec.update(fp=str(fp), n_pieces=len(comps), work=str(work),
                   ind_img=r["img"], ind_mask=str(h.normalise_mask(CASHEW_MASKS / f"{r['id']}.png", f"cashew_{r['id']}")),
                   cr_img=r["nn_img"], cr_mask=str(h.normalise_mask(donor_mask(r, master), f"{r['nn_src']}_{Path(r['nn_img']).stem}")))
        c = {"canvas": canvas, "mask_path": fp, "img": rec["ind_img"], "mask": rec["ind_mask"]}
        h.write_prep_case(c, k, a.out / "prep")
        cases.append(rec)
    json.dump(cases, open(a.out / "cases.json", "w"), indent=1)
    live = [c for c in cases if "skipped" not in c]
    print(f"{len(live)}/{len(cases)} cases placed; skipped: {[(c['id'], c['skipped']) for c in cases if 'skipped' in c]}", flush=True)

    # ---- generations: in-dist, cross, then blank (anomaly info removed) --------------------------------------
    pipeline, ip_adapter, t2i = gc.load_models(h.CKPTS["core-50k @20k"])
    gc.EXP = Path(tempfile.mkdtemp(prefix="classcal_"))

    def gen(c: dict, arm: str, img: str, mask: str, adapter) -> Path:
        ref_id = f"{arm}-{c['id']}"
        gc.generate_one(pipeline, ip_adapter, adapter, canvas_id=c["canvas"], ref_id=ref_id, difficulty="easy",
                        defect_map={ref_id: c["raw"]}, noise_strength=0.7, num_steps=50, guidance_scale=7.0,
                        band_mode=2, seed=a.seed + c["k"], device="cuda", layout="A", ref_img_override=Path(img),
                        ref_mask_override=Path(mask), placed_mask_override=Path(c["fp"]), save_raw=True,
                        canvas_override=gc.resolve_canvas_image(c["canvas"]))
        out = Path(c["work"]) / f"{arm}.png"
        shutil.copy(gc.EXP / "anomaly/imgs/easy" / f"{ref_id}.png", out)
        return out

    for c in live:
        c["gen_in"] = str(gen(c, "in", c["ind_img"], c["ind_mask"], t2i))
        c["gen_cross"] = str(gen(c, "cross", c["cr_img"], c["cr_mask"], t2i))
        print(f"  {c['cls']:12s} {c['id']}: in-dist + cross done", flush=True)
    real_encode, real_single = ip_adapter.encode_image, gc.generate_anomagic_single

    def no_caption(*x, **y):
        y["caption"] = " "                     # blank prompt: truthy, so no fallback to the defect-type caption
        return real_single(*x, **y)
    gc.generate_anomagic_single = no_caption
    for c in ([] if a.skip_nocap else live):
        c["gen_in_nocap"] = str(gen(c, "in_nocap", c["ind_img"], c["ind_mask"], t2i))
        c["gen_cross_nocap"] = str(gen(c, "cross_nocap", c["cr_img"], c["cr_mask"], t2i))
        print(f"  {c['cls']:12s} {c['id']}: no-caption arms done", flush=True)
    gc.generate_anomagic_single = real_single
    ip_adapter.encode_image = lambda *x, **y: torch.zeros_like(real_encode(*x, **y))       # no visual anomaly info

    def neutral(*x, **y):
        y["caption"] = "a photo of a cashew"                                                # no textual anomaly info
        return real_single(*x, **y)
    gc.generate_anomagic_single = neutral
    for c in live:
        c["gen_blank"] = str(gen(c, "blank", c["ind_img"], c["ind_mask"], None))
    ip_adapter.encode_image, gc.generate_anomagic_single = real_encode, real_single
    del pipeline, ip_adapter, t2i
    torch.cuda.empty_cache()

    # ---- the T=17 rule on all three --------------------------------------------------------------------------
    from compute_refined_masks import _load_dinov3_cnx
    dino = _load_dinov3_cnx("cuda")
    for c in live:
        prep = a.out / "prep" / f"{c['k']:03d}"
        for arm in arms:
            lab = refine(dino, Path(c[f"gen_{arm}"]), prep, a.threshold, min_blob_px=a.min_blob_px)
            p = Path(c["work"]) / f"{arm}_label.png"
            Image.fromarray((lab * 255).astype(np.uint8)).save(p)
            c[f"label_{arm}"] = str(p); c[f"label_px_{arm}"] = int(lab.sum())
    json.dump(cases, open(a.out / "cases.json", "w"), indent=1)

    print(f"\n{'class':12s}{'n':>4s}" + "".join(f"{arm:>14s}" for arm in arms) + "   (accepted = label non-empty)")
    for cls in sorted({c["cls"] for c in live}) + ["ALL"]:
        g = [c for c in live if cls in ("ALL", c["cls"])]
        counts = []
        for arm in arms:
            n_acc = sum(c["label_px_" + arm] > 0 for c in g)
            counts.append(f"{n_acc}/{len(g)}")
        print(f"{cls:12s}{len(g):>4d}" + "".join(f"{x:>14s}" for x in counts))


if __name__ == "__main__":
    main()
