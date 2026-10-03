#!/usr/bin/env python3
"""Contact sheets of cross-object donor CANDIDATES per cashew anomaly type (user 2026-10-03): for judging whether more
products can be added to the donor pools. One page per type; rows = the cashew defects themselves (reference), the
donors used now, and the candidates; per row: the whole image of the first example, then 6 randomly drawn defect
crops (seeded, not hand-picked) with the GT mask as a thin cyan outline. Nothing is added to any pool by this script.

  python scripts/donor_candidates_sheet.py   -> results/open_set_v2/sheets/donor_candidates/<type>.png
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import pregenerate_synthetic as P  # noqa: E402
from sheet_utils import mask_resize  # noqa: E402

REST = ROOT / "anomverse_extension/datasets/VisA_validation_dataset/restVisA/VisA_20220922"
D = ROOT / "anomverse_extension/datasets/full_training_dataset"
OUT = ROOT / "results/open_set_v2/sheets/donor_candidates"
TILE, N, LABW = 230, 6, 250
CANON = {"small holes": "holes", "corner or edge breakage": "breakage", "corner and edge breakage": "breakage", "middle breakage": "breakage",
         "small cracks": "breakage", "small scratches": "scratches", "burnt": "burnt", "same colour spot": "colour_same",
         "similar colour spot": "colour_same", "different colour spot": "colour_diff"}
# labels that are NOT in the pairing script's canonical map: shown as candidates, marked with *
EXTRA = {"color spot similar to the object": "colour_same", "different color spot": "colour_diff", "small chip around edge": "breakage",
         "chip around edge and corner": "breakage", "chunk of gum missing": "breakage", "corner missing": "breakage", "scratches": "scratches",
         "breakage down the middle": "breakage"}
VISA = ["fryum", "macaroni1", "macaroni2", "chewinggum", "candle"]
REALIAD = {"holes": [("vcpill", "pit")], "breakage": [("vcpill", "missing_parts")]}


def visa_rows(obj: str) -> dict[str, list[tuple[Path, Path, str]]]:
    out: dict[str, list] = {}
    for r in csv.reader(open(REST / obj / "image_anno.csv")):
        if len(r) < 2 or "Anomaly" not in r[0]:
            continue
        labs = [a.strip().lower() for a in r[1].split(",") if a.strip()]
        if len(labs) != 1:
            continue
        cls = CANON.get(labs[0]) or EXTRA.get(labs[0])
        stem = Path(r[0]).stem; img = REST / obj / "Data/Images/Anomaly" / f"{stem}.JPG"; msk = REST / obj / "Data/Masks/Anomaly" / f"{stem}.png"
        if cls and img.exists() and msk.exists():
            out.setdefault(cls, []).append((img, msk, labs[0] + ("" if labs[0] in CANON else "*")))
    return out


def tile(img_p: Path, msk_p: Path) -> Image.Image:
    im = Image.open(img_p).convert("RGB"); m = np.array(Image.open(msk_p).convert("L")) > 0
    if m.shape != (im.height, im.width):
        m = mask_resize(m, (im.height, im.width))
    if not m.any():
        return im.resize((TILE, TILE), Image.LANCZOS)
    ys, xs = np.nonzero(m); H, W = m.shape
    side = int(min(max(0.22 * min(H, W), 1.8 * max(np.ptp(ys), np.ptp(xs)) + 0.04 * min(H, W)), min(H, W)))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - side // 2, 0, H - side)); x0 = int(np.clip(cx - side // 2, 0, W - side))
    t = np.array(im.crop((x0, y0, x0 + side, y0 + side)).resize((TILE, TILE), Image.LANCZOS)).astype(np.float32)
    mm = mask_resize(m[y0:y0 + side, x0:x0 + side], TILE); ring = nd.binary_dilation(mm, iterations=1) & ~mm      # thin, outside the defect
    t[ring] = 0.4 * t[ring] + 0.6 * np.array([0, 255, 255], np.float32)
    return Image.fromarray(t.astype(np.uint8))


def whole(img_p: Path) -> Image.Image:
    im = Image.open(img_p).convert("RGB"); im.thumbnail((TILE, TILE), Image.LANCZOS)
    bg = Image.new("RGB", (TILE, TILE), (25, 25, 25)); bg.paste(im, ((TILE - im.width) // 2, (TILE - im.height) // 2)); return bg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); refs = P.load_refs(); pools = P.donor_pools(); font = ImageFont.truetype("arial.ttf", 15)
    small = ImageFont.truetype("arial.ttf", 12); visa = {o: visa_rows(o) for o in VISA}
    master = json.load(open(D / "master_training.json", encoding="utf-8")); rng = np.random.default_rng(7)
    for cls in P.CLASSES:
        rows = [("cashew (the defects to imitate)", [(Path(r["ind_img"]), Path(r["gt_mask"]), r["raw"]) for r in refs[cls]], "REFERENCE")]
        by = {}
        for x in pools[cls]:
            by.setdefault(x["donor"], []).append((Path(x["cr_img"]), Path(x["cr_mask"]), ""))
        rows += [(src, v, "USED NOW") for src, v in sorted(by.items())]
        rows += [(o, visa[o][cls], "CANDIDATE") for o in VISA if cls in visa[o] and not (o in by and all("*" in x[2] for x in visa[o][cls]))]
        for prod, dt in REALIAD.get(cls, []):
            ms = [m for m in master if m["product"] == prod and (m.get("defect_type") or "").lower() == dt and m.get("mask_path")]
            seen, items = set(), []
            for m in ms:                                              # one camera view per physical defect
                k = re.sub(r"_C\d_.*$", "", Path(m["image_path"]).stem)
                if k not in seen and (D / m["mask_path"]).exists():
                    seen.add(k); items.append((D / m["image_path"], D / m["mask_path"], dt))
            rows.append((f"{prod} (Real-IAD)", items, "CANDIDATE"))
        sheet = Image.new("RGB", (LABW + (N + 1) * TILE, 34 + len(rows) * TILE), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
        dr.text((8, 8), f"{cls}: donor candidates. Column 1 = whole image of the first example; then {N} random defect crops (cyan = GT mask). "
                        "* = label not in the pairing script's class map", fill=(230, 230, 230), font=small)
        for r, (name, items, status) in enumerate(rows):
            y = 34 + r * TILE; labs = sorted({x[2] for x in items if x[2]})
            col = {"REFERENCE": (0, 255, 255), "USED NOW": (140, 220, 140), "CANDIDATE": (255, 200, 90)}[status]
            dr.text((8, y + 8), name, fill=(240, 240, 240), font=font); dr.text((8, y + 30), f"{status}  |  n = {len(items)}", fill=col, font=small)
            for j, lab in enumerate(labs[:6]):
                dr.text((8, y + 50 + 16 * j), lab[:36], fill=(170, 170, 170), font=small)
            pick = [items[int(k)] for k in rng.permutation(len(items))[:N]]
            if pick:
                sheet.paste(whole(pick[0][0]), (LABW, y))
            for c, (ip, mp, _) in enumerate(pick):
                sheet.paste(tile(ip, mp), (LABW + (c + 1) * TILE, y))
        sheet.save(OUT / f"{P.CLASSES.index(cls) + 1}_{cls}.png"); print(OUT / f"{P.CLASSES.index(cls) + 1}_{cls}.png", sheet.size, [(n, len(i), s) for n, i, s in rows])


if __name__ == "__main__":
    main()
