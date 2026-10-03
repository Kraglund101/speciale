#!/usr/bin/env python3
"""Noise-strength sweep (user 2026-10-03): the same reference, canvas, placed mask and seed rendered at noise strength
0.1 ... 1.0 (everything else = the pipeline's settings: clean-20k, visual CFG 7, 50 steps, no caption).
Rows = anomalies (first stage-1 step of the chosen types, seed 42); columns = real reference (cyan = GT mask), clean
canvas (cyan = placed mask), then the ten renderings. Target-object and cross-object versions on separate sheets.

  python scripts/noise_strength_sweep.py [--types breakage burnt colour_diff] [--arm in|cross]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import open_set_steps as S  # noqa: E402
from sheet_utils import mask_resize  # noqa: E402

P = S.P
STRENGTHS = [round(0.1 * k, 1) for k in range(1, 11)]
TILE, LABW, SEED = 190, 110, 42


def box(M: np.ndarray, lo: int = 150) -> tuple[int, int, int]:
    ys, xs = np.nonzero(M); H, W = M.shape
    side = int(min(max(lo, 2.2 * max(np.ptp(ys), np.ptp(xs)) + 40), min(H, W)))
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    return int(np.clip(cy - side // 2, 0, H - side)), int(np.clip(cx - side // 2, 0, W - side)), side


def crop(img: np.ndarray, b: tuple[int, int, int], outline: np.ndarray | None = None) -> Image.Image:
    y0, x0, s = b; t = np.array(Image.fromarray(img[y0:y0 + s, x0:x0 + s]).resize((TILE, TILE), Image.LANCZOS)).astype(np.float32)
    if outline is not None:
        m = mask_resize(outline[y0:y0 + s, x0:x0 + s], TILE); ring = nd.binary_dilation(m, iterations=1) & ~m
        t[ring] = 0.45 * t[ring] + 0.55 * np.array([0, 255, 255], np.float32)
    return Image.fromarray(t.astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--types", nargs="+", default=["breakage", "burnt", "colour_diff"])
    ap.add_argument("--arm", choices=["in", "cross"], default="in"); a = ap.parse_args()
    man = S.manifest(SEED); refs = S.load_refs(); byid = {r["id"]: r for k in refs for r in refs[k]}
    P.gc.setup_experiment("ResNet"); pipe, ip, t2i = P.gc.load_models(P.h.CKPTS[P.CKPT]); P.gc.EXP = Path(tempfile.mkdtemp(prefix="nsweep_"))
    font = ImageFont.truetype("arial.ttf", 14); small = ImageFont.truetype("arial.ttf", 12)
    sheet = Image.new("RGB", (LABW + (2 + len(STRENGTHS)) * TILE, 30 + len(a.types) * TILE), (25, 25, 25)); dr = ImageDraw.Draw(sheet)
    for c, h in enumerate(["real reference", "clean canvas"] + [f"noise {x:.1f}" + (" (pipeline)" if x == 0.7 else "") for x in STRENGTHS]):
        dr.text((LABW + c * TILE + 6, 8), h, fill=(230, 230, 230), font=small)
    out_dir = S.OUT / "sheets" / "noise_sweep"; out_dir.mkdir(parents=True, exist_ok=True)
    for r, cls in enumerate(a.types):
        st = man["sets"]["s2"][cls][0]; prep = S.pdir(SEED, "s2", cls, st["i"]); ref = byid[st["ref"]]; dn = man["pairing"][cls][st["ref"]]
        seed = json.loads((prep / "meta.json").read_text())["seed"]
        img_ref, mask_ref = (ref["ind_img"], ref["ind_mask"]) if a.arm == "in" else (dn["cr_img"], dn["cr_mask"])
        M = mask_resize(np.array(Image.open(prep / "placed_mask.png").convert("L")) > 127, 512); b = box(M); y = 30 + r * TILE
        dr.text((6, y + 8), cls, fill=(240, 240, 240), font=font)
        dr.text((6, y + 28), f"ref {st['ref']}" if a.arm == "in" else f"{dn['donor']}", fill=(170, 170, 170), font=small)
        rim = np.array(Image.open(img_ref).convert("RGB")); rm = np.array(Image.open(mask_ref).convert("L")) > 0
        if rm.shape != rim.shape[:2]:
            rm = mask_resize(rm, rim.shape[:2])
        sheet.paste(crop(rim, box(rm, lo=int(0.25 * min(rm.shape))), rm), (LABW, y))
        can = np.array(Image.open(P.NORMAL_DIR / f"{st['canvas']}.JPG").convert("RGB").resize((512, 512), Image.LANCZOS))
        sheet.paste(crop(can, b, M), (LABW + TILE, y))
        for c, ns in enumerate(STRENGTHS):
            dst = out_dir / f"{a.arm}_{cls}_{ns:.1f}.png"
            if not dst.exists():
                rid = f"ns-{a.arm}-{cls}-{int(ns * 10)}"
                P.gc.generate_one(pipe, ip, t2i, canvas_id=st["canvas"], ref_id=rid, difficulty="easy", defect_map={rid: "defect"}, seed=seed,
                                  device="cuda", layout="A", ref_img_override=Path(img_ref), ref_mask_override=Path(mask_ref),
                                  placed_mask_override=prep / "placed_mask.png", save_raw=True,
                                  canvas_override=P.NORMAL_DIR / f"{st['canvas']}.JPG", caption_override=" ",
                                  **{**P.GEN, "noise_strength": ns})
                shutil.copy(P.gc.EXP / "anomaly/imgs/easy" / f"{rid}.png", dst)
            sheet.paste(crop(np.array(Image.open(dst).convert("RGB")), b), (LABW + (2 + c) * TILE, y))
        print(f"{cls}: done", flush=True)
    out = S.OUT / "sheets" / f"noise_strength_sweep_{a.arm}.png"; sheet.save(out); print(out, sheet.size)


if __name__ == "__main__":
    main()
