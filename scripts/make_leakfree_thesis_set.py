#!/usr/bin/env python3
"""Fold the leak fix into the thesis-split data so a set is simply 94 correct synthetics (user 2026-10-04).

Until now synthetic 067 existed twice: the original on canvas 472 (a TEST normal; never used since 2026-10-02) inside
the set, and its clean re-synthesis (canvas 293) in <set>/leak_fix/067/, which the runner's leak guard swapped in.
This writes a layout without that indirection; the training input is byte-identical (same image, mask and canvas files).

  results/cashew_100_leakfree/prep/NNN        copy of results/cashew_100_rp/prep with 067 = the re-synthesis's folder
  results/thesis_set_clean20k_leakfree/       generated/cfg7_vis/NNN.png + refined_masks_f025/NNN.png, 067 = the re-synthesis
  results/noise_strength_test/ns_*            in place: 067 image / mask replaced by the set's leak_fix version, leak_fix removed

  python scripts/make_leakfree_thesis_set.py            (idempotent)
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_PREP, OLD_SET = ROOT / "results/cashew_100_rp/prep", ROOT / "results/thesis_set_clean20k_B"
PREP, SET = ROOT / "results/cashew_100_leakfree/prep", ROOT / "results/thesis_set_clean20k_leakfree"
MASKS = "refined_masks_f025"
PREP_FILES = ("canvas.png", "placed_mask.png", "ref_image.png", "ref_mask.png", "meta.json")


def fold(set_dir: Path, fx: Path) -> None:
    shutil.copyfile(fx / "image.png", set_dir / "generated/cfg7_vis" / f"{fx.name}.png")
    shutil.copyfile(fx / f"{MASKS}.png", set_dir / MASKS / f"{fx.name}.png")


def main() -> None:
    fixes = sorted(p for p in (OLD_SET / "leak_fix").iterdir() if p.is_dir())
    assert fixes, "no leak_fix folder in the source set"
    if not PREP.exists():
        shutil.copytree(OLD_PREP, PREP)
    for sub in ("generated/cfg7_vis", MASKS):
        (SET / sub).mkdir(parents=True, exist_ok=True)
        for f in (OLD_SET / sub).glob("*.png"):
            shutil.copyfile(f, SET / sub / f.name)
    for fx in fixes:
        for f in PREP_FILES:
            shutil.copyfile(fx / f, PREP / fx.name / f)
        fold(SET, fx)
        m = json.loads((PREP / fx.name / "meta.json").read_text())
        print(f"{fx.name}: canvas {m['replaces_canvas']} -> {m['normal']} folded into {SET.name} and {PREP.parent.name}/prep")
    (SET / "SOURCE.txt").write_text((OLD_SET / "SOURCE.txt").read_text() + f"leak-free layout: {[f.name for f in fixes]} = the re-synthesis on a "
                                    f"clean canvas (scripts/make_leakfree_thesis_set.py); prep = {PREP.relative_to(ROOT).as_posix()}; masks = {MASKS}\n")
    n = 0
    for d in sorted((ROOT / "results/noise_strength_test").glob("ns_*")):
        lf = d / "leak_fix"
        if lf.is_dir():
            for fx in sorted(p for p in lf.iterdir() if p.is_dir()):
                fold(d, fx)
            shutil.rmtree(lf); n += 1
    print(f"noise-strength sets converted in place: {n}")
    print(f"sets now hold {len(list((SET / 'generated/cfg7_vis').glob('*.png')))} images, no leak_fix folder")


if __name__ == "__main__":
    main()
