#!/usr/bin/env python3
"""Final supplementary type-inclusive table with the 2/8 software-environment pattern matched between conditions
(2026-10-24). Output: results/open_set_v2/typeincl_2of8/TYPEINCL_2of8.md (nothing else is written).

Open-set (A):         seeds 42, 123 -> repro_a262/open/   (the ORIGINAL runs reproduced bit-exactly: anomalib 2.6.2,
                                                           MIG 1g.23gb; verified value by value vs runs/)
                      other 8 seeds -> score_dist_c1/      (reruns bit-identical to the original runs: anomalib 2.2.0)
Type-inclusive (I):   seeds 42, 123 -> repro_a262/ti/      (new, same environment as their open-set runs)
                      other 8 seeds -> typeincl/runs/      (2.2.0, whole B200, 2026-10-23)
"Previous" version:   A = score_dist_c1 for all 10 seeds, I = typeincl/runs for all 10 seeds (TYPEINCL_TABLES.md).
All: reduced test set (3 reserved anomalies dropped, same 50 normals), image score = adaptive top-k mean
('standard_paper', = img_aurocs of the main tables), best-0x epoch re-selected on the reduced set, 1-4x = mean of 1x-4x
at that epoch. Mean +- SE over seeds; differences paired on seed and fold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_steps as S  # noqa: E402
import open_set_typeincl as TI  # noqa: E402

O = S.OUT; OUT = O / "typeincl_2of8"; SEEDS = S.SEEDS10; ORIG_ENV = (42, 123)
ARMS = {"diffusion_in": "target-object", "diffusion_cross": "cross-object"}


def src(version: str, cond: str, arm: str, s: int, k: str) -> Path:
    if cond == "A":
        if version == "new" and s in ORIG_ENV:
            return O / "repro_a262" / "open" / f"{arm}_s1" / f"seed_{s}" / f"fold_{k}" / "image_scores.json"
        return O / "score_dist_c1" / f"{arm}_s1" / f"seed_{s}" / f"fold_{k}" / "image_scores.json"
    if version == "new" and s in ORIG_ENV:
        return O / "repro_a262" / "ti" / f"{arm}_ti" / f"seed_{s}" / f"fold_{k}" / "image_scores.json"
    return O / "typeincl" / "runs" / f"{arm}_ti" / f"seed_{s}" / f"fold_{k}" / "image_scores.json"


def collect(version: str) -> dict:
    res = TI.reserved(); R = {}
    for arm in ARMS:
        for s in SEEDS:
            for k in S.CLASSES:
                for c in "AI":
                    f = src(version, c, arm, s, k)
                    assert f.exists(), f"missing {f}"
                    R[(arm, s, k, c)] = TI.pick(TI.scores_auroc(f, set(res[k])))
    return R


se = lambda z: z.std(ddof=1) / np.sqrt(len(z))
fm = lambda z: f"{z.mean():.3f} ± {se(z):.3f}"
fd = lambda z: f"{z.mean():+.3f} ± {se(z):.3f}"


def table(R: dict, arm: str, idx: int) -> tuple[list[str], dict]:
    rows = ["| held-out type | open-set | type-inclusive | type-incl − open-set |", "|---|---|---|---|"]; summ = {}
    for k in S.CLASSES:
        a = np.array([R[(arm, s, k, "A")][idx] for s in SEEDS]); b = np.array([R[(arm, s, k, "I")][idx] for s in SEEDS])
        rows.append(f"| {k} | {fm(a)} | {fm(b)} | {fd(b - a)} |"); summ[k] = (a.mean(), b.mean(), (b - a).mean(), se(b - a))
    a = np.array([np.mean([R[(arm, s, k, "A")][idx] for k in S.CLASSES]) for s in SEEDS])
    b = np.array([np.mean([R[(arm, s, k, "I")][idx] for k in S.CLASSES]) for s in SEEDS])
    rows.append(f"| **macro (6 folds)** | {fm(a)} | {fm(b)} | {fd(b - a)} |"); summ["macro"] = (a.mean(), b.mean(), (b - a).mean(), se(b - a))
    return rows, summ


def main() -> None:
    new, old = collect("new"), collect("old")
    out = ["# Supplementary: type-inclusive vs open-set, 2/8 environment pattern matched (configuration 1)", "",
           __doc__.strip(), "", f"Reserved per type: {json.dumps(TI.reserved())}", ""]
    for arm, lab in ARMS.items():
        for idx, lvl in ((0, "0×"), (1, "mean 1–4×")):
            rows, sn = table(new, arm, idx); _, so = table(old, arm, idx)
            out += [f"## {lab}, {lvl} AUROC (new, 2/8-matched)", "", *rows, "",
                    f"Change vs previous version (all-2.2.0 open-set reruns): paired difference type-incl − open-set", "",
                    "| held-out type | previous diff | new diff | change |", "|---|---|---|---|"]
            for key in [*S.CLASSES, "macro"]:
                out.append(f"| {key} | {so[key][2]:+.3f} ± {so[key][3]:.3f} | {sn[key][2]:+.3f} ± {sn[key][3]:.3f} | {sn[key][2] - so[key][2]:+.3f} |")
            out.append("")
        for k in ("colour_same", "scratches"):
            out += [f"**{lab}, {k}, per seed (new): open-set → type-inclusive** (0×; 1–4×; selected epochs)", ""]
            for s in SEEDS:
                a, b = new[(arm, s, k, "A")], new[(arm, s, k, "I")]
                tag = " (orig. env 2.6.2/MIG)" if s in ORIG_ENV else ""
                out.append(f"- seed {s}{tag}: 0× {a[0]:.3f} → {b[0]:.3f}; 1–4× {a[1]:.3f} → {b[1]:.3f}; epochs {a[2]} → {b[2]}")
            out.append("")
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "TYPEINCL_2of8.md").write_text("\n".join(out) + "\n"); print("\n".join(out))


if __name__ == "__main__":
    main()
