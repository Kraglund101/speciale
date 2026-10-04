#!/usr/bin/env python3
"""Reference-group table for the report (Appendix Table tab:reference_groups), recomputed from the code's own sources:
real cashew defects per type, cross-object donor sources and pool sizes, and how many distinct donors the seeds
actually use. Donors are counted as PHYSICAL defects (Real-IAD fire_hood photographs one defect from up to 5 cameras;
the pairing draws one view per defect). Re-run whenever the design changes (seeds, budget):

  python scripts/open_set_reference_table.py            -> results/open_set_v2/reference_groups_table.md (+ LaTeX)
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import open_set_steps as S  # noqa: E402

P = S.P
SRC = {"fire_hood": "FH", "pipe_fryum": "PF", "macaroni1": "M1", "macaroni2": "M2", "fryum": "FR"}
LABEL = {"holes": "Holes", "breakage": "Breakage", "burnt": "Burnt", "colour_diff": "Colour diff.", "colour_same": "Colour same",
         "scratches": "Scratches"}


def phys(d: dict) -> str:
    return d["donor"] + ":" + re.sub(r"_C\d_.*$", "", Path(d["cr_img"]).stem)


def main() -> None:
    refs = S.load_refs(); pools = P.donor_pools(); plan = S.anomaly_plan(refs); seeds = S.SEEDS10; ext = S.ext_pools()
    pair = {s: P.pairing(s, refs) for s in seeds}; B = S.BUDGET; canv = P.normal_pools()[2]
    man = {s: S.build_manifest(s, refs, canv) for s in seeds}          # schedules only (no placement); gives the stage-5 draws
    src_of = lambda keys: ", ".join(SRC.get(k, k) for k, _ in collections.Counter(x.split(":")[0] for x in keys).most_common())
    rows = []
    for t in P.CLASSES:
        pool = {phys(x) for x in pools[t]}; s5 = {s: {phys(d) for d in man[s]["s5_donor"][t]} for s in seeds}
        rows.append({"type": t, "real": len(refs[t]), "src": src_of(pool), "pool": len(pool),
                     "d4": len({phys(pair[s][t][c]) for s in seeds for c in pair[s][t]}),
                     "d13": len({phys(pair[s][t][c]) for s in seeds for c in plan[str(s)][t]}),
                     "r13": len({c for s in seeds for c in plan[str(s)][t]}),
                     "src5": src_of(ext[t]), "pool5": len(ext[t]), "s5_seed": min(len(v) for v in s5.values()),
                     "d5": len(set().union(*s5.values()))})
    tot = {k: sum(r[k] for r in rows) for k in ("real", "pool", "d4", "d13", "r13", "pool5", "s5_seed", "d5")}
    n = len(seeds)
    md = [f"# Reference groups (recomputed from the code; {n} seeds {seeds}; budget {B} per type at stages 1-3)", "",
          "Donors are counted as physical defects (a Real-IAD fire_hood defect photographed by several cameras = one).",
          "Stages 1-3: per seed the target-object arm uses the budget's real defects and the cross-object arm their paired donors (1-to-1).",
          "Stage 4: all real defects / one donor each. Stage 5 (cross-object only): donors drawn from the extended pool.", "",
          f"| type | real defects | stages 1-3: refs per seed | real defects used over {n} seeds | donors used over {n} seeds | "
          f"stage 4: refs per seed | donors used over {n} seeds | matched source | matched pool | stage-5 source | stage-5 pool | "
          f"stage-5 donors per seed (min) | stage-5 donors over {n} seeds |", "|" + "---|" * 13]
    md += [f"| {r['type']} | {r['real']} | {B} | {r['r13']} | {r['d13']} | {r['real']} | {r['d4']} | {r['src']} | {r['pool']} | {r['src5']} | "
           f"{r['pool5']} | {r['s5_seed']} | {r['d5']} |" for r in rows]
    md += [f"| **total** | {tot['real']} | {B * len(rows)} | {tot['r13']} | {tot['d13']} | {tot['real']} | {tot['d4']} | | {tot['pool']} | | "
           f"{tot['pool5']} | {tot['s5_seed']} | {tot['d5']} |", "", "## LaTeX rows", "", "```latex"]
    md += [f"    {LABEL[r['type']]:13s} & {r['real']:2d} & {B} & {r['r13']:2d} & {r['d13']:2d} & {r['real']:2d} & {r['d4']:3d} & {r['src']:6s} & {r['pool']:3d} "
           f"& {r['src5']:14s} & {r['pool5']:3d} \\\\" for r in rows]
    md += [f"    Total         & {tot['real']} & {B * len(rows)} & {tot['r13']} & {tot['d13']} & {tot['real']} & {tot['d4']} &        & {tot['pool']} &                & {tot['pool5']} \\\\",
           "```", "", "Columns: type & real defects & stages 1-3 refs per seed & real defects used (10 seeds) & donors used (10 seeds) & "
           "stage 4 refs per seed & donors used (10 seeds) & matched source & matched pool & stage-5 source & stage-5 pool."]
    out = S.OUT / "reference_groups_table.md"; out.write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md)); print(out)


if __name__ == "__main__":
    main()
