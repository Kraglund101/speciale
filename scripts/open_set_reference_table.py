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
SRC = {"fire_hood": "FH", "pipe_fryum": "PF", "macaroni1": "MA"}
LABEL = {"holes": "Holes", "breakage": "Breakage", "burnt": "Burnt", "colour_diff": "Colour diff.", "colour_same": "Colour same",
         "scratches": "Scratches"}


def phys(d: dict) -> str:
    return d["donor"] + ":" + re.sub(r"_C\d_.*$", "", Path(d["cr_img"]).stem)


def main() -> None:
    refs = P.load_refs(); pools = P.donor_pools(); plan = S.anomaly_plan(refs); seeds = S.SEEDS10
    pair = {s: P.pairing(s, refs) for s in seeds}; budget = len(plan[str(seeds[0])][P.CLASSES[0]])
    rows = []
    for t in P.CLASSES:
        pool = {phys(x) for x in pools[t]}; src = collections.Counter(k.split(":")[0] for k in pool)
        rows.append({"type": t, "real": len(refs[t]), "source": ", ".join(SRC.get(k, k) for k in sorted(src, key=lambda k: -src[k])),
                     "pool_images": len(pools[t]), "pool": len(pool),
                     "distinct_all": len({phys(pair[s][t][c]) for s in seeds for c in pair[s][t]}),
                     "distinct_budget": len({phys(pair[s][t][c]) for s in seeds for c in plan[str(s)][t]}),
                     "real_used_budget": len({c for s in seeds for c in plan[str(s)][t]})})
    tot = {k: sum(r[k] for r in rows) for k in ("real", "pool_images", "pool", "distinct_all", "distinct_budget", "real_used_budget")}
    md = [f"# Reference groups (recomputed {len(seeds)} seeds {seeds}; fixed budget in anomaly_plan.json = {budget} per type)", "",
          "Donors counted as physical defects. `distinct` = different donor defects used over all seeds.", "",
          "| type | real cashew defects | source | donor pool (physical defects) | donor pool (images) | distinct donors, all cashew defects | "
          f"distinct donors, budget {budget} | cashew defects used with budget {budget} |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {r['type']} | {r['real']} | {r['source']} | {r['pool']} | {r['pool_images']} | {r['distinct_all']} | {r['distinct_budget']} | "
           f"{r['real_used_budget']} |" for r in rows]
    md += [f"| **total** | {tot['real']} | | {tot['pool']} | {tot['pool_images']} | {tot['distinct_all']} | {tot['distinct_budget']} | {tot['real_used_budget']} |", "",
           "## LaTeX rows (no-budget regime: one donor per cashew defect, per seed)", "", "```latex"]
    md += [f"    {LABEL[r['type']]:13s} & {r['real']:2d} & {r['real']:2d} & {r['source']:6s} & {r['pool']:3d} & {r['distinct_all']:3d} \\\\" for r in rows]
    md += [f"    Total         & {tot['real']} & {tot['real']} &        & {tot['pool']} & {tot['distinct_all']} \\\\", "```", "",
           f"## LaTeX rows (fixed-budget regimes: {budget} references per type and seed)", "", "```latex"]
    md += [f"    {LABEL[r['type']]:13s} & {r['real']:2d} & {budget:2d} & {r['source']:6s} & {r['pool']:3d} & {r['distinct_budget']:3d} \\\\" for r in rows]
    md += [f"    Total         & {tot['real']} & {budget * len(rows)} &        & {tot['pool']} & {tot['distinct_budget']} \\\\", "```",
           "", "Columns: type & real defects & donors per seed & source & donor pool & distinct donors over the seeds."]
    out = S.OUT / "reference_groups_table.md"; out.write_text("\n".join(md) + "\n", encoding="utf-8"); print("\n".join(md)); print(out)


if __name__ == "__main__":
    main()
