#!/usr/bin/env python3
"""Score histograms, normals vs anomalies, of the score_dist_c1 runs (scripts/open_set_scores.py).

One sheet per (seed, held-out type): rows = Normal-only, CutMix, DTD, Target-object, Cross-object (configuration 1);
columns = 0x and 1-4x (the four shift levels pooled). Every panel is read at the run's best-0x epoch (the epoch of the
highest 0x image AUROC, as in the tables). The AUROC recomputed from the dumped scores must equal results.json.
  python scripts/open_set_score_hist.py --seeds 42      -> results/open_set_v2/figures/score_hist_c1/seed_<s>_<type>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_steps as S  # noqa: E402

ARMS = [("Normal-only", "m1"), ("CutMix", "cutmix_s1"), ("DTD", "dtd_s1"), ("Target-object", "diffusion_in_s1"),
        ("Cross-object", "diffusion_cross_s1")]
LV = ["1x", "2x", "3x", "4x"]


def load(arm: str, s: int, k: str, src: Path) -> tuple[int, dict]:
    d = src / arm / f"seed_{s}" / f"fold_{k}"; r = json.load(open(d / "results.json", encoding="utf-8"))
    key = [x for x in r if x.startswith("Mode") and " @ " not in x][0]; e = int(np.argmax(r[f"{key} @ 0x"]["img_aurocs"]))
    primary = r[key].get("primary_agg"); sc = {}
    for x in json.load(open(d / "image_scores.json")):
        if x["epoch"] == e and x["shift"] in ["0x"] + LV and (primary is None or x["agg"] == primary):
            sc[x["shift"]] = (np.array(x["labels"]), np.array(x["scores"]))
    for lv in ["0x"] + LV:  # dumped scores reproduce the logged AUROC
        a = roc_auc_score(*sc[lv]); assert abs(a - r[f"{key} @ {lv}"]["img_aurocs"][e]) < 1e-6, (arm, s, k, lv, a)
    return e, sc


def panel(ax, groups: list, title: str) -> None:
    y = np.concatenate([g[0] for g in groups]); v = np.concatenate([g[1] for g in groups])
    bins = np.linspace(v.min(), v.max(), 31)
    ax.hist(v[y == 0], bins, color="#3b6fb6", alpha=.6, label=f"normal (n={int((y == 0).sum())})")
    ax.hist(v[y == 1], bins, color="#c8453c", alpha=.6, label=f"anomaly (n={int((y == 1).sum())})")
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, frameon=False); ax.tick_params(labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42]); ap.add_argument("--folds", nargs="+", default=S.CLASSES)
    ap.add_argument("--src", type=Path, default=S.OUT / "score_dist_c1")
    ap.add_argument("--out", type=Path, default=S.OUT / "figures" / "score_hist_c1")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        for k in a.folds:
            fig, axs = plt.subplots(len(ARMS), 2, figsize=(10, 3.1 * len(ARMS)), dpi=150)
            for (name, arm), row in zip(ARMS, axs):
                e, sc = load(arm, s, k, a.src)
                panel(row[0], [sc["0x"]], f"{name}  0x   AUROC {roc_auc_score(*sc['0x']):.3f}  (best-0x epoch {e + 1})")
                panel(row[1], [sc[lv] for lv in LV], f"{name}  1-4x pooled   mean AUROC {np.mean([roc_auc_score(*sc[lv]) for lv in LV]):.3f}")
            fig.suptitle(f"Image anomaly scores, held-out {k}, seed {s}, configuration 1", fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .985)); f = a.out / f"seed_{s}_{k}.png"; fig.savefig(f); plt.close(fig); print(f)


if __name__ == "__main__":
    main()
