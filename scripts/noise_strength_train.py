#!/usr/bin/env python3
"""Noise-strength test, training stage (cross-platform; user 2026-10-03). Thesis split, final model
(train_uninet.py --hero caao --no-photometric, fixed test shifts, leak guard), 5 training seeds per arm.

Arms = the image sets of results/noise_strength_test made by noise_strength_thesis_sets.py --fixed-steps 50:
  ns_0025_s50 ... ns_090_s50, ns_100   the baseline's 44 training synthetics re-rendered at noise strength 0.025 ... 1.0
                                       with 50 denoising steps (0.025: 21, 0.05: 41 = every timestep below the start)
  mix                                  per epoch, each of the 44 synthetics is drawn uniformly from its 12 versions above
                                       (same defect, canvas and placement; image + its own refined mask), via
                                       --synthetic-epochs. The draw is seeded by the training seed.
  base                                 the pipeline's own setting, 0.7 / 35 steps = results/thesis_set_clean20k_B (reference)
  070                                  the same setting re-rendered on this machine (removes the machine difference between
                                       "base" and the other arms):  noise_strength_thesis_sets.py --strengths 0.7  then  --arms 070
The image sets are made first with:
  python scripts/noise_strength_thesis_sets.py --fixed-steps 50 --strengths 0.025 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
  python scripts/noise_strength_thesis_sets.py --strengths 1.0

  python scripts/noise_strength_train.py [--seeds 42 123 7 99 256] [--arms mix 010_s50 ...] [--train-slots N] [--stagger S]
  python scripts/noise_strength_train.py --table-only

Resumable (runs with 20 epochs in results.json are skipped); checkpoints are deleted after every run. GPUs / MIG slices
are detected as in open_set_launch.py (one training pinned per device slot). Log: results/noise_strength_test/progress.log;
table: results/noise_strength_test/TABLES.md.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import shutil
import statistics as st
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_launch as L  # noqa: E402  (detect_gpus, run, done)

T = ROOT / "results/noise_strength_test"
PREP = ROOT / "results/cashew_100_rp/prep"
BASE = ROOT / "results/thesis_set_clean20k_B"                     # arm "base": the pipeline's own 0.7 / 35-step set
RUNNER = (ROOT / "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/experiment_UniNet"
          / "temp_uninet_experiment/run_uninet_cashew.py")
SETS = ["0025_s50", "005_s50", "010_s50", "020_s50", "030_s50", "040_s50", "050_s50", "060_s50", "070_s50", "080_s50", "090_s50", "100"]
LABEL = {"0025_s50": "0.025 (21 steps)", "005_s50": "0.05 (41 steps)", "100": "1.0 (50 steps)", "mix": "mixed (random strength per synthetic and epoch)"}
EPOCHS, LEVELS = 20, ["0x", "1x", "2x", "3x", "4x"]
_lock = threading.Lock()


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M')}  {msg}"
    with _lock:
        print(line, flush=True)
        with open(T / "progress.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")


def train_indices() -> list[str]:
    """The runner's 44 training synthetics (its own rule: easy cases shuffled with random.Random(SEED), first N_TRAIN_EASY)."""
    src = RUNNER.read_text(encoding="utf-8")
    seed = int(re.search(r"^SEED\s*=\s*(\d+)", src, re.M).group(1)); n = int(re.search(r"^N_TRAIN_EASY\s*=\s*(\d+)", src, re.M).group(1))
    assert int(re.search(r"^N_TRAIN_HARD\s*=\s*(\d+)", src, re.M).group(1)) == 0
    easy, hard = [], []
    for d in sorted(PREP.iterdir()):
        if d.is_dir():
            m = json.load(open(d / "meta.json")); (hard if m["is_hard"] else easy).append(m["idx"])
    rng = random.Random(seed); e, h = easy[:], hard[:]; rng.shuffle(e); rng.shuffle(h)
    return [f"{i:03d}" for i in sorted(e[:n])]


def triple(tag: str, idx: str) -> dict:
    d = T / f"ns_{tag}"; fx = d / "leak_fix" / idx
    if (fx / "image.png").exists():                               # the leak guard's replacement (067 on a clean canvas)
        t = (fx / "image.png", fx / "refined_masks_f025.png", fx / "canvas.png")
    else:
        t = (d / "generated/cfg7_vis" / f"{idx}.png", d / "refined_masks_f025" / f"{idx}.png", PREP / idx / "canvas.png")
    for p in t:
        assert p.exists(), f"missing {p}"
    return dict(zip(("image", "mask", "canvas"), map(str, t)))


def mix_epochs(seed: int) -> Path:
    out = T / "mix" / f"epochs_seed_{seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(100_000 + seed); idx = train_indices(); picks = [[rng.choice(SETS) for _ in idx] for _ in range(EPOCHS)]
    json.dump({"epochs": [[triple(tag, i) for tag, i in zip(row, idx)] for row in picks], "sets": picks, "indices": idx},
              open(out, "w"), indent=0)
    return out


def command(arm: str, seed: int) -> list[str]:
    base = "100" if arm == "mix" else arm; d = BASE if arm == "base" else T / f"ns_{base}"
    cmd = [sys.executable, "-u", str(ROOT / "scripts/train_uninet.py"), "--hero", "caao", "--no-photometric", "--cashew-100-dir", str(d),
           "--prep-dir", str(PREP), "--mask-dir", str(d / "refined_masks_f025"),
           "--output-dir", str(run_dir(arm) / f"seed_{seed}"), "--seed", str(seed)]
    return cmd + (["--", "--synthetic-epochs", str(mix_epochs(seed))] if arm == "mix" else [])


def run_dir(arm: str) -> Path:
    return T / "runs" / (arm if arm in ("mix", "base") else f"ns_{arm}")


def metrics(f: Path) -> dict | None:
    if not L.done(f, EPOCHS):
        return None
    r = json.load(open(f, encoding="utf-8")); k = [k for k in r if k.startswith("Mode") and " @ " not in k][0]
    if any(f"{k} @ {lv}" not in r for lv in LEVELS):
        return None
    a = [r[f"{k} @ {lv}"]["img_aurocs"][:EPOCHS] for lv in LEVELS]; c = [st.mean(col) for col in zip(*a)]; b = max(range(EPOCHS), key=lambda e: a[0][e])
    return {"0x best": a[0][b], "0x ep 15-20": st.mean(a[0][14:]), "0x last": a[0][-1], "composite ep 15-20": st.mean(c[14:]),
            "composite last": c[-1], "composite at best-0x epoch": c[b], "composite, per-level max (thesis style)": st.mean(max(x) for x in a)}


def table(seeds: list[int]) -> None:
    rows = [(LABEL.get(t, f"0.{t[1:3]} (50 steps)"), T / "runs" / f"ns_{t}") for t in SETS] + [(LABEL["mix"], T / "runs" / "mix"),
            ("0.7 pipeline setting (35 steps), rendered on this machine", T / "runs" / "ns_070"),
            ("0.7 pipeline setting (35 steps), shipped baseline set (rendered on the RTX 4090)", T / "runs" / "base")]
    cols = None; out = ["# Noise-strength test (image AUROC, thesis split, mean ± std over seeds)", ""]
    for lab, d in rows:
        ms = [m for m in (metrics(d / f"seed_{s}" / "results.json") for s in seeds) if m]
        if not ms:
            continue
        if cols is None:
            cols = list(ms[0]); out += ["| noise strength | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        out.append(f"| {lab} | {len(ms)} | " + " | ".join(f"{st.mean(m[c] for m in ms):.3f} ± {st.pstdev([m[c] for m in ms]):.3f}" for c in cols) + " |")
    out += ["", "Thesis-style composite = mean over the five shift levels of each level's best epoch (optimistic). "
            "All rows: --hero caao (projection weight 0.2, no stop-gradient), no photometric augmentation, fixed test shifts."]
    (T / "TABLES.md").write_text("\n".join(out) + "\n", encoding="utf-8"); print("\n".join(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7, 99, 256]); ap.add_argument("--arms", nargs="+", default=SETS + ["mix", "base"])
    ap.add_argument("--train-slots", type=int); ap.add_argument("--stagger", type=int, default=60)
    ap.add_argument("--gpu-ids", nargs="+"); ap.add_argument("--table-only", action="store_true"); a = ap.parse_args()
    if a.table_only:
        return table(a.seeds)
    found, mig = L.detect_gpus(); ids = a.gpu_ids or found; pin = len(ids) > 1 or bool(a.gpu_ids) or mig
    slots = a.train_slots or 2 * len(ids)                         # 2 per device = what a 24 GB GPU holds; raise on a whole B200
    env = dict(os.environ, PYTHONIOENCODING="utf-8"); free: queue.Queue = queue.Queue(); gate = threading.Lock()
    for i in range(slots):
        free.put(ids[i % len(ids)])
    jobs = [(arm, s) for s in a.seeds for arm in a.arms]          # seed-major: every arm gets its first seed early
    log(f"=== noise-strength training: {len(jobs)} runs ({len(a.arms)} arms x {len(a.seeds)} seeds), {slots} at a time, devices {ids}"
        f"{' (MIG slices)' if mig else ''}; {len(train_indices())} training synthetics")

    def one(job: tuple) -> None:
        arm, s = job; out = run_dir(arm) / f"seed_{s}"
        if L.done(out / "results.json", EPOCHS):
            return
        with gate:
            cmd = command(arm, s); time.sleep(a.stagger)
        g = free.get()
        try:
            t0 = time.time(); rc = L.run(cmd, T / "logs" / f"train_{arm}_seed_{s}.log", dict(env, CUDA_VISIBLE_DEVICES=g) if pin else env)
        finally:
            free.put(g)
        shutil.rmtree(out / "checkpoints", ignore_errors=True)
        log(f"{arm} seed {s}: rc {rc}, {(time.time() - t0) / 60:.1f} min")

    with ThreadPoolExecutor(slots) as ex:
        list(ex.map(one, jobs))
    table(a.seeds); log("=== noise-strength training finished")


if __name__ == "__main__":
    main()
