#!/usr/bin/env python3
"""Configuration 6 "mix-synthetic": the four synthetic methods mixed equally, open-set as every other arm.

A held-out-k run has, per epoch, the 45 slots of the five TRAINING types (9 per type; the held-out type never occurs).
Every slot has four fillings that share canvas + placed footprint: CutMix (the images made for fold k), DTD,
Target-object, Cross-object = the four configuration-<base> run lists of that run (scripts/open_set_steps.run_lists,
exactly what the launcher trains on). Epoch e (0..19) gives slot j (training types in CLASSES order, item 0..8) the
method METHODS[(j + e) % 4]: per epoch 11-12 slots of each method; over the 20 epochs every slot gets every method
exactly 5 times (225 slot-epochs per method). Everything else as the other arms: same train normals and test anomalies,
--hero caao, no photometric, 20 epochs, same seed. Base configuration 1 (default): the same 45 images per method in
every epoch, so the mix only changes which method fills a slot.

  python scripts/open_set_mix.py build --seeds 42 123 7 99 256 11 22 33 44 55        # lists + checks only
  python scripts/open_set_mix.py run   --seeds 42 123 7 99 256 11 22 33 44 55 --slots 12 --stagger 10
Runs: results/open_set_v2/runs/mix_synthetic_s6/seed_<s>/fold_<k>/results.json; logs results/open_set_v2/logs/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_steps as S  # noqa: E402

METHODS = ["cutmix", "dtd", "diffusion_in", "diffusion_cross"]
NAME = "mix_synthetic_s6"


def slot_key(x: dict) -> tuple:
    """(type, item) of a list entry, from its canvas path .../prep/seed_<s>/<set>/<type>/<iii>/canvas.png."""
    p = Path(x["canvas"]); return p.parent.parent.name, p.parent.name


def mix(lists: dict[str, list], fold: str) -> list[list[dict]]:
    """lists: method -> 20 epochs x 45 entries. Checks that the four methods fill the same slots, then mixes."""
    types = [t for t in S.CLASSES if t != fold]; E = len(lists[METHODS[0]])
    assert all(len(lists[m]) == E for m in METHODS)
    out = []
    for e in range(E):
        keys = [slot_key(x) for x in lists[METHODS[0]][e]]
        assert len(keys) == 5 * S.K and [k[0] for k in keys] == [t for t in types for _ in range(S.K)], (fold, e, "slot order")
        assert fold not in {k[0] for k in keys}, (fold, e, "held-out type in training slots")
        for m in METHODS:
            assert [slot_key(x) for x in lists[m][e]] == keys, (fold, e, m, "slots differ between methods")
            assert [x["canvas"] for x in lists[m][e]] == [x["canvas"] for x in lists[METHODS[0]][e]], (fold, e, m, "canvas")
        out.append([dict(lists[METHODS[(j + e) % 4]][e][j], method=METHODS[(j + e) % 4]) for j in range(len(keys))])
    n = {m: sum(x["method"] == m for ep in out for x in ep) for m in METHODS}
    assert len(set(n.values())) == 1, n
    return out


def build(seed: int, fold: str, base: int) -> Path:
    """Write epochs_mix_s6.json next to the run lists of this run (the four method lists are (re)written by run_lists)."""
    lists = {}
    for m in METHODS:
        r = S.run_lists(seed, fold, m, base); assert r["missing"] == 0, (seed, fold, m, r["missing"], "missing files")
        lists[m] = json.load(open(r["epochs"]))["epochs"]
    ep = mix(lists, fold); f = r["dir"] / f"epochs_mix_s6.json"
    json.dump({"base_step": base, "methods": METHODS, "epochs": ep}, open(f, "w"))
    return f


def command(seed: int, fold: str) -> list[str]:
    d = S.OUT / "lists" / f"seed_{seed}" / f"fold_{fold}"
    return [sys.executable, "-u", str(ROOT / "scripts/train_uninet.py"), "--hero", "caao", "--no-photometric",
            "--cashew-100-dir", str(ROOT / S.THESIS["set"]), "--prep-dir", str(ROOT / S.THESIS["prep"]), "--mask-dir", str(ROOT / S.THESIS["masks"]),
            "--output-dir", str(S.OUT / "runs" / NAME / f"seed_{seed}" / f"fold_{fold}"), "--seed", str(seed), "--",
            "--train-normal-list", str(d / "train_normals.txt"), "--test-anomaly-list", str(d / "test_anomalies.json"),
            "--fg-anomaly-dir", str(S.CASHEW / "birefnet_masks_anomaly"), "--synthetic-epochs", str(d / "epochs_mix_s6.json")]


def done(seed: int, fold: str) -> bool:
    p = S.OUT / "runs" / NAME / f"seed_{seed}" / f"fold_{fold}" / "results.json"
    try:
        r = json.load(open(p, encoding="utf-8")); k = [k for k in r if k.startswith("Mode") and " @ " not in k]
        return bool(k) and len(r[k[0]]["img_aurocs"]) >= S.E
    except Exception:
        return False


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M')}  [mix_s6] {msg}"; print(line, flush=True)
    with open(S.OUT / "launch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["build", "run"]); ap.add_argument("--seeds", type=int, nargs="+", default=S.SEEDS10)
    ap.add_argument("--folds", nargs="+", default=S.CLASSES); ap.add_argument("--base-step", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--stagger", type=int, default=90)
    a = ap.parse_args()
    for s in a.seeds:
        for k in a.folds:
            build(s, k, a.base_step)
        log(f"seed {s}: mix lists built + checked (base configuration {a.base_step}, folds {a.folds})")
    if a.stage == "build":
        return
    gate = threading.Lock(); env = dict(os.environ, PYTHONIOENCODING="utf-8")

    def one(job: tuple) -> None:
        s, k = job
        if done(s, k):
            return
        with gate:
            time.sleep(a.stagger)
        t0 = time.time(); lf = S.OUT / "logs" / f"train_{NAME}_seed{s}_{k}.log"; lf.parent.mkdir(parents=True, exist_ok=True)
        with open(lf, "a", encoding="utf-8") as f:
            rc = subprocess.call(command(s, k), stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        shutil.rmtree(S.OUT / "runs" / NAME / f"seed_{s}" / f"fold_{k}" / "checkpoints", ignore_errors=True)
        log(f"seed {s} fold {k}: rc {rc}, {(time.time() - t0) / 60:.1f} min")

    jobs = [(s, k) for s in a.seeds for k in a.folds]; log(f"training {len(jobs)} runs, {a.slots} at a time")
    with ThreadPoolExecutor(a.slots) as ex:
        list(ex.map(one, jobs))
    log("all runs finished")


if __name__ == "__main__":
    main()
