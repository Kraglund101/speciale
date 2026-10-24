#!/usr/bin/env python3
"""Reproduce the ORIGINAL seed-42/123 open-set runs (configuration 1, Mode 4 caao) in their original environment, with
per-image scores, and train the matching type-inclusive runs in the same environment (2026-10-24).

Original environment (verified by a 1-epoch probe that matched bit-exactly): anomalib 2.6.2 (.venv_a262, otherwise the
pinned package list), NVIDIA B200 MIG 1g.23gb slice, no OMP/MKL thread cap, no MPS, same code / runner / lists.
A whole B200 does NOT reproduce these runs (different from epoch 1; anomalib / threads / MPS / parallelism shown
irrelevant) - see report.

  python scripts/repro_a262.py gate                 # seed 42: cross colour_same + target holes, compared to runs/
  python scripts/repro_a262.py open                 # remaining open-set reproductions (seeds 42, 123; 24 total)
  python scripts/repro_a262.py ti                   # type-inclusive (typeincl lists), seeds 42, 123 (24 runs)
  python scripts/repro_a262.py verify               # compare every reproduced open-set run with its original
Output: results/open_set_v2/repro_a262/{open,ti}/<arm>/seed_<s>/fold_<k>/ (results.json, image_scores.json, stdout.log).
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import open_set_scores as OS  # noqa: E402
import open_set_steps as S  # noqa: E402
import open_set_typeincl as TI  # noqa: E402

OUT = S.OUT / "repro_a262"
ARMS = ["diffusion_in", "diffusion_cross"]
SEEDS = [42, 123]
GATE = [("diffusion_cross", 42, "colour_same"), ("diffusion_in", 42, "holes")]


def out_dir(kind: str, arm: str, s: int, k: str) -> Path:
    return OUT / kind / f"{arm}_{'s1' if kind == 'open' else 'ti'}" / f"seed_{s}" / f"fold_{k}"


def cmd(kind: str, arm: str, s: int, k: str) -> list[str]:
    c = OS.command(arm, s, k, "probe", []) if kind == "open" else TI.command(arm, s, k, "probe", [], "ti")
    c[c.index("--output-dir") + 1] = str(out_dir(kind, arm, s, k))
    assert Path(c[0]).resolve() == (ROOT / ".venv_a262/bin/python").resolve() or ".venv_a262" in c[0], c[0]
    return c


def done(d: Path) -> bool:
    """Complete = 20 epochs in results.json AND the full score dump (20 epochs x 6 levels x 2 aggregations = 240).
    (A run killed after training but before the dump would otherwise be resumed by the runner with an empty dump.)"""
    try:
        r = json.load(open(d / "results.json")); k = [x for x in r if x.startswith("Mode") and " @ " not in x][0]
        return len(r[k]["img_aurocs"]) == S.E and len(json.load(open(d / "image_scores.json"))) == 240
    except Exception:
        return False


def mig_ids() -> list[str]:
    import re
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    ids = re.findall(r"UUID:\s*(MIG-[0-9A-Za-z-]+)", out); assert ids, "no MIG slices"
    return ids


def run_all(jobs: list[tuple], per_slice: int, stagger: int, busy: dict | None = None, release_when: Path | None = None) -> None:
    """busy = {slice index: slots already used by another job}; those slots are added once `release_when` contains
    'GATE SCRIPT EXITED' (memory: ~7.5 GB per run, 20.5 GB per slice -> at most 2 per slice)."""
    ids = mig_ids(); free: queue.Queue = queue.Queue(); busy = busy or {}
    for i in range(per_slice * len(ids)):
        sl = i % len(ids)
        if i // len(ids) >= per_slice - busy.get(sl, 0):
            continue
        free.put(ids[sl])
    if busy:
        def release():
            while "GATE SCRIPT EXITED" not in (release_when.read_text() if release_when.exists() else ""):
                time.sleep(30)
            for sl, n in busy.items():
                for _ in range(n):
                    free.put(ids[sl])
            print(f"{time.strftime('%m-%d %H:%M')}  [repro_a262] gate finished: {sum(busy.values())} more slots", flush=True)
        threading.Thread(target=release, daemon=True).start()
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_MPS_PIPE_DIRECTORY")}
    env.update(PYTHONIOENCODING="utf-8", SPECIALE_ROOT=str(ROOT)); gate = threading.Lock()

    def one(job):
        kind, arm, s, k = job; d = out_dir(kind, arm, s, k)
        if done(d):
            return
        with gate:
            c = cmd(kind, arm, s, k); time.sleep(stagger)
        g = free.get(); d.mkdir(parents=True, exist_ok=True); t0 = time.time()
        try:
            with open(d / "stdout.log", "w") as f:
                rc = subprocess.call(c, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT), env=dict(env, CUDA_VISIBLE_DEVICES=g))
        finally:
            free.put(g)
        shutil.rmtree(d / "checkpoints", ignore_errors=True)
        print(f"{time.strftime('%m-%d %H:%M')}  [repro_a262] {kind} {arm} seed {s} fold {k} on {g[:12]}: rc {rc}, "
              f"{(time.time() - t0) / 60:.1f} min, scores {'ok' if done(d) else 'MISSING'}", flush=True)

    print(f"[repro_a262] {len(jobs)} runs, {per_slice} per slice on {len(ids)} slices", flush=True)
    with ThreadPoolExecutor(per_slice * len(ids)) as ex:
        list(ex.map(one, jobs))


def compare(arm: str, s: int, k: str) -> tuple[bool, list[str]]:
    """Every key and value of the original results.json vs the reproduction (exact equality)."""
    o = json.load(open(S.OUT / "runs" / f"{arm}_s1" / f"seed_{s}" / f"fold_{k}" / "results.json"))
    r = json.load(open(out_dir("open", arm, s, k) / "results.json")); diffs = []

    def walk(a, b, p):
        if type(a) != type(b):
            diffs.append(f"{p}: type {type(a).__name__} vs {type(b).__name__}"); return
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    diffs.append(f"{p}/{key}: only in {'original' if key in a else 'reproduction'}")
                else:
                    walk(a[key], b[key], f"{p}/{key}")
        elif isinstance(a, list):
            if len(a) != len(b):
                diffs.append(f"{p}: length {len(a)} vs {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{p}[{i}]")
        elif a != b:
            diffs.append(f"{p}: {a!r} vs {b!r}")
    walk(o, r, "")
    return not diffs, diffs


def main() -> None:
    what = sys.argv[1]
    if what == "gate":
        run_all([("open", a, s, k) for a, s, k in GATE], per_slice=1, stagger=5)
        for a, s, k in GATE:
            ok, d = compare(a, s, k)
            print(f"GATE {a} seed {s} fold {k}: {'EXACT' if ok else 'DIFFERENT'} ({len(d)} differing values)", *d[:15], sep="\n  ")
    elif what == "full":                                           # everything except the 2 gate runs (running separately)
        js = [("open", a, s, k) for s in SEEDS for a in ARMS for k in S.CLASSES if (a, s, k) not in GATE] + \
             [("ti", a, s, k) for s in SEEDS for a in ARMS for k in S.CLASSES]
        run_all(js, per_slice=2, stagger=10, busy={0: 1, 1: 1}, release_when=OUT / "gate.log")
    elif what == "open":
        run_all([("open", a, s, k) for s in SEEDS for a in ARMS for k in S.CLASSES], per_slice=2, stagger=10)
    elif what == "ti":
        run_all([("ti", a, s, k) for s in SEEDS for a in ARMS for k in S.CLASSES], per_slice=2, stagger=10)
    elif what == "verify":
        bad = 0
        for s in SEEDS:
            for a in ARMS:
                for k in S.CLASSES:
                    ok, d = compare(a, s, k); bad += not ok
                    print(f"{a} seed {s} fold {k}: {'EXACT' if ok else f'DIFFERENT ({len(d)} values), e.g. ' + (d[0] if d else '')}")
        print(f"VERIFY: {24 - bad}/24 exact"); raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
