#!/usr/bin/env python3
"""Noise-strength test on the thesis split (user 2026-10-03): the baseline synthetic set (results/thesis_set_clean20k_B:
94 easy placements of results/cashew_100_rp/prep, clean-20k, visual CFG 7, 50 steps, no caption, raw-canvas blend,
masks 0.25 x p90 matched reference, leak-fixed 067) re-rendered with ONLY the noise strength changed.
Same canvas, placed mask, reference and seed per image as the baseline. 0.7 is the existing baseline set (not redone).

  <out>/ns_XX/generated/cfg7_vis/NNN.png, <out>/ns_XX/refined_masks_f025/NNN.png, <out>/ns_XX/leak_fix/067/...

  python scripts/noise_strength_thesis_sets.py [--strengths 0.1 0.2 ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pregenerate_synthetic as P  # noqa: E402

PREP = ROOT / "results/cashew_100_rp/prep"
BASE = ROOT / "results/thesis_set_clean20k_B"
OUT = ROOT / "results/noise_strength_test"
DEFAULT = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0]


FULL = False     # --every-timestep: same start noise as the 50-step run, but every timestep below it is denoised


def gen_kw(ns: float) -> dict:
    """Generation settings for strength ns. FULL: the 1000-step DDIM schedule (timesteps 1000..1), started at the SAME
    timestep the 50-step run starts at (981 - 20 * int(50 * (1 - ns))), i.e. 81 / 41 / 21 steps for 0.1 / 0.05 / 0.025."""
    if S35:                                                       # the custom scheduler holds exactly the steps to run
        return {**P.GEN, "num_steps": NSTEPS, "noise_strength": 1.0}
    if not FULL:
        return {**P.GEN, "noise_strength": ns}
    t = 981 - 20 * int(50 * (1 - ns))
    return {**P.GEN, "num_steps": 1000, "noise_strength": (t - 0.5) / 1000}


S35 = False      # --fixed-steps N: same start noise, but always N denoising steps (user 2026-10-03: 50)
NSTEPS = 50


def s35_timesteps(ns: float) -> list[int]:
    """NSTEPS timesteps, evenly spaced from the 50-step run's start timestep down to 1 (fewer only if the start timestep
    is below NSTEPS: every timestep then; 0.025 starts at 21, 0.05 at 41). With NSTEPS 35, 0.7 gives exactly the
    pipeline's own list (681, 661, ..., 1); with NSTEPS 50, 1.0 gives the standard 50-step list (981, ..., 1)."""
    t0 = 981 - 20 * int(50 * (1 - ns)); n = min(NSTEPS, t0)
    ts = [int(x) for x in np.round(np.linspace(t0, 1, n))]
    assert len(set(ts)) == n and ts[0] == t0 and ts[-1] == 1
    return ts


def fixed_scheduler(config):
    """DDIM (eta 0) on an explicit timestep list; identical to DDIMScheduler.step on the standard list (checked)."""
    from diffusers import DDIMScheduler
    from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
    import torch

    class FixedStepDDIM(DDIMScheduler):
        custom: list[int] = []

        def set_timesteps(self, num_inference_steps, device=None, **kw):
            self.num_inference_steps = len(self.custom); self.timesteps = torch.tensor(self.custom, dtype=torch.long, device=device)

        def step(self, model_output, timestep, sample, **kw):
            i = self.custom.index(int(timestep)); a_t = float(self.alphas_cumprod[self.custom[i]])
            a_p = float(self.alphas_cumprod[self.custom[i + 1]]) if i + 1 < len(self.custom) else float(self.final_alpha_cumprod)
            x0 = (sample - (1 - a_t) ** 0.5 * model_output) / a_t ** 0.5
            return DDIMSchedulerOutput(prev_sample=a_p ** 0.5 * x0 + (1 - a_p) ** 0.5 * model_output, pred_original_sample=x0)

    return FixedStepDDIM.from_config(config)


def set_dir(ns: float) -> Path:
    if FULL or S35:
        return _set_dir(ns).with_name(_set_dir(ns).name + ("_alltimesteps" if FULL else f"_s{NSTEPS}"))
    return _set_dir(ns)


def _set_dir(ns: float) -> Path:
    t = int(round(ns * 1000))                                   # 0.05 -> ns_005, 0.7 -> ns_070, 0.025 -> ns_0025
    return OUT / (f"ns_{t // 10:03d}" if t % 10 == 0 else f"ns_{t:04d}")


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--strengths", type=float, nargs="+", default=DEFAULT)
    ap.add_argument("--every-timestep", action="store_true"); ap.add_argument("--fixed-steps", type=int, default=0); a = ap.parse_args()
    global FULL, S35, NSTEPS; FULL, S35 = a.every_timestep, a.fixed_steps > 0; NSTEPS = a.fixed_steps or NSTEPS
    assert P.GEN["noise_strength"] == 0.7 and P.CKPT == "clean-20k" and P.MASK_FACTOR == 0.25, "baseline settings changed"
    cases = [p for p in sorted(PREP.iterdir()) if p.is_dir() and not json.loads((p / "meta.json").read_text())["is_hard"]]
    fixes = sorted(p for p in (BASE / "leak_fix").iterdir() if p.is_dir())
    jobs = []                                                   # (strength, prep folder, canvas id, output image)
    for ns in a.strengths:
        d = set_dir(ns); (d / "generated/cfg7_vis").mkdir(parents=True, exist_ok=True); (d / "refined_masks_f025").mkdir(exist_ok=True)
        jobs += [(ns, p, d / "generated/cfg7_vis" / f"{p.name}.png") for p in cases]
        for fx in fixes:
            dst = d / "leak_fix" / fx.name; dst.mkdir(parents=True, exist_ok=True)
            for f in ("canvas.png", "placed_mask.png", "ref_image.png", "ref_mask.png", "meta.json"):
                shutil.copy(fx / f, dst / f)
            jobs.append((ns, dst, dst / "image.png"))
    todo = [j for j in jobs if not j[2].exists()]
    print(f"{len(jobs)} images, {len(todo)} to generate", flush=True)
    if todo:
        P.gc.setup_experiment("ResNet"); pipe, ip, t2i = P.gc.load_models(P.h.CKPTS[P.CKPT]); P.gc.EXP = Path(tempfile.mkdtemp(prefix="nstest_"))
        if S35:
            pipe.scheduler = fixed_scheduler(pipe.scheduler.config)
        for n, (ns, p, dst) in enumerate(todo):
            meta = json.loads((p / "meta.json").read_text()); rid = f"nst{'f' if FULL else f's{NSTEPS}' if S35 else ''}-{int(round(ns * 1000))}-{p.name}"
            if S35:
                pipe.scheduler.custom = s35_timesteps(ns)
            P.gc.generate_one(pipe, ip, t2i, canvas_id=meta["normal"].split(".")[0], ref_id=rid, difficulty="easy", defect_map={rid: "defect"},
                              seed=meta["seed"], device="cuda", layout="A", ref_img_override=p / "ref_image.png",
                              ref_mask_override=p / "ref_mask.png", placed_mask_override=p / "placed_mask.png", save_raw=True,
                              canvas_override=p / "canvas.png", caption_override=" ", **gen_kw(ns))
            shutil.copy(P.gc.EXP / "anomaly/imgs/easy" / f"{rid}.png", dst)
            if n % 50 == 0:
                print(f"generated {n + 1}/{len(todo)}", flush=True)
        del pipe, ip, t2i
        import torch; torch.cuda.empty_cache()
    from compute_refined_masks import _load_dinov3_cnx
    from refine_inprocess import refine
    dino = None
    for ns, p, img in jobs:
        dst = img.parent / "refined_masks_f025.png" if img.name == "image.png" else set_dir(ns) / "refined_masks_f025" / img.name
        if not dst.exists():
            dino = dino or _load_dinov3_cnx("cuda")
            m = refine(dino, img, p, None, factor=P.MASK_FACTOR, thesis_reference=True)
            Image.fromarray(m.astype(np.uint8) * 255).save(dst)
    for ns in a.strengths:
        d = set_dir(ns); cov = [(np.array(Image.open(f).convert("L")) > 127).mean() * 100 for f in sorted((d / "refined_masks_f025").glob("*.png"))]
        (d / "SOURCE.txt").write_text(f"baseline set {BASE} re-rendered with noise_strength {ns} (settings {gen_kw(ns)}{', timesteps ' + str(s35_timesteps(ns)) if S35 else ''}); everything else identical "
                                      f"(GEN {P.GEN}, ckpt {P.CKPT}, masks {P.MASK_FACTOR} x p90 thesis_reference)\n")
        print(f"noise {ns:.1f}: {len(cov)} images + {len(fixes)} leak fix | mask area median {np.median(cov):.2f}% of image", flush=True)
    print("SETS DONE", flush=True)


if __name__ == "__main__":
    main()
