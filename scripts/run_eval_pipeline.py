#!/usr/bin/env python3
"""Automated post-training evaluation pipeline.

Runs all evaluation steps for a trained Anomagic checkpoint:
  1a. Generate 100 cashews _rp (all 3 CFGs come free: 1.0, 3.5, 7.0)
  1b. Generate 100 cashews _mo (mimic-original placement, all 3 CFGs)
  2.  Compute perceptually-refined masks for both placements × both CFG candidates.
      Default pipeline: DINOv3-CNX backbone + α-blend VAE roundtrip canvas + σ=4 + factor=0.6
      (changed 2026-04-10 after visual inspection — was VGG + no-roundtrip + factor=0.65).
      See compute_refined_masks.py docstring for rationale.
  3.  ResNet real-vs-synthetic for BOTH CFGs using _mo → auto-select winner
  4.  ResNet supervised AD using _rp (winner CFG)
  5.  UniNet multi-shift evaluation using _rp (winner CFG)
  6.  Aggregate all metrics → eval_summary.json

Placement usage:
  - _mo (mimic-original): ResNet RvS — natural placement for fairest quality test
  - _rp (random-position): ResNet AD + UniNet — varied placement for training data

Each step is idempotent — skips if output already exists.

Usage:
    python scripts/run_eval_pipeline.py \
        --checkpoint important_results/<run>/checkpoint_final \
        --output-dir important_results/<run> \
        [--dust N] [--skip-generation] [--skip-masks] [--skip-resnet] [--skip-uninet]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/home/fpv297/venv/bin/python3"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Shared prep dirs (created once, reused by all runs)
PREP_RP = PROJECT_ROOT / "results" / "cashew_100_rp" / "prep"
PREP_MO = PROJECT_ROOT / "results" / "cashew_100_mo" / "prep"

# Birefnet FG masks
FG_DIR = PROJECT_ROOT / "anomverse_extension" / "datasets" / "VisA_validation_dataset" / "datasets" / "easy_test" / "cashew" / "birefnet_masks" / "Normal"

# Cashew base (for VisA source images)
CASHEW_BASE = (
    PROJECT_ROOT / "anomverse_extension" / "datasets" / "VisA_validation_dataset"
    / "datasets" / "easy_test" / "cashew"
)

# UniNet script location
UNINET_SCRIPT = CASHEW_BASE / "experiment_UniNet" / "temp_uninet_experiment" / "run_uninet_cashew.py"

# CFG candidates for auto-selection
CFG_CANDIDATES = [
    {"cfg": 3.5, "variant": "cfg35_vis"},
    {"cfg": 7.0, "variant": "cfg7_vis"},
]

# All sample IDs
ALL_EASY = [f"{i:03d}" for i in range(94)]
ALL_HARD = [f"{i:03d}" for i in range(94, 100)]


_GPU_LOG_PATH: Path | None = None  # set by main(); gpu_snapshot() appends to this


def gpu_snapshot(label: str) -> None:
    """Append a JSONL snapshot of per-GPU memory + all GPU processes (with owners).

    Writes to the path set in _GPU_LOG_PATH. No-op if unset. Tolerates nvidia-smi
    failures. Enough info to later distinguish our OOMs from external-contention
    OOMs on a shared GPU.
    """
    if _GPU_LOG_PATH is None:
        return
    import datetime

    entry: dict = {"time": datetime.datetime.now().isoformat(), "label": label}

    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        entry["gpus"] = [line.strip() for line in r.stdout.strip().splitlines()]
    except Exception as e:
        entry["gpus_error"] = repr(e)

    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-compute-apps=pid,gpu_name,used_memory,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        procs = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 4:
                continue
            pid = parts[0]
            try:
                ps = subprocess.run(
                    ["ps", "-p", pid, "-o", "user=,cmd="],
                    capture_output=True, text=True, timeout=5,
                )
                owner_cmd = ps.stdout.strip() or "?"
            except Exception:
                owner_cmd = "?"
            procs.append({
                "pid": pid,
                "gpu_name": parts[1],
                "used_memory": parts[2],
                "process_name": parts[3],
                "owner_cmd": owner_cmd,
            })
        entry["processes"] = procs
    except Exception as e:
        entry["processes_error"] = repr(e)

    try:
        with open(_GPU_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def run_cmd(cmd: list[str], step_name: str) -> bool:
    """Run a subprocess command. Returns True on success."""
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}")
    sys.stdout.flush()

    gpu_snapshot(f"before: {step_name}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0
    gpu_snapshot(f"after rc={result.returncode}: {step_name}")

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) in {elapsed:.0f}s")
        return False
    print(f"  OK in {elapsed:.0f}s")
    return True


def step_exists(path: Path) -> bool:
    """Check if a step's output already exists."""
    if path.is_file():
        return True
    if path.is_dir():
        return any(path.iterdir())
    return False


def setup_resnet_experiment_dir(
    experiment_dir: Path,
    gen_dir: Path,
) -> None:
    """Create experiment dir structure with symlinks for train_resnet.py.

    Creates:
        experiment_dir/real_anomalies/{easy,hard}/imgs/ → VisA source
        experiment_dir/anomaly/generated/{easy,hard}/ → generated images
        experiment_dir/splits_test2.json → copied from canonical source
    """
    visa_easy = CASHEW_BASE / "Data" / "Images" / "Anomaly" / "easy_imgs"
    visa_hard = CASHEW_BASE / "Data" / "Images" / "Anomaly" / "hard_imgs"

    # Real anomalies
    for diff, ids, src_dir in [("easy", ALL_EASY, visa_easy), ("hard", ALL_HARD, visa_hard)]:
        dst_dir = experiment_dir / "real_anomalies" / diff / "imgs"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for aid in ids:
            src = src_dir / f"{aid}.JPG"
            dst = dst_dir / f"{aid}.JPG"
            if not dst.exists() and not dst.is_symlink() and src.exists():
                os.symlink(src.resolve(), dst)

    # Synthetic anomalies
    for diff, ids in [("easy", ALL_EASY), ("hard", ALL_HARD)]:
        dst_dir = experiment_dir / "anomaly" / "generated" / diff
        dst_dir.mkdir(parents=True, exist_ok=True)
        for aid in ids:
            src = gen_dir / f"{aid}.png"
            dst = dst_dir / f"{aid}.png"
            if not dst.exists() and not dst.is_symlink() and src.exists():
                os.symlink(src.resolve(), dst)

    # VisA anomaly masks — train_resnet.py looks for these at
    # experiment_dir.parent / "Data" / "Masks" / "Anomaly" for real images
    # in --crop-to-anomaly and --mask-out-anomaly modes.
    visa_mask_src = CASHEW_BASE / "Data" / "Masks" / "Anomaly"
    visa_mask_dst = experiment_dir.parent / "Data" / "Masks" / "Anomaly"
    if visa_mask_src.exists() and not visa_mask_dst.exists():
        visa_mask_dst.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(visa_mask_src.resolve(), visa_mask_dst)

    # Splits file
    canonical_splits = CASHEW_BASE / "experiment_ResNet" / "splits_test2.json"
    dst_splits = experiment_dir / "splits_test2.json"
    if not dst_splits.exists() and canonical_splits.exists():
        import shutil
        shutil.copy2(canonical_splits, dst_splits)


def main():
    parser = argparse.ArgumentParser(description="Automated post-training evaluation pipeline")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for all evaluation results")
    parser.add_argument("--dust", type=int, default=0,
                        help="Number of dust particles for UniNet (default: 0)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip cashew generation (assumes already done)")
    parser.add_argument("--skip-masks", action="store_true",
                        help="Skip LPIPS mask computation")
    parser.add_argument("--skip-resnet", action="store_true",
                        help="Skip ResNet evaluations")
    parser.add_argument("--skip-resnet-rvs", action="store_true",
                        help="Skip ResNet real-vs-synthetic only")
    parser.add_argument("--skip-resnet-ad", action="store_true",
                        help="Skip ResNet supervised AD only")
    parser.add_argument("--skip-resnet-real-ad", action="store_true",
                        help="Skip ResNet real-train AD (upper bound)")
    parser.add_argument("--skip-uninet", action="store_true",
                        help="Skip UniNet evaluations")
    parser.add_argument("--uninet-modes", type=int, nargs="+", default=None,
                        help="Which UniNet modes to run (e.g. --uninet-modes 1 4). Default: all.")
    parser.add_argument("--uninet-epochs", type=int, default=30,
                        help="Number of UniNet training epochs (default: 30)")
    parser.add_argument("--resnet-epochs", type=int, default=30,
                        help="Number of ResNet training epochs (default: 30)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cj", type=float, default=0.1,
                        help="Training color jitter strength for UniNet + ResNet AD (default: 0.1)")
    parser.add_argument("--train-blur", type=float, default=0.5,
                        help="Training blur max sigma for UniNet + ResNet AD (default: 0.5)")
    parser.add_argument("--train-noise", type=float, default=0.005,
                        help="Training Gaussian noise std for UniNet + ResNet AD (default: 0.005)")
    parser.add_argument("--resnet-backbone", type=str, default="wrn50",
                        choices=["wrn50", "convnext_b"],
                        help="ResNet backbone: wrn50 (Wide-ResNet-50-2) or convnext_b (DINOv3 ConvNeXt-Base)")
    parser.add_argument("--uninet-backbone", type=str, default="wrn50",
                        choices=["wrn50", "convnext_b"],
                        help="UniNet backbone: wrn50 (Wide-ResNet-50-2) or convnext_b (DINOv3 ConvNeXt-Base)")
    parser.add_argument("--uninet-score-mode", type=str, default="max",
                        choices=["max", "paper"],
                        help="UniNet image-level scoring: 'max' (default, backward-compat) or 'paper' "
                             "(mean of adaptive top-k pixels as intended by the UniNet paper)")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    global _GPU_LOG_PATH
    _GPU_LOG_PATH = output_dir / "gpu_log.jsonl"
    gpu_snapshot("pipeline_start")

    if not checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)

    for prep_name, prep_path in [("RP", PREP_RP), ("MO", PREP_MO)]:
        if not prep_path.exists():
            print(f"ERROR: Shared prep dir not found: {prep_path}")
            mode = "rp" if prep_name == "RP" else "mo"
            print(f"  Run: {PYTHON} scripts/generate_100_cashews.py --prep-only "
                  f"--placement-mode {mode} --output-dir results/cashew_100_{mode} --seed 42")
            sys.exit(1)

    t_start = time.time()
    results_tracker = {}
    rp_dir = output_dir / "cashew_100_rp"
    mo_dir = output_dir / "cashew_100_mo"

    # ── Step 1a: Generate 100 cashews _rp ─────────────────────────────────
    # All 3 CFGs (1.0, 3.5, 7.0) are generated in one run automatically.
    rp_gen_check = rp_dir / "generated" / "cfg7_vis"
    if not args.skip_generation and not step_exists(rp_gen_check):
        ok = run_cmd([
            PYTHON, "scripts/generate_100_cashews.py",
            "--checkpoint", str(checkpoint),
            "--reuse-prep", str(PREP_RP),
            "--output-dir", str(rp_dir),
            "--seed", str(args.seed),
        ], "1a: Generate 100 cashews (random-position, all CFGs)")
        if not ok:
            print("Step 1a failed. Aborting.")
            sys.exit(1)
    else:
        print(f"\nSkip 1a: {rp_gen_check} {'exists' if step_exists(rp_gen_check) else '(--skip-generation)'}")

    # ── Step 1b: Generate 100 cashews _mo ─────────────────────────────────
    mo_gen_check = mo_dir / "generated" / "cfg7_vis"
    if not args.skip_generation and not step_exists(mo_gen_check):
        ok = run_cmd([
            PYTHON, "scripts/generate_100_cashews.py",
            "--checkpoint", str(checkpoint),
            "--reuse-prep", str(PREP_MO),
            "--output-dir", str(mo_dir),
            "--seed", str(args.seed),
        ], "1b: Generate 100 cashews (mimic-original, all CFGs)")
        if not ok:
            print("Step 1b failed. Aborting.")
            sys.exit(1)
    else:
        print(f"\nSkip 1b: {mo_gen_check} {'exists' if step_exists(mo_gen_check) else '(--skip-generation)'}")

    # ── Step 2: Compute perceptually-refined masks for both CFG candidates ─
    # Invokes compute_refined_masks.py with no flags → uses current defaults:
    # DINOv3-CNX + α-blend roundtrip + σ=4 + factor=0.6 (as of 2026-04-10).
    # Compute for both _rp and _mo
    for placement, p_dir, prep_dir in [("rp", rp_dir, PREP_RP), ("mo", mo_dir, PREP_MO)]:
        for cand in CFG_CANDIDATES:
            gen_dir = p_dir / "generated" / cand["variant"]
            mask_dir = p_dir / f"refined_masks_{cand['variant']}"
            if not args.skip_masks and step_exists(gen_dir) and not step_exists(mask_dir / "summary.json"):
                ok = run_cmd([
                    PYTHON, "scripts/compute_refined_masks.py",
                    "--gen-dir", str(gen_dir),
                    "--prep-dir", str(prep_dir),
                    "--output-dir", str(mask_dir),
                    "--fg-dir", str(FG_DIR),
                ], f"2: refined masks — DINOv3-CNX + α-blend ({placement}/{cand['variant']})")
                if not ok:
                    print(f"Step 2 failed for {placement}/{cand['variant']}. Continuing...")
            else:
                reason = "exists" if step_exists(mask_dir / "summary.json") else "(skipped or no gen)"
                print(f"\nSkip 2/{placement}/{cand['variant']}: {reason}")

    # ── Step 3: ResNet real-vs-synthetic for BOTH CFGs → pick winner ─────
    # Uses _mo (mimic-original) for fairest quality comparison — natural placement
    cfg_selection = {}
    if not args.skip_resnet and not args.skip_resnet_rvs:
        for cand in CFG_CANDIDATES:
            variant = cand["variant"]
            gen_dir = mo_dir / "generated" / variant  # _mo for RvS
            if not step_exists(gen_dir):
                print(f"\nSkip 3/{variant}: no generated data (mo)")
                continue

            rvs_dir = output_dir / "resnet_eval" / "real_vs_synthetic" / variant
            metrics_path = rvs_dir / "resnet_splits_test2_metrics.json"

            if not step_exists(metrics_path):
                # Create experiment dir with symlinks
                exp_dir = output_dir / f"_tmp_experiment_ResNet_{variant}"
                setup_resnet_experiment_dir(exp_dir, gen_dir)

                rvs_cmd = [
                    PYTHON, "scripts/train_resnet.py",
                    "--experiment-dir", str(exp_dir),
                    "--splits-file", "splits_test2.json",
                    "--epochs", str(args.resnet_epochs),
                    "--seed", str(args.seed),
                    "--backbone", args.resnet_backbone,
                    "--equalize-resolution", "512",
                    "--crop-to-anomaly",
                    "--prep-dir", str(PREP_MO),  # _mo prep for RvS
                    "--save-dir", str(rvs_dir),
                    "--checkpoint-name", checkpoint.name,
                ]
                rvs_mask_dir = mo_dir / f"refined_masks_{variant}"
                if step_exists(rvs_mask_dir):
                    rvs_cmd.extend(["--mask-dir", str(rvs_mask_dir)])
                ok = run_cmd(rvs_cmd, f"3: ResNet real-vs-synthetic ({variant}, mimic-original)")
            else:
                print(f"\nSkip 3/{variant}: {metrics_path} exists")

            # Read result
            if metrics_path.exists():
                with open(metrics_path) as f:
                    m = json.load(f)
                auroc = m.get("auroc", 1.0)
                cfg_selection[variant] = {
                    "cfg": cand["cfg"],
                    "variant": variant,
                    "auroc": auroc,
                    "f1": m.get("f1"),
                    "accuracy": m.get("accuracy"),
                    "confusion_matrix": m.get("confusion_matrix"),
                }
                print(f"  {variant}: AUROC={auroc:.4f}")

    # Select winner (lower AUROC = harder to discriminate = better quality)
    # Tie-break: prefer cfg35_vis (lower CFG → fewer artifacts)
    if cfg_selection:
        winner = min(cfg_selection.values(), key=lambda x: (x["auroc"], x["cfg"]))
        selected_cfg = winner["cfg"]
        selected_variant = winner["variant"]
        print(f"\n  CFG SELECTION: {selected_variant} wins (AUROC={winner['auroc']:.4f})")
    else:
        # Fallback: use CFG=7.0
        selected_cfg = 7.0
        selected_variant = "cfg7_vis"
        print(f"\n  CFG SELECTION: defaulting to {selected_variant} (no RvS results)")

    selected_gen = rp_dir / "generated" / selected_variant
    selected_masks = rp_dir / f"refined_masks_{selected_variant}"

    # ── Step 4: ResNet supervised AD (uses winner CFG) ───────────────────
    resnet_ad_dir = output_dir / "resnet_eval" / "syn_vs_normal"
    if not args.skip_resnet and not args.skip_resnet_ad and step_exists(selected_gen) and not step_exists(resnet_ad_dir / "metrics.json"):
        cmd = [
            PYTHON, "scripts/train_resnet_syn_vs_normal.py",
            "--synth-dir", str(selected_gen),
            "--prep-dir", str(PREP_RP),
            "--save-dir", str(resnet_ad_dir),
            "--epochs", str(args.resnet_epochs),
            "--seed", str(args.seed),
            "--backbone", args.resnet_backbone,
            "--train-cj", str(args.train_cj),
            "--train-blur", str(args.train_blur),
            "--train-noise", str(args.train_noise),
        ]
        if step_exists(selected_masks):
            cmd.extend(["--mask-dir", str(selected_masks)])
        ok = run_cmd(cmd, f"4: ResNet supervised AD ({selected_variant})")
        if ok:
            results_tracker["resnet_ad"] = str(resnet_ad_dir)
    else:
        reason = "exists" if step_exists(resnet_ad_dir / "metrics.json") else "(skipped)"
        print(f"\nSkip 4: ResNet supervised AD {reason}")

    # ── Step 4b: ResNet real-train AD (upper bound) ─────────────────────
    resnet_real_ad_dir = output_dir / "resnet_eval" / "real_vs_normal"
    if not args.skip_resnet and not args.skip_resnet_real_ad and not step_exists(resnet_real_ad_dir / "metrics.json"):
        cmd = [
            PYTHON, "scripts/train_resnet_syn_vs_normal.py",
            "--use-real-train",
            "--prep-dir", str(PREP_RP),
            "--save-dir", str(resnet_real_ad_dir),
            "--epochs", str(args.resnet_epochs),
            "--seed", str(args.seed),
            "--backbone", args.resnet_backbone,
            "--train-cj", str(args.train_cj),
            "--train-blur", str(args.train_blur),
            "--train-noise", str(args.train_noise),
        ]
        ok = run_cmd(cmd, "4b: ResNet real-train AD (upper bound)")
        if ok:
            results_tracker["resnet_real_ad"] = str(resnet_real_ad_dir)
    else:
        reason = "exists" if step_exists(resnet_real_ad_dir / "metrics.json") else "(skipped)"
        print(f"\nSkip 4b: ResNet real-train AD {reason}")

    # ── Step 5: UniNet multi-shift (uses winner CFG) ────────────────────
    if not args.skip_uninet and step_exists(selected_gen):
        uninet_out = output_dir / "uninet_eval"

        for seed in [42, 123]:
            seed_dir = uninet_out / f"seed_{seed}"
            if step_exists(seed_dir / "results.json"):
                print(f"\nSkip 5/seed_{seed}: exists")
                continue

            cmd = [
                PYTHON, str(UNINET_SCRIPT),
                "--placement", "rp",
                "--cfg-variant", selected_variant,
                "--cashew-100-dir", str(rp_dir),
                "--prep-dir", str(PREP_RP),
                "--output-dir", str(seed_dir),
                "--augment",
                "--epochs", str(args.uninet_epochs),
                "--seed", str(seed),
                "--train-cj", str(args.train_cj),
                "--train-blur", str(args.train_blur),
                "--train-noise", str(args.train_noise),
            ]
            if step_exists(selected_masks):
                cmd.extend(["--mask-dir", str(selected_masks)])
            if args.dust > 0:
                cmd.extend(["--train-dust", str(args.dust), "--test-dust", str(args.dust)])
            if args.uninet_modes:
                cmd.extend(["--modes"] + [str(m) for m in args.uninet_modes])
            if args.uninet_backbone != "wrn50":
                cmd.extend(["--backbone", args.uninet_backbone])
            if args.uninet_score_mode != "max":
                cmd.extend(["--score-mode", args.uninet_score_mode])

            ok = run_cmd(cmd, f"5/seed_{seed}: UniNet multi-shift ({selected_variant})")
            if ok:
                results_tracker[f"uninet_seed{seed}"] = str(seed_dir)
    elif args.skip_uninet:
        print("\nSkip 5: UniNet (--skip-uninet)")
    else:
        print("\nSkip 5: UniNet (no generated data)")

    # ── Step 6: Aggregate eval_summary.json ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP: 6. Aggregate eval_summary.json")
    print(f"{'='*70}")

    summary = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "cfg_selection": {
            "selected_cfg": selected_cfg,
            "selected_variant": selected_variant,
            "candidates": cfg_selection,
        },
    }

    # ResNet real-vs-synthetic — include BOTH CFG results
    if cfg_selection:
        summary["resnet_real_vs_synthetic"] = cfg_selection

    # ResNet supervised AD metrics (synthetic train)
    ad_metrics_path = resnet_ad_dir / "metrics.json"
    if ad_metrics_path.exists():
        with open(ad_metrics_path) as f:
            ad = json.load(f)
        summary["resnet_supervised_ad"] = {
            "cfg_used": selected_variant,
            "auroc": ad.get("auroc"),
            "f1": ad.get("f1"),
            "accuracy": ad.get("accuracy"),
            "confusion_matrix": ad.get("confusion_matrix"),
            "multi_shift": ad.get("multi_shift"),
            "img_aurocs": ad.get("img_aurocs"),
        }

    # ResNet real-train AD metrics (upper bound)
    real_ad_metrics_path = resnet_real_ad_dir / "metrics.json"
    if real_ad_metrics_path.exists():
        with open(real_ad_metrics_path) as f:
            rad = json.load(f)
        summary["resnet_real_ad"] = {
            "auroc": rad.get("auroc"),
            "f1": rad.get("f1"),
            "accuracy": rad.get("accuracy"),
            "confusion_matrix": rad.get("confusion_matrix"),
            "multi_shift": rad.get("multi_shift"),
            "img_aurocs": rad.get("img_aurocs"),
        }

    # UniNet metrics — aggregate across seeds (mean ± std)
    import statistics
    uninet_summary = {}
    seed_results_all = {}
    for seed in [42, 123]:
        results_path = output_dir / "uninet_eval" / f"seed_{seed}" / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                seed_results_all[seed] = json.load(f)

    if seed_results_all:
        # Collect all mode names across seeds
        all_mode_names = set()
        for uni in seed_results_all.values():
            for k, v in uni.items():
                if isinstance(v, dict) and "img_aurocs" in v:
                    all_mode_names.add(k)

        for mode_name in sorted(all_mode_names):
            final_aurocs = []
            best_aurocs = []
            final_pix = []
            best_pix = []
            final_obj_pix = []
            best_obj_pix = []
            for seed, uni in seed_results_all.items():
                mode_data = uni.get(mode_name)
                if mode_data and isinstance(mode_data, dict) and mode_data.get("img_aurocs"):
                    final_aurocs.append(mode_data["img_aurocs"][-1])
                    best_aurocs.append(max(mode_data["img_aurocs"]))
                    pix = mode_data.get("pix_aurocs") or [0]
                    final_pix.append(pix[-1])
                    best_pix.append(max(pix))
                    opx = mode_data.get("obj_pix_aurocs") or []
                    if opx:
                        final_obj_pix.append(opx[-1])
                        best_obj_pix.append(max(opx))

            if final_aurocs:
                entry = {
                    "final_img_auroc_mean": float(statistics.mean(final_aurocs)),
                    "best_img_auroc_mean": float(statistics.mean(best_aurocs)),
                    "final_pix_auroc_mean": float(statistics.mean(final_pix)),
                    "best_pix_auroc_mean": float(statistics.mean(best_pix)),
                    "n_seeds": len(final_aurocs),
                    "per_seed": {str(s): {"final": f, "best": b}
                                 for s, f, b in zip(seed_results_all.keys(), final_aurocs, best_aurocs)},
                }
                if final_obj_pix:
                    entry["final_obj_pix_auroc_mean"] = float(statistics.mean(final_obj_pix))
                    entry["best_obj_pix_auroc_mean"] = float(statistics.mean(best_obj_pix))
                if len(final_aurocs) > 1:
                    entry["final_img_auroc_std"] = float(statistics.stdev(final_aurocs))
                    entry["best_img_auroc_std"] = float(statistics.stdev(best_aurocs))
                    entry["final_pix_auroc_std"] = float(statistics.stdev(final_pix))
                    entry["best_pix_auroc_std"] = float(statistics.stdev(best_pix))
                    if len(final_obj_pix) > 1:
                        entry["final_obj_pix_auroc_std"] = float(statistics.stdev(final_obj_pix))
                        entry["best_obj_pix_auroc_std"] = float(statistics.stdev(best_obj_pix))
                uninet_summary[mode_name] = entry

    if uninet_summary:
        summary["uninet"] = {
            "cfg_used": selected_variant,
            **uninet_summary,
        }

    # Save
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {total_time / 60:.1f} min total")
    print(f"  Selected CFG: {selected_cfg} ({selected_variant})")
    print(f"  Summary: {summary_path}")
    print(f"{'='*70}")

    # ── Save results grid plots ──────────────────────────────────────────
    _save_results_grid(summary, seed_results_all, output_dir)


# ---------------------------------------------------------------------------
# Results grid plot — Mode × Shift, both final and best, iAUROC and pAUROC
# ---------------------------------------------------------------------------

_SHIFTS = ["0x", "1x", "2x", "3x", "4x"]

# Short display labels
_SHORT = {
    "Mode 1 — Baseline": "M1 Unsupervised",
    "Mode 2 — Supervised (real)": "M2 Real sup.",
    "Mode 3 — Supervised (synthetic)": "M3 Synth sup.",
    "Mode 4 — Extension": "M4 Extension",
}

# Colors matching run_uninet_cashew.py
_COLORS = {
    "Mode 1 — Baseline": "#888888",
    "Mode 2 — Supervised (real)": "#2a9d8f",
    "Mode 3 — Supervised (synthetic)": "#e63946",
    "Mode 4 — Extension": "#457b9d",
    "ResNet Syn-train": "#ff7f0e",
    "ResNet Real (upper)": "#1f77b4",
}


def _collect_uninet_grid(seed_results_all: dict) -> dict:
    """Collect UniNet data into {mode_key -> {shift -> {img_final, img_best, pix_final, pix_best}}}."""
    base_modes = set()
    for data in seed_results_all.values():
        for key, val in data.items():
            if isinstance(val, dict) and "img_aurocs" in val and "@" not in key:
                base_modes.add(key)

    grid = {}
    for base_mode in sorted(base_modes):
        grid[base_mode] = {}
        for shift in _SHIFTS:
            key = base_mode if shift == "0x" else f"{base_mode} @ {shift}"

            img_final, img_best = [], []
            pix_final, pix_best = [], []
            opx_final, opx_best = [], []
            for data in seed_results_all.values():
                mode_data = data.get(key)
                if not mode_data:
                    continue
                # All shifts now store per-epoch lists (img_aurocs / pix_aurocs / obj_pix_aurocs)
                if mode_data.get("img_aurocs"):
                    img_final.append(mode_data["img_aurocs"][-1])
                    img_best.append(max(mode_data["img_aurocs"]))
                if mode_data.get("pix_aurocs"):
                    pix_final.append(mode_data["pix_aurocs"][-1])
                    pix_best.append(max(mode_data["pix_aurocs"]))
                if mode_data.get("obj_pix_aurocs"):
                    opx_final.append(mode_data["obj_pix_aurocs"][-1])
                    opx_best.append(max(mode_data["obj_pix_aurocs"]))

            if img_final:
                _mean = lambda lst: sum(lst) / len(lst) if lst else None
                grid[base_mode][shift] = {
                    "img_final": _mean(img_final),
                    "img_best": _mean(img_best),
                    "pix_final": _mean(pix_final),
                    "pix_best": _mean(pix_best),
                    "opx_final": _mean(opx_final),
                    "opx_best": _mean(opx_best),
                }
    return grid


def _collect_uninet_trajectories(seed_results_all: dict) -> dict:
    """Collect per-epoch trajectories for ALL keys (base modes + shifts).

    Returns {result_key -> {seed -> {img_aurocs, pix_aurocs}}}.
    Keys include both base modes ("Mode 1 — Baseline") and shift keys
    ("Mode 1 — Baseline @ 1x").
    """
    trajectories = {}
    for seed, data in seed_results_all.items():
        for key, val in data.items():
            if isinstance(val, dict) and val.get("img_aurocs"):
                if key not in trajectories:
                    trajectories[key] = {}
                trajectories[key][seed] = {
                    "img_aurocs": val["img_aurocs"],
                    "pix_aurocs": val["pix_aurocs"],
                }
    return trajectories


def _collect_resnet_grid(summary: dict, agg: str = "best") -> dict:
    """Collect ResNet AD data into {row_label -> {shift -> auroc}}.

    agg: "best" uses max of per-epoch img_aurocs, "final" uses last epoch.
    Falls back to stored auroc if no per-epoch data.
    """
    _pick = max if agg == "best" else (lambda lst: lst[-1])
    grid = {}
    for sum_key, row_label in [("resnet_supervised_ad", "ResNet Syn-train"),
                                ("resnet_real_ad", "ResNet Real (upper)")]:
        ad = summary.get(sum_key)
        if not ad:
            continue
        epoch_aurocs = ad.get("img_aurocs")
        grid[row_label] = {"0x": _pick(epoch_aurocs) if epoch_aurocs else ad.get("auroc", 0)}
        for s, ms in (ad.get("multi_shift") or {}).items():
            shift_aurocs = ms.get("img_aurocs")
            grid[row_label][s] = _pick(shift_aurocs) if shift_aurocs else ms.get("auroc", 0)
    return grid


def _collect_resnet_trajectories(summary: dict) -> dict:
    """Collect ResNet per-epoch trajectories from summary.

    Returns {row_label -> {shift -> {"img_aurocs": [...]}}}
    matching the structure expected by the trajectory plotter.
    """
    trajectories = {}
    for sum_key, row_label in [("resnet_supervised_ad", "ResNet Syn-train"),
                                ("resnet_real_ad", "ResNet Real (upper)")]:
        ad = summary.get(sum_key)
        if not ad:
            continue
        traj = {}
        epoch_aurocs = ad.get("img_aurocs")
        if epoch_aurocs:
            traj["0x"] = epoch_aurocs
        for s, ms in (ad.get("multi_shift") or {}).items():
            shift_aurocs = ms.get("img_aurocs")
            if shift_aurocs:
                traj[s] = shift_aurocs
        if traj:
            trajectories[row_label] = traj
    return trajectories


def _save_results_grid(
    summary: dict,
    seed_results_all: dict,
    output_dir: Path,
) -> None:
    """Save cross plots + trajectory plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not seed_results_all:
        return

    uninet_grid = _collect_uninet_grid(seed_results_all)
    trajectories = _collect_uninet_trajectories(seed_results_all)
    resnet_trajectories = _collect_resnet_trajectories(summary)

    # ── 4 cross plots: {final, best} × {img, pix} ────────────────────
    for agg_key, agg_label in [("final", "Final Epoch"), ("best", "Best Epoch")]:
        resnet_grid = _collect_resnet_grid(summary, agg=agg_key)
        for metric_key, metric_label in [("img", "Image AUROC"), ("pix", "Pixel AUROC")]:
            val_key = f"{metric_key}_{agg_key}"

            row_labels, row_vals, row_colors = [], [], []

            for mode_key in sorted(uninet_grid.keys()):
                vals = []
                for s in _SHIFTS:
                    cell = uninet_grid[mode_key].get(s)
                    vals.append(cell[val_key] if cell and cell.get(val_key) is not None else None)
                if any(v is not None for v in vals):
                    row_labels.append(_SHORT.get(mode_key, mode_key))
                    row_vals.append(vals)
                    row_colors.append(_COLORS.get(mode_key, "#333333"))

            # ResNet rows (img only — no pixel AUROC for ResNet)
            if metric_key == "img":
                for rl in ["ResNet Syn-train", "ResNet Real (upper)"]:
                    if rl in resnet_grid:
                        vals = [resnet_grid[rl].get(s) for s in _SHIFTS]
                        if any(v is not None for v in vals):
                            row_labels.append(rl)
                            row_vals.append(vals)
                            row_colors.append(_COLORS.get(rl, "#333333"))

            if not row_vals:
                continue

            n_rows = len(row_vals)
            n_shifts = len(_SHIFTS)
            x = np.arange(n_shifts)
            bar_w = 0.8 / n_rows

            fig, ax = plt.subplots(figsize=(max(10, n_shifts * 2), 5.5))

            for i, (vals, label, color) in enumerate(zip(row_vals, row_labels, row_colors)):
                positions = x + (i - n_rows / 2 + 0.5) * bar_w
                bar_vals = [v if v is not None else 0 for v in vals]
                alphas = [1.0 if v is not None else 0.15 for v in vals]
                bars = ax.bar(positions, bar_vals, bar_w, label=label, color=color,
                              edgecolor="white", linewidth=0.5)
                for bar, v, a in zip(bars, vals, alphas):
                    bar.set_alpha(a)
                    if v is not None and v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                                f"{v:.3f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels(_SHIFTS, fontsize=11)
            ax.set_xlabel("Distribution Shift", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.set_title(f"{metric_label} ({agg_label}) — Mode × Shift", fontsize=13)
            ax.set_ylim(0, 1.08)
            ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()

            fname = output_dir / f"eval_grid_{agg_key}_{metric_key}_auroc.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {fname}")

    # ── Trajectory plots: per-epoch curves, one subplot per shift ────────
    # Discover base modes (keys without "@")
    base_modes = sorted({k for k in trajectories if "@" not in k})

    for metric_key, metric_label in [("img", "Image AUROC"), ("pix", "Pixel AUROC")]:
        auroc_key = f"{metric_key}_aurocs"

        fig, axes = plt.subplots(1, len(_SHIFTS), figsize=(5 * len(_SHIFTS), 5), sharey=True)

        for ax_idx, shift in enumerate(_SHIFTS):
            ax = axes[ax_idx]

            for mode_key in base_modes:
                color = _COLORS.get(mode_key, "#333333")
                short = _SHORT.get(mode_key, mode_key)
                traj_key = mode_key if shift == "0x" else f"{mode_key} @ {shift}"

                if traj_key not in trajectories:
                    continue

                all_curves = []
                for seed, traj in trajectories[traj_key].items():
                    if traj.get(auroc_key):
                        all_curves.append(traj[auroc_key])
                if not all_curves:
                    continue

                min_len = min(len(c) for c in all_curves)
                trimmed = [c[:min_len] for c in all_curves]
                mean_curve = [sum(vals) / len(vals) for vals in zip(*trimmed)]
                epochs = list(range(1, len(mean_curve) + 1))
                ax.plot(epochs, mean_curve, color=color, label=short, linewidth=1.5)
                if len(trimmed) > 1:
                    lo = [min(vals) for vals in zip(*trimmed)]
                    hi = [max(vals) for vals in zip(*trimmed)]
                    ax.fill_between(epochs, lo, hi, color=color, alpha=0.15)

            # ResNet trajectory lines (img only, single run — no seed bands)
            if metric_key == "img":
                for rl in ["ResNet Syn-train", "ResNet Real (upper)"]:
                    rt = resnet_trajectories.get(rl, {})
                    curve = rt.get(shift)
                    if curve:
                        epochs_r = list(range(1, len(curve) + 1))
                        ax.plot(epochs_r, curve, color=_COLORS.get(rl, "#333333"),
                                label=rl, linewidth=1.5, linestyle="--")

            ax.set_title(shift, fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch")
            if ax_idx == 0:
                ax.set_ylabel(metric_label, fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9)

        fig.suptitle(f"{metric_label} — Training Trajectories per Shift", fontsize=14)
        fig.tight_layout()
        fname = output_dir / f"eval_trajectory_{metric_key}_auroc.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")


if __name__ == "__main__":
    main()
