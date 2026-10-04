#!/usr/bin/env python3
"""Data package for running the open-set experiment on another machine (B200 / UCloud). Code comes from git; this
collects everything else the pipeline reads: datasets, donor data, the generator checkpoint, the thesis-split set the
UniNet runner loads at start-up, and the pretrained models (Hugging Face + torch hub caches, de-referenced so the
snapshot folders hold real files).

  python scripts/build_b200_package.py stage --dst C:/b200_test/speciale     copy the data into a project tree
  python scripts/build_b200_package.py zip   --dst C:/b200_test/speciale --zip C:/b200_test/speciale_b200_data.zip

Layout inside the zip = paths relative to the project root, plus _cache/huggingface and _cache/torch.
On the target:  unzip inside the cloned repo (folder must be named "speciale"), then
  export HF_HOME=$PWD/_cache/huggingface TORCH_HOME=$PWD/_cache/torch HF_HUB_OFFLINE=1
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
C = "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew"
V = "anomverse_extension/datasets/VisA_validation_dataset/restVisA/VisA_20220922"
F = "anomverse_extension/datasets/full_training_dataset"
ITEMS = [  # project-relative files / folders
    f"{C}/Data", f"{C}/birefnet_masks", f"{C}/birefnet_masks_anomaly", f"{C}/image_anno.csv",
    f"{C}/experiment_UniNet/splits.json", f"{C}/experiment_UniNet/source", f"{C}/experiment_UniNet/synthetic",
    f"{C}/experiment_UniNet/temp_uninet_experiment/run_uninet_cashew.py",
    f"{C}/experiment_ResNet/source", f"{C}/experiment_ResNet/splits.json",
    f"{V}/fryum", f"{V}/macaroni1", f"{V}/macaroni2", f"{V}/pipe_fryum",
    f"{F}/master_training.json", f"{F}/captions_from_master.json", f"{F}/realiad_1024/fire_hood",
    "datasets/dtd",
    "results/generator_nocashew_20k_constlr/checkpoint_20000",
    "results/masked_patch_knn/pairing_constrained.json",
    "results/thesis_set_clean20k_leakfree", "results/cashew_100_leakfree/prep",
    "results/open_set_v2/PLAN.md", "results/open_set_v2/GENERATION_PLAN.md", "results/open_set_v2/anomaly_plan_b3.json",
]
HF = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
HF_MODELS = ["models--runwayml--stable-diffusion-inpainting", "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
             "models--h94--IP-Adapter", "models--timm--convnext_base.dinov3_lvd1689m"]
HF_SKIP = ("sdxl_models",)                                     # sub-folders of a snapshot that the SD 1.5 pipeline never reads
TORCH = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch")) / "hub/checkpoints"
TORCH_FILES = ["wide_resnet50_2-95faca4d.pth"]
DATA_TOP = ["anomverse_extension", "datasets", "results", "_cache"]   # what goes into the zip (code is in git)


def copy(src: Path, dst: Path) -> int:
    """Copy a file or tree, following symlinks, skipping caches; returns bytes copied (existing same-size files kept)."""
    n = 0
    if src.is_file():
        files = [(src, dst)]
    else:
        files = [(p, dst / p.relative_to(src)) for p in src.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    for s, d in files:
        size = s.stat().st_size
        if not (d.exists() and d.stat().st_size == size):
            d.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(s, d)
        n += size
    return n


def stage(dst: Path) -> None:
    total = 0
    for it in ITEMS:
        assert (ROOT / it).exists(), f"missing {it}"
        b = copy(ROOT / it, dst / it); total += b; print(f"{b / 1e6:9.0f} MB  {it}", flush=True)
    for m in HF_MODELS:
        src = HF / m; assert (src / "snapshots").is_dir(), f"missing HF model {m}"; b = 0
        b += copy(src / "refs", dst / "_cache/huggingface/hub" / m / "refs")
        for p in (src / "snapshots").rglob("*"):
            if p.is_file() and not any(s in p.parts for s in HF_SKIP):
                b += copy(p, dst / "_cache/huggingface/hub" / m / "snapshots" / p.relative_to(src / "snapshots"))
        total += b; print(f"{b / 1e6:9.0f} MB  _cache/huggingface/hub/{m}", flush=True)
    for f in TORCH_FILES:
        b = copy(TORCH / f, dst / "_cache/torch/hub/checkpoints" / f); total += b; print(f"{b / 1e6:9.0f} MB  _cache/torch/hub/checkpoints/{f}")
    print(f"staged {total / 1e9:.1f} GB under {dst}")


def make_zip(dst: Path, out: Path) -> None:
    keep = {str((dst / it)).lower() for it in ITEMS}; n = b = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for it in ITEMS + ["_cache"]:                            # only the packaged inputs, never run outputs
            p = dst / it
            for f in ([p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())):
                z.write(f, f.relative_to(dst).as_posix()); n += 1; b += f.stat().st_size
    print(f"{out}: {n} files, {b / 1e9:.1f} GB ({out.stat().st_size / 1e9:.1f} GB on disk)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["stage", "zip"]); ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--zip", type=Path); a = ap.parse_args()
    stage(a.dst) if a.cmd == "stage" else make_zip(a.dst, a.zip)
