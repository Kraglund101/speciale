"""Build a zip package with training code + 10K sample images for server deployment."""
import json
import glob
import os
import shutil
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    base_data = project_root / "anomverse_extension" / "datasets" / "realiad_1024"
    concepts_dir = base_data / "concepts_10k"
    staging = project_root / "package_staging"

    print("=== Building AnomalyDiffusion Server Package ===\n")

    # Clean staging
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # --- 1. Code files ---
    print("1. Copying source code...")
    code_dirs = ["src"]
    for d in code_dirs:
        src = project_root / d
        dst = staging / d
        if src.exists():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            n = sum(1 for _ in dst.rglob("*.py"))
            print(f"   {d}/: {n} .py files")

    # --- 2. Training scripts ---
    print("2. Copying training scripts...")
    scripts_dst = staging / "scripts"
    scripts_dst.mkdir()
    essential_scripts = [
        "train_multi_anomaly.py",
        "train_anomagic.py",
        "train_scratch_full.py",
        "end_to_end_test.py",
        "validate_dataset.py",
        "generate_anomalies.py",
        "evaluate_generation.py",
        "live_loss.py",
        "generate_captions_v2.py",
        "compress_captions.py",
        "build_combined_json.py",
    ]
    for s in essential_scripts:
        src = project_root / "scripts" / s
        if src.exists():
            shutil.copy2(src, scripts_dst / s)
            print(f"   scripts/{s}")

    # --- 3. Configs ---
    print("3. Copying configs...")
    configs_src = project_root / "configs"
    if configs_src.exists():
        shutil.copytree(configs_src, staging / "configs",
                        ignore=shutil.ignore_patterns("__pycache__"))
        n = sum(1 for _ in (staging / "configs").rglob("*.yaml"))
        print(f"   configs/: {n} .yaml files")

    # --- 4. Concept JSONs ---
    print("4. Copying concept JSONs...")
    concepts_dst = staging / "data" / "concepts_10k"
    concepts_dst.mkdir(parents=True)
    for jf in sorted(concepts_dir.glob("*.json")):
        shutil.copy2(jf, concepts_dst / jf.name)
        print(f"   {jf.name}")

    # --- 5. Copy referenced images ---
    print("\n5. Copying 10K referenced images + masks...")
    images_dst = staging / "data" / "images"
    images_dst.mkdir(parents=True)

    total_copied = 0
    total_size = 0
    for jf in sorted(concepts_dir.glob("*.json")):
        with open(jf) as f:
            data = json.load(f)
        atype = data["meta"]["defect_type"]
        count = 0
        for entry in data["samples"]:
            for key in ["image_path", "mask_path"]:
                src_path = base_data / entry[key]
                dst_path = images_dst / entry[key]
                if src_path.exists() and not dst_path.exists():
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    total_size += src_path.stat().st_size
            count += 1
        total_copied += count
        print(f"   {atype}: {count} samples")

    print(f"   Total: {total_copied} images, {total_size / 1e9:.1f} GB")

    # --- 6. Captions ---
    print("\n6. Copying captions...")
    captions_dst = staging / "data"
    for cap_name in ["captions.json", "captions_v2.json"]:
        cap_src = base_data / cap_name
        if cap_src.exists():
            shutil.copy2(cap_src, captions_dst / cap_name)
            print(f"   {cap_name} ({cap_src.stat().st_size / 1024:.0f} KB)")

    # Paths in JSONs are already relative (e.g., "bottle_cap/NG/HS/..."),
    # just need to set --data-root to data/images/ when training

    # --- 7. Requirements ---
    print("\n7. Writing requirements.txt...")
    reqs = staging / "requirements.txt"
    reqs.write_text(
        "torch==2.3.0\n"
        "torchvision==0.18.0\n"
        "diffusers==0.32.2\n"
        "transformers==4.44.2\n"
        "accelerate>=0.24\n"
        "pillow>=9.0\n"
        "numpy>=1.21\n"
        "scipy>=1.7\n"
        "tqdm>=4.60\n"
        "pyyaml>=5.4\n"
        "prodigyopt>=1.0\n"
        "open-clip-torch>=2.20\n"
        "matplotlib>=3.5\n"
    )

    # --- 8. README ---
    print("8. Writing README...")
    readme = staging / "README.md"
    readme.write_text(
        "# AnomalyDiffusion Training Package\n\n"
        "## Quick Start\n\n"
        "```bash\n"
        "pip install -r requirements.txt\n\n"
        "# Phase 1: Textual Inversion (paper settings)\n"
        "python scripts/train_multi_anomaly.py \\\n"
        "    --splits-dir data/concepts_10k \\\n"
        "    --data-root data/images \\\n"
        "    --save-dir results/phase1 \\\n"
        "    --steps 100000 \\\n"
        "    --lr 0.005 \\\n"
        "    --batch-size 4\n\n"
        "# Phase 2: Anomagic (IP-Adapter + concept tokens)\n"
        "python scripts/train_anomagic.py \\\n"
        "    --splits-dir data/concepts_10k \\\n"
        "    --data-root data/images \\\n"
        "    --captions-file data/captions_v2.json \\\n"
        "    --phase1-dir results/phase1 \\\n"
        "    --save-dir results/phase2 \\\n"
        "    --ip-adapter-type plus \\\n"
        "    --ip-adapter-k 4\n"
        "```\n\n"
        "## Dataset\n\n"
        "- 10K anomaly images across 8 types and 32 products\n"
        "- Types: abrasion, contamination, damage, deformation, "
        "foreign_objects, missing_parts, pit, scratch\n"
        "- Images: 1024x1024 JPEG with binary PNG masks\n"
        "- Captions: Per-image text descriptions (captions_v2.json)\n\n"
        "## Structure\n\n"
        "```\n"
        "src/             - Model code (spatial encoder, SD pipeline, IP-Adapter, trainer)\n"
        "scripts/         - Training entry points\n"
        "configs/         - YAML configs\n"
        "data/\n"
        "  concepts_10k/  - Per-type JSON annotations\n"
        "  images/        - Product directories with NG images + masks\n"
        "  captions*.json - Per-image text captions\n"
        "```\n"
    )

    # --- 9. Zip it ---
    print(f"\n9. Creating zip archive...")
    zip_path = project_root / "anomaly_diffusion_package"
    shutil.make_archive(str(zip_path), "zip", str(staging))
    zip_size = (project_root / "anomaly_diffusion_package.zip").stat().st_size
    print(f"   Created: anomaly_diffusion_package.zip ({zip_size / 1e9:.1f} GB)")

    # Cleanup staging
    print("\n10. Cleaning up staging directory...")
    shutil.rmtree(staging)

    print(f"\nDone! Package at: {project_root / 'anomaly_diffusion_package.zip'}")


if __name__ == "__main__":
    main()
