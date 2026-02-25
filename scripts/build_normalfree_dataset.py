"""Build a normal-free zip of the dataset for server transfer.

Zips only anomaly images + masks referenced in master_training.json,
plus JSON metadata, captions, splits, and the VisA validation dataset.
Skips all normal/good images (~211 GB savings).

The zip preserves the exact directory structure under anomverse_extension/datasets/
so extracting on the server gives the same paths.

Output: anomverse_extension_normalfree.zip (~25 GB)

Usage:
    python scripts/build_normalfree_dataset.py [--output anomverse_extension_normalfree.zip]
"""
import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build normal-free dataset zip")
    parser.add_argument("--output", type=str,
                        default="anomverse_extension_normalfree.zip",
                        help="Output zip filename (relative to project root)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    datasets_root = project_root / "anomverse_extension" / "datasets"
    training_root = datasets_root / "full_training_dataset"
    visa_root = datasets_root / "VisA_validation_dataset"
    out_path = project_root / args.output

    # --- Collect anomaly files from master_training.json ---
    print("Loading master_training.json...")
    with open(training_root / "master_training.json", encoding="utf-8") as f:
        entries = json.load(f)

    files_to_include = set()
    for entry in entries:
        files_to_include.add(training_root / entry["image_path"])
        files_to_include.add(training_root / entry["mask_path"])

    print(f"  {len(entries)} entries -> {len(files_to_include)} unique image+mask files")

    # --- Add JSON metadata ---
    for jf in ["master_training.json", "captions_from_master.json"]:
        p = training_root / jf
        if p.exists():
            files_to_include.add(p)

    # Add splits_by_type/*.json
    splits_dir = training_root / "splits_by_type"
    if splits_dir.exists():
        for p in splits_dir.glob("*.json"):
            files_to_include.add(p)

    # --- Add VisA validation dataset (everything) ---
    visa_files = []
    if visa_root.exists():
        visa_files = [p for p in visa_root.rglob("*") if p.is_file()]
        print(f"  VisA validation: {len(visa_files)} files")
        for p in visa_files:
            files_to_include.add(p)
    else:
        print(f"  WARNING: VisA validation not found at {visa_root}")

    # --- Build zip ---
    # All paths stored relative to anomverse_extension/
    anomverse_root = project_root / "anomverse_extension"
    total_files = len(files_to_include)
    print(f"\nCreating {out_path.name} with {total_files} files...")

    written = 0
    missing = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_STORED) as zf:
        for fpath in sorted(files_to_include):
            if not fpath.exists():
                missing += 1
                continue
            arcname = fpath.relative_to(anomverse_root)
            zf.write(fpath, str(arcname))
            written += 1
            if written % 10000 == 0:
                print(f"  {written}/{total_files} files...")

    size = out_path.stat().st_size
    print(f"\nDone: {out_path}")
    print(f"  {written} files written, {missing} missing")
    print(f"  Size: {size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
