"""Export only the files needed for training to a flat file list.

Reads master_training.json and writes a list of all image + mask paths
that actually get loaded during training. Use this list with rsync or
scp to transfer only what's needed (~22 GB instead of ~250 GB).

Usage:
    python scripts/export_training_files.py --output training_files.txt
    python scripts/export_training_files.py --output training_files.txt --check-missing
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export training file list")
    parser.add_argument("--output", type=str, default="training_files.txt")
    parser.add_argument("--data-root", type=str,
                        default="anomverse_extension/datasets/full_training_dataset")
    parser.add_argument("--check-missing", action="store_true",
                        help="Report missing files")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / args.data_root
    master_path = data_root / "master_training.json"

    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    files: set[str] = set()
    missing = 0
    for entry in master:
        for key in ("image_path", "mask_path"):
            rel = entry[key]
            files.add(rel)
            if args.check_missing and not (data_root / rel).exists():
                missing += 1
                print(f"  MISSING: {rel}")

    # Sort for deterministic output
    sorted_files = sorted(files)

    output_path = project_root / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        for p in sorted_files:
            f.write(p + "\n")

    print(f"Total unique files: {len(sorted_files)}")
    print(f"Written to: {output_path}")
    if args.check_missing:
        print(f"Missing files: {missing}")


if __name__ == "__main__":
    main()
