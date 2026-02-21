"""Build a zip with remaining ~42K anomaly images + concept JSONs for server."""
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def main():
    project_root = Path(__file__).parent.parent
    base = project_root / "anomverse_extension" / "datasets" / "realiad_1024"
    concepts_10k = base / "concepts_10k"
    staging = project_root / "remaining_data_staging"

    CODE_TO_TYPE = {
        "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
        "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
    }

    print("=== Building Remaining Data Package ===\n")

    # 1. Collect paths already in 10K package
    pkg_paths = set()
    for jf in concepts_10k.glob("*.json"):
        data = json.load(open(jf))
        for s in data["samples"]:
            pkg_paths.add(s["image_path"].replace("\\", "/"))
    print(f"Already in 10K package: {len(pkg_paths)} samples")

    # 2. Find ALL remaining NG images with masks
    remaining = defaultdict(list)
    total_remaining = 0
    no_mask = 0
    no_code = 0

    for jpg in sorted(base.rglob("*.jpg")):
        rel = str(jpg.relative_to(base)).replace("\\", "/")
        if "/NG/" not in rel:
            continue
        if rel in pkg_paths:
            continue

        png = jpg.with_suffix(".png")
        if not png.exists():
            no_mask += 1
            continue

        parts = jpg.relative_to(base).parts
        if len(parts) < 3:
            no_code += 1
            continue

        defect_code = parts[2]
        if defect_code not in CODE_TO_TYPE:
            no_code += 1
            continue

        anomaly_type = CODE_TO_TYPE[defect_code]
        product = parts[0]
        mask_rel = str(png.relative_to(base)).replace("\\", "/")

        remaining[anomaly_type].append({
            "product": product,
            "anomaly_code": defect_code,
            "anomaly_type": anomaly_type,
            "image_path": rel,
            "mask_path": mask_rel,
        })
        total_remaining += 1

    print(f"Remaining to package: {total_remaining} samples")
    print(f"Skipped (no mask): {no_mask}, (unknown code): {no_code}")
    for atype in sorted(remaining):
        print(f"  {atype}: {len(remaining[atype])}")

    # 3. Create staging directory
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # 4. Write concept JSONs for the remaining data
    print(f"\nWriting concept JSONs...")
    concepts_dst = staging / "data" / "concepts_remaining"
    concepts_dst.mkdir(parents=True)
    for atype, samples in sorted(remaining.items()):
        out = {
            "meta": {"defect_type": atype, "count": len(samples)},
            "samples": samples,
        }
        jpath = concepts_dst / f"{atype}.json"
        json.dump(out, open(jpath, "w"), indent=2)
        print(f"  {atype}.json: {len(samples)} samples")

    # 5. Copy images + masks
    print(f"\nCopying {total_remaining} image+mask pairs...")
    images_dst = staging / "data" / "images"
    images_dst.mkdir(parents=True)
    total_size = 0
    copied = 0
    for atype, samples in remaining.items():
        for entry in samples:
            for key in ["image_path", "mask_path"]:
                src = base / entry[key]
                dst = images_dst / entry[key]
                if src.exists() and not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    total_size += src.stat().st_size
            copied += 1
            if copied % 5000 == 0:
                print(f"  {copied}/{total_remaining} ({total_size / 1e9:.1f} GB)")

    print(f"  Total: {copied} samples, {total_size / 1e9:.2f} GB")

    # 6. Zip it
    print(f"\nCreating zip archive...")
    zip_path = project_root / "remaining_data_package"
    shutil.make_archive(str(zip_path), "zip", str(staging))
    zip_size = (project_root / "remaining_data_package.zip").stat().st_size
    print(f"  Created: remaining_data_package.zip ({zip_size / 1e9:.2f} GB)")

    # 7. Cleanup
    print("Cleaning up staging...")
    shutil.rmtree(staging)

    print(f"\nDone! Package: {project_root / 'remaining_data_package.zip'}")


if __name__ == "__main__":
    main()
