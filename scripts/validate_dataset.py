"""
Validate dataset integrity: detect images that are actually masks.

Checks every image in the splits for:
1. Binary images (only 0/255 values) = mask loaded as image
2. Very low unique pixel count (< 10 unique values)
3. Mean pixel value < 30 (mostly black)
4. Image path containing 'mask' or 'ground_truth' keywords
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent


def check_image(image_path: str) -> dict:
    """Check if an image looks like a mask."""
    full_path = PROJECT_ROOT / image_path

    if not full_path.exists():
        return {"exists": False, "error": f"File not found: {full_path}"}

    try:
        img = Image.open(full_path)
        arr = np.array(img)
    except Exception as e:
        return {"exists": False, "error": str(e)}

    # Convert to grayscale for analysis
    if len(arr.shape) == 3:
        gray = arr.mean(axis=2)
    else:
        gray = arr.astype(float)

    unique_vals = len(np.unique(arr))
    mean_val = gray.mean()

    # Check for binary image (mask-like)
    is_binary = unique_vals <= 5
    is_mostly_black = mean_val < 20
    is_low_unique = unique_vals < 10

    # Check for suspicious path patterns
    path_lower = image_path.lower()
    has_mask_in_path = "mask" in path_lower
    has_gt_in_path = "ground_truth" in path_lower

    issues = []
    if is_binary:
        issues.append(f"BINARY: only {unique_vals} unique values")
    if is_mostly_black:
        issues.append(f"MOSTLY_BLACK: mean={mean_val:.1f}")
    if is_low_unique and not is_binary:
        issues.append(f"LOW_UNIQUE: {unique_vals} unique values")
    if has_mask_in_path:
        issues.append(f"PATH_CONTAINS_MASK")
    if has_gt_in_path:
        issues.append(f"PATH_CONTAINS_GROUND_TRUTH")

    return {
        "exists": True,
        "unique_vals": unique_vals,
        "mean_val": mean_val,
        "size": arr.shape,
        "mode": img.mode,
        "issues": issues,
        "is_suspicious": len(issues) > 0,
    }


def main():
    splits_dir = PROJECT_ROOT / "data" / "concepts"
    exclude_sources = {"MANTA_TINY_256", "MulSen_AD"}

    json_files = sorted(splits_dir.glob("*.json"))

    all_issues = []
    stats = defaultdict(int)

    for json_path in json_files:
        anomaly_type = json_path.stem

        with open(json_path) as f:
            data = json.load(f)

        images = data.get("images", [])
        checked = 0
        type_issues = 0

        for img_data in images:
            source = img_data.get("source_dataset", "")
            if source in exclude_sources:
                continue

            image_path = img_data.get("image_path_full") or img_data.get("image_path_abs") or img_data.get("image_path")
            mask_path = img_data.get("mask_path_full") or img_data.get("mask_path_abs") or img_data.get("mask_path")

            if not image_path:
                continue

            checked += 1
            result = check_image(image_path)
            stats["total_checked"] += 1

            if not result["exists"]:
                stats["missing_files"] += 1
                all_issues.append({
                    "type": anomaly_type,
                    "source": source,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "issue": result.get("error", "MISSING"),
                })
                type_issues += 1
                continue

            if result["is_suspicious"]:
                stats["suspicious"] += 1
                all_issues.append({
                    "type": anomaly_type,
                    "source": source,
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "issues": result["issues"],
                    "unique_vals": result["unique_vals"],
                    "mean_val": result["mean_val"],
                    "size": result["size"],
                    "mode": result["mode"],
                })
                type_issues += 1

        status = f"  {anomaly_type}: checked {checked}"
        if type_issues > 0:
            status += f" - {type_issues} ISSUES FOUND"
        print(status)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total images checked: {stats['total_checked']}")
    print(f"Missing files: {stats['missing_files']}")
    print(f"Suspicious images: {stats['suspicious']}")

    if all_issues:
        print(f"\n{'=' * 70}")
        print(f"DETAILED ISSUES ({len(all_issues)} total)")
        print("=" * 70)

        for issue in all_issues:
            print(f"\n  Type: {issue['type']}")
            print(f"  Source: {issue['source']}")
            print(f"  Image: {issue['image_path']}")
            print(f"  Mask:  {issue.get('mask_path', 'N/A')}")
            if "issues" in issue:
                for iss in issue["issues"]:
                    print(f"    -> {iss}")
                print(f"    Unique vals: {issue.get('unique_vals')}, Mean: {issue.get('mean_val', 0):.1f}, Size: {issue.get('size')}, Mode: {issue.get('mode')}")
            else:
                print(f"    -> {issue.get('issue', 'UNKNOWN')}")
    else:
        print("\nNo issues found! All images look legitimate.")

    # Save results
    out_path = PROJECT_ROOT / "results" / "dataset_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "stats": dict(stats),
            "issues": all_issues,
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
