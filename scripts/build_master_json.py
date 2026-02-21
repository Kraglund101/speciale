"""
Build master_training.json for Phase 2 training.

Collects ALL datasets from full_training_dataset/ into one unified JSON:
- 8 AnomVerse subdatasets (BTech, DAGM, eyecandies, MPDD, MulSen_AD, mvtec, mvtec3d, mvtec_ad_2)
- RealIAD (51K+ NG samples)

Output: full_training_dataset/master_training.json

Schema per entry:
{
    "dataset": str,         # e.g. "mvtec", "realiad"
    "product": str,         # e.g. "bottle", "audiojack"
    "defect_type": str,     # e.g. "contamination", "deformation"
    "image_path": str,      # relative to full_training_dataset/
    "mask_path": str,       # relative to full_training_dataset/
    "caption_short": str,   # 1-sentence caption
    "caption_long": str     # paragraph-length caption
}
"""
import json
import os
from pathlib import Path


def build_anomverse(base_dir: Path) -> list[dict]:
    """Read all 8 AnomVerse subdatasets."""
    # Map outer folder → inner folder name
    datasets = {
        "BTech": "BTech",
        "DAGM": "DAGM_anomaly_detection",
        "eyecandies": "eyecandies_preprocessed",
        "MPDD": "MPDD",
        "MulSen_AD": "MulSen_AD",
        "mvtec": "mvtec",
        "mvtec3d": "mvtec3d",
        "mvtec_ad_2": "mvtec_ad_2",
    }

    results = []
    for outer, inner in datasets.items():
        dataset_dir = base_dir / outer / inner
        info_path = dataset_dir / "dataset_info.json"
        if not info_path.exists():
            print(f"  WARN: {info_path} not found, skipping {outer}")
            continue

        with open(info_path) as f:
            data = json.load(f)

        images = data["images"]
        n_ok, n_skip, n_caption_miss = 0, 0, 0
        rel_prefix = f"AnomVerse_data_filtered/{outer}/{inner}"

        for img in images:
            img_path = dataset_dir / img["image_path"]
            mask_path = dataset_dir / img["mask_path"]

            if not img_path.exists() or not mask_path.exists():
                n_skip += 1
                continue

            # Read captions from text files
            caption_long = ""
            caption_short = ""

            analysis_file = img.get("analysis_files", "")
            if analysis_file:
                af_path = dataset_dir / analysis_file
                if af_path.exists():
                    caption_long = af_path.read_text(encoding="utf-8").strip()
                else:
                    n_caption_miss += 1

            analysis_short = img.get("analysis_files_short", "")
            if analysis_short:
                as_path = dataset_dir / analysis_short
                if as_path.exists():
                    caption_short = as_path.read_text(encoding="utf-8").strip()

            results.append({
                "dataset": outer,
                "product": img.get("category", ""),
                "defect_type": img.get("defect_name", ""),
                "image_path": f"{rel_prefix}/{img['image_path']}",
                "mask_path": f"{rel_prefix}/{img['mask_path']}",
                "caption_short": caption_short,
                "caption_long": caption_long,
            })
            n_ok += 1

        print(f"  {outer}: {n_ok} added, {n_skip} skipped (missing files), "
              f"{n_caption_miss} missing long captions")

    return results


def build_realiad(base_dir: Path) -> list[dict]:
    """Read RealIAD from captions_nano_clean.json (has short + long captions)."""
    realiad_dir = base_dir / "realiad_1024"

    # Primary source: captions_nano_clean.json (51,295 entries with short + long)
    nano_path = realiad_dir / "captions_nano_clean.json"
    if not nano_path.exists():
        print(f"  WARN: {nano_path} not found!")
        return []

    with open(nano_path, encoding="utf-8") as f:
        nano_data = json.load(f)

    # Also load captions.json for fallback (51,329 entries, original captions)
    fallback_path = realiad_dir / "captions.json"
    fallback_by_path = {}
    if fallback_path.exists():
        with open(fallback_path) as f:
            for entry in json.load(f):
                fallback_by_path[entry["image_path"]] = entry

    # Build index of nano entries by image_path
    nano_by_path = {}
    for entry in nano_data:
        nano_by_path[entry["image_path"]] = entry

    results = []
    n_ok, n_skip = 0, 0

    # Process nano entries (primary — has both caption types)
    for entry in nano_data:
        img_path = realiad_dir / entry["image_path"]
        mask_path = realiad_dir / entry["mask_path"]

        if not img_path.exists() or not mask_path.exists():
            n_skip += 1
            continue

        results.append({
            "dataset": "realiad",
            "product": entry.get("product", ""),
            "defect_type": entry.get("defect_name", ""),
            "image_path": f"realiad_1024/{entry['image_path']}",
            "mask_path": f"realiad_1024/{entry['mask_path']}",
            "caption_short": entry.get("caption_short", ""),
            "caption_long": entry.get("caption_long", ""),
        })
        n_ok += 1

    # Add fallback entries that weren't in nano (34 entries)
    n_fallback = 0
    for img_key, entry in fallback_by_path.items():
        if img_key in nano_by_path:
            continue
        img_path = realiad_dir / entry["image_path"]
        mask_path = realiad_dir / entry["mask_path"]
        if not img_path.exists() or not mask_path.exists():
            continue
        results.append({
            "dataset": "realiad",
            "product": entry.get("product", ""),
            "defect_type": entry.get("defect_name", ""),
            "image_path": f"realiad_1024/{entry['image_path']}",
            "mask_path": f"realiad_1024/{entry['mask_path']}",
            "caption_short": "",  # no short caption available
            "caption_long": entry.get("caption", ""),  # use original as long
        })
        n_fallback += 1

    print(f"  realiad: {n_ok} from nano_clean, {n_fallback} from fallback, "
          f"{n_skip} skipped (missing files)")

    return results


def main():
    root = Path(__file__).parent.parent
    ft_dir = root / "anomverse_extension" / "datasets" / "full_training_dataset"

    print("Building master_training.json...")
    print()

    # AnomVerse
    print("AnomVerse datasets:")
    anomverse = build_anomverse(ft_dir / "AnomVerse_data_filtered")
    print(f"  TOTAL AnomVerse: {len(anomverse)}")
    print()

    # RealIAD
    print("RealIAD:")
    realiad = build_realiad(ft_dir)
    print(f"  TOTAL RealIAD: {len(realiad)}")
    print()

    # Combine
    master = anomverse + realiad
    print(f"GRAND TOTAL: {len(master)} samples")

    # Stats
    datasets = {}
    for entry in master:
        d = entry["dataset"]
        datasets[d] = datasets.get(d, 0) + 1
    print("\nPer dataset:")
    for d, n in sorted(datasets.items()):
        print(f"  {d}: {n}")

    n_short = sum(1 for e in master if e["caption_short"])
    n_long = sum(1 for e in master if e["caption_long"])
    print(f"\nCaption coverage:")
    print(f"  caption_short: {n_short}/{len(master)} ({100*n_short/len(master):.1f}%)")
    print(f"  caption_long:  {n_long}/{len(master)} ({100*n_long/len(master):.1f}%)")

    # Save
    out_path = ft_dir / "master_training.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
