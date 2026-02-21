"""Build a zip with remaining RealIAD + ALL MVTec anomaly images with concept JSONs."""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image


# MVTec fine-grained defect -> concept type mapping (from old data/concepts/*.json)
MVTEC_DEFECT_TO_CONCEPT = {
    "bent": "bent", "bent_lead": "bent", "bent_wire": "bent",
    "broken": "broken", "broken_large": "broken", "broken_small": "broken",
    "broken_teeth": "broken",
    "contamination": "contamination", "metal_contamination": "contamination",
    "crack": "crack",
    "fold": "crease",
    "cut": "cut", "cut_inner_insulation": "cut", "cut_lead": "cut",
    "cut_outer_insulation": "cut",
    "damaged_case": "damage", "combined": "damage", "defective": "damage",
    "fabric_border": "damage", "fabric_interior": "damage",
    "manipulated_front": "damage",
    "squeeze": "dent", "squeezed_teeth": "dent",
    "glue": "glue", "glue_strip": "glue",
    "hole": "hole", "poke": "hole", "poke_insulation": "hole",
    "liquid": "liquid", "oil": "liquid",
    "cable_swap": "misalignment", "flip": "misalignment",
    "misplaced": "misalignment",
    "missing_cable": "missing", "missing_wire": "missing",
    "faulty_imprint": "print_defect", "gray_stroke": "print_defect",
    "print": "print_defect",
    "rough": "rough",
    "scratch": "scratch", "scratch_head": "scratch", "scratch_neck": "scratch",
    "split_teeth": "split",
    "color": "stain", "pill_type": "stain",
    "thread": "thread", "thread_side": "thread", "thread_top": "thread",
}

# RealIAD defect code -> concept type
REALIAD_CODE_TO_TYPE = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects",
    "ZW": "contamination",
}


def main():
    project_root = Path(__file__).parent.parent
    realiad_base = project_root / "anomverse_extension" / "datasets" / "realiad_1024"
    mvtec_base = project_root / "anomverse_extension" / "datasets" / "mvtec"
    concepts_10k = realiad_base / "concepts_10k"
    staging = project_root / "remaining_data_staging"
    min_size = 500

    print("=== Building Combined Remaining Package (RealIAD + MVTec) ===\n")

    # Clean staging
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    # ---- PART 1: Remaining RealIAD ----
    print("--- Part 1: Remaining RealIAD ---")

    # Collect 10K paths already on server
    pkg_paths = set()
    for jf in concepts_10k.glob("*.json"):
        data = json.load(open(jf))
        for s in data["samples"]:
            pkg_paths.add(s["image_path"].replace("\\", "/"))
    print(f"Already on server: {len(pkg_paths)} RealIAD samples")

    # Scan remaining RealIAD
    realiad_concepts = defaultdict(list)
    realiad_count = 0
    for jpg in sorted(realiad_base.rglob("*.jpg")):
        parts = jpg.relative_to(realiad_base).parts
        if len(parts) < 3 or parts[1] != "NG":
            continue
        rel = str(jpg.relative_to(realiad_base)).replace("\\", "/")
        if rel in pkg_paths:
            continue
        png = jpg.with_suffix(".png")
        if not png.exists():
            continue
        code = parts[2]
        if code not in REALIAD_CODE_TO_TYPE:
            continue

        anomaly_type = REALIAD_CODE_TO_TYPE[code]
        realiad_concepts[anomaly_type].append({
            "product": parts[0],
            "anomaly_type": anomaly_type,
            "source": "realiad",
            "image_path": f"realiad/{rel}",
            "mask_path": f"realiad/{str(png.relative_to(realiad_base)).replace(chr(92), '/')}",
            "_src_img": str(jpg),
            "_src_mask": str(png),
        })
        realiad_count += 1

    print(f"Remaining RealIAD: {realiad_count} samples")
    for atype in sorted(realiad_concepts):
        print(f"  {atype}: {len(realiad_concepts[atype])}")

    # ---- PART 2: MVTec (ALL, >= 500x500) ----
    print(f"\n--- Part 2: MVTec (>= {min_size}x{min_size}) ---")

    mvtec_concepts = defaultdict(list)
    mvtec_count = 0
    mvtec_skipped = 0

    for img in sorted(mvtec_base.rglob("*.png")):
        if img.stem.endswith("_mask"):
            continue
        mask = img.parent / f"{img.stem}_mask.png"
        if not mask.exists():
            continue

        # Check size
        with Image.open(img) as im:
            w, h = im.size
        if w < min_size or h < min_size:
            mvtec_skipped += 1
            continue

        parts = img.relative_to(mvtec_base).parts
        product = parts[0]
        defect_type = parts[1]
        concept = MVTEC_DEFECT_TO_CONCEPT.get(defect_type)
        if not concept:
            print(f"  WARNING: unmapped defect '{defect_type}' in {product}")
            continue

        rel_img = str(img.relative_to(mvtec_base)).replace("\\", "/")
        rel_mask = str(mask.relative_to(mvtec_base)).replace("\\", "/")

        mvtec_concepts[concept].append({
            "product": product,
            "anomaly_type": concept,
            "original_defect": defect_type,
            "source": "mvtec",
            "image_path": f"mvtec/{rel_img}",
            "mask_path": f"mvtec/{rel_mask}",
            "_src_img": str(img),
            "_src_mask": str(mask),
        })
        mvtec_count += 1

    print(f"MVTec valid: {mvtec_count} samples (skipped {mvtec_skipped} < {min_size}px)")
    for ctype in sorted(mvtec_concepts):
        print(f"  {ctype}: {len(mvtec_concepts[ctype])}")

    # ---- PART 3: Write concept JSONs ----
    print(f"\n--- Writing concept JSONs ---")
    concepts_dst = staging / "data" / "concepts_extra"
    concepts_dst.mkdir(parents=True)

    all_types = set(list(realiad_concepts.keys()) + list(mvtec_concepts.keys()))
    for atype in sorted(all_types):
        samples = []
        for s in realiad_concepts.get(atype, []):
            entry = {k: v for k, v in s.items() if not k.startswith("_")}
            samples.append(entry)
        for s in mvtec_concepts.get(atype, []):
            entry = {k: v for k, v in s.items() if not k.startswith("_")}
            samples.append(entry)

        out = {
            "meta": {
                "defect_type": atype,
                "count": len(samples),
                "sources": list(set(s["source"] for s in samples)),
            },
            "samples": samples,
        }
        json.dump(out, open(concepts_dst / f"{atype}.json", "w"), indent=2)
        src_breakdown = defaultdict(int)
        for s in samples:
            src_breakdown[s["source"]] += 1
        parts_str = ", ".join(f"{k}={v}" for k, v in sorted(src_breakdown.items()))
        print(f"  {atype}.json: {len(samples)} ({parts_str})")

    # ---- PART 4: Copy images ----
    print(f"\n--- Copying images ---")
    images_dst = staging / "data" / "images"
    images_dst.mkdir(parents=True)

    total_size = 0
    copied = 0
    all_samples = []
    for atype in all_types:
        all_samples.extend(realiad_concepts.get(atype, []))
        all_samples.extend(mvtec_concepts.get(atype, []))

    total_to_copy = len(all_samples)
    for s in all_samples:
        for src_key, path_key in [("_src_img", "image_path"), ("_src_mask", "mask_path")]:
            src = Path(s[src_key])
            dst = images_dst / s[path_key]
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                total_size += src.stat().st_size
        copied += 1
        if copied % 5000 == 0:
            print(f"  {copied}/{total_to_copy} ({total_size / 1e9:.1f} GB)")

    print(f"  Total: {copied} samples, {total_size / 1e9:.2f} GB")

    # ---- PART 5: Zip ----
    print(f"\nCreating zip...")
    zip_path = project_root / "remaining_plus_mvtec_package"
    shutil.make_archive(str(zip_path), "zip", str(staging))
    zip_size = (project_root / "remaining_plus_mvtec_package.zip").stat().st_size
    print(f"  Created: remaining_plus_mvtec_package.zip ({zip_size / 1e9:.2f} GB)")

    print("\nCleaning up staging...")
    shutil.rmtree(staging)

    print(f"\nDone!")
    print(f"  RealIAD remaining: {realiad_count}")
    print(f"  MVTec: {mvtec_count}")
    print(f"  Combined: {realiad_count + mvtec_count} samples")


if __name__ == "__main__":
    main()
