"""Restructure experiment_ResNet for 3-way split.

Current: 50 refs (47e+3h) + 50 test (47e+3h)
New:     40 refs (38e+2h) + 40 real_train (38e+2h) + 20 test (18e+2h)

Selects new refs as subset of current refs so existing placed masks stay valid.
"""

import json
import random
import shutil
from pathlib import Path

CASHEW = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy_test\cashew"
)
EXP = CASHEW / "experiment_ResNet"
ORIG_EASY_IMGS = CASHEW / "Data" / "Images" / "Anomaly" / "easy_imgs"
ORIG_HARD_IMGS = CASHEW / "Data" / "Images" / "Anomaly" / "hard_imgs"
ORIG_MASKS = CASHEW / "Data" / "Masks" / "Anomaly"
SEED = 42


def get_stems(d, ext="JPG"):
    return sorted(
        p.stem for p in Path(d).glob(f"*.{ext}")
        if p.is_file()
    )


def copy_anomaly_img(stem, difficulty, dst_dir):
    """Copy anomaly image from original VisA data."""
    src_dir = ORIG_EASY_IMGS if difficulty == "easy" else ORIG_HARD_IMGS
    src = src_dir / f"{stem}.JPG"
    if not src.exists():
        # Try alternate extensions
        for ext in [".jpg", ".png"]:
            alt = src_dir / f"{stem}{ext}"
            if alt.exists():
                src = alt
                break
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def copy_anomaly_mask(stem, dst_dir):
    """Copy anomaly mask from original VisA data."""
    src = ORIG_MASKS / f"{stem}.png"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / src.name)


def main():
    # ── 1. Read current split ────────────────────────────────────────────
    cur_ref_easy = get_stems(EXP / "source" / "reference_anomalies" / "easy" / "imgs")
    cur_ref_hard = get_stems(EXP / "source" / "reference_anomalies" / "hard" / "imgs")
    cur_test_easy = get_stems(EXP / "test_ResNet" / "anomaly" / "easy")
    cur_test_hard = get_stems(EXP / "test_ResNet" / "anomaly" / "hard")

    print(f"Current: refs={len(cur_ref_easy)}e+{len(cur_ref_hard)}h, "
          f"test={len(cur_test_easy)}e+{len(cur_test_hard)}h")
    print(f"Total: {len(cur_ref_easy)+len(cur_test_easy)} easy + "
          f"{len(cur_ref_hard)+len(cur_test_hard)} hard = "
          f"{len(cur_ref_easy)+len(cur_test_easy)+len(cur_ref_hard)+len(cur_test_hard)}")

    # ── 2. New 3-way split (select refs from current refs to preserve placements) ─
    rng = random.Random(SEED)

    ref_easy_shuf = cur_ref_easy.copy()
    rng.shuffle(ref_easy_shuf)
    new_ref_easy = sorted(ref_easy_shuf[:38])
    dropped_ref_easy = sorted(ref_easy_shuf[38:])  # 9

    ref_hard_shuf = cur_ref_hard.copy()
    rng.shuffle(ref_hard_shuf)
    new_ref_hard = sorted(ref_hard_shuf[:2])
    dropped_ref_hard = sorted(ref_hard_shuf[2:])  # 1

    # Pool for real_train + test
    pool_easy = sorted(dropped_ref_easy + cur_test_easy)  # 9 + 47 = 56
    pool_hard = sorted(dropped_ref_hard + cur_test_hard)  # 1 + 3 = 4

    rng2 = random.Random(SEED + 1)
    pool_easy_shuf = pool_easy.copy()
    rng2.shuffle(pool_easy_shuf)
    pool_hard_shuf = pool_hard.copy()
    rng2.shuffle(pool_hard_shuf)

    train_real_easy = sorted(pool_easy_shuf[:38])
    train_real_hard = sorted(pool_hard_shuf[:2])
    test_easy = sorted(pool_easy_shuf[38:])   # 18
    test_hard = sorted(pool_hard_shuf[2:])    # 2

    print(f"\nNew split:")
    print(f"  Refs (synthetic): {len(new_ref_easy)}e + {len(new_ref_hard)}h = {len(new_ref_easy)+len(new_ref_hard)}")
    print(f"  Real train:       {len(train_real_easy)}e + {len(train_real_hard)}h = {len(train_real_easy)+len(train_real_hard)}")
    print(f"  Test:             {len(test_easy)}e + {len(test_hard)}h = {len(test_easy)+len(test_hard)}")
    print(f"  Total:            {len(new_ref_easy)+len(train_real_easy)+len(test_easy)}e + "
          f"{len(new_ref_hard)+len(train_real_hard)+len(test_hard)}h = "
          f"{len(new_ref_easy)+len(train_real_easy)+len(test_easy)+len(new_ref_hard)+len(train_real_hard)+len(test_hard)}")

    # Verify no overlap
    all_easy = set(new_ref_easy) | set(train_real_easy) | set(test_easy)
    all_hard = set(new_ref_hard) | set(train_real_hard) | set(test_hard)
    assert len(all_easy) == 94, f"Easy overlap! {len(all_easy)} != 94"
    assert len(all_hard) == 6, f"Hard overlap! {len(all_hard)} != 6"
    assert not (set(new_ref_easy) & set(train_real_easy)), "Ref/train easy overlap!"
    assert not (set(new_ref_easy) & set(test_easy)), "Ref/test easy overlap!"
    assert not (set(train_real_easy) & set(test_easy)), "Train/test easy overlap!"
    assert not (set(new_ref_hard) & set(train_real_hard)), "Ref/train hard overlap!"
    assert not (set(new_ref_hard) & set(test_hard)), "Ref/test hard overlap!"
    assert not (set(train_real_hard) & set(test_hard)), "Train/test hard overlap!"
    print("  No overlaps — split is clean.")

    # ── 3. Filter existing manifests ─────────────────────────────────────
    with open(EXP / "anomaly" / "masks" / "easy" / "manifest.json") as f:
        old_easy_manifest = json.load(f)
    with open(EXP / "anomaly" / "masks" / "hard" / "manifest.json") as f:
        old_hard_manifest = json.load(f)

    new_ref_easy_set = set(new_ref_easy)
    new_ref_hard_set = set(new_ref_hard)

    new_easy_manifest = [e for e in old_easy_manifest if e["ref_id"] in new_ref_easy_set]
    new_hard_manifest = [e for e in old_hard_manifest if e["ref_id"] in new_ref_hard_set]

    kept_easy_canvas = {e["canvas_id"] for e in new_easy_manifest}
    kept_hard_canvas = {e["canvas_id"] for e in new_hard_manifest}

    print(f"\nManifests: {len(new_easy_manifest)} easy + {len(new_hard_manifest)} hard kept")
    print(f"  Easy canvases: {sorted(kept_easy_canvas)}")
    print(f"  Hard canvases: {sorted(kept_hard_canvas)}")

    # ── 4. Backup old structure ──────────────────────────────────────────
    backup = EXP / "_backup_old_structure"
    if backup.exists():
        print(f"\nWARNING: Backup already exists at {backup}, skipping backup")
    else:
        print(f"\nBacking up test_ResNet/ -> _backup_old_structure/test_ResNet/")
        backup.mkdir()
        shutil.copytree(EXP / "test_ResNet", backup / "test_ResNet")
        # Also backup current manifests
        (backup / "manifests").mkdir()
        shutil.copy2(EXP / "anomaly" / "masks" / "easy" / "manifest.json",
                      backup / "manifests" / "easy_manifest.json")
        shutil.copy2(EXP / "anomaly" / "masks" / "hard" / "manifest.json",
                      backup / "manifests" / "hard_manifest.json")

    # ── 5. Clean up reference_anomalies (remove dropped refs) ────────────
    print("\nUpdating reference_anomalies/...")
    for stem in dropped_ref_easy:
        for subdir in ["imgs", "masks"]:
            d = EXP / "source" / "reference_anomalies" / "easy" / subdir
            for ext in ["JPG", "jpg", "png"]:
                f = d / f"{stem}.{ext}"
                if f.exists():
                    f.unlink()
                    print(f"  Removed {f.name} from easy/{subdir}")
    for stem in dropped_ref_hard:
        for subdir in ["imgs", "masks"]:
            d = EXP / "source" / "reference_anomalies" / "hard" / subdir
            for ext in ["JPG", "jpg", "png"]:
                f = d / f"{stem}.{ext}"
                if f.exists():
                    f.unlink()
                    print(f"  Removed {f.name} from hard/{subdir}")

    # Verify
    remaining_ref_easy = get_stems(EXP / "source" / "reference_anomalies" / "easy" / "imgs")
    remaining_ref_hard = get_stems(EXP / "source" / "reference_anomalies" / "hard" / "imgs")
    print(f"  Refs now: {len(remaining_ref_easy)}e + {len(remaining_ref_hard)}h")

    # ── 6. Clean up placed masks (remove dropped canvases) ───────────────
    print("\nUpdating anomaly/masks/...")
    easy_mask_dir = EXP / "anomaly" / "masks" / "easy"
    hard_mask_dir = EXP / "anomaly" / "masks" / "hard"

    # Remove easy masks for dropped refs
    for entry in old_easy_manifest:
        if entry["ref_id"] not in new_ref_easy_set:
            mask_file = easy_mask_dir / f"{entry['canvas_id']}.png"
            if mask_file.exists():
                mask_file.unlink()
                print(f"  Removed easy mask {mask_file.name} (ref={entry['ref_id']})")

    # Remove hard masks for dropped refs
    for entry in old_hard_manifest:
        if entry["ref_id"] not in new_ref_hard_set:
            mask_file = hard_mask_dir / f"{entry['canvas_id']}.png"
            if mask_file.exists():
                mask_file.unlink()
                print(f"  Removed hard mask {mask_file.name} (ref={entry['ref_id']})")

    # Write updated manifests
    with open(easy_mask_dir / "manifest.json", "w") as f:
        json.dump(new_easy_manifest, f, indent=2)
    with open(hard_mask_dir / "manifest.json", "w") as f:
        json.dump(new_hard_manifest, f, indent=2)
    print(f"  Updated manifests: {len(new_easy_manifest)} easy + {len(new_hard_manifest)} hard")

    # ── 7. Create train_real/ ────────────────────────────────────────────
    print("\nCreating train_real/...")
    train_real_base = EXP / "train_real"
    if train_real_base.exists():
        shutil.rmtree(train_real_base)

    for stem in train_real_easy:
        copy_anomaly_img(stem, "easy", train_real_base / "easy" / "imgs")
        copy_anomaly_mask(stem, train_real_base / "easy" / "masks")
    for stem in train_real_hard:
        copy_anomaly_img(stem, "hard", train_real_base / "hard" / "imgs")
        copy_anomaly_mask(stem, train_real_base / "hard" / "masks")
    print(f"  Created: {len(train_real_easy)}e + {len(train_real_hard)}h")

    # ── 8. Rebuild test/ (replace test_ResNet/) ──────────────────────────
    print("\nRebuilding test/...")
    old_test = EXP / "test_ResNet"
    new_test = EXP / "test"

    if new_test.exists():
        shutil.rmtree(new_test)

    # Copy test normals from old structure
    new_test_normal = new_test / "normal"
    new_test_normal.mkdir(parents=True)
    for f in sorted((old_test / "normal").glob("*.JPG")):
        shutil.copy2(f, new_test_normal / f.name)
    print(f"  Copied {len(list(new_test_normal.glob('*.JPG')))} test normals")

    # Copy test anomalies from original data
    for stem in test_easy:
        copy_anomaly_img(stem, "easy", new_test / "anomaly" / "easy" / "imgs")
        copy_anomaly_mask(stem, new_test / "anomaly" / "easy" / "masks")
    for stem in test_hard:
        copy_anomaly_img(stem, "hard", new_test / "anomaly" / "hard" / "imgs")
        copy_anomaly_mask(stem, new_test / "anomaly" / "hard" / "masks")
    print(f"  Test anomalies: {len(test_easy)}e + {len(test_hard)}h")

    # Remove old test_ResNet/
    shutil.rmtree(old_test)
    print("  Removed old test_ResNet/")

    # ── 9. Save splits.json ──────────────────────────────────────────────
    splits = {
        "seed": SEED,
        "description": "3-way split: refs for synthetic generation, real training, test",
        "reference": {
            "easy": new_ref_easy,
            "hard": new_ref_hard,
            "total": len(new_ref_easy) + len(new_ref_hard),
        },
        "train_real": {
            "easy": train_real_easy,
            "hard": train_real_hard,
            "total": len(train_real_easy) + len(train_real_hard),
        },
        "test": {
            "easy": test_easy,
            "hard": test_hard,
            "total": len(test_easy) + len(test_hard),
        },
        "normals": {
            "train": sorted(get_stems(EXP / "normal")),
            "test": sorted(p.stem for p in (new_test / "normal").glob("*.JPG")),
            "canvas": sorted(get_stems(EXP / "source" / "canvas_normals" / "imgs")),
        },
    }
    with open(EXP / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved splits.json")

    # ── 10. Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Restructuring complete!")
    print(f"  Refs:       {len(new_ref_easy)}e + {len(new_ref_hard)}h = {len(new_ref_easy)+len(new_ref_hard)}")
    print(f"  Placed:     {len(new_easy_manifest)}e + {len(new_hard_manifest)}h = {len(new_easy_manifest)+len(new_hard_manifest)}")
    print(f"  Real train: {len(train_real_easy)}e + {len(train_real_hard)}h = {len(train_real_easy)+len(train_real_hard)}")
    print(f"  Test:       {len(test_easy)}e + {len(test_hard)}h = {len(test_easy)+len(test_hard)}")
    print(f"  Normals:    100 train, 50 test, 50 canvas")
    print(f"\nRef easy IDs:        {new_ref_easy}")
    print(f"Ref hard IDs:        {new_ref_hard}")
    print(f"Dropped ref easy:    {dropped_ref_easy}")
    print(f"Dropped ref hard:    {dropped_ref_hard}")
    print(f"Train real easy IDs: {train_real_easy}")
    print(f"Train real hard IDs: {train_real_hard}")
    print(f"Test easy IDs:       {test_easy}")
    print(f"Test hard IDs:       {test_hard}")


if __name__ == "__main__":
    main()
