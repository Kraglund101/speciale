"""Build combined_datasets.json from all dataset subsets with captions."""
import os
import json

EXT_BASE = "C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets"
ANOMVERSE = "C:/Users/frede/Desktop/kandidat/speciale/data/AnomVerse_data"


def read_caption(path: str) -> str:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def build_mvtec_samples() -> list[dict]:
    src = f"{ANOMVERSE}/mvtec/mvtec"
    dst = f"{EXT_BASE}/mvtec"
    prompt_long = f"{src}/prompt_annotated"
    prompt_short = f"{src}/prompt_short_annotated"

    categories = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
        "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
        "transistor", "wood", "zipper",
    ]

    samples = []
    for cat in categories:
        test_path = f"{src}/{cat}/test"
        if not os.path.exists(test_path):
            continue
        for defect in sorted(os.listdir(test_path)):
            if defect == "good" or not os.path.isdir(f"{test_path}/{defect}"):
                continue
            for img_file in sorted(os.listdir(f"{test_path}/{defect}")):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = os.path.splitext(img_file)[0]
                mask_check = f"{dst}/{cat}/{defect}/{stem}_mask.png"
                if not os.path.exists(mask_check):
                    continue

                caption_long = read_caption(f"{prompt_long}/{cat}/test/{defect}/{stem}_analysis.txt")
                caption_short = read_caption(f"{prompt_short}/{cat}/test/{defect}/{stem}_analysis_short.txt")

                samples.append({
                    "dataset": "mvtec",
                    "product": cat,
                    "defect_type": defect,
                    "image_path": f"mvtec/{cat}/{defect}/{stem}.png",
                    "mask_path": f"mvtec/{cat}/{defect}/{stem}_mask.png",
                    "caption": caption_long,
                    "caption_short": caption_short,
                })
    return samples


def build_mvtec_ad2_samples() -> list[dict]:
    src = f"{ANOMVERSE}/mvtec_ad_2/mvtec_ad_2"
    dst = f"{EXT_BASE}/mvtec_ad2"
    prompt_long = f"{src}/prompt_annotated"
    prompt_short = f"{src}/prompt_short_annotated"

    categories = ["can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"]

    samples = []
    for cat in categories:
        bad_path = f"{src}/{cat}/test_public/bad"
        if not os.path.exists(bad_path):
            continue
        for img_file in sorted(os.listdir(bad_path)):
            if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(img_file)[0]
            mask_check = f"{dst}/{cat}/{stem}_mask.png"
            if not os.path.exists(mask_check):
                continue

            caption_long = read_caption(f"{prompt_long}/{cat}/test_public/bad/{stem}_analysis.txt")
            caption_short = read_caption(f"{prompt_short}/{cat}/test_public/bad/{stem}_analysis_short.txt")

            samples.append({
                "dataset": "mvtec_ad2",
                "product": cat,
                "defect_type": "bad",
                "image_path": f"mvtec_ad2/{cat}/{img_file}",
                "mask_path": f"mvtec_ad2/{cat}/{stem}_mask.png",
                "caption": caption_long,
                "caption_short": caption_short,
            })
    return samples


def build_mvtec3d_samples() -> list[dict]:
    src = f"{ANOMVERSE}/mvtec3d/mvtec3d"
    dst = f"{EXT_BASE}/mvtec3d"
    prompt_long = f"{src}/prompt_annotated"
    prompt_short = f"{src}/prompt_short_annotated"

    categories = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"]

    samples = []
    for cat in categories:
        test_path = f"{src}/{cat}/test"
        if not os.path.exists(test_path):
            continue
        for defect in sorted(os.listdir(test_path)):
            if defect == "good" or not os.path.isdir(f"{test_path}/{defect}"):
                continue
            rgb_dir = f"{test_path}/{defect}/rgb"
            if not os.path.exists(rgb_dir):
                continue
            for img_file in sorted(os.listdir(rgb_dir)):
                if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                stem = os.path.splitext(img_file)[0]
                mask_check = f"{dst}/{cat}/{defect}/{stem}_mask.png"
                if not os.path.exists(mask_check):
                    continue

                # mvtec3d captions are under .../test/{defect}/rgb/
                caption_long = read_caption(f"{prompt_long}/{cat}/test/{defect}/rgb/{stem}_analysis.txt")
                caption_short = read_caption(f"{prompt_short}/{cat}/test/{defect}/rgb/{stem}_analysis_short.txt")

                samples.append({
                    "dataset": "mvtec3d",
                    "product": cat,
                    "defect_type": defect,
                    "image_path": f"mvtec3d/{cat}/{defect}/{stem}.png",
                    "mask_path": f"mvtec3d/{cat}/{defect}/{stem}_mask.png",
                    "caption": caption_long,
                    "caption_short": caption_short,
                })
    return samples


def build_realiad_samples() -> list[dict]:
    captions_path = f"{EXT_BASE}/realiad_1024/captions_v2.json"
    with open(captions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for entry in data:
        samples.append({
            "dataset": "realiad",
            "product": entry.get("product", ""),
            "defect_type": entry.get("defect_name", ""),
            "image_path": f"realiad_1024/{entry['image_path']}",
            "mask_path": f"realiad_1024/{entry['mask_path']}",
            "caption": entry.get("caption", ""),
            "caption_short": "",
        })
    return samples


def main() -> None:
    print("Building MVTec AD samples...")
    mvtec = build_mvtec_samples()
    print(f"  MVTec AD: {len(mvtec)} samples")

    print("Building MVTec AD 2 samples...")
    ad2 = build_mvtec_ad2_samples()
    print(f"  MVTec AD 2: {len(ad2)} samples")

    print("Building MVTec 3D samples...")
    m3d = build_mvtec3d_samples()
    print(f"  MVTec 3D: {len(m3d)} samples")

    print("Building RealIAD samples...")
    realiad = build_realiad_samples()
    print(f"  RealIAD: {len(realiad)} samples")

    all_samples = realiad + mvtec + ad2 + m3d

    combined = {
        "meta": {
            "total_samples": len(all_samples),
            "datasets": {
                "realiad": len(realiad),
                "mvtec": len(mvtec),
                "mvtec_ad2": len(ad2),
                "mvtec3d": len(m3d),
            },
        },
        "samples": all_samples,
    }

    out_path = f"{EXT_BASE}/combined_datasets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    print(f"\nTotal: {len(all_samples)} samples")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
