"""
Run 20 diverse examples through the captioning pipeline and save results for quality review.
Tests: template adherence, English-only output, no Chinese, token count within 77.
"""

import base64
import io
import json
import glob
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw
from scipy import ndimage

OLLAMA_URL = "http://localhost:11434/api/generate"

DEFECT_CODE_MAP = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
}


def get_bboxes_from_mask(mask_path: str):
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    labeled, num_features = ndimage.label(binary)
    bboxes = []
    for i in range(1, num_features + 1):
        comp_ys, comp_xs = np.where(labeled == i)
        if len(comp_xs) < 5:
            continue
        x1, y1 = int(comp_xs.min()), int(comp_ys.min())
        x2, y2 = int(comp_xs.max()), int(comp_ys.max())
        bboxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return bboxes if bboxes else None


def draw_bboxes(image_path, bboxes):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        for offset in range(3):
            draw.rectangle(
                [bb["x1"] - offset, bb["y1"] - offset, bb["x2"] + offset, bb["y2"] + offset],
                outline="red",
            )
    return img


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(product: str) -> str:
    product_name = product.replace("_", " ")
    return (
        "You are a visual inspector of industrial defects, output solely in English "
        "IMPORTANTLY no Chinese. "
        "The defects you will observe are highlighted with bounding boxes. "
        f"The object being inspected is a '{product_name}'. "
        "For each image, you will have to output:\n"
        "The image depicts [general description of the object], with a [type of defect] "
        "observed [location]. The defect is characterized by [detailed description] "
        "and exhibits [notable features]."
    )


def query_ollama(img_b64, prompt, model="qwen2.5vl:7b"):
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 77},
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "text": data.get("response", "").strip(),
        "eval_count": data.get("eval_count", -1),
        "total_duration_ms": data.get("total_duration", 0) / 1e6,
    }


def has_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))


def follows_template(text):
    return text.lower().startswith("the image depicts")


def main():
    random.seed(42)
    base_path = Path("C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets/realiad_1024")

    # Discover all images with masks
    jpg_files = glob.glob(str(base_path / "*/NG/**/*.jpg"), recursive=True)
    entries = []
    for jpg_path in jpg_files:
        png_path = jpg_path.replace(".jpg", ".png")
        if not os.path.exists(png_path):
            continue
        rel = os.path.relpath(jpg_path, base_path)
        product = rel.split(os.sep)[0]
        path_parts = Path(jpg_path).parts
        ng_idx = [i for i, p in enumerate(path_parts) if p == "NG"]
        defect_code = path_parts[ng_idx[0] + 1] if ng_idx else "unknown"
        entries.append({
            "image_path": jpg_path, "mask_path": png_path,
            "product": product, "defect_code": defect_code,
        })

    print(f"Total images with masks: {len(entries)}")

    # Pick 20 diverse examples - spread across products and defect types
    by_product = {}
    for e in entries:
        key = f"{e['product']}_{e['defect_code']}"
        by_product.setdefault(key, []).append(e)

    selected = []
    keys = sorted(by_product.keys())
    random.shuffle(keys)
    for key in keys:
        if len(selected) >= 20:
            break
        selected.append(random.choice(by_product[key]))

    if len(selected) < 20:
        remaining = [e for e in entries if e not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[:20 - len(selected)])

    print(f"Selected {len(selected)} diverse examples\n")

    # Run through 7B model
    results = []
    out_dir = Path("C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/bbox_examples/quality_test_20")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(selected):
        print(f"[{i+1}/20] {entry['product']}/{entry['defect_code']}...", end=" ", flush=True)

        bboxes = get_bboxes_from_mask(entry["mask_path"])
        if bboxes:
            img = draw_bboxes(entry["image_path"], bboxes)
        else:
            img = Image.open(entry["image_path"]).convert("RGB")

        # Save the bbox image for visual review
        img.save(str(out_dir / f"{i+1:02d}_{entry['product']}_{entry['defect_code']}_bbox.jpg"))

        # Save the mask too
        mask_img = Image.open(entry["mask_path"])
        mask_img.save(str(out_dir / f"{i+1:02d}_{entry['product']}_{entry['defect_code']}_mask.png"))

        img_b64 = pil_to_base64(img)
        prompt = build_prompt(entry["product"])

        t0 = time.time()
        resp = query_ollama(img_b64, prompt)
        elapsed = time.time() - t0

        caption = resp["text"]
        tokens = resp["eval_count"]
        chinese = has_chinese(caption)
        template_ok = follows_template(caption)

        print(f"OK ({elapsed:.1f}s, {tokens} tokens, chinese={chinese}, template={template_ok})")
        print(f"  >> {caption[:120]}{'...' if len(caption)>120 else ''}\n")

        results.append({
            "index": i + 1,
            "product": entry["product"],
            "defect_code": entry["defect_code"],
            "defect_name": DEFECT_CODE_MAP.get(entry["defect_code"], entry["defect_code"]),
            "caption": caption,
            "eval_count": tokens,
            "has_chinese": chinese,
            "follows_template": template_ok,
            "time_seconds": round(elapsed, 1),
            "num_bboxes": len(bboxes) if bboxes else 0,
            "image_file": f"{i+1:02d}_{entry['product']}_{entry['defect_code']}_bbox.jpg",
        })

    # Summary stats
    n_chinese = sum(1 for r in results if r["has_chinese"])
    n_template = sum(1 for r in results if r["follows_template"])
    avg_tokens = sum(r["eval_count"] for r in results) / len(results)
    max_tokens = max(r["eval_count"] for r in results)
    n_over_77 = sum(1 for r in results if r["eval_count"] > 77)
    avg_time = sum(r["time_seconds"] for r in results) / len(results)

    summary = {
        "total_examples": len(results),
        "follows_template": f"{n_template}/20",
        "has_chinese": f"{n_chinese}/20",
        "avg_tokens": round(avg_tokens, 1),
        "max_tokens": max_tokens,
        "over_77_tokens": f"{n_over_77}/20",
        "avg_time_seconds": round(avg_time, 1),
    }

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    # Save full results
    output = {"summary": summary, "results": results}
    out_file = str(out_dir / "quality_test_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
