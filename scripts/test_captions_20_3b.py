"""Run same 20 examples through 3B model for comparison with 7B results."""

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


def get_bboxes_from_mask(mask_path):
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


def build_prompt(product):
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


def query_ollama(img_b64, prompt, model):
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


def is_broken(text):
    """Check if output contains raw template brackets or is garbage."""
    return "[" in text and "]" in text and text.count("[") >= 2


def main():
    random.seed(42)  # Same seed as 7B test = same 20 examples
    base_path = Path("C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/datasets/realiad_1024")

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

    out_dir = Path("C:/Users/frede/Desktop/kandidat/speciale/anomverse_extension/bbox_examples/quality_test_20")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load 7B results for comparison
    results_7b_path = out_dir / "quality_test_results.json"
    results_7b = {}
    if results_7b_path.exists():
        with open(results_7b_path) as f:
            data_7b = json.load(f)
        for r in data_7b["results"]:
            results_7b[r["index"]] = r

    # Run 3B
    results_3b = []
    for i, entry in enumerate(selected):
        print(f"[{i+1}/20] 3B: {entry['product']}/{entry['defect_code']}...", end=" ", flush=True)

        bboxes = get_bboxes_from_mask(entry["mask_path"])
        if bboxes:
            img = draw_bboxes(entry["image_path"], bboxes)
        else:
            img = Image.open(entry["image_path"]).convert("RGB")

        img_b64 = pil_to_base64(img)
        prompt = build_prompt(entry["product"])

        t0 = time.time()
        resp = query_ollama(img_b64, prompt, "qwen2.5vl:3b")
        elapsed = time.time() - t0

        caption = resp["text"]
        tokens = resp["eval_count"]
        chinese = has_chinese(caption)
        template_ok = follows_template(caption)
        broken = is_broken(caption)

        print(f"OK ({elapsed:.1f}s, {tokens} tok, chinese={chinese}, template={template_ok}, broken={broken})")
        print(f"  3B >> {caption[:120]}{'...' if len(caption)>120 else ''}")

        # Show 7B result for comparison
        if i + 1 in results_7b:
            r7b = results_7b[i + 1]
            print(f"  7B >> {r7b['caption'][:120]}{'...' if len(r7b['caption'])>120 else ''}")
        print()

        results_3b.append({
            "index": i + 1,
            "product": entry["product"],
            "defect_code": entry["defect_code"],
            "defect_name": DEFECT_CODE_MAP.get(entry["defect_code"], entry["defect_code"]),
            "caption": caption,
            "eval_count": tokens,
            "has_chinese": chinese,
            "follows_template": template_ok,
            "is_broken": broken,
            "time_seconds": round(elapsed, 1),
        })

    # Summary
    n_chinese_3b = sum(1 for r in results_3b if r["has_chinese"])
    n_template_3b = sum(1 for r in results_3b if r["follows_template"])
    n_broken_3b = sum(1 for r in results_3b if r["is_broken"])
    avg_tokens_3b = sum(r["eval_count"] for r in results_3b) / len(results_3b)
    max_tokens_3b = max(r["eval_count"] for r in results_3b)
    avg_time_3b = sum(r["time_seconds"] for r in results_3b) / len(results_3b)

    # 7B stats from saved results
    results_7b_list = [results_7b[i] for i in sorted(results_7b.keys())] if results_7b else []

    # Compute 7B stats
    if results_7b_list:
        n_template_7b = sum(1 for r in results_7b_list if r.get("follows_template"))
        n_chinese_7b = sum(1 for r in results_7b_list if r.get("has_chinese"))
        avg_tokens_7b = sum(r["eval_count"] for r in results_7b_list) / len(results_7b_list)
        max_tokens_7b = max(r["eval_count"] for r in results_7b_list)
        avg_time_7b = sum(r["time_seconds"] for r in results_7b_list) / len(results_7b_list)
    else:
        n_template_7b = n_chinese_7b = avg_tokens_7b = max_tokens_7b = avg_time_7b = "N/A"

    print("\n" + "=" * 70)
    print("COMPARISON: 3B vs 7B")
    print("=" * 70)
    fmt = "{:<25} {:>10} {:>10}"
    print(fmt.format("Metric", "3B", "7B"))
    print("-" * 45)
    print(fmt.format("Follows template", f"{n_template_3b}/20", f"{n_template_7b}/20" if results_7b_list else "N/A"))
    print(fmt.format("Has Chinese", f"{n_chinese_3b}/20", f"{n_chinese_7b}/20" if results_7b_list else "N/A"))
    print(fmt.format("Broken output", f"{n_broken_3b}/20", "N/A"))
    print(fmt.format("Avg tokens", f"{avg_tokens_3b:.1f}", f"{avg_tokens_7b:.1f}" if results_7b_list else "N/A"))
    print(fmt.format("Max tokens", str(max_tokens_3b), str(max_tokens_7b) if results_7b_list else "N/A"))
    print(fmt.format("Avg time (s)", f"{avg_time_3b:.1f}", f"{avg_time_7b:.1f}" if results_7b_list else "N/A"))

    # Save
    output = {
        "model": "qwen2.5vl:3b",
        "summary_3b": {
            "follows_template": f"{n_template_3b}/20",
            "has_chinese": f"{n_chinese_3b}/20",
            "broken_output": f"{n_broken_3b}/20",
            "avg_tokens": round(avg_tokens_3b, 1),
            "max_tokens": max_tokens_3b,
            "avg_time_seconds": round(avg_time_3b, 1),
        },
        "results": results_3b,
    }
    out_file = str(out_dir / "quality_test_results_3b.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n3B results saved to: {out_file}")


if __name__ == "__main__":
    main()
