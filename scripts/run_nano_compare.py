"""Run GPT-5-nano (high detail) on the same 36 images from the high-detail GPT-4o-mini run,
then output a side-by-side comparison of all three: 4o-mini low, 4o-mini high, 5-nano high."""

import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

DEFECT_CODE_MAP = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
}


def get_union_crop(image_path, mask_path, pad_frac=0.2):
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    bw, bh = x2 - x1, y2 - y1
    pad = max(1, int(max(bw, bh) * pad_frac))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
    img = Image.open(image_path).convert("RGB")
    return img.crop((x1, y1, x2 + 1, y2 + 1))


def draw_bboxes(image_path, mask_path):
    from scipy import ndimage
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    bboxes = []
    for i in range(1, n + 1):
        cy, cx = np.where(labeled == i)
        bboxes.append({"x1": int(cx.min()), "y1": int(cy.min()),
                        "x2": int(cx.max()), "y2": int(cy.max())})
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        for off in range(3):
            draw.rectangle([bb["x1"]-off, bb["y1"]-off, bb["x2"]+off, bb["y2"]+off], outline="red")
    return img


def pil_to_base64(img, max_size=1024):
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(product, defect_type):
    return (
        "You are a visual inspector of industrial defects. "
        "These captions will be used as text conditioning in a Stable Diffusion "
        "inpainting model for synthetic anomaly generation — describe the visual "
        "appearance precisely so the diffusion model can reproduce it.\n\n"
        "You are given two images of the same object:\n"
        "- Image 1: the full object with red bounding boxes marking defect regions "
        "(describe the defect, not the annotations).\n"
        "- Image 2: a close-up crop of the defect area.\n\n"
        f"Object: '{product.replace('_',' ')}'. Defect type: '{defect_type.replace('_',' ')}'.\n\n"
        "Generate TWO captions:\n"
        "SHORT (max 60 words): A single dense sentence for CLIP text encoding.\n"
        "LONG (max 150 words): 2-3 sentences with richer detail about texture, "
        "color, shape, size, and spatial context.\n\n"
        "Output format (exactly):\n"
        "SHORT: <your short caption>\n"
        "LONG: <your long caption>\n\n"
        "Do not include any Chinese characters. Do not describe the bounding boxes."
    )


def parse_response(text):
    short_cap, long_cap = None, None
    lines = text.strip().split("\n")
    current = None
    current_lines = []
    for line in lines:
        ls = line.strip()
        if ls.upper().startswith("SHORT:"):
            if current == "long":
                long_cap = " ".join(current_lines).strip()
            current = "short"
            current_lines = [ls[6:].strip()]
        elif ls.upper().startswith("LONG:"):
            if current == "short":
                short_cap = " ".join(current_lines).strip()
            current = "long"
            current_lines = [ls[5:].strip()]
        elif current:
            current_lines.append(ls)
    if current == "short":
        short_cap = " ".join(current_lines).strip()
    elif current == "long":
        long_cap = " ".join(current_lines).strip()
    return short_cap, long_cap


async def caption_one(client, entry, base_path, semaphore):
    """Caption one image with GPT-5-nano at high detail."""
    async with semaphore:
        img_path = os.path.join(base_path, entry["image_path"])
        mask_path = os.path.join(base_path, entry["mask_path"])

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return {"filename": entry["filename"], "error": "file not found"}

        try:
            full_img = draw_bboxes(img_path, mask_path)
            crop_img = get_union_crop(img_path, mask_path)
            if crop_img is None:
                return {"filename": entry["filename"], "error": "no anomaly pixels"}

            full_b64 = pil_to_base64(full_img)
            crop_b64 = pil_to_base64(crop_img)
            prompt = build_prompt(entry["product"], entry["defect_name"])

            t0 = time.time()
            response = await client.chat.completions.create(
                model="gpt-5-nano",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{full_b64}", "detail": "high"}},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"}},
                    ],
                }],
                max_completion_tokens=2000,
            )
            elapsed = time.time() - t0
            text = (response.choices[0].message.content or "").strip()
            short_cap, long_cap = parse_response(text)

            return {
                "filename": entry["filename"],
                "image_path": entry["image_path"],
                "mask_path": entry["mask_path"],
                "product": entry["product"],
                "defect_name": entry["defect_name"],
                "caption_short": short_cap or "",
                "caption_long": long_cap or "",
                "raw_response": text,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "elapsed_s": round(elapsed, 2),
            }
        except Exception as e:
            return {"filename": entry["filename"], "error": str(e)}


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    project_root = Path(__file__).parent.parent
    base_path = str(project_root / "anomverse_extension" / "datasets" / "realiad_1024")
    output_dir = project_root / "anomverse_extension" / "datasets" / "realiad_1024"

    # Load the 36 entries from high-detail run (these are the ones we want to match)
    high_detail = json.load(open(output_dir / "captions_openai_high_detail.json"))
    print(f"Running GPT-5-nano on {len(high_detail)} images (high detail)...\n")

    semaphore = asyncio.Semaphore(3)  # concurrency limit
    start = time.time()

    tasks = [caption_one(client, entry, base_path, semaphore) for entry in high_detail]
    results = await asyncio.gather(*tasks)

    # Separate successes and failures
    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]

    elapsed = time.time() - start
    total_in = sum(r.get("tokens_in", 0) for r in successes)
    total_out = sum(r.get("tokens_out", 0) for r in successes)
    cost_in = total_in * 0.05 / 1_000_000
    cost_out = total_out * 0.40 / 1_000_000

    print(f"Done in {elapsed:.0f}s")
    print(f"Success: {len(successes)}/{len(results)}")
    print(f"Failures: {len(failures)}")
    if failures:
        for f in failures:
            print(f"  FAIL: {f['filename']} -> {f['error'][:80]}")
    print(f"Tokens: {total_in:,} in + {total_out:,} out")
    print(f"Cost: ${cost_in + cost_out:.4f}")

    # Save nano results
    nano_path = output_dir / "captions_openai_nano_high.json"
    with open(nano_path, "w") as f:
        json.dump(successes, f, indent=2)
    print(f"\nSaved {len(successes)} captions to {nano_path}")

    # Now print comparison for first few
    low_detail = json.load(open(output_dir / "captions_openai.json"))
    low_by_name = {e["filename"]: e for e in low_detail}
    high_by_name = {e["filename"]: e for e in high_detail}
    nano_by_name = {e["filename"]: e for e in successes}

    print(f"\n{'='*80}")
    print("SIDE-BY-SIDE COMPARISON (first 5)")
    print(f"{'='*80}")

    shown = 0
    for entry in high_detail:
        fn = entry["filename"]
        if fn not in nano_by_name:
            continue
        if shown >= 5:
            break

        lo = low_by_name.get(fn, {})
        hi = high_by_name.get(fn, {})
        na = nano_by_name.get(fn, {})

        print(f"\n--- {fn} ({entry['product']} / {entry['defect_name']}) ---")
        print(f"\n  4o-mini LOW  ({lo.get('tokens_in','?'):,} tok in):")
        print(f"    SHORT: {lo.get('caption_short','N/A')}")
        print(f"    LONG:  {lo.get('caption_long','N/A')}")
        print(f"\n  4o-mini HIGH ({hi.get('tokens_in','?'):,} tok in):")
        print(f"    SHORT: {hi.get('caption_short','N/A')}")
        print(f"    LONG:  {hi.get('caption_long','N/A')}")
        print(f"\n  5-nano HIGH  ({na.get('tokens_in','?'):,} tok in, {na.get('tokens_out','?'):,} tok out):")
        print(f"    SHORT: {na.get('caption_short','N/A')}")
        print(f"    LONG:  {na.get('caption_long','N/A')}")
        shown += 1


if __name__ == "__main__":
    asyncio.run(main())
