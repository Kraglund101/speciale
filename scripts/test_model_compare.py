"""Quick comparison: GPT-4o-mini vs GPT-5-nano on ONE image, high detail."""

import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

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


async def caption_one(client, model, full_b64, crop_b64, prompt):
    try:
        t0 = time.time()
        kwargs = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{full_b64}", "detail": "high"}},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{crop_b64}", "detail": "high"}},
                ],
            }],
            "max_completion_tokens": 2000,
        }
        # GPT-5 models don't support custom temperature
        if not model.startswith("gpt-5"):
            kwargs["temperature"] = 0.7
        response = await client.chat.completions.create(**kwargs)
        elapsed = time.time() - t0
        msg = response.choices[0].message
        text = (msg.content or "").strip()
        # Debug: show raw message structure if text is empty
        if not text:
            print(f"  DEBUG {model}: content={msg.content!r}")
            print(f"  DEBUG {model}: finish_reason={response.choices[0].finish_reason}")
            if hasattr(msg, 'refusal') and msg.refusal:
                print(f"  DEBUG {model}: refusal={msg.refusal}")
            # Check for other attributes
            for attr in dir(msg):
                if not attr.startswith('_') and attr not in ('content', 'role', 'function_call', 'tool_calls'):
                    val = getattr(msg, attr)
                    if val is not None:
                        print(f"  DEBUG {model}: {attr}={val!r}")
        return {
            "model": model,
            "text": text,
            "tokens_in": response.usage.prompt_tokens,
            "tokens_out": response.usage.completion_tokens,
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as e:
        return {"model": model, "error": str(e)}


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    project_root = Path(__file__).parent.parent
    base_path = project_root / "anomverse_extension" / "datasets" / "realiad_1024"

    # Pick first valid image
    import glob as glob_mod
    jpg_files = sorted(glob_mod.glob(str(base_path / "*/NG/**/*.jpg"), recursive=True))
    sample = None
    for f in jpg_files:
        png = f.replace(".jpg", ".png")
        if os.path.exists(png):
            rel = os.path.relpath(f, str(base_path))
            product = rel.split(os.sep)[0]
            parts = Path(f).parts
            ng_idx = [i for i, p in enumerate(parts) if p == "NG"]
            code = parts[ng_idx[0] + 1] if ng_idx else "??"
            sample = {"image": f, "mask": png, "product": product,
                      "defect": DEFECT_CODE_MAP.get(code, code)}
            break

    if not sample:
        print("No valid image found!")
        return

    print(f"Image: {os.path.basename(sample['image'])}")
    print(f"Product: {sample['product']}  |  Defect: {sample['defect']}\n")

    full_img = draw_bboxes(sample["image"], sample["mask"])
    crop_img = get_union_crop(sample["image"], sample["mask"])
    full_b64 = pil_to_base64(full_img)
    crop_b64 = pil_to_base64(crop_img)
    prompt = build_prompt(sample["product"], sample["defect"])

    # Run both in parallel
    r1, r2 = await asyncio.gather(
        caption_one(client, "gpt-4o-mini", full_b64, crop_b64, prompt),
        caption_one(client, "gpt-5-nano", full_b64, crop_b64, prompt),
    )

    for r in [r1, r2]:
        print(f"{'='*70}")
        print(f"  MODEL: {r['model']}")
        print(f"{'='*70}")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Tokens in:  {r['tokens_in']:,}")
            print(f"  Tokens out: {r['tokens_out']:,}")
            print(f"  Time:       {r['elapsed_s']}s")
            print(f"  Text length: {len(r['text'])} chars")
            print(f"\n{r['text']}\n")
            sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
