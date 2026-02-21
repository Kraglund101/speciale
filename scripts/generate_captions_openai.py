"""
Generate text captions for anomaly images using OpenAI GPT-4o mini.

Sends TWO images per request:
  1. Full image with red bounding boxes highlighting defect regions
  2. Crop around the union bounding box of all anomaly pixels + 20% padding

Generates TWO captions per image:
  - Short (≤77 CLIP tokens): for SD 1.5 text encoder conditioning
  - Long (≤200 CLIP tokens): for extended conditioning / metadata

Supports both real-time API and batch API (50% cheaper, async).

Usage:
    # Test 50 images
    python scripts/generate_captions_openai.py --max-images 50

    # Full run with concurrency
    python scripts/generate_captions_openai.py --concurrency 10

    # Resume interrupted run
    python scripts/generate_captions_openai.py --resume

    # Use batch API (50% cheaper, async)
    python scripts/generate_captions_openai.py --batch
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Load .env file if present (for OPENAI_API_KEY)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# RealIAD defect code translations
DEFECT_CODE_MAP = {
    "AK": "pit",
    "BX": "deformation",
    "CH": "abrasion",
    "HS": "scratch",
    "PS": "damage",
    "QS": "missing_parts",
    "YW": "foreign_objects",
    "ZW": "contamination",
}

MODEL = "gpt-4o-mini"


def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
    """Extract bounding boxes from binary mask using connected components."""
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None

        h, w = mask.shape
        from scipy import ndimage
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
    except Exception as e:
        logger.warning(f"Failed to read mask {mask_path}: {e}")
        return None


def get_union_crop(image_path: str, mask_path: str, pad_frac: float = 0.2) -> Optional[Image.Image]:
    """Crop image to union bounding box of anomaly pixels + padding."""
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None

        h, w = mask.shape
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        bw, bh = x2 - x1, y2 - y1
        pad = max(1, int(max(bw, bh) * pad_frac))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        img = Image.open(image_path).convert("RGB")
        return img.crop((x1, y1, x2 + 1, y2 + 1))
    except Exception as e:
        logger.warning(f"Failed to crop {image_path}: {e}")
        return None


def draw_bboxes_on_image(image_path: str, bboxes: list[dict]) -> Image.Image:
    """Draw red bounding boxes on the image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        for offset in range(3):
            draw.rectangle(
                [bb["x1"] - offset, bb["y1"] - offset, bb["x2"] + offset, bb["y2"] + offset],
                outline="red",
            )
    return img


def pil_to_base64(img: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL Image to base64 JPEG. Resize if larger than max_size."""
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(product: str, defect_type: str) -> str:
    """Build the two-image captioning prompt."""
    product_name = product.replace("_", " ")
    defect_name = defect_type.replace("_", " ")
    return (
        "You are a visual inspector of industrial defects. "
        "These captions will be used as text conditioning in a Stable Diffusion "
        "inpainting model for synthetic anomaly generation — describe the visual "
        "appearance precisely so the diffusion model can reproduce it.\n\n"
        "You are given two images of the same object:\n"
        "- Image 1: the full object with red bounding boxes marking defect regions "
        "(describe the defect, not the annotations).\n"
        "- Image 2: a close-up crop of the defect area.\n\n"
        f"Object: '{product_name}'. Defect type: '{defect_name}'.\n\n"
        "Generate TWO captions:\n"
        "SHORT (max 60 words): A single dense sentence for CLIP text encoding.\n"
        "LONG (max 150 words): 2-3 sentences with richer detail about texture, "
        "color, shape, size, and spatial context.\n\n"
        "Output format (exactly):\n"
        "SHORT: <your short caption>\n"
        "LONG: <your long caption>\n\n"
        "Do not include any Chinese characters. Do not describe the bounding boxes."
    )


def parse_response(text: str) -> tuple[Optional[str], Optional[str]]:
    """Parse SHORT and LONG captions from model response."""
    short_cap = None
    long_cap = None

    lines = text.strip().split("\n")
    current = None
    current_lines = []

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith("SHORT:"):
            if current == "long":
                long_cap = " ".join(current_lines).strip()
            current = "short"
            current_lines = [line_stripped[6:].strip()]
        elif line_stripped.upper().startswith("LONG:"):
            if current == "short":
                short_cap = " ".join(current_lines).strip()
            current = "long"
            current_lines = [line_stripped[5:].strip()]
        elif current:
            current_lines.append(line_stripped)

    # Flush last
    if current == "short":
        short_cap = " ".join(current_lines).strip()
    elif current == "long":
        long_cap = " ".join(current_lines).strip()

    return short_cap, long_cap


async def query_openai(
    client,
    img_full_b64: str,
    img_crop_b64: str,
    prompt: str,
    max_retries: int = 5,
) -> Optional[dict]:
    """Send two images + prompt to OpenAI API with retry on rate limit."""
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_full_b64}",
                                    "detail": "low",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_crop_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=300,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** attempt + 1  # 2, 3, 5, 9, 17 seconds
                logger.info(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                logger.warning(f"OpenAI API error: {e}")
                return None
    logger.warning("Max retries exceeded for rate limit")
    return None


def discover_realiad_images(base_path: str) -> list[dict]:
    """Find all RealIAD NG images that have matching masks."""
    import glob as glob_mod

    logger.info(f"Scanning RealIAD at {base_path}...")
    entries = []

    jpg_files = glob_mod.glob(os.path.join(base_path, "*/NG/**/*.jpg"), recursive=True)
    logger.info(f"Found {len(jpg_files)} total NG jpg files")

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
            "image_path": rel.replace(os.sep, "/"),
            "mask_path": os.path.relpath(png_path, base_path).replace(os.sep, "/"),
            "image_path_abs": jpg_path,
            "mask_path_abs": png_path,
            "product": product,
            "defect_code": defect_code,
            "defect_name": DEFECT_CODE_MAP.get(defect_code, defect_code),
            "filename": os.path.basename(jpg_path),
        })

    logger.info(f"Found {len(entries)} NG images with masks")
    return entries


def prepare_entry(entry: dict) -> Optional[dict]:
    """Prepare images + prompt for one entry."""
    img_abs = entry["image_path_abs"]
    mask_abs = entry["mask_path_abs"]

    bboxes = get_bboxes_from_mask(mask_abs)
    if not bboxes:
        return None

    img_full = draw_bboxes_on_image(img_abs, bboxes)
    img_crop = get_union_crop(img_abs, mask_abs)
    if img_crop is None:
        return None

    prompt = build_prompt(entry["product"], entry["defect_name"])
    img_full_b64 = pil_to_base64(img_full)
    img_crop_b64 = pil_to_base64(img_crop)

    return {
        "prompt": prompt,
        "img_full_b64": img_full_b64,
        "img_crop_b64": img_crop_b64,
        "bboxes": bboxes,
    }


async def process_batch(client, entries: list[dict], semaphore: asyncio.Semaphore) -> list[dict]:
    """Process entries concurrently with semaphore for rate limiting."""
    results = []

    async def process_one(entry):
        async with semaphore:
            prepared = prepare_entry(entry)
            if prepared is None:
                return None

            response = await query_openai(
                client,
                prepared["img_full_b64"],
                prepared["img_crop_b64"],
                prepared["prompt"],
            )
            if response is None:
                return None

            short_cap, long_cap = parse_response(response["text"])

            return {
                "filename": entry["filename"],
                "image_path": entry["image_path"],
                "mask_path": entry["mask_path"],
                "product": entry["product"],
                "defect_code": entry["defect_code"],
                "defect_name": entry["defect_name"],
                "bboxes": prepared["bboxes"],
                "caption_short": short_cap or "",
                "caption_long": long_cap or "",
                "raw_response": response["text"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
            }

    tasks = [process_one(entry) for entry in entries]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def save_results(results: list[dict], output_dir: Path):
    """Save short and long captions to separate files + combined."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combined (all data)
    combined_path = output_dir / "captions_openai.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2)

    # Short captions only (for SD 1.5 CLIP)
    short_captions = [
        {
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
            "product": r["product"],
            "defect_name": r["defect_name"],
            "caption": r["caption_short"],
        }
        for r in results
    ]
    short_path = output_dir / "captions_short_77.json"
    with open(short_path, "w") as f:
        json.dump(short_captions, f, indent=2)

    # Long captions (for extended conditioning)
    long_captions = [
        {
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
            "product": r["product"],
            "defect_name": r["defect_name"],
            "caption": r["caption_long"],
        }
        for r in results
    ]
    long_path = output_dir / "captions_long_200.json"
    with open(long_path, "w") as f:
        json.dump(long_captions, f, indent=2)

    logger.info(f"Saved: {combined_path} ({len(results)} entries)")
    logger.info(f"Saved: {short_path}")
    logger.info(f"Saved: {long_path}")


async def run_realtime(args):
    """Run captioning with real-time API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    project_root = Path(__file__).parent.parent
    base_path = str(project_root / "anomverse_extension" / "datasets" / "realiad_1024")
    output_dir = project_root / "anomverse_extension" / "datasets" / "realiad_1024"

    entries = discover_realiad_images(base_path)

    # Handle resume
    combined_path = output_dir / "captions_openai.json"
    results = []
    done_filenames = set()
    if args.resume and combined_path.exists():
        with open(combined_path) as f:
            results = json.load(f)
        done_filenames = {r["filename"] for r in results}
        logger.info(f"Resuming: {len(done_filenames)} already done")

    entries = [e for e in entries if e["filename"] not in done_filenames]

    if args.max_images > 0:
        entries = entries[:args.max_images]

    total = len(entries)
    logger.info(f"Captioning {total} images with {MODEL}, concurrency={args.concurrency}")

    semaphore = asyncio.Semaphore(args.concurrency)
    start_time = time.time()

    # Process in chunks for progress reporting + saving
    chunk_size = args.save_every
    for chunk_start in range(0, total, chunk_size):
        chunk = entries[chunk_start:chunk_start + chunk_size]
        chunk_results = await process_batch(client, chunk, semaphore)
        results.extend(chunk_results)

        elapsed = time.time() - start_time
        processed = chunk_start + len(chunk)
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (total - processed) / rate if rate > 0 else 0

        total_tokens_in = sum(r.get("tokens_in", 0) for r in results)
        total_tokens_out = sum(r.get("tokens_out", 0) for r in results)
        cost_in = total_tokens_in * 0.15 / 1_000_000
        cost_out = total_tokens_out * 0.60 / 1_000_000
        cost = cost_in + cost_out

        logger.info(
            f"[{processed}/{total}] {rate:.1f} img/s | "
            f"ETA: {eta/60:.0f}min | ${cost:.2f} spent | "
            f"short_ok: {sum(1 for r in results if r['caption_short'])}/{len(results)} | "
            f"long_ok: {sum(1 for r in results if r['caption_long'])}/{len(results)}"
        )

        save_results(results, output_dir)

    # Final save
    save_results(results, output_dir)
    elapsed = time.time() - start_time

    total_tokens_in = sum(r.get("tokens_in", 0) for r in results)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in results)
    cost = total_tokens_in * 0.15 / 1_000_000 + total_tokens_out * 0.60 / 1_000_000

    logger.info(
        f"Done! {len(results)} captions in {elapsed/60:.1f}min. "
        f"Cost: ${cost:.2f}. "
        f"Tokens: {total_tokens_in:,} in + {total_tokens_out:,} out"
    )


def main():
    parser = argparse.ArgumentParser(description="Generate captions with OpenAI GPT-4o mini")
    parser.add_argument("--max-images", type=int, default=0, help="0=all")
    parser.add_argument("--concurrency", type=int, default=3, help="Concurrent API requests (keep low for 200K TPM tier)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY not set! Get one from https://platform.openai.com/api-keys\n"
            "Then: set OPENAI_API_KEY=sk-your-key-here"
        )
        sys.exit(1)

    asyncio.run(run_realtime(args))


if __name__ == "__main__":
    main()
