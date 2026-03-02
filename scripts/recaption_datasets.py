"""Re-caption all non-RealIAD datasets using GPT-4o-mini.

Same approach as generate_captions_openai.py:
  - Image 1: full image with red bounding boxes on defect regions
  - Image 2: close-up crop of the defect area (union bbox + 20% padding)

Patches captions_from_master.json and master_training.json in-place.
Saves progress incrementally so interrupted runs can resume.

Usage:
    python scripts/recaption_datasets.py
    python scripts/recaption_datasets.py --resume          # resume interrupted run
    python scripts/recaption_datasets.py --concurrency 5   # faster (default 3)
    python scripts/recaption_datasets.py --dry-run          # just show counts
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

# Load .env
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FT_DIR = PROJECT_ROOT / "anomverse_extension" / "datasets" / "full_training_dataset"

# Datasets to re-caption (everything except RealIAD)
TARGET_DATASETS = {"mvtec", "DAGM", "mvtec3d", "mvtec_ad_2", "MulSen_AD", "BTech", "MPDD", "eyecandies"}


# ---------------------------------------------------------------------------
# Image utilities (same as recaption_bad.py / generate_captions_openai.py)
# ---------------------------------------------------------------------------


def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
    """Extract bounding boxes from binary mask using connected components."""
    try:
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
        img = img.copy()
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
        "IMPORTANT: There IS a defect in the image — the red bounding boxes show "
        "where it is. Look carefully at the crop (Image 2) to identify the "
        "anomaly even if it is subtle. Never say 'no visible defect'.\n\n"
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

    if current == "short":
        short_cap = " ".join(current_lines).strip()
    elif current == "long":
        long_cap = " ".join(current_lines).strip()

    return short_cap, long_cap


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------


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
                wait = 2 ** attempt + 1
                logger.info(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                logger.warning(f"OpenAI API error: {e}")
                return None
    logger.warning("Max retries exceeded for rate limit")
    return None


# ---------------------------------------------------------------------------
# Process one entry
# ---------------------------------------------------------------------------


def prepare_entry(entry: dict) -> Optional[dict]:
    """Prepare images + prompt for one master entry."""
    image_abs = str(FT_DIR / entry["image_path"])
    mask_abs = str(FT_DIR / entry["mask_path"])

    bboxes = get_bboxes_from_mask(mask_abs)
    if not bboxes:
        return None

    img_full = draw_bboxes_on_image(image_abs, bboxes)
    img_crop = get_union_crop(image_abs, mask_abs)
    if img_crop is None:
        return None

    prompt = build_prompt(entry["product"], entry["defect_type"])
    img_full_b64 = pil_to_base64(img_full)
    img_crop_b64 = pil_to_base64(img_crop)

    return {
        "prompt": prompt,
        "img_full_b64": img_full_b64,
        "img_crop_b64": img_crop_b64,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_batch(
    client, entries: list[dict], semaphore: asyncio.Semaphore
) -> list[dict]:
    """Process entries concurrently with semaphore."""

    async def process_one(entry: dict) -> Optional[dict]:
        async with semaphore:
            prepared = prepare_entry(entry)
            if prepared is None:
                return {"image_path": entry["image_path"], "caption_short": "", "caption_long": "", "failed": True}

            response = await query_openai(
                client,
                prepared["img_full_b64"],
                prepared["img_crop_b64"],
                prepared["prompt"],
            )
            if response is None:
                return {"image_path": entry["image_path"], "caption_short": "", "caption_long": "", "failed": True}

            short_cap, long_cap = parse_response(response["text"])

            return {
                "image_path": entry["image_path"],
                "caption_short": short_cap or "",
                "caption_long": long_cap or "",
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
                "failed": False,
            }

    tasks = [process_one(e) for e in entries]
    results = await asyncio.gather(*tasks)
    return list(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args):
    """Main processing loop."""

    # Load master
    master_path = FT_DIR / "master_training.json"
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    # Filter to target datasets
    target_entries = [e for e in master if e["dataset"] in TARGET_DATASETS]
    logger.info(f"Target entries: {len(target_entries)} across {TARGET_DATASETS}")

    if args.dry_run:
        from collections import Counter
        ds_counts = Counter(e["dataset"] for e in target_entries)
        for ds, count in ds_counts.most_common():
            logger.info(f"  {ds}: {count}")
        logger.info(f"Estimated cost: ~${len(target_entries) * 0.0019:.2f}")
        return

    # Progress file for resume
    progress_path = FT_DIR / "_recaption_progress.json"
    done_results: list[dict] = []
    done_paths: set[str] = set()

    if args.resume and progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            done_results = json.load(f)
        done_paths = {r["image_path"] for r in done_results}
        logger.info(f"Resuming: {len(done_paths)} already done")

    remaining = [e for e in target_entries if e["image_path"] not in done_paths]
    total = len(remaining)
    logger.info(f"Remaining to caption: {total}")

    if total == 0:
        logger.info("All done — applying patches from progress file")
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

        semaphore = asyncio.Semaphore(args.concurrency)
        start_time = time.time()

        chunk_size = args.save_every
        for chunk_start in range(0, total, chunk_size):
            chunk = remaining[chunk_start:chunk_start + chunk_size]
            chunk_results = await process_batch(client, chunk, semaphore)
            done_results.extend(chunk_results)

            elapsed = time.time() - start_time
            processed = chunk_start + len(chunk)
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            ok_count = sum(1 for r in done_results if not r.get("failed") and r["caption_short"])
            fail_count = sum(1 for r in done_results if r.get("failed"))
            total_tokens_in = sum(r.get("tokens_in", 0) for r in done_results)
            total_tokens_out = sum(r.get("tokens_out", 0) for r in done_results)
            cost = total_tokens_in * 0.15 / 1_000_000 + total_tokens_out * 0.60 / 1_000_000

            logger.info(
                f"[{processed}/{total}] {rate:.1f} img/s | "
                f"ETA: {eta/60:.0f}min | ${cost:.2f} | "
                f"ok: {ok_count} fail: {fail_count}"
            )

            # Save progress
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(done_results, f, ensure_ascii=False)

        elapsed = time.time() - start_time
        total_tokens_in = sum(r.get("tokens_in", 0) for r in done_results)
        total_tokens_out = sum(r.get("tokens_out", 0) for r in done_results)
        cost = total_tokens_in * 0.15 / 1_000_000 + total_tokens_out * 0.60 / 1_000_000
        logger.info(f"API done in {elapsed/60:.1f}min. Cost: ${cost:.2f}")

    # Build replacement map
    replacements = {}
    for r in done_results:
        if not r.get("failed") and r["caption_short"]:
            replacements[r["image_path"]] = {
                "short": r["caption_short"],
                "long": r.get("caption_long", ""),
            }

    logger.info(f"Applying {len(replacements)} caption replacements...")

    # Patch captions_from_master.json
    captions_path = FT_DIR / "captions_from_master.json"
    with open(captions_path, encoding="utf-8") as f:
        caps = json.load(f)

    patched_caps = 0
    for item in caps:
        if item["image_path"] in replacements:
            item["caption"] = replacements[item["image_path"]]["short"]
            patched_caps += 1

    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched_caps} in captions_from_master.json")

    # Patch master_training.json
    patched_master = 0
    for entry in master:
        if entry["image_path"] in replacements:
            entry["caption_short"] = replacements[entry["image_path"]]["short"]
            entry["caption_long"] = replacements[entry["image_path"]]["long"]
            patched_master += 1

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched_master} in master_training.json")

    # Clean up progress file
    if progress_path.exists():
        progress_path.unlink()
        logger.info("Removed progress file")

    logger.info("Done!")


def main():
    parser = argparse.ArgumentParser(description="Re-caption non-RealIAD datasets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
