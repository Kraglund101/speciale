"""Generate crop-relative text captions for anomaly images using OpenAI.

During training, images larger than 512px have a random 512x512 crop extracted
around the anomaly. The existing captions were generated from full images and
contain spatial references ("on the left side") that don't match the crop view.

This script:
1. Classifies each image as "crop" (>512px, anomaly fits in 512) or "keep"
2. For crop cases: extracts a random 512x512 crop containing the anomaly,
   applies random flips, and sends to OpenAI for crop-relative captioning
3. For keep cases: copies the existing caption unchanged
4. Outputs a merged captions file usable via --captions-file

Usage:
    # Test 10 images
    python scripts/generate_crop_captions_openai.py --max-images 10

    # Full run with concurrency
    python scripts/generate_crop_captions_openai.py --concurrency 10

    # Resume interrupted run
    python scripts/generate_crop_captions_openai.py --resume
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Load .env file if present (for OPENAI_API_KEY)
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

MODEL = "gpt-5-nano"


# ---------------------------------------------------------------------------
# Image / mask utilities
# ---------------------------------------------------------------------------


def get_bboxes_from_mask_array(mask_np: np.ndarray) -> Optional[list[dict]]:
    """Extract bounding boxes from binary mask array using connected components."""
    binary = (mask_np > 127).astype(np.uint8)
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


def draw_bboxes_on_pil(img: Image.Image, bboxes: list[dict]) -> Image.Image:
    """Draw red bounding boxes on a PIL image (returns a copy)."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        for offset in range(3):
            draw.rectangle(
                [bb["x1"] - offset, bb["y1"] - offset,
                 bb["x2"] + offset, bb["y2"] + offset],
                outline="red",
            )
    return img


def get_union_crop_from_pil(
    img: Image.Image, mask_np: np.ndarray, pad_frac: float = 0.2
) -> Optional[Image.Image]:
    """Crop image to union bounding box of anomaly pixels + padding."""
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return None

    h, w = mask_np.shape
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    bw, bh = x2 - x1, y2 - y1
    pad = max(1, int(max(bw, bh) * pad_frac))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    return img.crop((x1, y1, x2 + 1, y2 + 1))


def pil_to_base64(img: Image.Image, max_size: int = 1024) -> str:
    """Convert PIL Image to base64 JPEG. Resize if larger than max_size."""
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Crop logic (mirrors future training augmentation)
# ---------------------------------------------------------------------------


def get_anomaly_crop_512(
    image_pil: Image.Image,
    mask_pil: Image.Image,
    target: int = 512,
) -> Optional[Tuple[Image.Image, Image.Image]]:
    """Extract a random target x target crop that fully contains the anomaly.

    Returns (cropped_image, cropped_mask) or None if:
    - Image is already <= target in both dimensions (no crop needed)
    - Anomaly bbox doesn't fit in target x target (downsample case)
    """
    w, h = image_pil.size

    # Image already fits — no crop needed
    if h <= target and w <= target:
        return None

    # Get anomaly union bbox
    mask_np = np.array(mask_pil.convert("L"))
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return None

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    bbox_w = x2 - x1 + 1
    bbox_h = y2 - y1 + 1

    # Anomaly too large for target crop → downsample case
    if bbox_w > target or bbox_h > target:
        return None

    # Compute valid top-left corner range so crop fully contains the anomaly
    left_min = max(0, x2 - target + 1)
    left_max = min(w - target, x1)
    top_min = max(0, y2 - target + 1)
    top_max = min(h - target, y1)

    if left_min > left_max or top_min > top_max:
        return None  # Safety check

    left = random.randint(left_min, left_max)
    top = random.randint(top_min, top_max)

    crop_img = image_pil.crop((left, top, left + target, top + target))
    crop_mask = mask_pil.crop((left, top, left + target, top + target))

    return crop_img, crop_mask


def apply_random_flips(
    img: Image.Image, mask: Image.Image
) -> Tuple[Image.Image, Image.Image]:
    """Apply random horizontal and vertical flips jointly to image + mask."""
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    return img, mask


# ---------------------------------------------------------------------------
# Prompt / parsing
# ---------------------------------------------------------------------------


def build_crop_prompt(product: str, defect_type: str) -> str:
    """Build prompt for crop-relative captioning."""
    product_name = product.replace("_", " ")
    defect_name = defect_type.replace("_", " ")
    return (
        "You are a visual inspector of industrial defects. "
        "These captions will be used as text conditioning in a Stable Diffusion "
        "inpainting model for synthetic anomaly generation — describe the visual "
        "appearance precisely so the diffusion model can reproduce it.\n\n"
        "You are given two images:\n"
        "- Image 1: a cropped region of an industrial object, centered around a "
        "defect area, with red bounding boxes marking defect regions "
        "(describe the defect, not the annotations). "
        "This is NOT the full object — it is a random crop showing a portion "
        "of the surface.\n"
        "- Image 2: a close-up crop of the defect area within this region.\n\n"
        f"Object: '{product_name}'. Defect type: '{defect_name}'.\n\n"
        "Generate TWO captions:\n"
        "SHORT (max 60 words): A single dense sentence for CLIP text encoding. "
        "Describe the defect appearance in the context of the visible surface region.\n"
        "LONG (max 150 words): 2-3 sentences with richer detail about texture, "
        "color, shape, size, and surface context.\n\n"
        "Output format (exactly):\n"
        "SHORT: <your short caption>\n"
        "LONG: <your long caption>\n\n"
        "Do not include any Chinese characters. Do not describe the bounding boxes."
    )


def parse_response(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse SHORT and LONG captions from model response."""
    short_cap = None
    long_cap = None

    lines = text.strip().split("\n")
    current = None
    current_lines: list[str] = []

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
    img1_b64: str,
    img2_b64: str,
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
                                    "url": f"data:image/jpeg;base64,{img1_b64}",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img2_b64}",
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=5000,
            )
            text = response.choices[0].message.content.strip()
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens
            return {"text": text, "tokens_in": tokens_in, "tokens_out": tokens_out}
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = 2 ** attempt + 1
                logger.info(
                    f"Rate limited, waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait)
            else:
                logger.warning(f"OpenAI API error: {e}")
                return None
    logger.warning("Max retries exceeded for rate limit")
    return None


# ---------------------------------------------------------------------------
# Entry classification and preparation
# ---------------------------------------------------------------------------


def classify_entry(image_path: str, mask_path: str, target: int = 512) -> str:
    """Classify an entry as 'crop', 'keep', or 'skip'.

    crop  — image > target AND anomaly bbox fits in target
    keep  — image <= target OR anomaly too large for target crop
    skip  — cannot read image/mask or no anomaly pixels
    """
    try:
        img = Image.open(image_path)
        w, h = img.size
    except Exception as e:
        logger.warning(f"Cannot open image {image_path}: {e}")
        return "skip"

    if h <= target and w <= target:
        return "keep"

    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return "skip"

        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        bbox_w = x2 - x1 + 1
        bbox_h = y2 - y1 + 1

        if bbox_w > target or bbox_h > target:
            return "keep"  # Downsample case — keep existing caption

        return "crop"
    except Exception as e:
        logger.warning(f"Cannot read mask {mask_path}: {e}")
        return "skip"


def save_crop_images(
    crop_img: Image.Image,
    crop_mask: Image.Image,
    image_path: str,
    crops_dir: Path,
) -> Tuple[str, str]:
    """Save crop image + mask to disk, mirroring the original path structure.

    Returns (crop_image_relpath, crop_mask_relpath) relative to crops_dir parent.
    """
    # Build output path: crops_512/{original_relative_path}
    crop_image_rel = f"crops_512/{image_path}"
    # Mask: same path but always .png
    base, _ = os.path.splitext(image_path)
    crop_mask_rel = f"crops_512/{base}_mask.png"

    crop_image_abs = crops_dir / image_path
    crop_mask_abs = crops_dir / f"{base}_mask.png"

    crop_image_abs.parent.mkdir(parents=True, exist_ok=True)

    # Save image as JPEG if original was .jpg, else PNG
    ext = Path(image_path).suffix.lower()
    if ext in (".jpg", ".jpeg"):
        crop_img.save(str(crop_image_abs), format="JPEG", quality=95)
    else:
        crop_img.save(str(crop_image_abs), format="PNG")

    # Mask always PNG
    crop_mask.save(str(crop_mask_abs), format="PNG")

    return crop_image_rel, crop_mask_rel


def prepare_crop_entry(
    image_path: str,
    mask_path: str,
    image_path_rel: str,
    product: str,
    defect_type: str,
    crops_dir: Path,
) -> Optional[dict]:
    """Prepare a crop-case entry for API call + save crop to disk.

    Returns dict with img1_b64, img2_b64, prompt, bboxes, crop paths
    — or None on failure.
    """
    try:
        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path).convert("L")
    except Exception as e:
        logger.warning(f"Failed to open {image_path}: {e}")
        return None

    # Extract random 512 crop
    result = get_anomaly_crop_512(image_pil, mask_pil)
    if result is None:
        return None
    crop_img, crop_mask = result

    # Apply random flips
    crop_img, crop_mask = apply_random_flips(crop_img, crop_mask)

    # Save crop images to disk
    crop_image_rel, crop_mask_rel = save_crop_images(
        crop_img, crop_mask, image_path_rel, crops_dir
    )

    crop_mask_np = np.array(crop_mask)

    # Image 1: crop with red bboxes
    bboxes = get_bboxes_from_mask_array(crop_mask_np)
    if not bboxes:
        return None
    img1 = draw_bboxes_on_pil(crop_img, bboxes)

    # Image 2: close-up of anomaly within the crop
    img2 = get_union_crop_from_pil(crop_img, crop_mask_np)
    if img2 is None:
        return None

    prompt = build_crop_prompt(product, defect_type)
    img1_b64 = pil_to_base64(img1)
    img2_b64 = pil_to_base64(img2)

    return {
        "img1_b64": img1_b64,
        "img2_b64": img2_b64,
        "prompt": prompt,
        "bboxes": bboxes,
        "crop_image_path": crop_image_rel,
        "crop_mask_path": crop_mask_rel,
    }


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


async def process_batch(
    client, entries: list[dict], semaphore: asyncio.Semaphore,
    crops_dir: Path,
) -> list[dict]:
    """Process crop entries concurrently with semaphore for rate limiting."""

    async def process_one(entry: dict) -> Optional[dict]:
        async with semaphore:
            prepared = prepare_crop_entry(
                entry["image_path_abs"],
                entry["mask_path_abs"],
                entry["image_path"],
                entry["product"],
                entry["defect_type"],
                crops_dir,
            )
            if prepared is None:
                return None

            response = await query_openai(
                client,
                prepared["img1_b64"],
                prepared["img2_b64"],
                prepared["prompt"],
            )
            if response is None:
                return None

            short_cap, long_cap = parse_response(response["text"])

            return {
                "image_path": entry["image_path"],
                "mask_path": entry["mask_path"],
                "crop_image_path": prepared["crop_image_path"],
                "crop_mask_path": prepared["crop_mask_path"],
                "dataset": entry["dataset"],
                "product": entry["product"],
                "defect_type": entry["defect_type"],
                "bboxes": prepared["bboxes"],
                "caption_short": short_cap or "",
                "caption_long": long_cap or "",
                "raw_response": response["text"],
                "tokens_in": response["tokens_in"],
                "tokens_out": response["tokens_out"],
            }

    tasks = [process_one(e) for e in entries]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_crop_results(results: list[dict], output_path: Path) -> None:
    """Save crop caption results (API responses only)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(results)} crop results to {output_path}")



def build_crop_samples(crop_results: list[dict], master_entries: list[dict]) -> list[dict]:
    """Build crop_samples.json — additional training samples from crops.

    Each entry mirrors the master_training.json format so it can be
    ingested by the dataset / splits pipeline.
    """
    # Lookup master entries by image_path for metadata
    master_map = {e["image_path"]: e for e in master_entries}

    samples = []
    for r in crop_results:
        if not r["caption_short"]:
            continue
        master = master_map.get(r["image_path"], {})
        samples.append({
            "dataset": r["dataset"],
            "product": r["product"],
            "defect_type": r["defect_type"],
            "image_path": r["crop_image_path"],
            "mask_path": r["crop_mask_path"],
            "caption_short": r["caption_short"],
            "caption_long": r.get("caption_long", ""),
            "source_image_path": r["image_path"],
        })

    return samples



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args):
    """Main processing loop."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()

    project_root = Path(__file__).resolve().parent.parent
    ft_dir = project_root / "anomverse_extension" / "datasets" / "full_training_dataset"
    data_root = ft_dir  # image paths in master_training.json are relative to this

    # Load master training data
    master_path = ft_dir / "master_training.json"
    logger.info(f"Loading master training data from {master_path}...")
    with open(master_path, encoding="utf-8") as f:
        master_entries = json.load(f)
    logger.info(f"Loaded {len(master_entries)} entries")

    # Output paths
    crop_results_path = ft_dir / "captions_crop.json"
    crop_samples_path = ft_dir / "crop_samples.json"
    crops_dir = ft_dir / "crops_512"
    crops_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume
    crop_results: list[dict] = []
    done_image_paths: set[str] = set()
    if args.resume and crop_results_path.exists():
        with open(crop_results_path, encoding="utf-8") as f:
            crop_results = json.load(f)
        done_image_paths = {r["image_path"] for r in crop_results}
        logger.info(f"Resuming: {len(done_image_paths)} already done")

    # Classify all entries
    logger.info("Classifying entries (crop vs keep)...")
    crop_entries = []
    keep_count = 0
    skip_count = 0

    for entry in master_entries:
        image_path = entry["image_path"]
        if image_path in done_image_paths:
            continue

        image_abs = str(data_root / image_path)
        mask_abs = str(data_root / entry["mask_path"])

        classification = classify_entry(image_abs, mask_abs)

        if classification == "crop":
            crop_entries.append({
                **entry,
                "image_path_abs": image_abs,
                "mask_path_abs": mask_abs,
            })
        elif classification == "keep":
            keep_count += 1
        else:
            skip_count += 1

    logger.info(
        f"Classification: {len(crop_entries)} crop, {keep_count} keep, "
        f"{skip_count} skip, {len(done_image_paths)} already done"
    )

    if args.max_images > 0:
        crop_entries = crop_entries[: args.max_images]

    total = len(crop_entries)
    if total == 0:
        logger.info("No new crop entries to process — rebuilding final files from existing results")
    else:
        logger.info(
            f"Captioning {total} crop images with {MODEL}, "
            f"concurrency={args.concurrency}"
        )

        semaphore = asyncio.Semaphore(args.concurrency)
        start_time = time.time()

        # Process in chunks for progress reporting + intermediate saves
        chunk_size = args.save_every
        for chunk_start in range(0, total, chunk_size):
            chunk = crop_entries[chunk_start : chunk_start + chunk_size]
            chunk_results = await process_batch(client, chunk, semaphore, crops_dir)
            crop_results.extend(chunk_results)

            elapsed = time.time() - start_time
            processed = chunk_start + len(chunk)
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0

            total_tokens_in = sum(r.get("tokens_in", 0) for r in crop_results)
            total_tokens_out = sum(r.get("tokens_out", 0) for r in crop_results)
            cost_in = total_tokens_in * 0.05 / 1_000_000
            cost_out = total_tokens_out * 0.40 / 1_000_000
            cost = cost_in + cost_out

            logger.info(
                f"[{processed}/{total}] {rate:.1f} img/s | "
                f"ETA: {eta / 60:.0f}min | ${cost:.2f} spent | "
                f"ok: {sum(1 for r in crop_results if r['caption_short'])}"
                f"/{len(crop_results)}"
            )

            save_crop_results(crop_results, crop_results_path)

    # Final saves
    save_crop_results(crop_results, crop_results_path)

    crop_samples = build_crop_samples(crop_results, master_entries)
    save_crop_results(crop_samples, crop_samples_path)

    # Build captions_from_master_crop.json:
    # copy of captions_from_master.json + crop entries appended
    existing_captions_path = ft_dir / "captions_from_master.json"
    merged_path = ft_dir / "captions_from_master_crop.json"
    with open(existing_captions_path, encoding="utf-8") as f:
        merged = json.load(f)  # original entries, untouched
    for s in crop_samples:
        merged.append({
            "image_path": s["image_path"],
            "caption": s["caption_short"],
        })
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Saved {len(merged)} entries to {merged_path} "
        f"({len(merged) - len(crop_samples)} original + {len(crop_samples)} crop)"
    )

    elapsed = time.time() - start_time
    total_tokens_in = sum(r.get("tokens_in", 0) for r in crop_results)
    total_tokens_out = sum(r.get("tokens_out", 0) for r in crop_results)
    cost = (
        total_tokens_in * 0.05 / 1_000_000
        + total_tokens_out * 0.40 / 1_000_000
    )

    logger.info(
        f"Done! {len(crop_results)} crop captions in {elapsed / 60:.1f}min. "
        f"Cost: ${cost:.2f}. "
        f"Tokens: {total_tokens_in:,} in + {total_tokens_out:,} out"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate crop-relative captions with OpenAI"
    )
    parser.add_argument("--max-images", type=int, default=0, help="0=all")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Concurrent API requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing captions_crop.json",
    )
    parser.add_argument(
        "--save-every", type=int, default=500, help="Save results every N images"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-5-nano", help="OpenAI model to use"
    )
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY not set! Get one from "
            "https://platform.openai.com/api-keys\n"
            "Then: set OPENAI_API_KEY=sk-your-key-here"
        )
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
