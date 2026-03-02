"""Re-caption the ~54 samples whose captions say 'no visible defect' etc.

Uses the same GPT-4o-mini pipeline as generate_captions_openai.py:
  - Image 1: full image with red bounding boxes on defect regions
  - Image 2: close-up crop of the defect area

After generating new captions, patches captions_from_master.json in-place
(with a backup saved as captions_from_master_backup.json).

Usage:
    # Dry run — show which captions would be replaced
    python scripts/recaption_bad.py --dry-run

    # Actually re-caption and patch
    python scripts/recaption_bad.py
"""

import asyncio
import base64
import io
import json
import logging
import os
import shutil
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


# ---------------------------------------------------------------------------
# Image utilities (same as generate_captions_openai.py)
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
# Find bad captions
# ---------------------------------------------------------------------------


def find_bad_captions(caps: list[dict]) -> list[dict]:
    """Find captions that say 'no visible defect' or similar."""
    bad = []
    for item in caps:
        cl = item["caption"].lower()
        if "no visible" in cl:
            # Skip RealIAD ones describing component state (not defect absence)
            if any(x in cl for x in [
                "no visible leads", "no visible contact",
                "no visible component", "no visible solder",
            ]):
                continue
            bad.append(item)
    return bad


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
# Main
# ---------------------------------------------------------------------------


async def run(dry_run: bool = False):
    """Find bad captions, re-caption them, patch the JSON."""

    # Load data
    captions_path = FT_DIR / "captions_from_master.json"
    master_path = FT_DIR / "master_training.json"

    with open(captions_path, encoding="utf-8") as f:
        caps = json.load(f)
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    master_map = {e["image_path"]: e for e in master}

    # Find bad ones
    bad = find_bad_captions(caps)
    logger.info(f"Found {len(bad)} bad captions to re-caption")

    if dry_run:
        for item in bad:
            m = master_map.get(item["image_path"], {})
            logger.info(
                f"  [{m.get('dataset','?')}/{m.get('product','?')}/{m.get('defect_type','?')}] "
                f"{item['caption'][:120]}"
            )
        logger.info("Dry run — no changes made.")
        return

    # API client
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    # Process each bad caption
    bad_image_paths = {item["image_path"] for item in bad}
    replacements = {}  # image_path -> new short caption
    total_tokens_in = 0
    total_tokens_out = 0

    for i, item in enumerate(bad):
        image_path = item["image_path"]
        m = master_map.get(image_path)
        if not m:
            logger.warning(f"No master entry for {image_path}, skipping")
            continue

        image_abs = str(FT_DIR / m["image_path"])
        mask_abs = str(FT_DIR / m["mask_path"])

        # Prepare images
        bboxes = get_bboxes_from_mask(mask_abs)
        if not bboxes:
            logger.warning(f"No bboxes for {image_path}, skipping")
            continue

        img_full = draw_bboxes_on_image(image_abs, bboxes)
        img_crop = get_union_crop(image_abs, mask_abs)
        if img_crop is None:
            logger.warning(f"No crop for {image_path}, skipping")
            continue

        prompt = build_prompt(m["product"], m["defect_type"])
        img_full_b64 = pil_to_base64(img_full)
        img_crop_b64 = pil_to_base64(img_crop)

        response = await query_openai(client, img_full_b64, img_crop_b64, prompt)
        if response is None:
            logger.warning(f"API failed for {image_path}")
            continue

        short_cap, long_cap = parse_response(response["text"])
        total_tokens_in += response["tokens_in"]
        total_tokens_out += response["tokens_out"]

        if short_cap:
            replacements[image_path] = short_cap
            logger.info(
                f"[{i+1}/{len(bad)}] {m['product']}/{m['defect_type']}\n"
                f"  OLD: {item['caption'][:100]}\n"
                f"  NEW: {short_cap[:100]}"
            )
        else:
            logger.warning(f"Failed to parse caption for {image_path}")

    logger.info(f"\nGenerated {len(replacements)}/{len(bad)} new captions")

    cost = total_tokens_in * 0.15 / 1_000_000 + total_tokens_out * 0.60 / 1_000_000
    logger.info(f"Cost: ${cost:.4f} ({total_tokens_in:,} in + {total_tokens_out:,} out)")

    if not replacements:
        logger.info("No replacements to apply.")
        return

    # Backup original
    backup_path = FT_DIR / "captions_from_master_backup.json"
    if not backup_path.exists():
        shutil.copy2(captions_path, backup_path)
        logger.info(f"Backed up original to {backup_path}")

    # Patch captions_from_master.json
    patched = 0
    for item in caps:
        if item["image_path"] in replacements:
            item["caption"] = replacements[item["image_path"]]
            patched += 1

    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched} captions in {captions_path}")

    # Also update master_training.json caption_short
    patched_master = 0
    for entry in master:
        if entry["image_path"] in replacements:
            entry["caption_short"] = replacements[entry["image_path"]]
            patched_master += 1

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched_master} caption_short in {master_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-caption bad 'no visible defect' captions")
    parser.add_argument("--dry-run", action="store_true", help="Just show bad captions, don't call API")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.dry_run:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)

    asyncio.run(run(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
