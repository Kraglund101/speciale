"""
Generate text captions for anomaly images — V2 (two-image input + improved prompt).

V1 (generate_captions_ollama.py): single image with red bboxes drawn on.
V2 (this file): TWO images sent to the model:
  1. Full image with red bounding boxes highlighting defect regions
  2. Crop around the union bounding box of all anomaly pixels + 20% padding
  Improved prompt: no bbox description leak, Chinese character warning.

Uses Qwen2.5-VL 7B via vLLM. Output: captions_v2.json

Usage:
    python scripts/generate_captions_v2.py --dataset realiad --backend vllm
    python scripts/generate_captions_v2.py --dataset realiad --backend vllm --max-images 100
    python scripts/generate_captions_v2.py --dataset realiad --backend vllm --resume
"""

import argparse
import base64
import glob
import io
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# RealIAD defect code translations (Chinese abbreviations)
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

VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"


def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
    """Extract bounding boxes from binary mask using connected components.
    Returns list of bbox dicts, or None if mask is empty."""
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
            bboxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h,
                "x1_pct": round(100 * x1 / w, 1),
                "y1_pct": round(100 * y1 / h, 1),
                "x2_pct": round(100 * x2 / w, 1),
                "y2_pct": round(100 * y2 / h, 1),
                "area_pct": round(100 * len(comp_xs) / (w * h), 2),
            })

        return bboxes if bboxes else None
    except ImportError:
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None
        h, w = mask.shape
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        return [{
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": w, "h": h,
            "x1_pct": round(100 * x1 / w, 1),
            "y1_pct": round(100 * y1 / h, 1),
            "x2_pct": round(100 * x2 / w, 1),
            "y2_pct": round(100 * y2 / h, 1),
            "area_pct": round(100 * len(xs) / (w * h), 2),
        }]
    except Exception as e:
        logger.warning(f"Failed to read mask {mask_path}: {e}")
        return None


def get_union_crop(image_path: str, mask_path: str, pad_frac: float = 0.2) -> Optional[Image.Image]:
    """Crop the image to the bounding box of all anomaly pixels + padding.

    Args:
        pad_frac: Fraction of bbox size to add as padding (0.2 = 20%).
    Returns cropped PIL Image, or None if mask is empty.
    """
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            return None

        h, w = mask.shape
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        # Add padding
        bw, bh = x2 - x1, y2 - y1
        pad = max(1, int(max(bw, bh) * pad_frac))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w - 1, x2 + pad)
        y2 = min(h - 1, y2 + pad)

        img = Image.open(image_path).convert("RGB")
        crop = img.crop((x1, y1, x2 + 1, y2 + 1))
        return crop
    except Exception as e:
        logger.warning(f"Failed to crop {image_path}: {e}")
        return None


def draw_bboxes_on_image(image_path: str, bboxes: list[dict]) -> Image.Image:
    """Draw red bounding boxes on the image and return the modified PIL Image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        for offset in range(3):
            draw.rectangle(
                [bb["x1"] - offset, bb["y1"] - offset, bb["x2"] + offset, bb["y2"] + offset],
                outline="red",
            )
    return img


def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 encoded JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt_v2(product: str) -> str:
    """Build the two-image prompt.

    Image 1 = full image with bounding boxes, Image 2 = tight crop of defect area.
    """
    product_name = product.replace("_", " ")
    prompt = (
        "You are a visual inspector of industrial defects. Output solely in English. "
        "You are given two images of the same object:\n"
        "- Image 1: the full object. Red bounding boxes indicate defect regions for "
        "your reference only — describe the defect itself, not the annotations.\n"
        "- Image 2: a close-up crop of the defect area.\n"
        f"The object being inspected is a '{product_name}'. "
        "Your output will be truncated at 77 tokens, so be concise. "
        "Be careful not to include any Chinese characters. "
        "Output exactly two sentences in this format:\n"
        "The image depicts [object description], with a [defect type] "
        "observed [location]. The defect is characterized by [description] "
        "and exhibits [features]."
    )
    return prompt


def init_vllm(model_name: str = VLLM_MODEL):
    """Initialize vLLM with Qwen2.5-VL."""
    from vllm import LLM
    logger.info(f"Loading vLLM model: {model_name}")
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    return llm


def query_vllm_batch_v2(
    llm,
    image_pairs: list[tuple[Image.Image, Image.Image]],
    prompts: list[str],
    max_tokens: int = 77,
) -> list[Optional[dict]]:
    """Process a batch of two-image inputs through vLLM."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
    )

    inputs = []
    for (img_full, img_crop), prompt in zip(image_pairs, prompts):
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs.append({
            "prompt": text,
            "multi_modal_data": {"image": [img_full, img_crop]},
        })

    try:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        results = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            n_tokens = len(output.outputs[0].token_ids)
            results.append({"text": text, "eval_count": n_tokens})
        return results
    except Exception as e:
        logger.warning(f"vLLM batch error: {e}")
        return [None] * len(image_pairs)


def query_ollama_v2(
    img_full_b64: str, img_crop_b64: str, prompt: str, timeout: int = 90
) -> Optional[dict]:
    """Send two base64 images + prompt to Ollama."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [img_full_b64, img_crop_b64],
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 77,
                },
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "text": data.get("response", "").strip(),
            "eval_count": data.get("eval_count", -1),
        }
    except requests.exceptions.Timeout:
        logger.warning("Timeout querying Ollama")
        return None
    except Exception as e:
        logger.warning(f"Ollama error: {e}")
        return None


def discover_realiad_images(base_path: str) -> list[dict]:
    """Find all RealIAD NG images that have matching masks."""
    logger.info(f"Scanning RealIAD at {base_path}...")
    entries = []

    jpg_files = glob.glob(os.path.join(base_path, "*/NG/**/*.jpg"), recursive=True)
    logger.info(f"Found {len(jpg_files)} total NG jpg files")

    for jpg_path in jpg_files:
        png_path = jpg_path.replace(".jpg", ".png")
        if not os.path.exists(png_path):
            continue

        fname = os.path.basename(jpg_path)
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
            "filename": fname,
        })

    logger.info(f"Found {len(entries)} NG images with masks")
    return entries


def load_progress(output_file: str) -> set:
    """Load already-captioned filenames from output JSON."""
    if not os.path.exists(output_file):
        return set()
    with open(output_file, "r") as f:
        data = json.load(f)
    return {entry["filename"] for entry in data}


def save_progress(output_file: str, results: list[dict]) -> None:
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate captions V2 (two-image input)")
    parser.add_argument("--dataset", choices=["realiad"], default="realiad")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="0=all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    if args.dataset == "realiad":
        base_path = str(project_root / "anomverse_extension" / "datasets" / "realiad_1024")
        entries = discover_realiad_images(base_path)
        default_output = str(
            project_root / "anomverse_extension" / "datasets" / "realiad_1024" / "captions_v2.json"
        )

    output_file = args.output or default_output

    # Handle resume
    done_filenames = set()
    results = []
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
        done_filenames = {r["filename"] for r in results}
        logger.info(f"Resuming: {len(done_filenames)} already done")

    entries = [e for e in entries if e["filename"] not in done_filenames]

    if args.max_images > 0:
        entries = entries[: args.max_images]

    total = len(entries)
    logger.info(f"V2 captioning: {total} images, backend={args.backend}, two-image input")

    # Init backend
    llm = None
    if args.backend == "ollama":
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5)
        except Exception:
            logger.error("Ollama is not running!")
            sys.exit(1)
    elif args.backend == "vllm":
        llm = init_vllm()

    start_time = time.time()
    token_counts = []
    skipped_no_mask = 0

    def prepare_entry_v2(entry):
        """Prepare two images + prompt. Returns (img_full, img_crop, prompt, bboxes) or None."""
        img_abs = entry.get("image_path_abs", entry["image_path"])
        mask_abs = entry.get("mask_path_abs", entry["mask_path"])

        bboxes = get_bboxes_from_mask(mask_abs)
        if not bboxes:
            return None

        img_full = draw_bboxes_on_image(img_abs, bboxes)
        img_crop = get_union_crop(img_abs, mask_abs)
        if img_crop is None:
            return None

        prompt = build_prompt_v2(entry["product"])
        return img_full, img_crop, prompt, bboxes

    if args.backend == "vllm":
        bs = args.batch_size
        for batch_start in range(0, total, bs):
            batch_entries = entries[batch_start : batch_start + bs]

            batch_pairs = []
            batch_prompts = []
            batch_bboxes = []
            batch_valid_entries = []

            for entry in batch_entries:
                prepared = prepare_entry_v2(entry)
                if prepared is None:
                    skipped_no_mask += 1
                    continue
                img_full, img_crop, prompt, bboxes = prepared
                batch_pairs.append((img_full, img_crop))
                batch_prompts.append(prompt)
                batch_bboxes.append(bboxes)
                batch_valid_entries.append(entry)

            if not batch_pairs:
                continue

            responses = query_vllm_batch_v2(llm, batch_pairs, batch_prompts)

            for entry, response, bboxes in zip(batch_valid_entries, responses, batch_bboxes):
                if response is None:
                    continue
                token_counts.append(response["eval_count"])
                results.append({
                    "filename": entry["filename"],
                    "image_path": entry["image_path"],
                    "mask_path": entry["mask_path"],
                    "product": entry["product"],
                    "defect_code": entry["defect_code"],
                    "defect_name": DEFECT_CODE_MAP.get(entry["defect_code"], entry["defect_code"]),
                    "bboxes": bboxes,
                    "num_defect_regions": len(bboxes) if bboxes else 0,
                    "caption": response["text"],
                    "token_count": response["eval_count"],
                    "version": "v2_two_image_padded",
                })

            processed = min(batch_start + bs, total)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            if processed <= bs or processed % 1000 < bs:
                logger.info(
                    f"[{processed}/{total}] {rate:.2f} img/s | "
                    f"ETA: {eta/3600:.1f}h | avg_tokens: {avg_tokens:.0f} | "
                    f"skipped_no_mask: {skipped_no_mask}"
                )

            if processed % args.save_every < bs:
                save_progress(output_file, results)
                logger.info(f"Progress saved: {len(results)} captions")

    elif args.backend == "ollama":
        consecutive_errors = 0
        for i, entry in enumerate(entries):
            prepared = prepare_entry_v2(entry)
            if prepared is None:
                skipped_no_mask += 1
                continue
            img_full, img_crop, prompt, bboxes = prepared

            response = query_ollama_v2(
                pil_to_base64(img_full), pil_to_base64(img_crop), prompt
            )

            if response is None:
                consecutive_errors += 1
                if consecutive_errors > 10:
                    logger.error("Too many consecutive errors, stopping.")
                    break
                continue
            else:
                consecutive_errors = 0

            token_counts.append(response["eval_count"])
            results.append({
                "filename": entry["filename"],
                "image_path": entry["image_path"],
                "mask_path": entry["mask_path"],
                "product": entry["product"],
                "defect_code": entry["defect_code"],
                "defect_name": DEFECT_CODE_MAP.get(entry["defect_code"], entry["defect_code"]),
                "bboxes": bboxes,
                "num_defect_regions": len(bboxes) if bboxes else 0,
                "caption": response["text"],
                "token_count": response["eval_count"],
                "version": "v2_two_image_padded",
            })

            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    f"[{i+1}/{total}] {rate:.2f} img/s | "
                    f"ETA: {eta/3600:.1f}h | avg_tokens: {avg_tokens:.0f} | "
                    f"{entry['product']}/{entry['defect_code']}"
                )

            if (i + 1) % args.save_every == 0:
                save_progress(output_file, results)
                logger.info(f"Progress saved: {len(results)} captions")

    # Final save
    save_progress(output_file, results)
    total_time = time.time() - start_time
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    logger.info(
        f"V2 Done! Generated {len(results)} captions in {total_time/3600:.1f}h. "
        f"Avg tokens: {avg_tokens:.0f}. Skipped (no mask): {skipped_no_mask}. "
        f"Saved to {output_file}"
    )


if __name__ == "__main__":
    main()
