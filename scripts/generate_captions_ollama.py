"""
Generate text captions for anomaly images using Qwen2.5-VL via Ollama or vLLM.

Iterates through RealIAD (or AnomVerse) images that have pixel masks,
draws red bounding boxes on the image to highlight defect regions,
and sends to Qwen2.5-VL 7B for captioning.

Usage:
    python scripts/generate_captions_ollama.py --dataset realiad
    python scripts/generate_captions_ollama.py --dataset realiad --backend vllm
    python scripts/generate_captions_ollama.py --dataset realiad --max-images 10  # test run
    python scripts/generate_captions_ollama.py --dataset realiad --resume  # continue from last
"""

import argparse
import base64
import glob
import io
import json
import logging
import os
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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5vl:7b"
VLLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
    """Extract bounding boxes from binary mask. Finds separate connected components.
    Returns list of bbox dicts, or None if mask is empty."""
    try:
        mask = np.array(Image.open(mask_path).convert("L"))
        binary = (mask > 127).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None

        h, w = mask.shape

        # Find connected components using simple flood-fill labeling
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)

        bboxes = []
        for i in range(1, num_features + 1):
            comp_ys, comp_xs = np.where(labeled == i)
            if len(comp_xs) < 5:  # Skip tiny noise regions (<5 pixels)
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
        # Fallback without scipy: single overall bbox
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


def draw_bboxes_on_image(image_path: str, bboxes: list[dict]) -> Image.Image:
    """Draw red bounding boxes on the image and return the modified PIL Image."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bb in bboxes:
        # Draw 3-pixel wide red rectangle
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


def build_prompt(product: str) -> str:
    """Build the prompt for the vision model.

    Does NOT leak the defect type - the model must infer it from the image.
    The product/object type is provided to help context.
    """
    product_name = product.replace("_", " ")
    prompt = (
        "You are a visual inspector of industrial defects, output solely in English "
        "IMPORTANTLY no Chinese. "
        "The defects you will observe are highlighted with bounding boxes. "
        f"The object being inspected is a '{product_name}'. "
        "For each image, you will have to output:\n"
        "The image depicts [general description of the object], with a [type of defect] "
        "observed [location]. The defect is characterized by [detailed description] "
        "and exhibits [notable features]."
    )
    return prompt


def query_ollama(
    img_b64: str, prompt: str, timeout: int = 90
) -> Optional[dict]:
    """Send base64 image + prompt to Ollama and return response dict with text and token count."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [img_b64],
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


def init_vllm(model_name: str = VLLM_MODEL):
    """Initialize vLLM with Qwen2.5-VL. Returns the LLM instance."""
    from vllm import LLM, SamplingParams  # noqa: F811
    logger.info(f"Loading vLLM model: {model_name}")
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=16,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )
    return llm


def query_vllm_batch(
    llm,
    images: list[Image.Image],
    prompts: list[str],
    prompt_template: str = "",
    max_tokens: int = 77,
) -> list[Optional[dict]]:
    """Process a batch of images through vLLM. Passes PIL images directly — no base64."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
    )

    # Build prompts with image placeholder — pass PIL images via multi_modal_data
    inputs = []
    for img, prompt in zip(images, prompts):
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs.append({"prompt": text, "multi_modal_data": {"image": img}})

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
        return [None] * len(images)


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

        # Parse filename: product__NNNN_NG_CODE_CN_timestamp.jpg
        fname = os.path.basename(jpg_path)
        parts = fname.split("_")
        # Extract product from path
        rel = os.path.relpath(jpg_path, base_path)
        product = rel.split(os.sep)[0]
        # Extract defect code from path
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


def discover_anomverse_images(splits_dir: str, data_root: str) -> list[dict]:
    """Find all AnomVerse images from splits JSON files."""
    logger.info(f"Scanning AnomVerse splits at {splits_dir}...")
    entries = []

    for json_file in glob.glob(os.path.join(splits_dir, "*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)

        concept = data.get("concept_name", "unknown")
        for img in data.get("images", []):
            img_abs = img.get("image_path_full") or img.get("image_path_abs", "")
            mask_abs = img.get("mask_path_full") or img.get("mask_path_abs", "")
            if not img_abs or not mask_abs:
                continue

            full_img = os.path.join(data_root, img_abs)
            full_mask = os.path.join(data_root, mask_abs)

            if os.path.exists(full_img) and os.path.exists(full_mask):
                entries.append({
                    "image_path": img_abs,
                    "mask_path": mask_abs,
                    "image_path_abs": full_img,
                    "mask_path_abs": full_mask,
                    "product": img.get("category", "unknown"),
                    "defect_code": concept,
                    "filename": os.path.basename(full_img),
                })

    logger.info(f"Found {len(entries)} AnomVerse images with masks")
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
    parser = argparse.ArgumentParser(description="Generate captions for anomaly images")
    parser.add_argument("--dataset", choices=["realiad", "anomverse"], default="realiad")
    parser.add_argument("--backend", choices=["ollama", "vllm"], default="ollama",
                        help="Inference backend: ollama (sequential) or vllm (batched)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for vLLM backend")
    parser.add_argument("--max-images", type=int, default=0, help="Max images to process (0=all)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run")
    parser.add_argument("--save-every", type=int, default=100, help="Save progress every N images")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: auto based on dataset)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Set paths based on dataset
    if args.dataset == "realiad":
        base_path = str(project_root / "anomverse_extension" / "datasets" / "realiad_1024")
        entries = discover_realiad_images(base_path)
        default_output = str(
            project_root / "anomverse_extension" / "datasets" / "realiad_1024" / "captions.json"
        )
    else:
        splits_dir = str(project_root / "data" / "concepts")
        data_root = str(project_root)
        entries = discover_anomverse_images(splits_dir, data_root)
        default_output = str(project_root / "data" / "captions.json")

    output_file = args.output or default_output

    # Handle resume
    done_filenames = set()
    results = []
    if args.resume and os.path.exists(output_file):
        with open(output_file, "r") as f:
            results = json.load(f)
        done_filenames = {r["filename"] for r in results}
        logger.info(f"Resuming: {len(done_filenames)} already done")

    # Filter out already done
    entries = [e for e in entries if e["filename"] not in done_filenames]

    if args.max_images > 0:
        entries = entries[: args.max_images]

    total = len(entries)
    logger.info(f"Processing {total} images with backend={args.backend}...")

    # Backend-specific init
    llm = None
    if args.backend == "ollama":
        try:
            requests.get("http://localhost:11434/api/tags", timeout=5)
        except Exception:
            logger.error("Ollama is not running! Start it first.")
            sys.exit(1)
    elif args.backend == "vllm":
        llm = init_vllm()

    start_time = time.time()
    consecutive_errors = 0
    token_counts = []

    def prepare_entry(entry):
        """Prepare image + prompt for an entry. Returns (img_pil, img_b64, prompt, bboxes)."""
        img_abs = entry.get("image_path_abs", entry["image_path"])
        mask_abs = entry.get("mask_path_abs", entry["mask_path"])
        bboxes = get_bboxes_from_mask(mask_abs)
        if bboxes:
            img_with_bbox = draw_bboxes_on_image(img_abs, bboxes)
        else:
            img_with_bbox = Image.open(img_abs).convert("RGB")
        prompt = build_prompt(entry["product"])
        return img_with_bbox, pil_to_base64(img_with_bbox), prompt, bboxes

    if args.backend == "ollama":
        # Sequential processing
        for i, entry in enumerate(entries):
            img_pil, img_b64, prompt, bboxes = prepare_entry(entry)
            response = query_ollama(img_b64, prompt)

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

    elif args.backend == "vllm":
        # Batch processing
        bs = args.batch_size
        for batch_start in range(0, total, bs):
            batch_entries = entries[batch_start : batch_start + bs]

            # Prepare all entries in this batch
            batch_imgs = []
            batch_prompts = []
            batch_bboxes = []
            for entry in batch_entries:
                img_pil, _, prompt, bboxes = prepare_entry(entry)
                batch_imgs.append(img_pil)
                batch_prompts.append(prompt)
                batch_bboxes.append(bboxes)

            # Run batch inference
            responses = query_vllm_batch(llm, batch_imgs, batch_prompts)

            for entry, response, bboxes in zip(batch_entries, responses, batch_bboxes):
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
                })

            processed = min(batch_start + bs, total)
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (total - processed) / rate if rate > 0 else 0
            avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
            # Log every 1000 images (or first batch)
            if processed <= bs or processed % 1000 < bs:
                logger.info(
                    f"[{processed}/{total}] {rate:.2f} img/s | "
                    f"ETA: {eta/3600:.1f}h | avg_tokens: {avg_tokens:.0f}"
                )

            if processed % args.save_every < bs:
                save_progress(output_file, results)
                logger.info(f"Progress saved: {len(results)} captions")

    # Final save
    save_progress(output_file, results)
    total_time = time.time() - start_time
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    logger.info(
        f"Done! Generated {len(results)} captions in {total_time/3600:.1f}h. "
        f"Avg tokens: {avg_tokens:.0f}. Saved to {output_file}"
    )


if __name__ == "__main__":
    main()
