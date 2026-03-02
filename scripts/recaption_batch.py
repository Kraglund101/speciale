"""Submit remaining re-caption jobs via OpenAI Batch API (no rate limits, 50% cheaper).

Reads _recaption_progress.json to find already-done image_paths,
submits the remaining ones as a batch, then polls until complete.

Usage:
    # Submit batch for remaining images
    python scripts/recaption_batch.py submit

    # Check batch status
    python scripts/recaption_batch.py status

    # Download results and apply patches
    python scripts/recaption_batch.py apply
"""

import argparse
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
TARGET_DATASETS = {"mvtec", "DAGM", "mvtec3d", "mvtec_ad_2", "MulSen_AD", "BTech", "MPDD", "eyecandies"}

BATCH_STATE_PATH = FT_DIR / "_batch_state.json"


# ---------------------------------------------------------------------------
# Image utilities (same as recaption_datasets.py)
# ---------------------------------------------------------------------------


def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
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
    if max(img.size) > max_size:
        img = img.copy()
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(product: str, defect_type: str) -> str:
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
# Find remaining entries
# ---------------------------------------------------------------------------


def get_remaining_entries() -> list[dict]:
    """Get master entries that still need captioning."""
    master_path = FT_DIR / "master_training.json"
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    target_entries = [e for e in master if e["dataset"] in TARGET_DATASETS]

    # Load progress from real-time run
    progress_path = FT_DIR / "_recaption_progress.json"
    done_paths = set()
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        # Only count successful ones as done
        done_paths = {r["image_path"] for r in progress if not r.get("failed") and r.get("caption_short")}

    remaining = [e for e in target_entries if e["image_path"] not in done_paths]
    logger.info(f"Total target: {len(target_entries)}, already done: {len(done_paths)}, remaining: {len(remaining)}")
    return remaining


# ---------------------------------------------------------------------------
# Submit batch
# ---------------------------------------------------------------------------


def submit_batch(entries: list[dict]):
    """Build JSONL file and submit to OpenAI Batch API."""
    from openai import OpenAI
    client = OpenAI()

    # Build JSONL requests
    jsonl_path = FT_DIR / "_batch_requests.jsonl"
    skipped = 0
    written = 0

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            image_abs = str(FT_DIR / entry["image_path"])
            mask_abs = str(FT_DIR / entry["mask_path"])

            bboxes = get_bboxes_from_mask(mask_abs)
            if not bboxes:
                skipped += 1
                continue

            img_full = draw_bboxes_on_image(image_abs, bboxes)
            img_crop = get_union_crop(image_abs, mask_abs)
            if img_crop is None:
                skipped += 1
                continue

            prompt = build_prompt(entry["product"], entry["defect_type"])
            img_full_b64 = pil_to_base64(img_full)
            img_crop_b64 = pil_to_base64(img_crop)

            request = {
                "custom_id": entry["image_path"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
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
                    "max_tokens": 300,
                    "temperature": 0.7,
                },
            }

            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            written += 1

            if written % 100 == 0:
                logger.info(f"Prepared {written} requests...")

    logger.info(f"JSONL ready: {written} requests, {skipped} skipped. File: {jsonl_path}")

    # Upload file
    logger.info("Uploading JSONL to OpenAI...")
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    logger.info(f"Uploaded file: {uploaded.id}")

    # Submit batch
    logger.info("Submitting batch...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": f"recaption {written} non-RealIAD images"},
    )
    logger.info(f"Batch submitted: {batch.id} (status: {batch.status})")

    # Save state
    state = {
        "batch_id": batch.id,
        "input_file_id": uploaded.id,
        "num_requests": written,
        "submitted_at": time.time(),
    }
    with open(BATCH_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved batch state to {BATCH_STATE_PATH}")


# ---------------------------------------------------------------------------
# Check status
# ---------------------------------------------------------------------------


def check_status():
    """Check batch status (supports single batch_id or list of batch_ids)."""
    from openai import OpenAI
    client = OpenAI()

    if not BATCH_STATE_PATH.exists():
        logger.error("No batch state file found. Run 'submit' first.")
        return

    with open(BATCH_STATE_PATH) as f:
        state = json.load(f)

    # Support both old (batch_id) and new (batch_ids) format
    batch_ids = state.get("batch_ids") or [state["batch_id"]]
    elapsed = time.time() - state["submitted_at"]

    all_completed = True
    total_ok = 0
    total_failed = 0
    output_file_ids = []

    for i, bid in enumerate(batch_ids):
        batch = client.batches.retrieve(bid)
        print(f"\n--- Batch {i+1}/{len(batch_ids)} ---")
        print(f"Batch ID:    {batch.id}")
        print(f"Status:      {batch.status}")
        if batch.request_counts:
            print(f"Completed:   {batch.request_counts.completed}")
            print(f"Failed:      {batch.request_counts.failed}")
            total_ok += batch.request_counts.completed
            total_failed += batch.request_counts.failed

        if batch.status != "completed":
            all_completed = False
        if batch.output_file_id:
            output_file_ids.append(batch.output_file_id)

    print(f"\n--- Summary ---")
    print(f"Elapsed:     {elapsed/60:.0f} min")
    print(f"Total req:   {state['num_requests']}")
    print(f"Completed:   {total_ok}")
    print(f"Failed:      {total_failed}")

    if all_completed:
        print(f"\nAll batches completed! Output files: {output_file_ids}")
        print("Run 'apply' to download results and patch captions.")
    else:
        print("\nNot all batches completed yet.")

    # Update state
    state["output_file_ids"] = output_file_ids
    state["all_completed"] = all_completed
    with open(BATCH_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Apply results
# ---------------------------------------------------------------------------


def apply_results():
    """Download batch results and merge with real-time progress, then patch JSONs."""
    from openai import OpenAI
    client = OpenAI()

    if not BATCH_STATE_PATH.exists():
        logger.error("No batch state file found.")
        return

    with open(BATCH_STATE_PATH) as f:
        state = json.load(f)

    # Support both old (batch_id) and new (batch_ids) format
    batch_ids = state.get("batch_ids") or [state["batch_id"]]

    # Verify all batches completed
    output_file_ids = []
    for bid in batch_ids:
        batch = client.batches.retrieve(bid)
        if batch.status != "completed":
            logger.error(f"Batch {bid} not completed yet (status: {batch.status}). Run 'status' to check.")
            return
        output_file_ids.append(batch.output_file_id)

    # Download and merge results from all batches
    results_path = FT_DIR / "_batch_results.jsonl"
    with open(results_path, "wb") as f:
        for i, ofid in enumerate(output_file_ids):
            logger.info(f"Downloading results from batch {i+1}/{len(output_file_ids)} ({ofid})...")
            content = client.files.content(ofid)
            data = content.read()
            f.write(data)
            # Ensure newline between files
            if not data.endswith(b"\n"):
                f.write(b"\n")
    logger.info(f"Saved merged results to {results_path}")

    # Parse results
    batch_replacements = {}
    ok_count = 0
    fail_count = 0

    with open(results_path, encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            image_path = result["custom_id"]

            if result.get("error"):
                fail_count += 1
                logger.warning(f"Batch error for {image_path}: {result['error']}")
                continue

            body = result["response"]["body"]
            text = body["choices"][0]["message"]["content"].strip()
            short_cap, long_cap = parse_response(text)

            if short_cap:
                batch_replacements[image_path] = {"short": short_cap, "long": long_cap or ""}
                ok_count += 1
            else:
                fail_count += 1

    logger.info(f"Batch results: {ok_count} ok, {fail_count} failed")

    # Load real-time progress replacements
    progress_path = FT_DIR / "_recaption_progress.json"
    rt_replacements = {}
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            progress = json.load(f)
        for r in progress:
            if not r.get("failed") and r.get("caption_short"):
                rt_replacements[r["image_path"]] = {
                    "short": r["caption_short"],
                    "long": r.get("caption_long", ""),
                }
    logger.info(f"Real-time progress: {len(rt_replacements)} successful")

    # Merge: real-time + batch
    all_replacements = {**rt_replacements, **batch_replacements}
    logger.info(f"Total replacements: {len(all_replacements)}")

    # Patch captions_from_master.json
    captions_path = FT_DIR / "captions_from_master.json"
    with open(captions_path, encoding="utf-8") as f:
        caps = json.load(f)

    patched_caps = 0
    for item in caps:
        if item["image_path"] in all_replacements:
            item["caption"] = all_replacements[item["image_path"]]["short"]
            patched_caps += 1

    with open(captions_path, "w", encoding="utf-8") as f:
        json.dump(caps, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched_caps} in captions_from_master.json")

    # Patch master_training.json
    master_path = FT_DIR / "master_training.json"
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)

    patched_master = 0
    for entry in master:
        if entry["image_path"] in all_replacements:
            entry["caption_short"] = all_replacements[entry["image_path"]]["short"]
            entry["caption_long"] = all_replacements[entry["image_path"]]["long"]
            patched_master += 1

    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)
    logger.info(f"Patched {patched_master} in master_training.json")

    logger.info("Done! All captions patched.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Batch API re-captioning for remaining images")
    parser.add_argument("action", choices=["submit", "status", "apply"],
                        help="submit=create batch, status=check progress, apply=download+patch")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    global MODEL
    MODEL = args.model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set!")
        sys.exit(1)

    if args.action == "submit":
        entries = get_remaining_entries()
        if not entries:
            logger.info("Nothing to submit — all done!")
            return
        submit_batch(entries)
    elif args.action == "status":
        check_status()
    elif args.action == "apply":
        apply_results()


if __name__ == "__main__":
    main()
