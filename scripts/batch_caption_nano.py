"""
Batch captioning of ~50K Real-IAD images using GPT-5-nano (high detail).

Usage:
    # Step 1: Generate JSONL batch files
    python scripts/batch_caption_nano.py prepare

    # Step 1.5: Re-split into smaller files (if token limit hit)
    python scripts/batch_caption_nano.py resplit

    # Step 2: Submit batches sequentially (one at a time, waits for completion)
    python scripts/batch_caption_nano.py submit

    # Step 3: Check status of all batches
    python scripts/batch_caption_nano.py status

    # Step 4: Download results and parse into final JSON
    python scripts/batch_caption_nano.py collect
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "anomverse_extension" / "datasets" / "realiad_1024"
BATCH_DIR = DATA_ROOT / "batch_nano"
OUTPUT_FILE = DATA_ROOT / "captions_nano.json"

MODEL = "gpt-5-nano"
MAX_REQUESTS_PER_FILE = 500  # Keep under 2M token enqueue limit (~500 × 1780 = 890K tokens)
MAX_COMPLETION_TOKENS = 5000

DEFECT_CODE_MAP = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
}


# ── Image processing (reused from existing scripts) ──────────────────────────

def get_bboxes_from_mask(mask_path: str) -> Optional[list[dict]]:
    from scipy import ndimage
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
            bboxes.append({
                "x1": int(comp_xs.min()), "y1": int(comp_ys.min()),
                "x2": int(comp_xs.max()), "y2": int(comp_ys.max()),
            })
        return bboxes
    except Exception:
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


def get_union_crop(image_path: str, mask_path: str, pad_frac: float = 0.2) -> Optional[Image.Image]:
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


def pil_to_base64(img: Image.Image, max_size: int = 1024) -> str:
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_prompt(product: str, defect_type: str) -> str:
    return (
        "You are a visual inspector of industrial defects. "
        "These captions will be used as text conditioning in a Stable Diffusion "
        "inpainting model for synthetic anomaly generation — describe the visual "
        "appearance precisely so the diffusion model can reproduce it.\n\n"
        "You are given two images of the same object:\n"
        "- Image 1: the full object with red bounding boxes marking defect regions "
        "(describe the defect, not the annotations).\n"
        "- Image 2: a close-up crop of the defect area.\n\n"
        f"Object: '{product.replace('_', ' ')}'. Defect type: '{defect_type.replace('_', ' ')}'.\n\n"
        "Generate TWO captions:\n"
        "SHORT (max 60 words): A single dense sentence describing the defect's "
        "visual appearance — texture, color, shape, material, and location on "
        "the object.\n"
        "LONG (max 150 words): 2-3 sentences following this structure: "
        "(1) what the defect looks like and where it is on the object, "
        "(2) detailed visual characteristics (texture, color, edges, surface finish), "
        "(3) how it contrasts with the surrounding normal surface.\n\n"
        "Do not include physical measurements (mm, cm, etc.) — use relative size "
        "descriptions instead (tiny, small, large, etc.).\n"
        "Do not include any Chinese characters.\n"
        "Do not describe the bounding boxes or annotations.\n\n"
        "Output format (exactly):\n"
        "SHORT: <caption>\n"
        "LONG: <caption>"
    )


def parse_response(text: str) -> tuple[Optional[str], Optional[str]]:
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


def discover_images() -> list[dict]:
    """Find all RealIAD NG images with masks."""
    import glob as glob_mod
    base_path = str(DATA_ROOT)
    jpg_files = glob_mod.glob(os.path.join(base_path, "*/NG/**/*.jpg"), recursive=True)
    logger.info(f"Found {len(jpg_files)} total NG jpg files")

    entries = []
    for jpg_path in sorted(jpg_files):
        png_path = jpg_path.replace(".jpg", ".png")
        if not os.path.exists(png_path):
            continue
        rel = os.path.relpath(jpg_path, base_path)
        product = rel.split(os.sep)[0]
        parts = Path(jpg_path).parts
        ng_idx = [i for i, p in enumerate(parts) if p == "NG"]
        defect_code = parts[ng_idx[0] + 1] if ng_idx else "unknown"
        entries.append({
            "filename": os.path.basename(jpg_path),
            "image_path": rel.replace(os.sep, "/"),
            "mask_path": os.path.relpath(png_path, base_path).replace(os.sep, "/"),
            "image_abs": jpg_path,
            "mask_abs": png_path,
            "product": product,
            "defect_code": defect_code,
            "defect_name": DEFECT_CODE_MAP.get(defect_code, defect_code),
        })
    logger.info(f"Found {len(entries)} NG images with masks")
    return entries


# ── Phase 1: PREPARE ─────────────────────────────────────────────────────────

def cmd_prepare(args):
    """Generate JSONL batch input files."""
    entries = discover_images()

    # Check for already-done entries (resume support)
    done_filenames = set()
    if OUTPUT_FILE.exists():
        existing = json.load(open(OUTPUT_FILE))
        done_filenames = {e["filename"] for e in existing}
        logger.info(f"Skipping {len(done_filenames)} already-captioned images")

    entries = [e for e in entries if e["filename"] not in done_filenames]
    logger.info(f"Preparing {len(entries)} images for batch processing")

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    # Also save the entry metadata for later (needed during collect)
    meta_path = BATCH_DIR / "entries_meta.json"
    meta = {e["filename"]: {
        "image_path": e["image_path"],
        "mask_path": e["mask_path"],
        "product": e["product"],
        "defect_code": e["defect_code"],
        "defect_name": e["defect_name"],
    } for e in entries}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    file_idx = 0
    request_count = 0
    skipped = 0
    current_file = None
    current_count = 0

    for i, entry in enumerate(entries):
        # Prepare image data
        bboxes = get_bboxes_from_mask(entry["mask_abs"])
        if not bboxes:
            skipped += 1
            continue

        full_img = draw_bboxes_on_image(entry["image_abs"], bboxes)
        crop_img = get_union_crop(entry["image_abs"], entry["mask_abs"])
        if crop_img is None:
            skipped += 1
            continue

        full_b64 = pil_to_base64(full_img)
        crop_b64 = pil_to_base64(crop_img)
        prompt = build_prompt(entry["product"], entry["defect_name"])

        # Build batch request
        request = {
            "custom_id": entry["filename"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "max_completion_tokens": MAX_COMPLETION_TOKENS,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{full_b64}",
                            "detail": "high",
                        }},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{crop_b64}",
                            "detail": "high",
                        }},
                    ],
                }],
            },
        }

        # Open new file if needed
        if current_file is None or current_count >= MAX_REQUESTS_PER_FILE:
            if current_file is not None:
                current_file.close()
                logger.info(f"  Wrote batch_{file_idx:03d}.jsonl ({current_count} requests)")
                file_idx += 1
            fpath = BATCH_DIR / f"batch_{file_idx:03d}.jsonl"
            current_file = open(fpath, "w")
            current_count = 0

        current_file.write(json.dumps(request) + "\n")
        current_count += 1
        request_count += 1

        if (i + 1) % 1000 == 0:
            logger.info(f"  Processed {i + 1}/{len(entries)} images...")

    # Close last file
    if current_file is not None:
        current_file.close()
        logger.info(f"  Wrote batch_{file_idx:03d}.jsonl ({current_count} requests)")

    total_files = file_idx + 1 if request_count > 0 else 0
    logger.info(f"\nDone! {request_count} requests in {total_files} files. Skipped {skipped}.")
    logger.info(f"JSONL files saved to: {BATCH_DIR}")

    # Save batch manifest
    manifest = {
        "total_requests": request_count,
        "total_files": total_files,
        "skipped": skipped,
        "model": MODEL,
        "files": [f"batch_{i:03d}.jsonl" for i in range(total_files)],
    }
    with open(BATCH_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


# ── Phase 1.5: RESPLIT ────────────────────────────────────────────────────────

def cmd_resplit(args):
    """Re-split existing JSONL files into smaller chunks (no re-prepare needed)."""
    manifest_path = BATCH_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json found. Run 'prepare' first.")
        sys.exit(1)

    manifest = json.load(open(manifest_path))
    old_files = manifest["files"]

    # Read all requests from existing files
    all_requests = []
    for fname in old_files:
        fpath = BATCH_DIR / fname
        if not fpath.exists():
            logger.warning(f"  {fname} not found, skipping")
            continue
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    all_requests.append(line)

    logger.info(f"Read {len(all_requests)} total requests from {len(old_files)} old files")

    # Delete old batch files
    for fname in old_files:
        fpath = BATCH_DIR / fname
        if fpath.exists():
            fpath.unlink()

    # Clear old batch_status.json (old batch IDs are invalid)
    status_path = BATCH_DIR / "batch_status.json"
    if status_path.exists():
        status_path.unlink()

    # Write new smaller files
    new_files = []
    for chunk_start in range(0, len(all_requests), MAX_REQUESTS_PER_FILE):
        chunk = all_requests[chunk_start:chunk_start + MAX_REQUESTS_PER_FILE]
        file_idx = len(new_files)
        fname = f"batch_{file_idx:03d}.jsonl"
        fpath = BATCH_DIR / fname
        with open(fpath, "w") as f:
            for req in chunk:
                f.write(req + "\n")
        new_files.append(fname)
        logger.info(f"  Wrote {fname} ({len(chunk)} requests)")

    # Update manifest
    manifest["files"] = new_files
    manifest["total_files"] = len(new_files)
    manifest["requests_per_file"] = MAX_REQUESTS_PER_FILE
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nResplit into {len(new_files)} files of {MAX_REQUESTS_PER_FILE} requests each")


# ── Phase 2: SUBMIT ──────────────────────────────────────────────────────────

def _wait_for_batch(client, batch_id: str, fname: str, poll_interval: int = 30) -> dict:
    """Poll a batch until it completes, fails, or expires."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        rc = batch.request_counts
        done = rc.completed if rc else 0
        total = rc.total if rc else 0
        failed = rc.failed if rc else 0

        if status in ("completed", "failed", "expired", "cancelled"):
            logger.info(f"  {fname}: {status} ({done}/{total} done, {failed} failed)")
            return {
                "status": status,
                "output_file_id": getattr(batch, "output_file_id", None),
                "error_file_id": getattr(batch, "error_file_id", None),
                "errors": batch.errors.model_dump() if batch.errors else None,
            }

        logger.info(f"  {fname}: {status} ({done}/{total} done) — waiting {poll_interval}s...")
        time.sleep(poll_interval)


def cmd_submit(args):
    """Upload JSONL files and create batches SEQUENTIALLY (one at a time, wait for completion)."""
    from openai import OpenAI
    client = OpenAI()

    manifest_path = BATCH_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json found. Run 'prepare' first.")
        sys.exit(1)

    manifest = json.load(open(manifest_path))
    status_path = BATCH_DIR / "batch_status.json"

    # Load existing status (for resume)
    batch_status = {}
    if status_path.exists():
        batch_status = json.load(open(status_path))

    total_files = len(manifest["files"])
    completed_count = sum(1 for v in batch_status.values() if v.get("status") == "completed")
    logger.info(f"Processing {total_files} batch files ({completed_count} already completed)...")

    start_time = time.time()

    for i, fname in enumerate(manifest["files"]):
        # Skip already completed batches
        if fname in batch_status and batch_status[fname].get("status") == "completed":
            logger.info(f"  [{i+1}/{total_files}] {fname}: already completed, skipping")
            continue

        # Skip already submitted but not yet checked — re-check status
        if fname in batch_status and batch_status[fname].get("batch_id"):
            old_status = batch_status[fname].get("status", "")
            if old_status in ("failed", "expired", "cancelled"):
                logger.info(f"  [{i+1}/{total_files}] {fname}: previously {old_status}, re-submitting...")
            else:
                # Check if it completed since last run
                logger.info(f"  [{i+1}/{total_files}] {fname}: checking existing batch...")
                result = _wait_for_batch(client, batch_status[fname]["batch_id"], fname)
                batch_status[fname].update(result)
                with open(status_path, "w") as f:
                    json.dump(batch_status, f, indent=2)
                if result["status"] == "completed":
                    continue
                logger.info(f"  [{i+1}/{total_files}] {fname}: {result['status']}, re-submitting...")

        fpath = BATCH_DIR / fname

        # Retry loop for token_limit_exceeded errors
        max_retries = 10
        retry_delay = 120  # seconds to wait before retrying
        for attempt in range(max_retries):
            logger.info(f"  [{i+1}/{total_files}] Uploading {fname} ({fpath.stat().st_size / 1024 / 1024:.1f} MB)...")

            # Upload file
            with open(fpath, "rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")
            logger.info(f"    File ID: {file_obj.id}")

            # Create batch
            batch = client.batches.create(
                input_file_id=file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            logger.info(f"    Batch ID: {batch.id} — waiting for completion...")

            batch_status[fname] = {
                "file_id": file_obj.id,
                "batch_id": batch.id,
                "status": batch.status,
            }

            # Save immediately after submission
            with open(status_path, "w") as f:
                json.dump(batch_status, f, indent=2)

            # Wait for this batch to complete before submitting next
            result = _wait_for_batch(client, batch.id, fname)
            batch_status[fname].update(result)

            # Save after completion
            with open(status_path, "w") as f:
                json.dump(batch_status, f, indent=2)

            if result["status"] == "completed":
                break  # Success, move to next file

            # Check if it's a token limit error (retryable)
            errors = result.get("errors") or {}
            error_data = errors.get("data", []) if isinstance(errors, dict) else []
            is_token_limit = any(e.get("code") == "token_limit_exceeded" for e in error_data)

            if is_token_limit and attempt < max_retries - 1:
                logger.warning(f"    Token limit hit — waiting {retry_delay}s before retry "
                               f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_delay)
                continue

            # Non-retryable error or exhausted retries
            logger.error(f"    BATCH FAILED: {result.get('errors')}")
            logger.error(f"    Stopping. Fix the issue and re-run submit to resume.")
            sys.exit(1)

        elapsed = time.time() - start_time
        done_so_far = sum(1 for v in batch_status.values() if v.get("status") == "completed")
        remaining = total_files - done_so_far
        avg_per_batch = elapsed / max(1, done_so_far - completed_count)
        eta_min = remaining * avg_per_batch / 60
        logger.info(f"    [{done_so_far}/{total_files}] done. ETA: ~{eta_min:.0f} min remaining")

        # Small cooldown to let OpenAI free token quota before next submission
        if remaining > 0:
            time.sleep(10)

    elapsed = time.time() - start_time
    logger.info(f"\nAll {total_files} batches completed in {elapsed/60:.1f} min!")
    logger.info(f"Status saved to: {status_path}")


# ── Phase 3: STATUS ──────────────────────────────────────────────────────────

def cmd_status(args):
    """Check status of all submitted batches."""
    from openai import OpenAI
    client = OpenAI()

    status_path = BATCH_DIR / "batch_status.json"
    if not status_path.exists():
        logger.error("No batch_status.json found. Run 'submit' first.")
        sys.exit(1)

    batch_status = json.load(open(status_path))

    counts = {"completed": 0, "in_progress": 0, "failed": 0, "other": 0}
    total_completed = 0
    total_failed = 0

    for fname, info in sorted(batch_status.items()):
        batch = client.batches.retrieve(info["batch_id"])
        info["status"] = batch.status
        info["output_file_id"] = getattr(batch, "output_file_id", None)
        info["error_file_id"] = getattr(batch, "error_file_id", None)

        rc = batch.request_counts
        completed = rc.completed if rc else 0
        failed = rc.failed if rc else 0
        total = rc.total if rc else 0
        total_completed += completed
        total_failed += failed

        status_str = batch.status
        if status_str == "completed":
            counts["completed"] += 1
        elif status_str in ("validating", "in_progress", "finalizing"):
            counts["in_progress"] += 1
        elif status_str in ("failed", "expired", "cancelled"):
            counts["failed"] += 1
        else:
            counts["other"] += 1

        logger.info(f"  {fname}: {status_str} ({completed}/{total} done, {failed} failed)")

    # Save updated status
    with open(status_path, "w") as f:
        json.dump(batch_status, f, indent=2)

    logger.info(f"\nSummary: {counts['completed']} completed, {counts['in_progress']} in progress, "
                f"{counts['failed']} failed")
    logger.info(f"Requests: {total_completed} completed, {total_failed} failed")


# ── Phase 4: COLLECT ─────────────────────────────────────────────────────────

def cmd_collect(args):
    """Download completed batch results and parse into final JSON."""
    from openai import OpenAI
    client = OpenAI()

    status_path = BATCH_DIR / "batch_status.json"
    meta_path = BATCH_DIR / "entries_meta.json"

    if not status_path.exists():
        logger.error("No batch_status.json found. Run 'submit' first.")
        sys.exit(1)

    batch_status = json.load(open(status_path))
    meta = json.load(open(meta_path))

    # Load existing results (for resume)
    results = []
    done_filenames = set()
    if OUTPUT_FILE.exists():
        results = json.load(open(OUTPUT_FILE))
        done_filenames = {r["filename"] for r in results}
        logger.info(f"Loaded {len(results)} existing results")

    total_new = 0
    total_errors = 0
    total_empty = 0
    total_tokens_in = 0
    total_tokens_out = 0

    for fname, info in sorted(batch_status.items()):
        if info.get("status") != "completed":
            # Re-check status
            batch = client.batches.retrieve(info["batch_id"])
            info["status"] = batch.status
            info["output_file_id"] = getattr(batch, "output_file_id", None)
            info["error_file_id"] = getattr(batch, "error_file_id", None)

        if info["status"] != "completed":
            logger.warning(f"  {fname}: not completed (status={info['status']}), skipping")
            continue

        if not info.get("output_file_id"):
            logger.warning(f"  {fname}: no output file, skipping")
            continue

        if info.get("collected"):
            logger.info(f"  {fname}: already collected")
            continue

        logger.info(f"  Downloading {fname} results...")
        content = client.files.content(info["output_file_id"])
        lines = content.text.strip().split("\n")

        file_new = 0
        file_errors = 0
        file_empty = 0

        for line in lines:
            resp = json.loads(line)
            custom_id = resp["custom_id"]  # = filename

            if custom_id in done_filenames:
                continue

            if resp.get("error"):
                file_errors += 1
                total_errors += 1
                continue

            body = resp["response"]["body"]
            msg = body["choices"][0]["message"]
            text = (msg.get("content") or "").strip()
            usage = body.get("usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)
            total_tokens_in += tokens_in
            total_tokens_out += tokens_out

            if not text:
                file_empty += 1
                total_empty += 1
                continue

            short_cap, long_cap = parse_response(text)
            entry_meta = meta.get(custom_id, {})

            results.append({
                "filename": custom_id,
                "image_path": entry_meta.get("image_path", ""),
                "mask_path": entry_meta.get("mask_path", ""),
                "product": entry_meta.get("product", ""),
                "defect_code": entry_meta.get("defect_code", ""),
                "defect_name": entry_meta.get("defect_name", ""),
                "caption_short": short_cap or "",
                "caption_long": long_cap or "",
                "raw_response": text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            })
            done_filenames.add(custom_id)
            file_new += 1
            total_new += 1

        info["collected"] = True
        logger.info(f"    {file_new} new, {file_errors} errors, {file_empty} empty")

    # Save updated status
    with open(status_path, "w") as f:
        json.dump(batch_status, f, indent=2)

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # Also save short/long split files
    short_captions = [{
        "image_path": r["image_path"], "mask_path": r["mask_path"],
        "product": r["product"], "defect_name": r["defect_name"],
        "caption": r["caption_short"],
    } for r in results]
    with open(DATA_ROOT / "captions_nano_short_77.json", "w") as f:
        json.dump(short_captions, f, indent=2)

    long_captions = [{
        "image_path": r["image_path"], "mask_path": r["mask_path"],
        "product": r["product"], "defect_name": r["defect_name"],
        "caption": r["caption_long"],
    } for r in results]
    with open(DATA_ROOT / "captions_nano_long_200.json", "w") as f:
        json.dump(long_captions, f, indent=2)

    cost_in = total_tokens_in * 0.025 / 1_000_000  # batch pricing
    cost_out = total_tokens_out * 0.20 / 1_000_000
    cost = cost_in + cost_out

    logger.info(f"\nCollected {total_new} new captions ({total_errors} errors, {total_empty} empty responses)")
    logger.info(f"Total captions: {len(results)}")
    logger.info(f"New tokens: {total_tokens_in:,} in + {total_tokens_out:,} out")
    logger.info(f"Estimated batch cost for new: ${cost:.2f}")
    logger.info(f"Saved to: {OUTPUT_FILE}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch caption Real-IAD with GPT-5-nano")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("prepare", help="Generate JSONL batch input files")
    sub.add_parser("resplit", help="Re-split existing JSONL files into smaller chunks")
    sub.add_parser("submit", help="Upload and create batches (sequential, one at a time)")
    sub.add_parser("status", help="Check batch status")
    sub.add_parser("collect", help="Download and parse results")

    args = parser.parse_args()

    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "resplit":
        cmd_resplit(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "collect":
        cmd_collect(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
