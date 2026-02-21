"""Test 20 samples with correct defect mapping to evaluate measurement quality."""
import json, os, io, base64, random, re
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()
DATA_ROOT = "anomverse_extension/datasets/realiad_1024"

DEFECT_CODE_MAP = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
}

# Approximate real-world object sizes in mm (longest dimension)
OBJECT_SIZE_MM = {
    "audiojack": 12, "bottle_cap": 30, "button_battery": 20, "end_cap": 40,
    "eraser": 45, "fire_hood": 35, "metal_cylinder": 25, "mint": 25,
    "pcb": 25, "phone_battery": 50, "plastic_nut": 20, "plastic_plug": 15,
    "porcelain_doll": 30, "regulator": 25, "rolled_strip_base": 60,
    "sim_card_set": 25, "switch": 20, "tape": 50, "terminalblock": 30,
    "toy": 60, "toothbrush": 180, "u_block": 30, "usb_adaptor": 40,
    "vcpill": 20,
}


def get_union_crop(image_path: str, mask_path: str, pad_frac: float = 0.2):
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


def draw_bboxes(image_path: str, mask_path: str):
    from scipy import ndimage
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    labeled, n = ndimage.label(binary)
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i in range(1, n + 1):
        cy, cx = np.where(labeled == i)
        for offset in range(3):
            draw.rectangle(
                [int(cx.min()) - offset, int(cy.min()) - offset,
                 int(cx.max()) + offset, int(cy.max()) + offset],
                outline="red",
            )
    return img


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
        "inpainting model for synthetic anomaly generation.\n\n"
        "You are given two images of the same object:\n"
        "- Image 1: the full object with red bounding boxes marking defect regions "
        "(describe the defect, not the annotations).\n"
        "- Image 2: a close-up crop of the defect area.\n\n"
        f"Object: '{product.replace('_', ' ')}'. Defect type: '{defect_type.replace('_', ' ')}'.\n\n"
        "Generate TWO captions:\n"
        "SHORT (max 60 words): A single dense sentence for CLIP text encoding.\n"
        "LONG (max 150 words): 2-3 sentences with richer detail about texture, "
        "color, shape, size, and spatial context.\n\n"
        "Output format (exactly):\n"
        "SHORT: <your short caption>\n"
        "LONG: <your long caption>\n\n"
        "Do not include any Chinese characters. Do not describe the bounding boxes."
    )


def estimate_real_mm(product: str, bbox_max_px: int, img_size: int = 1024) -> float:
    """Estimate real-world size in mm from pixel bbox."""
    obj_mm = OBJECT_SIZE_MM.get(product, 30)
    return bbox_max_px / img_size * obj_mm


def extract_measurements(text: str) -> list[float]:
    """Extract all mm/cm measurements from text."""
    mm_vals = [float(v) for v in re.findall(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)', text, re.I)]
    cm_vals = [float(v) * 10 for v in re.findall(r'(\d+(?:\.\d+)?)\s*(?:cm|centimeter)', text, re.I)]
    return mm_vals + cm_vals


# Pick 20 diverse samples
concepts_dir = os.path.join(DATA_ROOT, "concepts")
random.seed(42)
all_samples = []
for concept_file in sorted(os.listdir(concepts_dir)):
    if not concept_file.endswith(".json"):
        continue
    with open(os.path.join(concepts_dir, concept_file), encoding="utf-8") as f:
        data = json.load(f)
    all_samples.extend(data["samples"])

random.shuffle(all_samples)

# Pick diverse: spread across products and defect types
picks = []
used = set()
for s in all_samples:
    key = (s["product"], s["anomaly_code"])
    if key not in used:
        used.add(key)
        picks.append(s)
    if len(picks) >= 20:
        break

print(f"Testing {len(picks)} samples\n")

results = []
for i, sample in enumerate(picks):
    product = sample["product"]
    code = sample["anomaly_code"]
    defect_type = DEFECT_CODE_MAP[code]
    img_path = os.path.join(DATA_ROOT, sample["image_path"])
    mask_path = os.path.join(DATA_ROOT, sample["mask_path"])

    # Mask stats
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        continue
    ys, xs = np.where(binary > 0)
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    bbox_max = max(bbox_w, bbox_h)

    est_mm = estimate_real_mm(product, bbox_max)

    bbox_img = draw_bboxes(img_path, mask_path)
    crop_img = get_union_crop(img_path, mask_path)
    if crop_img is None:
        continue

    prompt = build_prompt(product, defect_type)
    b64_bbox = pil_to_base64(bbox_img)
    b64_crop = pil_to_base64(crop_img)

    response = client.chat.completions.create(
        model="gpt-5-nano",
        max_completion_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_bbox}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_crop}"}},
            ],
        }],
    )

    text = response.choices[0].message.content or ""
    claimed = extract_measurements(text)
    claimed_max = max(claimed) if claimed else None

    ratio = claimed_max / est_mm if claimed_max and est_mm > 0 else None

    verdict = "NO_MEASUREMENT"
    if ratio is not None:
        if 0.3 <= ratio <= 3.0:
            verdict = "OK"
        elif 0.1 <= ratio <= 10.0:
            verdict = "ROUGH"
        else:
            verdict = "BAD"

    results.append({
        "product": product, "defect": defect_type,
        "bbox_px": f"{bbox_w}x{bbox_h}", "est_mm": round(est_mm, 1),
        "claimed_mm": claimed, "claimed_max": claimed_max,
        "ratio": round(ratio, 2) if ratio else None, "verdict": verdict,
    })

    print(f"#{i+1:2d} [{product}/{defect_type}]  bbox={bbox_w}x{bbox_h}px  est={est_mm:.1f}mm  claimed={claimed}  ratio={f'{ratio:.2f}' if ratio else 'N/A'}  {verdict}")
    # Print short caption only
    for line in text.strip().split("\n"):
        if line.strip().upper().startswith("SHORT:"):
            print(f"     {line.strip()}")
            break
    print()

# Summary
print("=" * 80)
print("SUMMARY")
ok = sum(1 for r in results if r["verdict"] == "OK")
rough = sum(1 for r in results if r["verdict"] == "ROUGH")
bad = sum(1 for r in results if r["verdict"] == "BAD")
no_meas = sum(1 for r in results if r["verdict"] == "NO_MEASUREMENT")
total = len(results)

print(f"  OK (within 3x):      {ok}/{total}")
print(f"  ROUGH (within 10x):  {rough}/{total}")
print(f"  BAD (>10x off):      {bad}/{total}")
print(f"  No measurement:      {no_meas}/{total}")

ratios = [r["ratio"] for r in results if r["ratio"] is not None]
if ratios:
    print(f"  Median ratio (claimed/estimated): {sorted(ratios)[len(ratios)//2]:.2f}")
    print(f"  Mean ratio: {sum(ratios)/len(ratios):.2f}")
