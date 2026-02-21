"""Test 5 samples with correct defect mapping to evaluate measurement quality."""
import json, os, io, base64, random
import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

client = OpenAI()
DATA_ROOT = "anomverse_extension/datasets/realiad_1024"

# Correct mapping from concepts JSONs
DEFECT_CODE_MAP = {
    "AK": "pit", "BX": "deformation", "CH": "abrasion", "HS": "scratch",
    "PS": "damage", "QS": "missing_parts", "YW": "foreign_objects", "ZW": "contamination",
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


def build_prompt_with_measurements(product: str, defect_type: str) -> str:
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


def get_mask_stats(mask_path: str):
    mask = np.array(Image.open(mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8)
    h, w = mask.shape
    area = binary.sum()
    if area == 0:
        return None
    ys, xs = np.where(binary > 0)
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1
    return {
        "img_size": f"{w}x{h}",
        "bbox": f"{bbox_w}x{bbox_h}px",
        "area_px": int(area),
        "coverage_pct": f"{100 * area / (h * w):.3f}%",
    }


# Pick 5 diverse samples: different products, different defect types
concepts_dir = os.path.join(DATA_ROOT, "concepts")
random.seed(77)
picks = []
used_combos = set()

for concept_file in sorted(os.listdir(concepts_dir)):
    if not concept_file.endswith(".json"):
        continue
    with open(os.path.join(concepts_dir, concept_file), encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    random.shuffle(samples)
    for s in samples:
        combo = (s["product"], s["anomaly_code"])
        if combo not in used_combos:
            used_combos.add(combo)
            picks.append(s)
            break
    if len(picks) >= 5:
        break

print(f"Testing {len(picks)} samples with GPT-5-nano (correct mapping + measurements)\n")
print("=" * 100)

for i, sample in enumerate(picks):
    product = sample["product"]
    code = sample["anomaly_code"]
    defect_type = DEFECT_CODE_MAP[code]
    img_path = os.path.join(DATA_ROOT, sample["image_path"])
    mask_path = os.path.join(DATA_ROOT, sample["mask_path"])

    stats = get_mask_stats(mask_path)
    bbox_img = draw_bboxes(img_path, mask_path)
    crop_img = get_union_crop(img_path, mask_path)
    if crop_img is None:
        continue

    prompt = build_prompt_with_measurements(product, defect_type)
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
    print(f"\n#{i+1} [{product}/{defect_type}] code={code}")
    print(f"  Mask stats: {stats}")
    print(f"  Image: {sample['image_path']}")
    print(f"  Response:")
    for line in text.strip().split("\n"):
        print(f"    {line}")
    print()
