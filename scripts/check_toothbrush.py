"""Quick script to find the toothbrush measurement example."""
import json, re, os
from PIL import Image
import numpy as np

with open('anomverse_extension/datasets/realiad_1024/captions_nano.json', encoding='utf-8') as f:
    caps = json.load(f)

mm_pat = re.compile(r'(\d+(?:\.\d+)?)\s*(?:mm|millimeter)', re.I)
data_root = 'anomverse_extension/datasets/realiad_1024'

for c in caps:
    if c['product'] != 'toothbrush' or c['defect_name'] != 'scratch':
        continue
    long_text = c.get('caption_long', '')
    vals = mm_pat.findall(long_text)
    if '6' not in vals:
        continue

    mask_path = os.path.join(data_root, c['mask_path'])
    if not os.path.exists(mask_path):
        continue
    mask = np.array(Image.open(mask_path).convert('L'))
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        continue
    ys, xs = np.where(binary > 0)
    bbox_w = xs.max() - xs.min() + 1
    bbox_h = ys.max() - ys.min() + 1

    if bbox_w > 350 and bbox_h > 350:
        print(f'Product: {c["product"]}')
        print(f'Defect:  {c["defect_name"]}')
        print(f'Image:   {os.path.join(data_root, c["image_path"])}')
        print(f'Mask:    {mask_path}')
        print(f'Bbox:    {bbox_w}x{bbox_h}px (image is {mask.shape[1]}x{mask.shape[0]})')
        print(f'Area:    {binary.sum()}px ({100*binary.sum()/(mask.shape[0]*mask.shape[1]):.2f}%)')
        print()
        print(f'SHORT: {c["caption_short"]}')
        print()
        print(f'LONG:  {c["caption_long"]}')
        break
