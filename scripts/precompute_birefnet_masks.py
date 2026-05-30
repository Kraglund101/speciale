#!/usr/bin/env python3
"""Precompute BiRefNet salient object masks for all ResNet experiment images."""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from transformers import AutoModelForImageSegmentation

device = "cuda"

print("Loading BiRefNet...")
model = AutoModelForImageSegmentation.from_pretrained(
    "zhengpeng7/BiRefNet", trust_remote_code=True
)
model = model.to(device).float().eval()

transform = T.Compose([
    T.Resize((1024, 1024)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

exp = Path("anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/experiment_ResNet")
real_dir = exp / "real_anomalies"
synth_dir = exp / "anomaly" / "generated"

with open(exp / "splits_test2.json") as f:
    splits = json.load(f)

out_dir = Path("results/birefnet_masks")
out_real = out_dir / "real"
out_synth = out_dir / "synth"
out_real.mkdir(parents=True, exist_ok=True)
out_synth.mkdir(parents=True, exist_ok=True)

all_jobs = []  # (input_path, output_path)
for diff in ("easy", "hard"):
    for aid in splits["train"].get(diff, []) + splits["test"].get(diff, []):
        for ext in [".JPG", ".jpg", ".png"]:
            rp = real_dir / diff / "imgs" / f"{aid}{ext}"
            if rp.exists():
                all_jobs.append((str(rp), str(out_real / f"{aid}.png")))
                break
        for ext in [".JPG", ".jpg", ".png"]:
            sp = synth_dir / diff / f"{aid}{ext}"
            if sp.exists():
                all_jobs.append((str(sp), str(out_synth / f"{aid}.png")))
                break

print(f"Processing {len(all_jobs)} images...")

for i, (inp, outp) in enumerate(all_jobs):
    if i % 20 == 0:
        print(f"  {i}/{len(all_jobs)}")
    img = Image.open(inp).convert("RGB")
    orig_w, orig_h = img.size
    inp_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(inp_t)[-1].sigmoid()

    mask = preds.squeeze().cpu()
    mask_resized = torch.nn.functional.interpolate(
        mask.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
        mode="bilinear", align_corners=False,
    ).squeeze()
    mask_bin = (mask_resized > 0.5).numpy().astype(np.uint8) * 255
    Image.fromarray(mask_bin).save(outp)

print("Done!")
