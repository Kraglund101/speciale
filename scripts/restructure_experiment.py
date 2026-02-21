"""
Restructure experiment folders: flatten train/test into top-level dirs.

Before:
  dataset_real/train/normal, dataset_real/test/anomaly/easy, ...
After:
  dataset_real/normal, dataset_real/anomaly/easy, dataset_real/masks/easy, ...
"""

import shutil
from pathlib import Path

EXP = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy_test\cashew\experiment"
)

# ── Dataset Real ──────────────────────────────────────────────────────────
real = EXP / "dataset_real"

# Move train/normal → normal
if (real / "train" / "normal").exists():
    if (real / "normal").exists():
        shutil.rmtree(real / "normal")
    shutil.move(str(real / "train" / "normal"), str(real / "normal"))
    shutil.rmtree(real / "train")

# Move test/anomaly → anomaly
if (real / "test" / "anomaly").exists():
    if (real / "anomaly").exists():
        shutil.rmtree(real / "anomaly")
    shutil.move(str(real / "test" / "anomaly"), str(real / "anomaly"))

# Move test/masks → masks
if (real / "test" / "masks").exists():
    if (real / "masks").exists():
        shutil.rmtree(real / "masks")
    shutil.move(str(real / "test" / "masks"), str(real / "masks"))

# Remove empty test/
if (real / "test").exists():
    shutil.rmtree(real / "test")

# ── Dataset Synthetic ─────────────────────────────────────────────────────
synth = EXP / "dataset_synthetic"

# Move train/normal → normal
if (synth / "train" / "normal").exists():
    if (synth / "normal").exists():
        shutil.rmtree(synth / "normal")
    shutil.move(str(synth / "train" / "normal"), str(synth / "normal"))
    shutil.rmtree(synth / "train")

# Move test/anomaly → anomaly
if (synth / "test" / "anomaly").exists():
    if (synth / "anomaly").exists():
        shutil.rmtree(synth / "anomaly")
    shutil.move(str(synth / "test" / "anomaly"), str(synth / "anomaly"))

# Move test/masks → masks
if (synth / "test" / "masks").exists():
    if (synth / "masks").exists():
        shutil.rmtree(synth / "masks")
    shutil.move(str(synth / "test" / "masks"), str(synth / "masks"))

# Clean up test/ (remove viz too)
if (synth / "test").exists():
    shutil.rmtree(synth / "test")

print("Done. New structure:")
import subprocess
result = subprocess.run(
    ["find", str(EXP), "-maxdepth", "5", "-type", "d"],
    capture_output=True, text=True
)
for line in result.stdout.strip().split("\n"):
    # Show relative path
    rel = line.replace(str(EXP), "experiment")
    print(f"  {rel}")
