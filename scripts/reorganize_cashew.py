"""Reorganize cashew experiment into new split scheme.

New structure:
- 50 real anomalies (gen_anomalies: 47 easy + 3 hard)
- 50 synthetic (from remaining test_anomalies: 47 easy + 3 hard, placed on canvases)
- Each split: train (2 hard + 38 easy = 40) / test (1 hard + 9 easy = 10)
- Normals: 100 train, 50 test (shared), 50 canvas
"""
import json
import random
import shutil
from pathlib import Path

SEED = 42
random.seed(SEED)

CASHEW = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\validation\VisA\datasets\easy_test\cashew"
)
EXP = CASHEW / "experiment_UniNet"

# Load current splits
with open(EXP / "splits.json") as f:
    splits = json.load(f)

# Current anomaly groups
gen_easy = splits["gen_anomalies_easy"]    # 47
gen_hard = splits["gen_anomalies_hard"]    # 3
test_easy = splits["test_anomalies_easy"]  # 47
test_hard = splits["test_anomalies_hard"]  # 3

print(f"Gen anomalies:  {len(gen_easy)} easy + {len(gen_hard)} hard = {len(gen_easy)+len(gen_hard)}")
print(f"Test anomalies: {len(test_easy)} easy + {len(test_hard)} hard = {len(test_easy)+len(test_hard)}")

# Shuffle for random train/test split
random.shuffle(gen_easy)
random.shuffle(gen_hard)
random.shuffle(test_easy)
random.shuffle(test_hard)

# Split each group: 1 hard test + 9 easy test, rest train
# Real (from gen_anomalies)
real_test_hard = gen_hard[:1]     # 1
real_train_hard = gen_hard[1:]    # 2
real_test_easy = gen_easy[:9]     # 9
real_train_easy = gen_easy[9:]    # 38

# Synthetic (from test_anomalies = "remaining")
synth_test_hard = test_hard[:1]   # 1
synth_train_hard = test_hard[1:]  # 2
synth_test_easy = test_easy[:9]   # 9
synth_train_easy = test_easy[9:]  # 38

print(f"\nReal split:")
print(f"  Train: {len(real_train_easy)} easy + {len(real_train_hard)} hard = {len(real_train_easy)+len(real_train_hard)}")
print(f"  Test:  {len(real_test_easy)} easy + {len(real_test_hard)} hard = {len(real_test_easy)+len(real_test_hard)}")
print(f"\nSynthetic split:")
print(f"  Train: {len(synth_train_easy)} easy + {len(synth_train_hard)} hard = {len(synth_train_easy)+len(synth_train_hard)}")
print(f"  Test:  {len(synth_test_easy)} easy + {len(synth_test_hard)} hard = {len(synth_test_easy)+len(synth_test_hard)}")

# Save new splits
new_splits = {
    "seed": SEED,
    "train_normals": splits["train_normals"],
    "canvas_normals": splits["canvas_normals"],
    "test_normals": splits["test_normals"],
    "real": {
        "train_easy": real_train_easy,
        "train_hard": real_train_hard,
        "test_easy": real_test_easy,
        "test_hard": real_test_hard,
    },
    "synthetic": {
        "source_easy": test_easy,   # all 47 used as references
        "source_hard": test_hard,   # all 3 used as references
        "train_easy": synth_train_easy,
        "train_hard": synth_train_hard,
        "test_easy": synth_test_easy,
        "test_hard": synth_test_hard,
    },
}

out_path = EXP / "splits_v2.json"
with open(out_path, "w") as f:
    json.dump(new_splits, f, indent=2)
print(f"\nSaved new splits: {out_path}")

# Print specific IDs for verification
print(f"\nReal test hard: {real_test_hard}")
print(f"Real test easy: {real_test_easy}")
print(f"Synth test hard: {synth_test_hard}")
print(f"Synth test easy: {synth_test_easy}")
print(f"\nSynth source = test_anomalies (remaining 47e + 3h) — need mask placement on canvases")
