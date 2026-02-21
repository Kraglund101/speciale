"""Visualize candidate fixed samples."""
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

data_root = Path(r"C:\Users\frede\desktop\kandidat\speciale\anomverse_extension\datasets\full_training_dataset")

samples = [
    # Keep
    ("cut_lead\n(transistor)", "AnomVerse_data_filtered/mvtec/mvtec/transistor/test/cut_lead/005.png",
     "AnomVerse_data_filtered/mvtec/mvtec/transistor/ground_truth/cut_lead/005_mask.png"),
    ("broken_large\n(bottle)", "AnomVerse_data_filtered/mvtec/mvtec/bottle/test/broken_large/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/bottle/ground_truth/broken_large/011_mask.png"),
    ("fold\n(leather)", "AnomVerse_data_filtered/mvtec/mvtec/leather/test/fold/016.png",
     "AnomVerse_data_filtered/mvtec/mvtec/leather/ground_truth/fold/016_mask.png"),
    # Replacements for contamination/scratch/poke
    ("broken_teeth\n(zipper)", "AnomVerse_data_filtered/mvtec/mvtec/zipper/test/broken_teeth/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/zipper/ground_truth/broken_teeth/011_mask.png"),
    ("cut\n(hazelnut)", "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/test/cut/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/hazelnut/ground_truth/cut/011_mask.png"),
    ("defective\n(toothbrush)", "AnomVerse_data_filtered/mvtec/mvtec/toothbrush/test/defective/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/toothbrush/ground_truth/defective/011_mask.png"),
    ("thread\n(carpet)", "AnomVerse_data_filtered/mvtec/mvtec/carpet/test/thread/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/carpet/ground_truth/thread/011_mask.png"),
    ("faulty_imprint\n(pill)", "AnomVerse_data_filtered/mvtec/mvtec/pill/test/faulty_imprint/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/pill/ground_truth/faulty_imprint/011_mask.png"),
    ("flip\n(metal_nut)", "AnomVerse_data_filtered/mvtec/mvtec/metal_nut/test/flip/011.png",
     "AnomVerse_data_filtered/mvtec/mvtec/metal_nut/ground_truth/flip/011_mask.png"),
]

fig, axes = plt.subplots(2, len(samples), figsize=(3.5 * len(samples), 7))

for i, (atype, img_path, mask_path) in enumerate(samples):
    img = Image.open(data_root / img_path).convert("RGB")
    mask = Image.open(data_root / mask_path).convert("L")

    axes[0, i].imshow(img)
    axes[0, i].set_title(atype, fontsize=9, fontweight="bold")
    axes[0, i].axis("off")

    axes[1, i].imshow(mask, cmap="gray")
    axes[1, i].set_title("Mask", fontsize=8)
    axes[1, i].axis("off")

plt.suptitle("Fixed Samples: 3 kept (left) + 6 candidates (right)", fontsize=12)
plt.tight_layout()
out = Path(r"C:\Users\frede\desktop\kandidat\speciale\results\fixed_samples_candidates.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
