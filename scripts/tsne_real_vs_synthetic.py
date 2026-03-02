"""t-SNE visualization comparing real vs synthetic cashew anomalies.

Extracts features from both CLIP ViT-H/14 (1280-dim) and ResNet-50 (2048-dim),
runs t-SNE, and produces a 1x2 scatter plot with matched-pair lines.

Usage:
    python scripts/tsne_real_vs_synthetic.py \
        --experiment ResNet \
        --perplexity 15 \
        --output tsne_comparison.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────
CASHEW_ROOT = Path(
    r"C:\Users\frede\Desktop\kandidat\speciale\anomverse_extension"
    r"\datasets\VisA_validation_dataset\datasets\easy_test\cashew"
)

# ── CLIP normalization (must match training) ──────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ── ImageNet normalization (for ResNet-50) ────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_manifest(experiment: str) -> list[dict]:
    """Load the full manifest (all 100 entries) from manifest_all.json."""
    exp_dir = CASHEW_ROOT / f"experiment_{experiment}"
    manifest_path = exp_dir / "anomaly" / "manifest_all.json"
    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found. "
              f"Run generate_cashew.py --all first.")
        sys.exit(1)
    with open(manifest_path) as f:
        return json.load(f)


def load_image_tensor(path: Path, size: int = 224) -> torch.Tensor:
    """Load image, resize to size x size, return [3, H, W] float32 in [0, 1]."""
    img = Image.open(path).convert("RGB").resize(
        (size, size), Image.LANCZOS,
    )
    return torch.from_numpy(
        np.array(img).astype(np.float32) / 255.0,
    ).permute(2, 0, 1)  # [3, H, W]


def normalize_batch(
    batch: torch.Tensor, mean: list[float], std: list[float],
) -> torch.Tensor:
    """Normalize [B, 3, H, W] tensor with per-channel mean/std."""
    m = torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    return (batch - m) / s


@torch.no_grad()
def extract_clip_features(
    images: torch.Tensor, device: str = "cuda", batch_size: int = 32,
) -> np.ndarray:
    """Extract CLIP ViT-H/14 features: penultimate hidden → mean pool → 1280-dim.

    Args:
        images: [N, 3, 224, 224] in [0, 1]
    Returns:
        [N, 1280] numpy array
    """
    from transformers import CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ).to(device).eval()

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = normalize_batch(batch, CLIP_MEAN, CLIP_STD)
        out = model(batch, output_hidden_states=True)
        # Penultimate hidden states: [B, 257, 1280] → drop CLS → [B, 256, 1280]
        hidden = out.hidden_states[-2][:, 1:, :]
        # Mean pool patch tokens → [B, 1280]
        feats = hidden.mean(dim=1)
        all_feats.append(feats.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_resnet_features(
    images: torch.Tensor, device: str = "cuda", batch_size: int = 32,
) -> np.ndarray:
    """Extract ResNet-50 features: avgpool output → 2048-dim.

    Args:
        images: [N, 3, 224, 224] in [0, 1]
    Returns:
        [N, 2048] numpy array
    """
    import torchvision.models as models

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device).eval()

    # Hook to capture avgpool output
    features = []

    def hook_fn(module, input, output):
        features.append(output.squeeze(-1).squeeze(-1).cpu().numpy())

    handle = model.avgpool.register_forward_hook(hook_fn)

    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = normalize_batch(batch, IMAGENET_MEAN, IMAGENET_STD)
        model(batch)

    handle.remove()
    del model
    torch.cuda.empty_cache()
    return np.concatenate(features, axis=0)


def plot_tsne(
    clip_coords: np.ndarray,
    resnet_coords: np.ndarray,
    n_real: int,
    difficulties: list[str],
    perplexity: float,
    output_path: Path,
):
    """Create 1x2 scatter plot: CLIP t-SNE (left) + ResNet t-SNE (right).

    Blue = real, Red = synthetic. Circles = easy, Triangles = hard.
    Gray lines connect matched pairs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    def _scatter(ax, coords, title, feat_dim):
        n = n_real
        real = coords[:n]
        synth = coords[n:]

        # Draw matched-pair lines
        for j in range(n):
            ax.plot(
                [real[j, 0], synth[j, 0]],
                [real[j, 1], synth[j, 1]],
                color="gray", alpha=0.3, linewidth=0.8, zorder=1,
            )

        # Scatter by difficulty
        for j in range(n):
            diff = difficulties[j]
            marker = "o" if diff == "easy" else "^"
            ax.scatter(
                real[j, 0], real[j, 1],
                c="tab:blue", marker=marker, s=40, edgecolors="white",
                linewidths=0.5, zorder=2,
            )
            ax.scatter(
                synth[j, 0], synth[j, 1],
                c="tab:red", marker=marker, s=40, edgecolors="white",
                linewidths=0.5, zorder=2,
            )

        # Mean pairwise distance
        dists = np.linalg.norm(real - synth, axis=1)
        mean_dist = dists.mean()

        ax.set_title(
            f"{title} ({feat_dim}-d)\n"
            f"perplexity={perplexity}, n={n}, "
            f"mean pair dist={mean_dist:.2f}",
            fontsize=11,
        )
        ax.set_xticks([])
        ax.set_yticks([])

        # Legend
        handles = [
            mlines.Line2D([], [], color="tab:blue", marker="o",
                          linestyle="None", markersize=7, label="Real easy"),
            mlines.Line2D([], [], color="tab:blue", marker="^",
                          linestyle="None", markersize=7, label="Real hard"),
            mlines.Line2D([], [], color="tab:red", marker="o",
                          linestyle="None", markersize=7, label="Synth easy"),
            mlines.Line2D([], [], color="tab:red", marker="^",
                          linestyle="None", markersize=7, label="Synth hard"),
            mlines.Line2D([], [], color="gray", alpha=0.5,
                          linewidth=1, label="Matched pair"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")

        return mean_dist

    clip_dist = _scatter(ax1, clip_coords, "CLIP ViT-H/14", 1280)
    resnet_dist = _scatter(ax2, resnet_coords, "ResNet-50", 2048)

    fig.suptitle(
        "Real vs Synthetic Cashew Anomalies — t-SNE Feature Comparison",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {output_path}")
    print(f"  CLIP mean pair distance:   {clip_dist:.4f}")
    print(f"  ResNet mean pair distance:  {resnet_dist:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE: real vs synthetic cashew anomalies",
    )
    parser.add_argument(
        "--experiment", type=str, default="ResNet",
        help="Experiment name (default: ResNet)",
    )
    parser.add_argument(
        "--perplexity", type=float, default=15.0,
        help="t-SNE perplexity (default: 15)",
    )
    parser.add_argument(
        "--output", type=str, default="tsne_comparison.png",
        help="Output PNG path (default: tsne_comparison.png)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    args = parser.parse_args()

    exp_dir = CASHEW_ROOT / f"experiment_{args.experiment}"

    # 1. Load manifest
    manifest = load_manifest(args.experiment)
    print(f"Loaded manifest: {len(manifest)} entries")

    # 2. Load images (real + synthetic)
    real_tensors = []
    synth_tensors = []
    difficulties = []
    valid_entries = []

    for entry in manifest:
        ref_id = entry["ref_id"]
        diff = entry["difficulty"]

        # Real image: always in Data/Images/Anomaly/
        subdir = "easy_imgs" if diff == "easy" else "hard_imgs"
        real_path = (CASHEW_ROOT / "Data" / "Images" / "Anomaly"
                     / subdir / f"{ref_id}.JPG")

        # Synthetic image: anomaly/imgs/{difficulty}/{ref_id}.png
        synth_path = (exp_dir / "anomaly" / "imgs" / diff
                      / f"{ref_id}.png")

        if not real_path.exists():
            print(f"  WARNING: real image not found: {real_path}")
            continue
        if not synth_path.exists():
            print(f"  WARNING: synthetic image not found: {synth_path}")
            continue

        real_tensors.append(load_image_tensor(real_path))
        synth_tensors.append(load_image_tensor(synth_path))
        difficulties.append(diff)
        valid_entries.append(entry)

    n = len(real_tensors)
    print(f"Loaded {n} matched pairs "
          f"({sum(1 for d in difficulties if d == 'easy')} easy, "
          f"{sum(1 for d in difficulties if d == 'hard')} hard)")

    if n == 0:
        print("No valid pairs found. Run generate_cashew.py --all first.")
        sys.exit(1)

    # Stack: [N, 3, 224, 224] real then [N, 3, 224, 224] synthetic
    all_images = torch.stack(real_tensors + synth_tensors)  # [2N, 3, 224, 224]
    print(f"Image tensor: {all_images.shape}")

    # 3. Extract features
    print("Extracting CLIP ViT-H/14 features...")
    clip_feats = extract_clip_features(all_images)
    print(f"  CLIP features: {clip_feats.shape}")

    print("Extracting ResNet-50 features...")
    resnet_feats = extract_resnet_features(all_images)
    print(f"  ResNet features: {resnet_feats.shape}")

    # 4. Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    clip_tsne = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=args.seed, init="pca", learning_rate="auto",
    ).fit_transform(clip_feats)

    resnet_tsne = TSNE(
        n_components=2, perplexity=args.perplexity,
        random_state=args.seed, init="pca", learning_rate="auto",
    ).fit_transform(resnet_feats)

    # 5. Plot
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = exp_dir / output_path

    plot_tsne(
        clip_tsne, resnet_tsne,
        n_real=n,
        difficulties=difficulties,
        perplexity=args.perplexity,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
