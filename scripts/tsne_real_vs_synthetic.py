"""t-SNE visualization comparing real vs synthetic cashew anomalies + normals.

Extracts features from both CLIP ViT-H/14 (1280-dim) and ResNet-50 (2048-dim),
runs t-SNE per CFG variant with and without PCA preprocessing, and produces
a Nx4 grid: [CLIP raw, CLIP PCA-50, ResNet raw, ResNet PCA-50] per variant row.

Three point groups: blue=real anomaly, red=synthetic anomaly, green=normal.
Gray lines connect index-matched real↔synthetic pairs.

Usage:
    python scripts/tsne_real_vs_synthetic.py \
        --synth-dir results/cashew_100_mo/generated \
        --n-normals 25 \
        --output results/cashew_100_mo/tsne_comparison.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────
CASHEW_ROOT = Path(
    "anomverse_extension/datasets/VisA_validation_dataset"
    "/datasets/easy_test/cashew"
)

# ── CLIP normalization (must match training) ──────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ── ImageNet normalization (for ResNet-50) ────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Hard image indices (094-099)
HARD_INDICES = set(range(94, 100))

PCA_DIMS = 50  # PCA target dimensionality


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
    """Extract CLIP ViT-H/14 features: penultimate hidden -> mean pool -> 1280-dim."""
    from transformers import CLIPVisionModelWithProjection

    model = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ).to(device).eval()

    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size].to(device)
        batch = normalize_batch(batch, CLIP_MEAN, CLIP_STD)
        out = model(batch, output_hidden_states=True)
        hidden = out.hidden_states[-2][:, 1:, :]
        feats = hidden.mean(dim=1)
        all_feats.append(feats.cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(all_feats, axis=0)


@torch.no_grad()
def extract_resnet_features(
    images: torch.Tensor, device: str = "cuda", batch_size: int = 32,
) -> np.ndarray:
    """Extract ResNet-50 features: avgpool output -> 2048-dim."""
    import torchvision.models as models

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.to(device).eval()

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


def run_tsne(feats: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    """Run t-SNE on feature matrix, returns [N, 2] coords."""
    return TSNE(
        n_components=2, perplexity=perplexity,
        random_state=seed, init="pca", learning_rate="auto",
    ).fit_transform(feats)


def plot_tsne_grid(
    variant_data: list[dict],
    perplexity: float,
    output_path: Path,
):
    """Create Nx4 grid with real (blue), synthetic (red), normal (green) points.

    Layout: [real | synth | normal] concatenated in feature arrays.
    variant_data dicts have: name, clip_raw, clip_pca, resnet_raw, resnet_pca,
        n_real, n_synth, n_normal, difficulties
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    from scipy.spatial.distance import cdist

    n_variants = len(variant_data)
    fig, axes = plt.subplots(
        n_variants, 4, figsize=(28, 6 * n_variants),
    )
    if n_variants == 1:
        axes = axes.reshape(1, 4)

    def _scatter(ax, coords, n_real, n_synth, n_normal, difficulties, title):
        real = coords[:n_real]
        synth = coords[n_real:n_real + n_synth]
        normal = coords[n_real + n_synth:]

        # Index-matched lines (real[i] <-> synth[i])
        n_pairs = min(n_real, n_synth)
        for j in range(n_pairs):
            ax.plot(
                [real[j, 0], synth[j, 0]],
                [real[j, 1], synth[j, 1]],
                color="gray", alpha=0.3, linewidth=0.8, zorder=1,
            )

        # Normal points (green, behind)
        if n_normal > 0:
            ax.scatter(
                normal[:, 0], normal[:, 1],
                c="tab:green", marker="s", s=35, edgecolors="white",
                linewidths=0.5, zorder=2, alpha=0.7,
            )

        # Real anomaly points (blue)
        for diff_type, marker in [("easy", "o"), ("hard", "^")]:
            mask = [d == diff_type for d in difficulties]
            if any(mask):
                idx = np.array(mask)
                ax.scatter(
                    real[idx, 0], real[idx, 1],
                    c="tab:blue", marker=marker, s=40, edgecolors="white",
                    linewidths=0.5, zorder=3, alpha=0.7,
                )
                ax.scatter(
                    synth[idx, 0], synth[idx, 1],
                    c="tab:red", marker=marker, s=40, edgecolors="white",
                    linewidths=0.5, zorder=3, alpha=0.7,
                )

        # NN distances: synth<->real and normal<->real
        D_sr = cdist(synth, real)
        sr_nn = (D_sr.min(axis=1).mean() + D_sr.min(axis=0).mean()) / 2

        subtitle = f"NN: S-R={sr_nn:.1f}"
        if n_normal > 0:
            D_nr = cdist(normal, real)
            nr_nn = (D_nr.min(axis=1).mean() + D_nr.min(axis=0).mean()) / 2
            subtitle += f"  N-R={nr_nn:.1f}"

        ax.set_title(f"{title}\n{subtitle}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return sr_nn

    # Shared legend
    legend_handles = [
        mlines.Line2D([], [], color="tab:blue", marker="o",
                      linestyle="None", markersize=7, label="Real anomaly (easy)"),
        mlines.Line2D([], [], color="tab:blue", marker="^",
                      linestyle="None", markersize=7, label="Real anomaly (hard)"),
        mlines.Line2D([], [], color="tab:red", marker="o",
                      linestyle="None", markersize=7, label="Synth anomaly (easy)"),
        mlines.Line2D([], [], color="tab:red", marker="^",
                      linestyle="None", markersize=7, label="Synth anomaly (hard)"),
        mlines.Line2D([], [], color="tab:green", marker="s",
                      linestyle="None", markersize=7, label="Normal"),
        mlines.Line2D([], [], color="gray", alpha=0.5,
                      linewidth=1, label="Same-index pair"),
    ]

    col_headers = [
        f"CLIP ViT-H/14 (1280-d raw)",
        f"CLIP ViT-H/14 (PCA-{PCA_DIMS} -> t-SNE)",
        f"ResNet-50 (2048-d raw)",
        f"ResNet-50 (PCA-{PCA_DIMS} -> t-SNE)",
    ]

    for row, vd in enumerate(variant_data):
        keys = ["clip_raw", "clip_pca", "resnet_raw", "resnet_pca"]
        dists = []
        for col, key in enumerate(keys):
            title = f"{col_headers[col]}\n{vd['name']}"
            d = _scatter(
                axes[row, col], vd[key],
                vd["n_real"], vd["n_synth"], vd["n_normal"],
                vd["difficulties"], title,
            )
            dists.append(d)

        axes[row, 3].legend(handles=legend_handles, fontsize=7, loc="best")

        print(f"  {vd['name']}: avg NN dist (S-R) — "
              f"CLIP raw={dists[0]:.2f}, CLIP PCA={dists[1]:.2f}, "
              f"ResNet raw={dists[2]:.2f}, ResNet PCA={dists[3]:.2f}")

    n_r = variant_data[0]["n_real"]
    n_n = variant_data[0]["n_normal"]
    fig.suptitle(
        f"Real vs Synthetic Cashew Anomalies — t-SNE "
        f"(perplexity={perplexity}, {n_r} anomalies, {n_n} normals)",
        fontsize=14, fontweight="bold", y=1.0,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {output_path}")


def load_matched_pairs(
    real_dir: Path, synth_dir: Path,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str], int]:
    """Load matched real/synthetic image pairs by index (000-099).

    Returns: (real_tensors, synth_tensors, difficulties, count)
    """
    real_tensors = []
    synth_tensors = []
    difficulties = []

    for idx in range(100):
        tag = f"{idx:03d}"
        diff = "hard" if idx in HARD_INDICES else "easy"

        subdir = "hard_imgs" if diff == "hard" else "easy_imgs"
        real_path = real_dir / subdir / f"{tag}.JPG"
        synth_path = synth_dir / f"{tag}.png"

        if not real_path.exists():
            print(f"  WARNING: real image not found: {real_path}")
            continue
        if not synth_path.exists():
            print(f"  WARNING: synth image not found: {synth_path}")
            continue

        real_tensors.append(load_image_tensor(real_path))
        synth_tensors.append(load_image_tensor(synth_path))
        difficulties.append(diff)

    n = len(real_tensors)
    print(f"  Loaded {n} matched pairs "
          f"({sum(1 for d in difficulties if d == 'easy')} easy, "
          f"{sum(1 for d in difficulties if d == 'hard')} hard)")
    return real_tensors, synth_tensors, difficulties, n


def load_normal_images(
    normal_dir: Path, n: int, seed: int,
) -> list[torch.Tensor]:
    """Load n random normal images from directory."""
    all_normals = sorted(
        p for p in normal_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    rng = random.Random(seed)
    selected = rng.sample(all_normals, min(n, len(all_normals)))
    tensors = [load_image_tensor(p) for p in selected]
    print(f"  Loaded {len(tensors)} normal images")
    return tensors


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE: real vs synthetic cashew anomalies + normals",
    )
    parser.add_argument(
        "--synth-dir", type=str, required=True,
        help="Path to generated/ directory (contains CFG variant subdirs)",
    )
    parser.add_argument(
        "--real-dir", type=str, default=None,
        help="Path to real anomaly images dir (default: auto from CASHEW_ROOT)",
    )
    parser.add_argument(
        "--normal-dir", type=str, default=None,
        help="Path to normal images dir (default: CASHEW_ROOT/Data/Images/Normal)",
    )
    parser.add_argument(
        "--n-normals", type=int, default=0,
        help="Number of normal images to include (default: 0)",
    )
    parser.add_argument(
        "--variants", type=str, nargs="+", default=None,
        help="CFG variant subdirs to compare (default: all found in synth-dir)",
    )
    parser.add_argument(
        "--perplexity", type=float, default=15.0,
        help="t-SNE perplexity (default: 15)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path (default: <synth-dir>/../tsne_comparison.png)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for t-SNE (default: 42)",
    )
    args = parser.parse_args()

    synth_base = Path(args.synth_dir)
    if not synth_base.exists():
        print(f"ERROR: synth dir not found: {synth_base}")
        sys.exit(1)

    # Resolve real dir
    if args.real_dir:
        real_dir = Path(args.real_dir)
    else:
        real_dir = CASHEW_ROOT / "Data" / "Images" / "Anomaly"
    if not real_dir.exists():
        print(f"ERROR: real dir not found: {real_dir}")
        sys.exit(1)

    # Resolve normal dir
    if args.normal_dir:
        normal_dir = Path(args.normal_dir)
    else:
        normal_dir = CASHEW_ROOT / "Data" / "Images" / "Normal"

    # Discover CFG variants
    if args.variants:
        variants = args.variants
    else:
        variants = sorted([
            d.name for d in synth_base.iterdir()
            if d.is_dir() and any(d.glob("*.png"))
        ])
    if not variants:
        print(f"ERROR: no CFG variant subdirs with .png files in {synth_base}")
        sys.exit(1)
    print(f"CFG variants: {variants}")

    # Load real images
    print("\nLoading real anomaly images...")
    first_synth_dir = synth_base / variants[0]
    real_tensors, _, _, n_first = load_matched_pairs(real_dir, first_synth_dir)
    if n_first == 0:
        print("No valid pairs found.")
        sys.exit(1)

    # Load normal images
    normal_tensors = []
    n_normals = 0
    if args.n_normals > 0 and normal_dir.exists():
        print("Loading normal images...")
        normal_tensors = load_normal_images(normal_dir, args.n_normals, args.seed)
        n_normals = len(normal_tensors)

    # Extract real + normal features once
    real_stack = torch.stack(real_tensors)
    print(f"\nExtracting CLIP features for {n_first} real images...")
    real_clip = extract_clip_features(real_stack)
    print(f"Extracting ResNet features for {n_first} real images...")
    real_resnet = extract_resnet_features(real_stack)

    normal_clip = np.zeros((0, real_clip.shape[1]))
    normal_resnet = np.zeros((0, real_resnet.shape[1]))
    if n_normals > 0:
        normal_stack = torch.stack(normal_tensors)
        print(f"Extracting CLIP features for {n_normals} normal images...")
        normal_clip = extract_clip_features(normal_stack)
        print(f"Extracting ResNet features for {n_normals} normal images...")
        normal_resnet = extract_resnet_features(normal_stack)

    # Process each variant
    variant_data = []
    for vname in variants:
        print(f"\n--- Variant: {vname} ---")
        synth_dir = synth_base / vname
        _, synth_tensors, difficulties, n = load_matched_pairs(
            real_dir, synth_dir,
        )
        if n == 0:
            print(f"  Skipping {vname}: no valid pairs")
            continue

        synth_stack = torch.stack(synth_tensors)
        print(f"  Extracting CLIP features for {n} synthetic images...")
        synth_clip = extract_clip_features(synth_stack)
        print(f"  Extracting ResNet features for {n} synthetic images...")
        synth_resnet = extract_resnet_features(synth_stack)

        # Order: [real | synth | normal]
        all_clip = np.concatenate([real_clip, synth_clip, normal_clip], axis=0)
        all_resnet = np.concatenate([real_resnet, synth_resnet, normal_resnet], axis=0)

        # PCA-reduced versions
        clip_pca = PCA(n_components=PCA_DIMS, random_state=args.seed).fit_transform(all_clip)
        resnet_pca = PCA(n_components=PCA_DIMS, random_state=args.seed).fit_transform(all_resnet)

        # Run 4 t-SNE embeddings
        print(f"  t-SNE: CLIP raw ({all_clip.shape[1]}d)...")
        clip_raw_tsne = run_tsne(all_clip, args.perplexity, args.seed)
        print(f"  t-SNE: CLIP PCA-{PCA_DIMS}...")
        clip_pca_tsne = run_tsne(clip_pca, args.perplexity, args.seed)
        print(f"  t-SNE: ResNet raw ({all_resnet.shape[1]}d)...")
        resnet_raw_tsne = run_tsne(all_resnet, args.perplexity, args.seed)
        print(f"  t-SNE: ResNet PCA-{PCA_DIMS}...")
        resnet_pca_tsne = run_tsne(resnet_pca, args.perplexity, args.seed)

        variant_data.append({
            "name": vname,
            "clip_raw": clip_raw_tsne,
            "clip_pca": clip_pca_tsne,
            "resnet_raw": resnet_raw_tsne,
            "resnet_pca": resnet_pca_tsne,
            "n_real": n,
            "n_synth": n,
            "n_normal": n_normals,
            "difficulties": difficulties,
        })

    if not variant_data:
        print("No variants had valid pairs.")
        sys.exit(1)

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = synth_base.parent / "tsne_comparison.png"

    plot_tsne_grid(variant_data, args.perplexity, output_path)


if __name__ == "__main__":
    main()
