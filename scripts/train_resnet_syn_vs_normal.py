#!/usr/bin/env python3
"""Wide-ResNet-50-2 supervised anomaly detection: train on synthetic + normals, test on real + normals.

Evaluates whether synthetic anomalies are useful for training a real anomaly detector.
High test AUROC = synthetic data trains a good detector.

Data splits (44 train / 56 test, matching UniNet):
  - Train: 44 easy synthetic anomalies (label=1) + 44 real normals (label=0) = 88
  - Test:  50 easy + 6 hard = 56 REAL anomalies (label=1) + 56 real normals (label=0) = 112
  - Hard cases (094-099): TEST ONLY in both synthetic and real

Augmentation matches UniNet: 256×256, RandomHFlip, RandomVFlip, rot90, ColorJitter(0.1), GaussianBlur.

Usage:
    python scripts/train_resnet_syn_vs_normal.py \
        --synth-dir <run>/cashew_100_rp/generated/cfg7_vis \
        --mask-dir <run>/cashew_100_rp/refined_masks \
        --prep-dir results/cashew_100_rp/prep \
        --save-dir <run>/resnet_eval/syn_vs_normal \
        --epochs 50 --seed 42
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
import timm
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

CASHEW_BASE = "anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew"
EASY_IMG_DIR = "Data/Images/Anomaly/easy_imgs"
HARD_IMG_DIR = "Data/Images/Anomaly/hard_imgs"
MASK_DIR = "Data/Masks/Anomaly"
NORMAL_DIR = "Data/Images/Normal"
FG_MASKS_DIR = "birefnet_masks/Normal"

UNINET_SPLITS = "experiment_UniNet/splits.json"

# AD split: 44 train / 56 test (shared with UniNet)
N_TRAIN_EASY = 44
IMAGE_SIZE = 256
SEED_DEFAULT = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class AnomalyDetectionDataset(Dataset):
    """Binary classification: 0=normal, 1=anomaly.

    Samples: list of (image_path, label, difficulty, mask_path_or_None).
    """

    def __init__(
        self,
        samples: List[Tuple[str, int, str, Optional[str]]],
        transform: transforms.Compose,
        equalize_resolution: int = 256,
    ):
        self.samples = samples
        self.transform = transform
        self.equalize_resolution = equalize_resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _diff, _mask_path = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Equalize resolution
        sz = self.equalize_resolution
        if img.size != (sz, sz):
            img = img.resize((sz, sz), Image.LANCZOS)

        # JPEG-compress synthetics (PNG) to match real images (already JPEG)
        if not path.lower().endswith((".jpg", ".jpeg")):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_transforms(train: bool, train_cj: float = 0.1, train_blur: float = 0.5,
                      train_noise: float = 0.005,
                      test_cj: float = 0.0, test_blur: float = 0.0,
                      test_noise: float = 0.0) -> transforms.Compose:
    """Augmentation shared with UniNet: 256×256, flips, rot90, CJ, blur, noise.

    For test: optional distribution-shift perturbation (CJ, blur, noise).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    random_rot90 = transforms.RandomChoice([
        transforms.Lambda(lambda x: x),
        transforms.Lambda(lambda x: x.rotate(90, expand=True)),
        transforms.Lambda(lambda x: x.rotate(180, expand=True)),
        transforms.Lambda(lambda x: x.rotate(270, expand=True)),
    ])

    if train:
        tf_list = [
            random_rot90,
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
        if train_cj > 0:
            tf_list.append(transforms.ColorJitter(train_cj, train_cj, train_cj))
        if train_blur > 0:
            ks = max(3, int(train_blur * 6) | 1)
            tf_list.append(transforms.GaussianBlur(kernel_size=ks, sigma=(0.01, train_blur)))
        tf_list.append(transforms.ToTensor())
        if train_noise > 0:
            _noise = train_noise
            tf_list.append(transforms.Lambda(lambda x: (x + torch.randn_like(x) * _noise).clamp(0, 1)))
        tf_list.extend([transforms.Normalize(mean, std)])
        return transforms.Compose(tf_list)
    else:
        tf_list = [
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
        if test_cj > 0:
            tf_list.append(transforms.ColorJitter(test_cj, test_cj, test_cj))
        if test_blur > 0:
            ks = max(3, int(test_blur * 6) | 1)
            tf_list.append(transforms.GaussianBlur(kernel_size=ks, sigma=test_blur))
        tf_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return transforms.Compose(tf_list)


# ── Data assembly ─────────────────────────────────────────────────────────────

def build_splits(
    prep_dir: Path,
    seed: int = SEED_DEFAULT,
) -> Tuple[List[int], List[int], List[int]]:
    """Build 44 train / 56 test split from prep directory.

    Returns:
        (train_easy_indices, test_easy_indices, hard_indices)
    """
    easy_indices, hard_indices = [], []
    for idx_dir in sorted(prep_dir.iterdir()):
        if not idx_dir.is_dir():
            continue
        meta_path = idx_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        idx = meta["idx"]
        if meta.get("is_hard", False):
            hard_indices.append(idx)
        else:
            easy_indices.append(idx)

    rng = random.Random(seed)
    easy_shuffled = easy_indices[:]
    rng.shuffle(easy_shuffled)

    train_easy = sorted(easy_shuffled[:N_TRAIN_EASY])
    test_easy = sorted(easy_shuffled[N_TRAIN_EASY:])

    return train_easy, test_easy, hard_indices


def collect_samples(
    train_easy: List[int],
    test_easy: List[int],
    hard_indices: List[int],
    synth_dir: Optional[Path],
    mask_dir: Optional[Path],
    prep_dir: Path,
    cashew_base: Path,
    normal_dir: Path,
    splits_json: dict,
    seed: int = SEED_DEFAULT,
    use_real_train: bool = False,
    n_train_normals: int = 100,
) -> Tuple[List, List]:
    """Collect train and test samples.

    Train: 44 anomalies (synthetic or real) + n_train_normals normals (default 100,
        matching UniNet's training data regime → 144 total per epoch).
    Test: 56 real anomalies (50 easy + 6 hard) + 56 normals = 112
    """
    # Normals from UniNet splits.json
    train_normals = [normal_dir / n for n in splits_json["normals"]["train"]]
    test_normals = [normal_dir / n for n in splits_json["normals"]["test"]]

    rng = random.Random(seed)

    # Test normals: use the full 50 from splits.json (matching UniNet's test set).
    # Imbalanced w.r.t. test anomalies (56) but apples-to-apples with UniNet eval.
    test_normals_subset = list(test_normals)

    # Collect canvas normals to exclude from training
    canvas_normals = set()
    if prep_dir and prep_dir.exists():
        import os as _os
        for d in _os.listdir(prep_dir):
            meta_path = prep_dir / d / "meta.json"
            if meta_path.exists():
                import json as _json
                meta = _json.load(open(meta_path))
                canvas_normals.add(meta.get("normal", ""))

    test_normal_names = {p.name for p in test_normals_subset}
    excluded = canvas_normals | test_normal_names

    # Train normals: explicit count (default 100, matches UniNet)
    train_normals_shuffled = train_normals[:]
    rng.shuffle(train_normals_shuffled)
    train_normals_subset = [n for n in train_normals_shuffled if n.name not in excluded][:n_train_normals]

    # If we need more normals, pull from unassigned normals on disk
    if len(train_normals_subset) < n_train_normals:
        all_on_disk = sorted(normal_dir.iterdir())
        split_names = {p.name for p in train_normals + test_normals}
        used_names = {p.name for p in train_normals_subset} | test_normal_names | canvas_normals
        extra_pool = [p for p in all_on_disk if p.name not in used_names and p.suffix.upper() in (".JPG", ".PNG", ".JPEG")]
        rng.shuffle(extra_pool)
        need = n_train_normals - len(train_normals_subset)
        train_normals_subset.extend(extra_pool[:need])

    # ── Train samples ──
    train_samples = []
    easy_dir = cashew_base / EASY_IMG_DIR
    visa_mask_dir = cashew_base / MASK_DIR

    if use_real_train:
        # REAL anomalies for training (upper bound experiment)
        for idx in train_easy:
            s = f"{idx:03d}"
            img_path = easy_dir / f"{s}.JPG"
            mask_path = visa_mask_dir / f"{s}.png"
            mp = str(mask_path) if mask_path.exists() else None
            if img_path.exists():
                train_samples.append((str(img_path), 1, "easy", mp))
    else:
        # Synthetic anomalies (label=1)
        for idx in train_easy:
            s = f"{idx:03d}"
            syn_path = synth_dir / f"{s}.png"
            mask_path = None
            if mask_dir and (mask_dir / f"{s}.png").exists():
                mask_path = str(mask_dir / f"{s}.png")
            elif (prep_dir / s / "placed_mask.png").exists():
                mask_path = str(prep_dir / s / "placed_mask.png")
            if syn_path.exists():
                train_samples.append((str(syn_path), 1, "easy", mask_path))

    # Real normals (label=0)
    for np_ in train_normals_subset:
        if np_.exists():
            train_samples.append((str(np_), 0, "normal", None))

    # ── Test samples ──
    test_samples = []
    easy_dir = cashew_base / EASY_IMG_DIR
    hard_dir = cashew_base / HARD_IMG_DIR
    visa_mask_dir = cashew_base / MASK_DIR

    # Real anomalies — easy test (label=1)
    for idx in test_easy:
        s = f"{idx:03d}"
        img_path = easy_dir / f"{s}.JPG"
        mask_path = visa_mask_dir / f"{s}.png"
        mp = str(mask_path) if mask_path.exists() else None
        if img_path.exists():
            test_samples.append((str(img_path), 1, "easy", mp))

    # Real anomalies — hard (label=1)
    for idx in hard_indices:
        s = f"{idx:03d}"
        img_path = hard_dir / f"{s}.JPG"
        mask_path = visa_mask_dir / f"{s}.png"
        mp = str(mask_path) if mask_path.exists() else None
        if img_path.exists():
            test_samples.append((str(img_path), 1, "hard", mp))

    # Real normals (label=0)
    for np_ in test_normals_subset:
        if np_.exists():
            test_samples.append((str(np_), 0, "normal", None))

    return train_samples, test_samples


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(backbone: str = "wrn50") -> nn.Module:
    """Build pretrained backbone with single-logit head, last stage + head trainable.

    Args:
        backbone: "wrn50" for Wide-ResNet-50-2 (2048-dim, layer4+fc trainable)
                  "convnext_b" for DINOv3-distilled ConvNeXt-Base (1024-dim, stages.3+head trainable)
    """
    if backbone == "convnext_b":
        model = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True, num_classes=1)
        for name, param in model.named_parameters():
            if not (name.startswith("stages.3") or name.startswith("head")):
                param.requires_grad = False
    else:
        model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 1)
        for name, param in model.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Model [%s]: %d trainable / %d total (%.1f%%)", backbone, trainable, total, 100 * trainable / total)
    return model


# ── Training & evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def predict(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(-1)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def optimal_threshold(probs, labels) -> float:
    best_j, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 201):
        preds = (probs >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


def compute_metrics(probs, labels, diff_keys) -> Dict:
    metrics = {}
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")

    thresh = optimal_threshold(probs, labels)
    preds = (probs >= thresh).astype(int)
    metrics["threshold"] = float(thresh)
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))

    # Confusion matrix
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics["confusion_matrix"] = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}

    keys = np.array(diff_keys)
    for diff in ("easy", "hard", "normal"):
        mask = keys == diff
        if mask.sum() < 2:
            continue
        sub_labels = labels[mask]
        sub_probs = probs[mask]
        if len(set(sub_labels)) < 2:
            metrics[f"auroc_{diff}"] = float("nan")
            continue
        try:
            metrics[f"auroc_{diff}"] = float(roc_auc_score(sub_labels, sub_probs))
        except ValueError:
            metrics[f"auroc_{diff}"] = float("nan")

    report = classification_report(labels, preds, target_names=["normal", "anomaly"], zero_division=0)
    metrics["_report"] = report
    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def save_training_plot(history, save_path):
    epochs = list(range(1, len(history["loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("ResNet Supervised AD: Synthetic→Real Transfer", fontsize=13)

    ax1.plot(epochs, history["loss"], "b-", linewidth=1.2, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["auroc"], "r-", linewidth=1.2, label="Overall")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Test AUROC (higher = better AD transfer)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Error visualization ──────────────────────────────────────────────────────

def save_error_images(test_samples, probs, preds, errors_dir, mask_dir=None, prep_dir=None):
    """Save all test images with predictions for error analysis."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()

    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    S = 256
    TITLE_H = 28

    for i, (path, label, diff, sample_mask) in enumerate(test_samples):
        pred = preds[i]
        stem = Path(path).stem
        kind = "anomaly" if label == 1 else "normal"
        correct = (pred == label)

        img = Image.open(path).convert("RGB").resize((S, S), Image.LANCZOS)

        # Mask contour overlay for anomaly images
        mask_overlay = img.copy()
        if label == 1 and sample_mask and Path(sample_mask).exists():
            mask_raw = np.array(Image.open(sample_mask).convert("L"))
            mask_bin = (mask_raw > 0).astype(np.uint8)
            if mask_bin.shape != (S, S):
                mask_bin = np.array(
                    Image.fromarray(mask_bin * 255).resize((S, S), Image.NEAREST)
                ) > 127
                mask_bin = mask_bin.astype(np.uint8)
            outer = binary_dilation(mask_bin, iterations=1).astype(np.uint8)
            edge = outer - mask_bin
            arr = np.array(mask_overlay)
            arr[edge > 0] = [255, 0, 0]
            mask_overlay = Image.fromarray(arr)

        comp_w = S * 2 + 4
        comp_h = TITLE_H + S
        composite = Image.new("RGB", (comp_w, comp_h), (255, 255, 255))
        draw = ImageDraw.Draw(composite)

        verdict = "OK" if correct else ("FP" if pred == 1 else "FN")
        pred_label = "anomaly" if pred == 1 else "normal"
        color = (0, 128, 0) if correct else (200, 0, 0)
        title = f"{stem} ({diff}) | True: {kind} | Pred: {pred_label} (p={probs[i]:.3f}) | {verdict}"
        draw.text((comp_w // 2, TITLE_H // 2), title, fill=color, font=font, anchor="mm")

        composite.paste(img, (0, TITLE_H))
        composite.paste(mask_overlay, (S + 4, TITLE_H))

        if not correct:
            sub = "FP_normal_called_anomaly" if pred == 1 else "FN_anomaly_called_normal"
        else:
            sub = "correct"
        sub_dir = errors_dir / sub
        sub_dir.mkdir(parents=True, exist_ok=True)
        composite.save(sub_dir / f"{diff}_{stem}_{kind}_p{probs[i]:.3f}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ResNet supervised AD: train on synthetic+normal, test on real+normal"
    )
    parser.add_argument("--synth-dir", type=str, default=None,
                        help="Path to generated synthetic images (cfg7_vis)")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Path to refined masks (overrides placed_mask)")
    parser.add_argument("--prep-dir", type=str, required=True,
                        help="Path to prep directory")
    parser.add_argument("--use-real-train", action="store_true",
                        help="Train on REAL anomalies instead of synthetic (upper bound)")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--backbone", type=str, default="wrn50",
                        choices=["wrn50", "convnext_b"],
                        help="Backbone: wrn50 (Wide-ResNet-50-2) or convnext_b (ConvNeXt-Base)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-cj", type=float, default=0.1,
                        help="Training ColorJitter strength (default: 0.1, 0 to disable)")
    parser.add_argument("--train-blur", type=float, default=0.5,
                        help="Training blur max sigma (default: 0.5, 0 to disable)")
    parser.add_argument("--train-noise", type=float, default=0.005,
                        help="Training Gaussian noise std (default: 0.005, 0 to disable)")
    parser.add_argument("--n-train-normals", type=int, default=100,
                        help="Number of training normals (default 100 — matches UniNet's "
                             "100 normals + 44 anomalies = 144 sample regime).")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Training batch size (default 4 — matches UniNet's T2bs4 setup for "
                             "apples-to-apples comparison).")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--shift-levels", type=str, default=None,
                        help="Custom shift levels as JSON string, e.g. "
                             "'{\"1x\":{\"cj\":0.1,\"blur\":1.0,\"noise\":0.005}}'. "
                             "Default: 1x/2x/3x/4x standard levels.")
    args = parser.parse_args()

    if not args.use_real_train and not args.synth_dir:
        parser.error("--synth-dir is required when not using --use-real-train")

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    project_root = Path(__file__).parent.parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else project_root / p

    synth_dir = resolve(args.synth_dir) if args.synth_dir else None
    prep_dir = resolve(args.prep_dir)
    save_dir = resolve(args.save_dir)
    mask_dir = resolve(args.mask_dir) if args.mask_dir else None
    cashew_base = project_root / CASHEW_BASE
    normal_dir = cashew_base / NORMAL_DIR

    save_dir.mkdir(parents=True, exist_ok=True)

    # Load UniNet splits for normal train/test split
    splits_path = cashew_base / UNINET_SPLITS
    with open(splits_path) as f:
        splits_json = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Build AD split
    train_easy, test_easy, hard_indices = build_splits(prep_dir, seed=args.seed)
    log.info("Split: %d train easy, %d test easy, %d hard (test only)",
             len(train_easy), len(test_easy), len(hard_indices))

    train_samples, test_samples = collect_samples(
        train_easy, test_easy, hard_indices,
        synth_dir, mask_dir, prep_dir, cashew_base, normal_dir, splits_json,
        seed=args.seed,
        use_real_train=args.use_real_train,
        n_train_normals=args.n_train_normals,
    )

    n_train_anom = sum(1 for _, l, _, _ in train_samples if l == 1)
    n_train_norm = sum(1 for _, l, _, _ in train_samples if l == 0)
    n_test_anom = sum(1 for _, l, _, _ in test_samples if l == 1)
    n_test_norm = sum(1 for _, l, _, _ in test_samples if l == 0)
    log.info("Train: %d anomaly + %d normal = %d", n_train_anom, n_train_norm, len(train_samples))
    log.info("Test:  %d anomaly + %d normal = %d", n_test_anom, n_test_norm, len(test_samples))

    # Build datasets and loaders
    train_ds = AnomalyDetectionDataset(train_samples, build_transforms(
        train=True, train_cj=args.train_cj, train_blur=args.train_blur, train_noise=args.train_noise,
    ), equalize_resolution=IMAGE_SIZE)
    test_ds = AnomalyDetectionDataset(test_samples, build_transforms(train=False), equalize_resolution=IMAGE_SIZE)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    # Model
    model = build_model(backbone=args.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # Build shift dataloaders for per-epoch evaluation (matching UniNet)
    if args.shift_levels:
        import json as _json
        SHIFT_LEVELS = _json.loads(args.shift_levels)
    else:
        SHIFT_LEVELS = {
            "1x": {"cj": 0.1, "blur": 1.0, "noise": 0.005},
            "2x": {"cj": 0.2, "blur": 2.0, "noise": 0.005},
            "3x": {"cj": 0.3, "blur": 3.0, "noise": 0.005},
            "4x": {"cj": 0.4, "blur": 4.0, "noise": 0.005},
        }
    shift_loaders = {}
    for shift_label, shift_cfg in SHIFT_LEVELS.items():
        shift_tf = build_transforms(train=False, test_cj=shift_cfg["cj"],
                                     test_blur=shift_cfg["blur"], test_noise=shift_cfg["noise"])
        shift_ds = AnomalyDetectionDataset(test_samples, shift_tf, equalize_resolution=IMAGE_SIZE)
        shift_loaders[shift_label] = DataLoader(shift_ds, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=4, pin_memory=True, persistent_workers=True)
    log.info("Multi-shift evaluation per epoch (%d levels): %s", len(SHIFT_LEVELS), list(SHIFT_LEVELS.keys()))

    # Training with per-epoch shift evaluation
    best_auroc = 0.0
    best_state = None
    test_diff_keys = [d for _, _, d, _ in test_samples]
    history = {"loss": [], "auroc": []}
    # Per-epoch shift results: {shift_label: {"img_aurocs": [...]}}
    shift_history = {sl: {"img_aurocs": []} for sl in SHIFT_LEVELS}

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        probs, lbls = predict(model, test_loader, device)
        metrics_ep = compute_metrics(probs, lbls, test_diff_keys)
        auroc = metrics_ep["auroc"] if not np.isnan(metrics_ep["auroc"]) else 0.0

        history["loss"].append(loss)
        history["auroc"].append(metrics_ep["auroc"])

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Per-epoch shift evaluation
        for shift_label, shift_loader in shift_loaders.items():
            shift_probs, shift_lbls = predict(model, shift_loader, device)
            shift_metrics = compute_metrics(shift_probs, shift_lbls, test_diff_keys)
            shift_auroc = shift_metrics["auroc"] if not np.isnan(shift_metrics["auroc"]) else 0.0
            shift_history[shift_label]["img_aurocs"].append(shift_auroc)

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            shift_str = " | ".join(f"{sl}={shift_history[sl]['img_aurocs'][-1]:.3f}" for sl in SHIFT_LEVELS)
            log.info("Epoch %3d/%d | loss=%.4f | AUROC=%.4f | best=%.4f | %s",
                     epoch, args.epochs, loss, auroc, best_auroc, shift_str)

        if epoch % 5 == 0 or epoch == args.epochs:
            save_training_plot(history, save_dir / "plot.png")

    # Final evaluation with best model
    log.info("=" * 60)
    log.info("FINAL EVALUATION")
    log.info("=" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    probs, lbls = predict(model, test_loader, device)
    metrics = compute_metrics(probs, lbls, test_diff_keys)

    log.info("AUROC:    %.4f", metrics["auroc"])
    log.info("F1:       %.4f", metrics["f1"])
    log.info("Accuracy: %.4f", metrics["accuracy"])
    log.info("\n%s", metrics["_report"])

    # Save error images
    thresh = metrics["threshold"]
    preds_bin = (probs >= thresh).astype(int)
    save_error_images(test_samples, probs, preds_bin, save_dir / "errors", mask_dir, prep_dir)

    # Final shift evaluation with best model
    log.info("Multi-shift evaluation (best model):")
    shift_results = {}
    for shift_label, shift_loader in shift_loaders.items():
        shift_probs, shift_lbls = predict(model, shift_loader, device)
        shift_metrics = compute_metrics(shift_probs, shift_lbls, test_diff_keys)
        shift_results[shift_label] = {
            "auroc": shift_metrics["auroc"],
            "f1": shift_metrics["f1"],
            "accuracy": shift_metrics["accuracy"],
            "confusion_matrix": shift_metrics["confusion_matrix"],
            "img_aurocs": shift_history[shift_label]["img_aurocs"],
        }
        log.info("  %s: AUROC=%.4f  F1=%.4f", shift_label, shift_metrics["auroc"], shift_metrics["f1"])

    # Save metrics
    task_name = "real_ad_upper_bound" if args.use_real_train else "supervised_ad_transfer"
    results_dict = {
        "task": task_name,
        "use_real_train": args.use_real_train,
        "train_anomaly": n_train_anom,
        "train_normal": n_train_norm,
        "test_anomaly": n_test_anom,
        "test_normal": n_test_norm,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "auroc": metrics["auroc"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
        "train_easy_indices": train_easy,
        "test_easy_indices": test_easy,
        "hard_indices": hard_indices,
        "img_aurocs": history["auroc"],
    }
    results_dict["multi_shift"] = shift_results

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    log.info("Saved: %s", save_dir / "metrics.json")


if __name__ == "__main__":
    main()
