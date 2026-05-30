#!/usr/bin/env python3
"""Wide-ResNet-50-2 binary classifier: real (0) vs synthetic (1) anomaly discrimination.

If the CNN can't distinguish real from synthetic anomalies, the synthetic
data is high quality.

Two test scenarios (selected via --splits-file):
  - Test 1 (splits_test1.json): 50/50 disjoint references vs real.
    Train: 40S + 40R = 80, Test: 10S + 10R = 20.
  - Test 2 (splits_test2.json): full 200 — all 100 as both real and synthetic.
    Train: 80×2 = 160, Test: 20×2 = 40.

Usage:
    python scripts/train_resnet.py \
        --experiment-dir <path_to_experiment_ResNet> \
        --splits-file splits_test1.json \
        --epochs 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import io
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
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion
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

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RealVsSyntheticDataset(Dataset):
    """Binary classification: 0=real, 1=synthetic."""

    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        transform: transforms.Compose,
        crop_to_anomaly: bool = False,
        mask_out_anomaly: bool = False,
        crop_to_foreground: bool = False,
        fg_threshold: int = 30,
        birefnet_masks: Optional[Dict[str, str]] = None,
        mask_lookup: Optional[Dict[str, str]] = None,
        crop_padding: float = 0.2,
        equalize_resolution: int = 0,
    ):
        """
        Args:
            samples: list of (path, label, difficulty) tuples.
                label: 0=real, 1=synthetic.
                difficulty: "easy" or "hard".
            crop_to_anomaly: if True, crop image to anomaly bounding box before transforms.
            mask_out_anomaly: if True, black out anomaly region (test background discriminability).
            crop_to_foreground: if True, crop to foreground (cashew) bounding box.
                Removes the foreground-coverage confound (VisA normal ~53% vs anomaly ~64% FG).
            fg_threshold: grayscale threshold for foreground detection (default 30).
            birefnet_masks: dict mapping "stem_label" -> BiRefNet mask path.
                If provided, uses BiRefNet masks for crop-to-foreground instead of thresholding.
            mask_lookup: dict mapping image stem -> mask path (for crop/mask modes).
            crop_padding: fractional padding around anomaly bbox (default 0.2 = 20%).
            equalize_resolution: if >0, resize all images to this square size before
                any processing (removes resolution-based confounds). 0 = disabled.
        """
        self.samples = samples
        self.transform = transform
        self.crop_to_anomaly = crop_to_anomaly
        self.mask_out_anomaly = mask_out_anomaly
        self.crop_to_foreground = crop_to_foreground
        self.fg_threshold = fg_threshold
        self.birefnet_masks = birefnet_masks or {}
        self.mask_lookup = mask_lookup or {}
        self.crop_padding = crop_padding
        self.equalize_resolution = equalize_resolution

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _diff = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # JPEG-compress non-JPEG images (synthetics/PNGs) to match real JPEGs
        if not path.lower().endswith((".jpg", ".jpeg")):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # Equalize resolution: resize all images to the same size BEFORE any
        # processing.  Real VisA images are ~1274x1176 while synthetics are
        # 512x512 — the CNN can trivially distinguish based on downsampling
        # artifacts alone.  Resizing both to 512x512 here removes that confound.
        if self.equalize_resolution > 0:
            sz = self.equalize_resolution
            if img.size != (sz, sz):
                img = img.resize((sz, sz), Image.LANCZOS)

        # Crop to foreground bounding box if requested.  VisA anomaly photos
        # have ~64% cashew coverage vs normal photos ~53% (0% P10-P90 overlap).
        # Cropping to the tight foreground bbox removes dark background so the
        # CNN can't exploit background-area ratio as a class cue.
        if self.crop_to_foreground:
            stem = Path(path).stem
            key = f"{stem}_{label}"
            birefnet_path = self.birefnet_masks.get(key)
            if birefnet_path and Path(birefnet_path).exists():
                # Use precomputed BiRefNet saliency mask
                fg_mask = np.array(Image.open(birefnet_path).convert("L"))
                iw, ih = img.size
                H, W = fg_mask.shape
                if (H, W) != (ih, iw):
                    fg_mask = np.array(
                        Image.fromarray(fg_mask).resize((iw, ih), Image.NEAREST)
                    )
                fg = fg_mask > 127
            else:
                # Fallback: simple brightness threshold
                img_arr = np.array(img)
                gray = img_arr.mean(axis=2)
                fg = gray > self.fg_threshold
            ys, xs = np.where(fg)
            if len(ys) > 10:  # need some foreground
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                # Add small padding (5%) to avoid cutting right at edge
                h, w = y1 - y0, x1 - x0
                pad = int(max(h, w) * 0.05)
                iw, ih = img.size
                y0 = max(0, y0 - pad)
                x0 = max(0, x0 - pad)
                y1 = min(ih, y1 + pad)
                x1 = min(iw, x1 + pad)
                img = img.crop((x0, y0, x1, y1))

        # Crop to anomaly: square crop covering all anomaly pixels, min 256,
        # padded by 20%, resized to 256x256.
        if self.crop_to_anomaly:
            stem = Path(path).stem
            key = f"{stem}_{label}"
            mask_path = self.mask_lookup.get(key)
            if mask_path and Path(mask_path).exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                mask_bin = mask > (0 if label == 0 else 127)
                ys, xs = np.where(mask_bin)
                if len(ys) > 0:
                    H, W = mask.shape
                    iw, ih = img.size
                    # Anomaly bbox in image coords
                    my0, my1 = int(ys.min()), int(ys.max())
                    mx0, mx1 = int(xs.min()), int(xs.max())
                    by0 = int(my0 * ih / H)
                    by1 = int(my1 * ih / H)
                    bx0 = int(mx0 * iw / W)
                    bx1 = int(mx1 * iw / W)
                    # Bbox size + 10% padding
                    bh = by1 - by0
                    bw = bx1 - bx0
                    pad = int(max(bh, bw) * 0.1)
                    # Square side = max(bbox_h, bbox_w) + padding, minimum 256
                    side = max(bh + 2 * pad, bw + 2 * pad, 256)
                    # Center on bbox center
                    cy = (by0 + by1) // 2
                    cx = (bx0 + bx1) // 2
                    half = side // 2
                    y0 = max(0, min(cy - half, ih - side))
                    x0 = max(0, min(cx - half, iw - side))
                    y1 = y0 + side
                    x1 = x0 + side
                    # Clamp if image is smaller than side
                    if ih < side:
                        y0, y1 = 0, ih
                    if iw < side:
                        x0, x1 = 0, iw
                    img = img.crop((x0, y0, x1, y1))
                    # Resize to 256x256
                    if img.size != (256, 256):
                        img = img.resize((256, 256), Image.LANCZOS)

        # Mask out anomaly region if requested (black out anomaly, keep background)
        # Uses a "roundtripped" mask: downsample to latent res via maxpool (matches
        # what the UNet sees), dilate by 2px in latent space (covers band_mode=2
        # blending zone), then upsample back.  This blacks out EVERYTHING the
        # diffusion model touched, including the alpha-blend transition zone.
        if self.mask_out_anomaly:
            stem = Path(path).stem
            key = f"{stem}_{label}"
            mask_path = self.mask_lookup.get(key)
            if mask_path and Path(mask_path).exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                mask_bin = mask > (0 if label == 0 else 127)
                # Resize mask to image size if needed
                iw, ih = img.size
                H, W = mask_bin.shape
                if (H, W) != (ih, iw):
                    mask_pil = Image.fromarray(mask_bin.astype(np.uint8) * 255)
                    mask_pil = mask_pil.resize((iw, ih), Image.NEAREST)
                    mask_bin = np.array(mask_pil) > 127
                # Round-trip through latent resolution: maxpool down, dilate, upsample back
                mask_t = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                latent_h, latent_w = ih // 8, iw // 8
                mask_latent = torch.nn.functional.max_pool2d(
                    mask_t, kernel_size=(ih // latent_h, iw // latent_w),
                )  # [1,1,latent_h,latent_w]
                # Dilate 2px in latent space (covers band_mode=2 blend zone)
                mask_latent_np = (mask_latent.squeeze().numpy() > 0.5).astype(np.uint8)
                mask_latent_np = binary_dilation(mask_latent_np, iterations=2).astype(np.uint8)
                # Upsample back to image resolution
                mask_latent_t = torch.from_numpy(mask_latent_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mask_roundtrip = torch.nn.functional.interpolate(
                    mask_latent_t, size=(ih, iw), mode="nearest",
                ).squeeze().numpy() > 0.5
                img_arr = np.array(img)
                img_arr[mask_roundtrip] = 0
                img = Image.fromarray(img_arr)

        # Re-compress through JPEG to equalise compression artifacts
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_transforms(train: bool) -> transforms.Compose:
    """Minimal transforms for real-vs-synthetic quality test.

    No augmentation — pure quality signal. Crop-to-anomaly (in dataset)
    handles spatial confounds.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _find_image(directory: Path, stem: str) -> str | None:
    """Find an image file by stem in a directory (tries common extensions)."""
    for ext in [".JPG", ".jpg", ".png", ".jpeg"]:
        p = directory / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def collect_samples_test1(
    experiment_dir: Path,
    splits: dict,
    partition: str,
) -> List[Tuple[str, int, str]]:
    """Collect samples for Test 1 (50/50 disjoint).

    Args:
        partition: "train" or "test"

    Returns:
        list of (path, label, difficulty) — label 0=real, 1=synthetic
    """
    samples: List[Tuple[str, int, str]] = []
    real_dir = experiment_dir / "real_anomalies"
    synth_dir = experiment_dir / "anomaly" / "generated"

    part = splits[partition]

    # Real anomaly images (label=0)
    for diff in ("easy", "hard"):
        for aid in part["real"].get(diff, []):
            path = _find_image(real_dir / diff / "imgs", aid)
            if path:
                samples.append((path, 0, diff))
            else:
                log.warning("Real image not found: %s/%s", diff, aid)

    # Synthetic anomaly images (label=1)
    for diff in ("easy", "hard"):
        for aid in part["synthetic"].get(diff, []):
            path = _find_image(synth_dir / diff, aid)
            if path:
                samples.append((path, 1, diff))
            else:
                log.warning("Synthetic image not found: %s/%s", diff, aid)

    return samples


def collect_samples_test2(
    experiment_dir: Path,
    splits: dict,
    partition: str,
) -> List[Tuple[str, int, str]]:
    """Collect samples for Test 2 (full/200).

    Each ID in train/test has both a real and synthetic version.

    Args:
        partition: "train" or "test"

    Returns:
        list of (path, label, difficulty) — label 0=real, 1=synthetic
    """
    samples: List[Tuple[str, int, str]] = []
    real_dir = experiment_dir / "real_anomalies"
    synth_dir = experiment_dir / "anomaly" / "generated"

    part = splits[partition]

    for diff in ("easy", "hard"):
        for aid in part.get(diff, []):
            # Real version (label=0)
            path_real = _find_image(real_dir / diff / "imgs", aid)
            if path_real:
                samples.append((path_real, 0, diff))
            else:
                log.warning("Real image not found: %s/%s", diff, aid)

            # Synthetic version (label=1)
            path_synth = _find_image(synth_dir / diff, aid)
            if path_synth:
                samples.append((path_synth, 1, diff))
            else:
                log.warning("Synthetic image not found: %s/%s", diff, aid)

    return samples


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(unfreeze_all: bool = False, backbone: str = "wrn50") -> nn.Module:
    """Build pretrained backbone with single-logit head.

    Args:
        unfreeze_all: If True, all parameters are trainable.
        backbone: "wrn50" for Wide-ResNet-50-2 (2048-dim, layer4+fc trainable)
                  "convnext_b" for DINOv3-distilled ConvNeXt-Base (1024-dim, stages.3+head trainable)
    """
    if backbone == "convnext_b":
        model = timm.create_model("convnext_base.dinov3_lvd1689m", pretrained=True, num_classes=1)
        if not unfreeze_all:
            for name, param in model.named_parameters():
                if not (name.startswith("stages.3") or name.startswith("head")):
                    param.requires_grad = False
    else:
        model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(2048, 1)
        if not unfreeze_all:
            for name, param in model.named_parameters():
                if not (name.startswith("layer4") or name.startswith("fc")):
                    param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Model [%s]: %d trainable / %d total (%.1f%%)",
             backbone, trainable, total, 100.0 * trainable / total)
    return model


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

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
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, labels) arrays."""
    model.eval()
    all_probs, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(-1)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs), np.concatenate(all_labels)


def optimal_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    """Youden's J statistic for optimal binary threshold."""
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


def compute_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    diff_keys: List[str],
) -> Dict:
    """Compute AUROC, F1, accuracy overall and per-difficulty."""
    metrics: Dict = {}

    # Overall AUROC
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")

    # Optimal threshold + derived metrics
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

    # Per-difficulty AUROC
    keys = np.array(diff_keys)
    for diff in ("easy", "hard"):
        mask = keys == diff
        if mask.sum() < 2:
            metrics[f"auroc_{diff}"] = float("nan")
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

    report = classification_report(
        labels, preds, target_names=["real", "synthetic"], zero_division=0,
    )
    metrics["_report"] = report
    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_training_plot(
    history: Dict[str, List[float]],
    save_path: Path,
    test_name: str,
) -> None:
    """Save a 2-panel training plot: loss + AUROC over epochs."""
    epochs = list(range(1, len(history["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"ResNet Real-vs-Synthetic — {test_name}", fontsize=13)

    # Left: Loss
    ax1.plot(epochs, history["loss"], "b-", linewidth=1.2, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: AUROC
    ax2.plot(epochs, history["auroc"], "r-", linewidth=1.2, label="Overall")
    if any(not np.isnan(v) for v in history.get("auroc_easy", [])):
        ax2.plot(epochs, history["auroc_easy"], "g--", linewidth=1, alpha=0.7, label="Easy")
    if any(not np.isnan(v) for v in history.get("auroc_hard", [])):
        ax2.plot(epochs, history["auroc_hard"], "m--", linewidth=1, alpha=0.7, label="Hard")
    ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Random (0.5)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUROC")
    ax2.set_title("Test AUROC (lower = better synthetic quality)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wide-ResNet-50-2 real-vs-synthetic anomaly discriminator. "
                    "Low AUROC = good synthetic quality (CNN can't tell them apart).",
    )
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--splits-file", type=str, required=True,
                        help="Split file name (e.g. splits_test1.json)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbone", type=str, default="wrn50",
                        choices=["wrn50", "convnext_b"],
                        help="Backbone: wrn50 (Wide-ResNet-50-2) or convnext_b (DINOv3 ConvNeXt-Base)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-all", action="store_true",
                        help="Fine-tune entire network (default: only last stage + head)")
    parser.add_argument("--plot-every", type=int, default=5,
                        help="Save training plot every N epochs (default: 5)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override results directory (default: <experiment-dir>/results)")
    parser.add_argument("--checkpoint-name", type=str, default=None,
                        help="Name of checkpoint used for generation (tracked in metrics)")
    parser.add_argument("--prep-dir", type=str, default=None,
                        help="Path to prep/ dir with placed_mask.png per sample (for error overlays on synthetic images)")
    parser.add_argument("--mask-dir", type=str, default=None,
                        help="Path to refined masks dir (overrides placed_mask for synthetic samples in crop/mask modes)")
    parser.add_argument("--crop-to-anomaly", action="store_true",
                        help="Crop images to anomaly bounding box (removes seam context)")
    parser.add_argument("--crop-padding", type=float, default=0.2,
                        help="Fractional padding around anomaly bbox (default: 0.2)")
    parser.add_argument("--mask-out-anomaly", action="store_true",
                        help="Black out anomaly region in full image (tests if background/seam alone is discriminative)")
    parser.add_argument("--crop-to-foreground", action="store_true",
                        help="Crop to foreground (cashew) bounding box before transforms. "
                             "Removes the foreground-coverage confound (normal=53%% vs anomaly=64%% FG).")
    parser.add_argument("--fg-threshold", type=int, default=30,
                        help="Grayscale threshold for foreground detection (default: 30)")
    parser.add_argument("--birefnet-mask-dir", type=str, default=None,
                        help="Path to precomputed BiRefNet masks (real/ and synth/ subdirs). "
                             "Used with --crop-to-foreground for precise object-level cropping.")
    parser.add_argument("--equalize-resolution", type=int, default=512,
                        help="Resize all images to NxN before processing to remove resolution "
                             "confound (real=1274x1176 vs synth=512x512). Set 0 to disable. (default: 512)")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_dir = Path(args.experiment_dir)
    splits_path = experiment_dir / args.splits_file
    if not splits_path.exists():
        log.error("Splits file not found: %s", splits_path)
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_name = Path(args.splits_file).stem  # e.g. "splits_test1"
    log.info("Device: %s", device)
    log.info("Splits: %s", args.splits_file)
    log.info("Task: %s", splits.get("description", "real vs synthetic"))

    # ---- Detect test format and collect data ----
    is_test1 = "reference" in splits  # test1 has disjoint ref/real
    if is_test1:
        train_samples = collect_samples_test1(experiment_dir, splits, "train")
        test_samples = collect_samples_test1(experiment_dir, splits, "test")
    else:
        train_samples = collect_samples_test2(experiment_dir, splits, "train")
        test_samples = collect_samples_test2(experiment_dir, splits, "test")

    n_train_real = sum(1 for _, l, _ in train_samples if l == 0)
    n_train_synth = sum(1 for _, l, _ in train_samples if l == 1)
    n_test_real = sum(1 for _, l, _ in test_samples if l == 0)
    n_test_synth = sum(1 for _, l, _ in test_samples if l == 1)

    log.info("Train: %d real + %d synthetic = %d",
             n_train_real, n_train_synth, len(train_samples))
    log.info("Test:  %d real + %d synthetic = %d",
             n_test_real, n_test_synth, len(test_samples))

    from collections import Counter
    train_diff = Counter(d for _, _, d in train_samples)
    test_diff = Counter(d for _, _, d in test_samples)
    log.info("Train difficulty: %s", dict(train_diff))
    log.info("Test difficulty:  %s", dict(test_diff))

    if n_train_synth == 0:
        log.error("No synthetic training images found. "
                   "Generate them first (scripts/generate_cashew.py --all).")
        sys.exit(1)
    if n_train_real == 0:
        log.error("No real training images found. "
                   "Run setup_resnet_v2.py first.")
        sys.exit(1)

    # ---- Build mask lookup for crop-to-anomaly / mask-out modes ----
    # Key: "stem_label" to avoid collisions (same ID in both real and synthetic)
    mask_lookup: Dict[str, str] = {}
    if args.crop_to_anomaly or args.mask_out_anomaly:
        visa_mask_dir = experiment_dir.parent / "Data" / "Masks" / "Anomaly"
        prep_dir_path = Path(args.prep_dir) if args.prep_dir else None
        all_samples = train_samples + test_samples
        for path, label, _diff in all_samples:
            stem = Path(path).stem
            key = f"{stem}_{label}"
            if label == 0:  # real → VisA mask
                mp = visa_mask_dir / f"{stem}.png"
                if mp.exists():
                    mask_lookup[key] = str(mp)
            else:  # synthetic → refined mask (if --mask-dir) or placed mask
                mask_dir_path = Path(args.mask_dir) if args.mask_dir else None
                if mask_dir_path and (mask_dir_path / f"{stem}.png").exists():
                    mask_lookup[key] = str(mask_dir_path / f"{stem}.png")
                elif prep_dir_path:
                    mp = prep_dir_path / stem / "placed_mask.png"
                    if mp.exists():
                        mask_lookup[key] = str(mp)
        mode_name = "crop-to-anomaly" if args.crop_to_anomaly else "mask-out-anomaly"
        log.info("%s: %d/%d mask lookups found", mode_name, len(mask_lookup), len(all_samples))

    # ---- Build loaders ----
    if args.equalize_resolution > 0:
        log.info("Resolution equalization: all images → %dx%d before processing",
                 args.equalize_resolution, args.equalize_resolution)
    # ---- Build BiRefNet mask lookup for crop-to-foreground ----
    birefnet_masks: Dict[str, str] = {}
    if args.crop_to_foreground and args.birefnet_mask_dir:
        bdir = Path(args.birefnet_mask_dir)
        all_samples_for_lookup = train_samples + test_samples
        for path, label, _diff in all_samples_for_lookup:
            stem = Path(path).stem
            key = f"{stem}_{label}"
            sub = "real" if label == 0 else "synth"
            mp = bdir / sub / f"{stem}.png"
            if mp.exists():
                birefnet_masks[key] = str(mp)
        log.info("BiRefNet masks: %d/%d lookups found",
                 len(birefnet_masks), len(all_samples_for_lookup))
    if args.crop_to_foreground:
        log.info("Crop-to-foreground: %s — removing FG coverage confound",
                 "BiRefNet" if birefnet_masks else f"threshold={args.fg_threshold}")
    train_ds = RealVsSyntheticDataset(
        train_samples, build_transforms(train=True),
        crop_to_anomaly=args.crop_to_anomaly, mask_out_anomaly=args.mask_out_anomaly,
        crop_to_foreground=args.crop_to_foreground, fg_threshold=args.fg_threshold,
        birefnet_masks=birefnet_masks,
        mask_lookup=mask_lookup, crop_padding=args.crop_padding,
        equalize_resolution=args.equalize_resolution,
    )
    test_ds = RealVsSyntheticDataset(
        test_samples, build_transforms(train=False),
        crop_to_anomaly=args.crop_to_anomaly, mask_out_anomaly=args.mask_out_anomaly,
        crop_to_foreground=args.crop_to_foreground, fg_threshold=args.fg_threshold,
        birefnet_masks=birefnet_masks,
        mask_lookup=mask_lookup, crop_padding=args.crop_padding,
        equalize_resolution=args.equalize_resolution,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ---- Model ----
    model = build_model(unfreeze_all=args.unfreeze_all, backbone=args.backbone).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training ----
    best_auroc = 0.0
    best_state = None
    test_diff_keys = [d for _, _, d in test_samples]
    if args.save_dir:
        results_dir = Path(args.save_dir)
    else:
        results_dir = experiment_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / f"resnet_{test_name}_plot.png"

    history: Dict[str, List[float]] = {
        "loss": [], "auroc": [], "auroc_easy": [], "auroc_hard": [],
    }

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        probs, lbls = predict(model, test_loader, device)
        metrics_ep = compute_metrics(probs, lbls, test_diff_keys)
        auroc = metrics_ep["auroc"] if not np.isnan(metrics_ep["auroc"]) else 0.0

        history["loss"].append(loss)
        history["auroc"].append(metrics_ep["auroc"])
        history["auroc_easy"].append(metrics_ep.get("auroc_easy", float("nan")))
        history["auroc_hard"].append(metrics_ep.get("auroc_hard", float("nan")))

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            log.info("Epoch %3d/%d | loss=%.4f | AUROC=%.4f (easy=%.3f hard=%.3f) | best=%.4f",
                     epoch, args.epochs, loss, auroc,
                     metrics_ep.get("auroc_easy", float("nan")),
                     metrics_ep.get("auroc_hard", float("nan")),
                     best_auroc)

        if epoch % args.plot_every == 0 or epoch == args.epochs:
            save_training_plot(history, plot_path, test_name)
            log.info("  Plot saved → %s", plot_path)

    # ---- Final evaluation with best model ----
    log.info("=" * 60)
    log.info("FINAL EVALUATION (best model, %s)", test_name)
    log.info("=" * 60)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    probs, lbls = predict(model, test_loader, device)
    metrics = compute_metrics(probs, lbls, test_diff_keys)

    log.info("AUROC:      %.4f", metrics["auroc"])
    log.info("AUROC easy: %.4f", metrics.get("auroc_easy", float("nan")))
    log.info("AUROC hard: %.4f", metrics.get("auroc_hard", float("nan")))
    log.info("F1:         %.4f", metrics["f1"])
    log.info("Accuracy:   %.4f", metrics["accuracy"])
    log.info("Threshold:  %.3f", metrics["threshold"])
    log.info("\n%s", metrics["_report"])

    # Interpretation guide
    if metrics["auroc"] < 0.6:
        log.info("AUROC < 0.6 — CNN CANNOT distinguish real from synthetic (GOOD quality)")
    elif metrics["auroc"] < 0.75:
        log.info("AUROC 0.6-0.75 — CNN finds SOME differences (moderate quality)")
    else:
        log.info("AUROC > 0.75 — CNN EASILY distinguishes real from synthetic (poor quality)")

    # ---- Save ALL test images with mask edges, titled and labelled ----
    from PIL import ImageDraw, ImageFont
    try:
        _title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        _label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except OSError:
        _title_font = ImageFont.load_default()
        _label_font = _title_font

    thresh = metrics["threshold"]
    preds = (probs >= thresh).astype(int)
    errors_dir = results_dir / f"errors_{test_name}"
    if errors_dir.exists():
        shutil.rmtree(errors_dir)
    errors_dir.mkdir(parents=True, exist_ok=True)

    # Mask dirs for edge overlay
    visa_mask_dir = (experiment_dir.parent / "Data" / "Masks" / "Anomaly")
    prep_dir = Path(args.prep_dir) if args.prep_dir else None
    birefnet_dir = Path(args.birefnet_mask_dir) if args.birefnet_mask_dir else None

    def _preprocess_for_viz(path: str, label: int, stem: str) -> Image.Image:
        """Apply same preprocessing as Dataset.__getitem__ (minus transforms/JPEG)."""
        img = Image.open(path).convert("RGB")
        # Equalize resolution
        if args.equalize_resolution > 0:
            sz = args.equalize_resolution
            if img.size != (sz, sz):
                img = img.resize((sz, sz), Image.LANCZOS)
        # BiRefNet crop
        if args.crop_to_foreground and birefnet_dir:
            sub = "real" if label == 0 else "synth"
            mp = birefnet_dir / sub / f"{stem}.png"
            if mp.exists():
                iw, ih = img.size
                fg_mask = np.array(Image.open(mp).convert("L"))
                fg_mask = np.array(Image.fromarray(fg_mask).resize((iw, ih), Image.NEAREST))
                fg = fg_mask > 127
                ys, xs = np.where(fg)
                if len(ys) > 10:
                    y0, y1 = int(ys.min()), int(ys.max())
                    x0, x1 = int(xs.min()), int(xs.max())
                    h, w = y1 - y0, x1 - x0
                    pad = int(max(h, w) * 0.05)
                    y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
                    y1 = min(ih, y1 + pad); x1 = min(iw, x1 + pad)
                    img = img.crop((x0, y0, x1, y1))
        # Mask-out
        if args.mask_out_anomaly:
            key = f"{stem}_{label}"
            mask_path = mask_lookup.get(key)
            if mask_path and Path(mask_path).exists():
                iw, ih = img.size
                mask_raw = np.array(Image.open(mask_path).convert("L"))
                m_bin = mask_raw > (0 if label == 0 else 127)
                H, W = m_bin.shape
                if (H, W) != (ih, iw):
                    m_bin = np.array(Image.fromarray(m_bin.astype(np.uint8) * 255).resize((iw, ih), Image.NEAREST)) > 127
                mask_t = torch.from_numpy(m_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                lh, lw = ih // 8, iw // 8
                ml = torch.nn.functional.max_pool2d(mask_t, kernel_size=(ih // lh, iw // lw))
                ml_np = (ml.squeeze().numpy() > 0.5).astype(np.uint8)
                ml_np = binary_dilation(ml_np, iterations=2).astype(np.uint8)
                ml_t = torch.from_numpy(ml_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                mr = torch.nn.functional.interpolate(ml_t, size=(ih, iw), mode="nearest").squeeze().numpy() > 0.5
                arr = np.array(img)
                arr[mr] = 0
                img = Image.fromarray(arr)
        return img

    def _get_birefnet_crop_box(stem: str, label: int, eq_sz: int) -> Optional[Tuple[int, int, int, int]]:
        """Return (x0, y0, x1, y1) crop box from BiRefNet mask, or None."""
        if not birefnet_dir:
            return None
        sub = "real" if label == 0 else "synth"
        mp = birefnet_dir / sub / f"{stem}.png"
        if not mp.exists():
            return None
        fg_mask = np.array(Image.open(mp).convert("L"))
        fg_mask = np.array(Image.fromarray(fg_mask).resize((eq_sz, eq_sz), Image.NEAREST))
        fg = fg_mask > 127
        ys, xs = np.where(fg)
        if len(ys) <= 10:
            return None
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        h, w = y1 - y0, x1 - x0
        pad = int(max(h, w) * 0.05)
        y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
        y1 = min(eq_sz, y1 + pad); x1 = min(eq_sz, x1 + pad)
        return (x0, y0, x1, y1)

    def _load_anomaly_mask(stem: str, label: int, img_size: Tuple[int, int]) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """Load anomaly mask for visualization overlay.
        - Synthetic (label=1): returns (full_roundtripped, core_only) — both at img_size.
          core = maxpooled to latent, NO dilation. full = core + 2px latent dilation.
        - Real (label=0): returns (raw_mask, None).
        Applies the same BiRefNet crop as the image."""
        # Load raw mask
        if label == 0:
            p = visa_mask_dir / f"{stem}.png"
            if not p.exists():
                return None
            raw = np.array(Image.open(p).convert("L"))
            mask_bin = (raw > 0)
        else:
            if prep_dir is None:
                return None
            p = prep_dir / f"{stem}" / "placed_mask.png"
            if not p.exists():
                return None
            raw = np.array(Image.open(p).convert("L"))
            mask_bin = (raw > 127)
        # Resize to equalized resolution (same as image preprocessing)
        eq_sz = args.equalize_resolution if args.equalize_resolution > 0 else 512
        H, W = mask_bin.shape
        if (H, W) != (eq_sz, eq_sz):
            mask_bin = np.array(
                Image.fromarray(mask_bin.astype(np.uint8) * 255).resize((eq_sz, eq_sz), Image.NEAREST)
            ) > 127
        if label == 1:
            # Synthetic: compute core and full in latent space, upsample both
            mask_t = torch.from_numpy(mask_bin.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lh, lw = eq_sz // 8, eq_sz // 8
            # Core: maxpool to latent, NO dilation
            core_latent = torch.nn.functional.max_pool2d(mask_t, kernel_size=(eq_sz // lh, eq_sz // lw))
            core_np = (core_latent.squeeze().numpy() > 0.5).astype(np.uint8)
            # Full: core + 2px dilation in latent space
            full_np = binary_dilation(core_np, iterations=2).astype(np.uint8)
            # Upsample both to image resolution
            core_t = torch.from_numpy(core_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            full_t = torch.from_numpy(full_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            core_img = (torch.nn.functional.interpolate(core_t, size=(eq_sz, eq_sz), mode="nearest").squeeze().numpy() > 0.5).astype(np.uint8)
            full_img = (torch.nn.functional.interpolate(full_t, size=(eq_sz, eq_sz), mode="nearest").squeeze().numpy() > 0.5).astype(np.uint8)
            # Apply BiRefNet crop
            if args.crop_to_foreground:
                box = _get_birefnet_crop_box(stem, label, eq_sz)
                if box is not None:
                    x0, y0, x1, y1 = box
                    core_img = core_img[y0:y1, x0:x1]
                    full_img = full_img[y0:y1, x0:x1]
            # Resize to final image size
            iw, ih = img_size
            ch, cw = core_img.shape
            if (ch, cw) != (ih, iw):
                core_img = (np.array(Image.fromarray(core_img * 255).resize((iw, ih), Image.NEAREST)) > 127).astype(np.uint8)
                full_img = (np.array(Image.fromarray(full_img * 255).resize((iw, ih), Image.NEAREST)) > 127).astype(np.uint8)
            return (full_img, core_img)
        else:
            # Real: raw mask only
            result = mask_bin.astype(np.uint8)
            if args.crop_to_foreground:
                box = _get_birefnet_crop_box(stem, label, eq_sz)
                if box is not None:
                    x0, y0, x1, y1 = box
                    result = result[y0:y1, x0:x1]
            iw, ih = img_size
            ch, cw = result.shape
            if (ch, cw) != (ih, iw):
                result = (np.array(
                    Image.fromarray(result * 255).resize((iw, ih), Image.NEAREST)
                ) > 127).astype(np.uint8)
            return (result, None)

    def _draw_mask_edges(img: Image.Image, full_mask: np.ndarray, core_mask: Optional[np.ndarray]) -> Image.Image:
        """Draw mask edges on image.
        - If core_mask provided (synthetic): red edge = core boundary, yellow edge = dilation outer boundary.
        - If core_mask is None (real): red edge = anomaly boundary only."""
        iw, ih = img.size
        # Resize masks to display image size
        fh, fw = full_mask.shape
        if (fh, fw) != (ih, iw):
            full_mask = (np.array(Image.fromarray(full_mask * 255).resize((iw, ih), Image.NEAREST)) > 127).astype(np.uint8)
        if core_mask is not None:
            ch, cw = core_mask.shape
            if (ch, cw) != (ih, iw):
                core_mask = (np.array(Image.fromarray(core_mask * 255).resize((iw, ih), Image.NEAREST)) > 127).astype(np.uint8)
        arr = np.array(img).copy()
        if core_mask is not None:
            # Synthetic: two layers computed properly in latent space
            # Core edge (red)
            core_outer = binary_dilation(core_mask, iterations=1).astype(np.uint8)
            core_edge = core_outer - core_mask
            # Dilation zone outer edge (yellow)
            full_outer = binary_dilation(full_mask, iterations=1).astype(np.uint8)
            full_edge = full_outer - full_mask
            arr[full_edge > 0] = [255, 200, 0]
            arr[core_edge > 0] = [255, 0, 0]
        else:
            # Real: single boundary
            outer = binary_dilation(full_mask, iterations=1).astype(np.uint8)
            edge = outer - full_mask
            arr[edge > 0] = [255, 0, 0]
        return Image.fromarray(arr)

    # Save all test images: CNN input + roundtripped mask edge overlay
    n_fp, n_fn = 0, 0
    S = 256
    TITLE_H = 28
    for i, (path, label, diff) in enumerate(test_samples):
        pred = preds[i]
        stem = Path(path).stem
        kind = "real" if label == 0 else "synth"
        correct = (pred == label)

        # Apply same preprocessing the CNN saw
        img_processed = _preprocess_for_viz(path, label, stem)
        img_s = img_processed.resize((S, S), Image.LANCZOS)

        # Mask edge overlay (red=core, yellow=dilation for synth; red=boundary for real)
        mask_result = _load_anomaly_mask(stem, label, img_processed.size)
        if mask_result is not None:
            full_mask, core_mask = mask_result
            img_edges = _draw_mask_edges(img_s, full_mask, core_mask)
        else:
            img_edges = img_s

        # Composite with title
        comp_w = S * 2 + 4  # 4px gap
        comp_h = TITLE_H + S
        composite = Image.new("RGB", (comp_w, comp_h), (255, 255, 255))
        draw = ImageDraw.Draw(composite)

        verdict = "CORRECT" if correct else ("FP" if (pred == 1 and label == 0) else "FN")
        pred_label = "synthetic" if pred == 1 else "real"
        color = (0, 128, 0) if correct else (200, 0, 0)
        title = f"{stem} ({diff}) | True: {kind} | Pred: {pred_label} (p={probs[i]:.3f}) | {verdict}"
        draw.text((comp_w // 2, TITLE_H // 2), title, fill=color, font=_label_font, anchor="mm")

        composite.paste(img_s, (0, TITLE_H))
        composite.paste(img_edges, (S + 4, TITLE_H))

        if not correct:
            sub = "FP_real_called_synthetic" if (pred == 1 and label == 0) else "FN_synthetic_called_real"
            sub_dir = errors_dir / sub
            sub_dir.mkdir(parents=True, exist_ok=True)
            dst = sub_dir / f"{diff}_{stem}_{kind}_p{probs[i]:.3f}.png"
            composite.save(dst)
            if pred == 1 and label == 0:
                n_fp += 1
            else:
                n_fn += 1
        else:
            correct_dir = errors_dir / "correct"
            correct_dir.mkdir(parents=True, exist_ok=True)
            dst = correct_dir / f"{diff}_{stem}_{kind}_p{probs[i]:.3f}.png"
            composite.save(dst)

    log.info("Misclassified: %d FP (real→synth) + %d FN (synth→real) saved to %s",
             n_fp, n_fn, errors_dir)

    # ---- Save results ----
    results = {
        "task": "real_vs_synthetic",
        "checkpoint": args.checkpoint_name,
        "splits_file": args.splits_file,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "unfreeze_all": args.unfreeze_all,
        "equalize_resolution": args.equalize_resolution,
        "crop_to_foreground": args.crop_to_foreground,
        "train_real": n_train_real,
        "train_synthetic": n_train_synth,
        "test_real": n_test_real,
        "test_synthetic": n_test_synth,
        "auroc": metrics["auroc"],
        "auroc_easy": metrics.get("auroc_easy", None),
        "auroc_hard": metrics.get("auroc_hard", None),
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
        "confusion_matrix": metrics.get("confusion_matrix"),
    }

    metrics_path = results_dir / f"resnet_{test_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
