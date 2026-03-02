#!/usr/bin/env python3
"""
ResNet-18 binary classifier for anomaly detection: normal (0) vs anomaly (1).

Three training modes to measure synthetic data utility:
  - synthetic: 100 normals + 40 synthetic anomalies
  - real:      100 normals + 40 real anomalies
  - both:      100 normals + 80 anomalies (40 real + 40 synthetic)
  - Test always: 50 normals + 20 anomalies

Usage:
    python scripts/train_resnet.py \
        --experiment-dir <path_to_experiment_ResNet> \
        --mode real --epochs 50 --seed 42
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

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

class CashewBinaryDataset(Dataset):
    """Binary classification dataset: 0=normal, 1=anomaly."""

    def __init__(self, samples: List[Tuple[str, int]], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


def build_transforms(train: bool) -> transforms.Compose:
    """ImageNet-normalised transforms with augmentation for training."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _glob_images(directory: Path) -> List[str]:
    """Collect image files from a directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(
        str(p) for p in directory.iterdir()
        if p.suffix.lower() in EXTS
    )


def collect_train(experiment_dir: Path, mode: str) -> List[Tuple[str, int]]:
    """Collect training samples based on mode.

    Returns list of (path, label) where label: 0=normal, 1=anomaly.
    """
    samples: List[Tuple[str, int]] = []

    # Normals (always all 100)
    for p in _glob_images(experiment_dir / "normal"):
        samples.append((p, 0))

    # Anomalies depend on mode
    if mode in ("real", "both"):
        for diff in ("easy", "hard"):
            for p in _glob_images(experiment_dir / "train_real" / diff / "imgs"):
                samples.append((p, 1))

    if mode in ("synthetic", "both"):
        for diff in ("easy", "hard"):
            for p in _glob_images(experiment_dir / "anomaly" / "imgs" / diff):
                samples.append((p, 1))

    return samples


def collect_test(experiment_dir: Path) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Collect test samples (always the same regardless of mode).

    Returns (samples, difficulty_keys) where difficulty_keys tracks easy/hard.
    """
    samples: List[Tuple[str, int]] = []
    diff_keys: List[str] = []

    # Test normals
    for p in _glob_images(experiment_dir / "test" / "normal"):
        samples.append((p, 0))
        diff_keys.append("normal")

    # Test anomalies
    for diff in ("easy", "hard"):
        for p in _glob_images(experiment_dir / "test" / "anomaly" / diff / "imgs"):
            samples.append((p, 1))
            diff_keys.append(diff)

    return samples, diff_keys


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(unfreeze_all: bool = False) -> nn.Module:
    """ResNet-18 pretrained, fc replaced with single-logit head."""
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 1)

    if not unfreeze_all:
        # Freeze everything except layer4 + fc
        for name, param in model.named_parameters():
            if not (name.startswith("layer4") or name.startswith("fc")):
                param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info("Model params: %d trainable / %d total (%.1f%%)",
             trainable, total, 100.0 * trainable / total)
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

    # Per-difficulty AUROC (easy anomalies vs normals, hard anomalies vs normals)
    keys = np.array(diff_keys)
    for diff in ("easy", "hard"):
        mask = (keys == diff) | (keys == "normal")
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
        labels, preds, target_names=["normal", "anomaly"], zero_division=0,
    )
    metrics["_report"] = report
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 anomaly detector (normal vs anomaly). "
                    "Compare synthetic, real, and combined training.",
    )
    parser.add_argument("--experiment-dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True,
                        choices=["synthetic", "real", "both"],
                        help="Training data: synthetic-only, real-only, or both")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-all", action="store_true",
                        help="Fine-tune entire network (default: only layer4 + fc)")
    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        log.error("Experiment dir not found: %s", experiment_dir)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    log.info("Mode: %s", args.mode)

    # ---- Collect data ----
    train_samples = collect_train(experiment_dir, args.mode)
    test_samples, test_diff_keys = collect_test(experiment_dir)

    n_train_normal = sum(1 for _, l in train_samples if l == 0)
    n_train_anomaly = sum(1 for _, l in train_samples if l == 1)
    n_test_normal = sum(1 for _, l in test_samples if l == 0)
    n_test_anomaly = sum(1 for _, l in test_samples if l == 1)

    log.info("Train: %d normal + %d anomaly = %d",
             n_train_normal, n_train_anomaly, len(train_samples))
    log.info("Test:  %d normal + %d anomaly = %d",
             n_test_normal, n_test_anomaly, len(test_samples))

    from collections import Counter
    test_counts = Counter(test_diff_keys)
    log.info("Test breakdown: %s", dict(test_counts))

    if n_train_anomaly == 0:
        log.error("No training anomalies found for mode '%s'. "
                   "Run generate_cashew.py first if using synthetic mode.", args.mode)
        sys.exit(1)

    # ---- Build loaders ----
    train_ds = CashewBinaryDataset(train_samples, build_transforms(train=True))
    test_ds = CashewBinaryDataset(test_samples, build_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ---- Model ----
    model = build_model(unfreeze_all=args.unfreeze_all).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training ----
    best_auroc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        # Evaluate every epoch
        probs, lbls = predict(model, test_loader, device)
        try:
            auroc = float(roc_auc_score(lbls, probs))
        except ValueError:
            auroc = 0.0

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs or epoch == 1:
            log.info("Epoch %3d/%d | loss=%.4f | AUROC=%.4f | best=%.4f",
                     epoch, args.epochs, loss, auroc, best_auroc)

    # ---- Final evaluation with best model ----
    log.info("=" * 60)
    log.info("FINAL EVALUATION (best model, mode=%s)", args.mode)
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

    # ---- Save results ----
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    # Save checkpoint
    ckpt_path = results_dir / f"resnet_{args.mode}.pt"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        log.info("Saved checkpoint: %s", ckpt_path)

    # Save metrics
    results = {
        "task": "anomaly_detection",
        "mode": args.mode,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "unfreeze_all": args.unfreeze_all,
        "train_normal": n_train_normal,
        "train_anomaly": n_train_anomaly,
        "test_normal": n_test_normal,
        "test_anomaly": n_test_anomaly,
        "auroc": metrics["auroc"],
        "auroc_easy": metrics.get("auroc_easy", None),
        "auroc_hard": metrics.get("auroc_hard", None),
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "threshold": metrics["threshold"],
    }

    metrics_path = results_dir / f"resnet_{args.mode}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
