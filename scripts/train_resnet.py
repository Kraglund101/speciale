#!/usr/bin/env python3
"""
ResNet-18 binary classifier: can it tell real anomalies from synthetic ones?

Labels: 0 = real anomaly, 1 = synthetic anomaly.
AUROC ~0.5 means synthetics are indistinguishable from real (= success).
AUROC ~1.0 means the model easily tells them apart (= synthetics look fake).

Data pool:
  - Real anomalies: train_real/{easy,hard}/imgs/ + test/anomaly/{easy,hard}/imgs/
  - Synthetic anomalies: anomaly/imgs/{easy,hard}/

Method: stratified k-fold cross-validation (small dataset, ~80-100 images total).
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
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BinaryImageDataset(Dataset):
    """Binary classification dataset from (path, label) list."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: transforms.Compose,
    ):
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
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ])


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _glob_images(directory: Path) -> List[str]:
    """Collect jpg/png images from a directory (non-recursive)."""
    exts = {".jpg", ".jpeg", ".png"}
    if not directory.exists():
        return []
    return sorted(
        str(p) for p in directory.iterdir()
        if p.suffix.lower() in exts
    )


def collect_samples(
    experiment_dir: Path,
) -> Tuple[List[Tuple[str, int, str]], int, int]:
    """Collect all real and synthetic anomaly images.

    Returns:
        samples: list of (path, label, difficulty) — label 0=real, 1=synthetic
        n_real: count of real images
        n_syn: count of synthetic images
    """
    samples: List[Tuple[str, int, str]] = []

    # Real anomalies from train_real/
    for diff in ("easy", "hard"):
        imgs = _glob_images(experiment_dir / "train_real" / diff / "imgs")
        for p in imgs:
            samples.append((p, 0, diff))

    # Real anomalies from test/anomaly/
    for diff in ("easy", "hard"):
        imgs = _glob_images(experiment_dir / "test" / "anomaly" / diff / "imgs")
        for p in imgs:
            samples.append((p, 0, diff))

    n_real = sum(1 for _, l, _ in samples if l == 0)

    # Synthetic anomalies from anomaly/imgs/
    for diff in ("easy", "hard"):
        imgs = _glob_images(experiment_dir / "anomaly" / "imgs" / diff)
        for p in imgs:
            samples.append((p, 1, diff))

    n_syn = sum(1 for _, l, _ in samples if l == 1)

    log.info("Real anomalies: %d", n_real)
    log.info("Synthetic anomalies: %d", n_syn)
    log.info("Total: %d", len(samples))

    # Per-difficulty breakdown
    for diff in ("easy", "hard"):
        r = sum(1 for _, l, d in samples if l == 0 and d == diff)
        s = sum(1 for _, l, d in samples if l == 1 and d == diff)
        log.info("  %s: %d real, %d synthetic", diff, r, s)

    return samples, n_real, n_syn


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


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def run_fold(
    fold: int,
    train_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    test_difficulties: List[str],
    args: argparse.Namespace,
    device: torch.device,
) -> Dict:
    """Train and evaluate one fold. Returns metrics dict."""

    train_ds = BinaryImageDataset(train_samples, build_transforms(train=True))
    test_ds = BinaryImageDataset(test_samples, build_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = build_model(unfreeze_all=args.unfreeze_all).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # Train with early stopping on test AUROC
    best_auroc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        probs, labels = predict(model, test_loader, device)
        try:
            auroc = float(roc_auc_score(labels, probs))
        except ValueError:
            auroc = 0.0

        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == args.epochs:
            log.info("  Fold %d | Epoch %3d/%d | loss=%.4f | AUROC=%.4f | best=%.4f",
                     fold, epoch, args.epochs, loss, auroc, best_auroc)

    # Evaluate best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    probs, labels = predict(model, test_loader, device)

    metrics: Dict = {}
    try:
        metrics["auroc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auroc"] = float("nan")

    thresh = optimal_threshold(probs, labels)
    preds = (probs >= thresh).astype(int)
    metrics["threshold"] = float(thresh)
    metrics["accuracy"] = float(accuracy_score(labels, preds))
    metrics["f1"] = float(f1_score(labels, preds, zero_division=0))

    # Per-difficulty AUROC
    difficulties = np.array(test_difficulties)
    test_labels = np.array([l for _, l in test_samples])
    for diff in ("easy", "hard"):
        mask = difficulties == diff
        if mask.sum() == 0:
            metrics[f"auroc_{diff}"] = float("nan")
            continue
        try:
            metrics[f"auroc_{diff}"] = float(roc_auc_score(labels[mask], probs[mask]))
        except ValueError:
            metrics[f"auroc_{diff}"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ResNet-18 real-vs-synthetic classifier. AUROC ~0.5 = indistinguishable.",
    )
    parser.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Path to experiment_ResNet directory",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--unfreeze-all", action="store_true",
        help="Fine-tune entire network (default: freeze layers 1-3)",
    )
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
    log.info("Folds: %d | Epochs: %d | LR: %.1e | BS: %d | Unfreeze all: %s",
             args.folds, args.epochs, args.lr, args.batch_size, args.unfreeze_all)

    # ---- Collect all data ----
    all_samples, n_real, n_syn = collect_samples(experiment_dir)

    if n_syn == 0:
        log.error("No synthetic anomaly images found. Run generate_cashew.py first.")
        sys.exit(1)
    if n_real == 0:
        log.error("No real anomaly images found.")
        sys.exit(1)

    paths = [s[0] for s in all_samples]
    labels = np.array([s[1] for s in all_samples])
    difficulties = [s[2] for s in all_samples]

    # ---- K-fold cross-validation ----
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(paths, labels), 1):
        log.info("=" * 50)
        log.info("FOLD %d / %d", fold_idx, args.folds)
        log.info("=" * 50)

        train_samples = [(paths[i], int(labels[i])) for i in train_idx]
        test_samples = [(paths[i], int(labels[i])) for i in test_idx]
        test_diffs = [difficulties[i] for i in test_idx]

        n_train_real = sum(1 for _, l in train_samples if l == 0)
        n_train_syn = sum(1 for _, l in train_samples if l == 1)
        n_test_real = sum(1 for _, l in test_samples if l == 0)
        n_test_syn = sum(1 for _, l in test_samples if l == 1)
        log.info("Train: %d real + %d synthetic | Test: %d real + %d synthetic",
                 n_train_real, n_train_syn, n_test_real, n_test_syn)

        metrics = run_fold(fold_idx, train_samples, test_samples, test_diffs, args, device)
        fold_metrics.append(metrics)

        log.info("Fold %d results: AUROC=%.4f  F1=%.4f  Acc=%.4f",
                 fold_idx, metrics["auroc"], metrics["f1"], metrics["accuracy"])

    # ---- Aggregate results ----
    log.info("=" * 60)
    log.info("AGGREGATE RESULTS (%d-fold CV)", args.folds)
    log.info("=" * 60)

    for key in ("auroc", "auroc_easy", "auroc_hard", "f1", "accuracy"):
        vals = [m[key] for m in fold_metrics if not np.isnan(m.get(key, float("nan")))]
        if vals:
            mean, std = np.mean(vals), np.std(vals)
            log.info("%-12s  %.4f +/- %.4f  (per fold: %s)",
                     key, mean, std, ", ".join(f"{v:.4f}" for v in vals))
        else:
            log.info("%-12s  N/A", key)

    mean_auroc = np.mean([m["auroc"] for m in fold_metrics
                          if not np.isnan(m["auroc"])])
    log.info("")
    if mean_auroc < 0.6:
        log.info("VERDICT: Synthetics are INDISTINGUISHABLE from real (AUROC %.3f)", mean_auroc)
    elif mean_auroc < 0.75:
        log.info("VERDICT: Synthetics are SOMEWHAT distinguishable (AUROC %.3f)", mean_auroc)
    else:
        log.info("VERDICT: Synthetics are EASILY distinguishable (AUROC %.3f)", mean_auroc)

    # ---- Save results ----
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)

    results = {
        "task": "real_vs_synthetic_classification",
        "interpretation": "AUROC ~0.5 = indistinguishable, ~1.0 = easily separated",
        "folds": args.folds,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "unfreeze_all": args.unfreeze_all,
        "seed": args.seed,
        "n_real": n_real,
        "n_synthetic": n_syn,
        "mean_auroc": float(mean_auroc),
        "per_fold": fold_metrics,
    }

    out_path = results_dir / "resnet_real_vs_synthetic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved results: %s", out_path)


if __name__ == "__main__":
    main()
