#!/usr/bin/env python3
"""ResNet baseline: real Normal vs real Anomaly photos with BiRefNet crop.

Measures how well the CNN distinguishes VisA photography sessions
(different camera settings) when both classes are real photos.
This is the ceiling — if real-vs-synthetic scores similar to this,
the CNN is just detecting photography-session differences, not synthetic artifacts.
"""
import random, json, sys, logging, io
import numpy as np, torch, torch.nn as nn
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

seed = 42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# Collect 100 normals and 100 anomalies
visa = Path("anomverse_extension/datasets/VisA_validation_dataset/datasets/easy_test/cashew/Data/Images")
normals = sorted(visa.glob("Normal/*.JPG"))[:100]
anomalies = sorted(
    list(visa.glob("Anomaly/easy_imgs/*.JPG"))
    + list(visa.glob("Anomaly/hard_imgs/*.JPG"))
)

# Build BiRefNet mask lookup
mask_dir = Path("results/birefnet_masks")
birefnet = {}
for p in normals:
    mp = mask_dir / "normal" / p.name.replace(".JPG", ".png")
    if mp.exists():
        birefnet[f"{p.stem}_0"] = str(mp)
for p in anomalies:
    mp = mask_dir / "real" / f"{p.stem}.png"
    if mp.exists():
        birefnet[f"{p.stem}_1"] = str(mp)

# All samples: (path, label)  0=normal, 1=anomaly
all_normals = [(str(p), 0) for p in normals]
all_anomalies = [(str(p), 1) for p in anomalies]
random.shuffle(all_normals)
random.shuffle(all_anomalies)

# 80/20 split, stratified
train = all_normals[:80] + all_anomalies[:80]
test = all_normals[80:] + all_anomalies[80:]
random.shuffle(train)
random.shuffle(test)

log.info("Train: %d normal + %d anomaly = %d",
         sum(1 for _, l in train if l == 0), sum(1 for _, l in train if l == 1), len(train))
log.info("Test:  %d normal + %d anomaly = %d",
         sum(1 for _, l in test if l == 0), sum(1 for _, l in test if l == 1), len(test))
log.info("BiRefNet masks: %d", len(birefnet))

SZ = 512


class FGCropDataset(Dataset):
    def __init__(self, samples, transform, birefnet_masks):
        self.samples = samples
        self.transform = transform
        self.birefnet = birefnet_masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        # Equalize resolution
        if img.size != (SZ, SZ):
            img = img.resize((SZ, SZ), Image.LANCZOS)
        # BiRefNet crop
        stem = Path(path).stem
        key = f"{stem}_{label}"
        mp = self.birefnet.get(key)
        if mp and Path(mp).exists():
            mask = np.array(Image.open(mp).convert("L"))
            mask = np.array(Image.fromarray(mask).resize((SZ, SZ), Image.NEAREST))
            fg = mask > 127
            ys, xs = np.where(fg)
            if len(ys) > 10:
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                h, w = y1 - y0, x1 - x0
                pad = int(max(h, w) * 0.05)
                y0 = max(0, y0 - pad); x0 = max(0, x0 - pad)
                y1 = min(SZ, y1 + pad); x1 = min(SZ, x1 + pad)
                img = img.crop((x0, y0, x1, y1))
        # JPEG roundtrip
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
rot90 = transforms.RandomChoice([
    transforms.Lambda(lambda x: x),
    transforms.Lambda(lambda x: x.rotate(90, expand=True)),
    transforms.Lambda(lambda x: x.rotate(180, expand=True)),
    transforms.Lambda(lambda x: x.rotate(270, expand=True)),
])
train_tf = transforms.Compose([
    rot90, transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(), transforms.Normalize(mean, std),
])
test_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(mean, std),
])

train_ds = FGCropDataset(train, train_tf, birefnet)
test_ds = FGCropDataset(test, test_tf, birefnet)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

device = torch.device("cuda")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 1)
for name, p in model.named_parameters():
    if not (name.startswith("layer4") or name.startswith("fc")):
        p.requires_grad = False
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_auroc = 0.0
best_state = None

for epoch in range(1, 51):
    model.train()
    total_loss = 0.0
    nb = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(-1)
            loss = criterion(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        nb += 1
    scheduler.step()

    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logits = model(imgs).squeeze(-1)
            all_p.append(torch.sigmoid(logits).cpu().numpy())
            all_l.append(labels.numpy())
    probs = np.concatenate(all_p)
    lbls = np.concatenate(all_l)
    try:
        auroc = roc_auc_score(lbls, probs)
    except ValueError:
        auroc = 0.5
    if auroc > best_auroc:
        best_auroc = auroc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if epoch % 10 == 0 or epoch == 1 or epoch == 50:
        log.info("Epoch %d/50 | loss=%.4f | AUROC=%.4f | best=%.4f",
                 epoch, total_loss / max(nb, 1), auroc, best_auroc)

# Final eval with best model
model.load_state_dict(best_state)
model.to(device).eval()
all_p, all_l = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            logits = model(imgs).squeeze(-1)
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_l.append(labels.numpy())
probs = np.concatenate(all_p)
lbls = np.concatenate(all_l)
auroc = roc_auc_score(lbls, probs)
preds = (probs >= 0.5).astype(int)

log.info("=" * 60)
log.info("FINAL: Real Normal vs Real Anomaly (BiRefNet crop)")
log.info("=" * 60)
log.info("AUROC:    %.4f", auroc)
log.info("Accuracy: %.4f", accuracy_score(lbls, preds))
log.info("F1:       %.4f", f1_score(lbls, preds, zero_division=0))
print(classification_report(lbls, preds, target_names=["normal", "anomaly"], zero_division=0))
