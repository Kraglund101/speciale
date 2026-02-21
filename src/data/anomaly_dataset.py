"""
Shared anomaly dataset for joint training across anomaly types.

Used by Anomagic (train_anomagic.py) training script.
"""
import os
import json
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import numpy as np

from src.utils.crop_utils import clip_crop, clip_crop_multi


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _apply_unet_augmentation(
    image: Image.Image,
    mask: Image.Image,
    target_size: int = 512,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """UNet-path augmentation: color jitter only + bilinear downsample to target_size.

    No flips, rotations, or crops — captions contain spatial references that must
    match the image orientation and framing. Decorrelation is achieved by
    augmenting the CLIP path independently.

    Args:
        image: PIL RGB image at original resolution (e.g. 1024×1024)
        mask: PIL grayscale mask at original resolution
        target_size: Output size (512 for SD 1.5)

    Returns:
        (image_tensor [3, target_size, target_size] in [-1,1],
         mask_tensor [1, target_size, target_size] in {0,1})
    """
    # Brightness/contrast jitter ±15%
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)

    # Simple downsample to target_size (no crop — preserves spatial layout for captions)
    img_t = TF.to_tensor(image)                    # [3, H, W] in [0, 1]
    mask_t = TF.to_tensor(mask.convert("L"))        # [1, H, W]
    mask_t = (mask_t > 0.5).float()

    img_t = F.interpolate(img_t.unsqueeze(0), size=(target_size, target_size),
                          mode='bilinear', align_corners=False).squeeze(0)
    mask_t = F.interpolate(mask_t.unsqueeze(0), size=(target_size, target_size),
                           mode='nearest').squeeze(0)

    # Normalize to [-1, 1]
    img_t = img_t * 2.0 - 1.0
    mask_t = (mask_t > 0.5).float()

    return img_t, mask_t


def _clip_augmentation_transforms(
    image: Image.Image,
    mask: Image.Image,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CLIP-path augmentation transforms (flip, jitter, rotate) to PIL images.

    Returns full-resolution tensors BEFORE cropping, so callers can apply
    either single-crop (clip_crop) or multi-crop (clip_crop_multi).

    Args:
        image: PIL RGB image at original resolution
        mask: PIL grayscale mask at original resolution

    Returns:
        (image_tensor [3, H, W] in [0,1],
         mask_tensor [1, H, W] in {0,1})
    """
    # Horizontal flip p=0.5
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Vertical flip p=0.5
    if random.random() < 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # Brightness/contrast jitter ±15%
    b = random.uniform(*brightness_range)
    c = random.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)

    # Rotate by 0/90/180/270 (artifact-free discrete rotations)
    angle = random.choice([0, 90, 180, 270])
    if angle != 0:
        image = TF.rotate(image, angle, expand=False)
        mask = TF.rotate(mask, angle, expand=False)

    # Convert to tensors
    img_t = TF.to_tensor(image)              # [3, H, W] in [0, 1]
    mask_t = TF.to_tensor(mask.convert("L")) # [1, H, W]
    mask_t = (mask_t > 0.5).float()

    return img_t, mask_t


def _apply_clip_augmentation(
    image: Image.Image,
    mask: Image.Image,
    crop_size: int = 224,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLIP-path augmentation: flip, jitter, discrete rotate, crop_utils crop.

    Order: flip → jitter → rotate {0,90,180,270} → crop_utils → 224×224

    Independent draws from UNet augmentation.

    Args:
        image: PIL RGB image at original resolution
        mask: PIL grayscale mask at original resolution
        crop_size: Output size (224 for CLIP)

    Returns:
        (image_tensor [3, crop_size, crop_size] in [0,1],
         mask_tensor [1, crop_size, crop_size] in {0,1})
    """
    img_t, mask_t = _clip_augmentation_transforms(
        image, mask, brightness_range, contrast_range,
    )

    # crop_utils: component grouping + anomaly-aware crop → 224×224
    crop_img, crop_mask = clip_crop(img_t, mask_t, crop_size=crop_size, pad_frac=0.10)
    crop_mask = (crop_mask > 0.5).float()

    return crop_img, crop_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnomalyDataset(Dataset):
    """Dataset that loads anomaly types from split JSONs for joint training."""

    def __init__(
        self,
        splits_dir: Path,
        image_size: int = 512,
        anomaly_types: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        data_root: Optional[Path] = None,
        captions_file: Optional[Path] = None,
        return_reference: bool = False,
        reference_mode: str = "full",
        reference_crop_size: int = 224,
        augment: bool = False,
        multi_crop: bool = False,
    ):
        """
        Args:
            splits_dir: Directory containing per-anomaly-type JSON files
            image_size: Target image size for SD (default: 512 for SD 1.5)
            anomaly_types: Specific types to load (default: all)
            exclude_sources: Dataset sources to skip
            data_root: Root for resolving relative image paths
            captions_file: Path to captions.json
            return_reference: Whether to include reference image for CLIP
            reference_mode: 'full' = downscale entire image, 'crop' = anomaly bbox crop
            reference_crop_size: Target size for reference image (224 for CLIP)
            augment: Enable data augmentation (UNet + CLIP paths)
            multi_crop: Return 2 independent CLIP crops from different component groups
        """
        self.image_size = image_size
        self.splits_dir = Path(splits_dir)
        self.exclude_sources: Set[str] = set(exclude_sources or [])
        self.data_root = Path(data_root) if data_root else None
        self.return_reference = return_reference
        self.reference_mode = reference_mode
        self.reference_crop_size = reference_crop_size
        self.augment = augment
        self.multi_crop = multi_crop

        # Load captions if provided (keyed by image_path for uniqueness)
        self.captions: Dict[str, str] = {}
        if captions_file and Path(captions_file).exists():
            with open(captions_file, encoding="utf-8") as f:
                captions_data = json.load(f)
            for entry in captions_data:
                key = entry.get("image_path") or entry.get("filename", "")
                self.captions[key] = entry["caption"]
            print(f"Loaded {len(self.captions)} captions from {captions_file}")

        # Load all anomaly types
        self.samples: List[Dict] = []
        self.type_to_samples: Dict[str, List[int]] = defaultdict(list)

        json_files = list(self.splits_dir.glob("*.json"))

        if anomaly_types:
            json_files = [f for f in json_files if f.stem in anomaly_types]

        if self.exclude_sources:
            print(f"Excluding sources: {self.exclude_sources}")

        print(f"Loading {len(json_files)} anomaly types...")

        skipped_source = 0
        skipped_duplicate = 0
        for json_path in sorted(json_files):
            anomaly_type = json_path.stem

            try:
                with open(json_path) as f:
                    data = json.load(f)

                # Support both AnomVerse format ("images") and RealIAD format ("samples")
                entries = data.get("images") or data.get("samples", [])
                loaded = 0

                for img_data in entries:
                    # Skip bad data sources (AnomVerse only)
                    source = img_data.get("source_dataset", "")
                    if source and source in self.exclude_sources:
                        skipped_source += 1
                        continue

                    image_path = img_data.get("image_path_full") or img_data.get("image_path_abs") or img_data.get("image_path")
                    mask_path = img_data.get("mask_path_full") or img_data.get("mask_path_abs") or img_data.get("mask_path")

                    # Resolve relative paths against data_root if provided
                    if self.data_root and image_path and not os.path.isabs(image_path):
                        image_path = str(self.data_root / image_path)
                    if self.data_root and mask_path and not os.path.isabs(mask_path):
                        mask_path = str(self.data_root / mask_path)

                    # Skip entries where image == mask (e.g. MTD dataset bug)
                    if image_path and mask_path and image_path == mask_path:
                        skipped_duplicate += 1
                        continue

                    # Look up caption by relative image_path (fallback to filename)
                    raw_path = img_data.get("image_path_full") or img_data.get("image_path_abs") or img_data.get("image_path", "")
                    caption = self.captions.get(raw_path, "")
                    if not caption:
                        caption = self.captions.get(os.path.basename(raw_path), "")

                    sample = {
                        "anomaly_type": anomaly_type,
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "caption": caption,
                    }
                    self.samples.append(sample)
                    self.type_to_samples[anomaly_type].append(len(self.samples) - 1)
                    loaded += 1

                if loaded > 0:
                    print(f"  {anomaly_type}: {loaded} samples (skipped {len(entries) - loaded})")

            except Exception as e:
                print(f"  Error loading {json_path}: {e}")

        print(f"\nTotal: {len(self.samples)} samples across {len(self.type_to_samples)} types")
        if skipped_source > 0:
            print(f"Skipped {skipped_source} samples from excluded sources")
        if skipped_duplicate > 0:
            print(f"Skipped {skipped_duplicate} samples where image_path == mask_path (MTD bug)")

        self.anomaly_types = sorted(self.type_to_samples.keys())

        # Count captions
        n_captions = sum(1 for s in self.samples if s.get("caption"))
        print(f"Samples with captions: {n_captions}/{len(self.samples)}")

        # Non-augmented transforms (fallback when augment=False)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, _retries: int = 0) -> Dict:
        sample = self.samples[idx]

        try:
            image_pil = Image.open(sample["image_path"])
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
        except Exception:
            image_pil = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))

        try:
            mask_pil = Image.open(sample["mask_path"]).convert("L")
        except Exception:
            mask_pil = Image.new("L", (self.image_size, self.image_size), 0)

        if self.augment:
            # --- Augmented path ---
            # UNet: flip, 90° rotate, jitter, anomaly-aware 512 crop
            image, mask = _apply_unet_augmentation(
                image_pil, mask_pil, target_size=self.image_size,
            )

            result = {
                "image": image,
                "mask": mask,
                "anomaly_type": sample["anomaly_type"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "caption": sample.get("caption", ""),
            }

            # CLIP: independent flip, jitter, continuous rotate, crop_utils crop
            if self.return_reference:
                if self.multi_crop:
                    # Multi-crop: apply transforms then get 2 separate group crops
                    img_t, mask_t = _clip_augmentation_transforms(image_pil, mask_pil)
                    crops, crop_masks, valid = clip_crop_multi(
                        img_t, mask_t, crop_size=self.reference_crop_size,
                    )
                    result["reference"] = crops[0] * 2.0 - 1.0
                    result["clip_mask"] = (crop_masks[0] > 0.5).float()
                    result["reference_2"] = crops[1] * 2.0 - 1.0
                    result["clip_mask_2"] = (crop_masks[1] > 0.5).float()
                    result["group_valid"] = torch.tensor(
                        [float(valid[0]), float(valid[1])],
                    )
                else:
                    clip_img, clip_mask = _apply_clip_augmentation(
                        image_pil, mask_pil, crop_size=self.reference_crop_size,
                    )
                    # Reference in [-1, 1] to match existing pipeline expectation
                    result["reference"] = clip_img * 2.0 - 1.0
                    result["clip_mask"] = clip_mask

        else:
            # --- Non-augmented path (original behavior) ---
            image = self.transform(image_pil)
            mask = self.mask_transform(mask_pil)
            mask = (mask > 0.5).float()

            result = {
                "image": image,
                "mask": mask,
                "anomaly_type": sample["anomaly_type"],
                "image_path": sample["image_path"],
                "mask_path": sample["mask_path"],
                "caption": sample.get("caption", ""),
            }

            if self.return_reference:
                if self.multi_crop:
                    # Multi-crop: get 2 separate group crops from non-augmented image
                    img_t = TF.to_tensor(image_pil)  # [3, H, W] in [0, 1]
                    mask_t = TF.to_tensor(mask_pil.convert("L"))
                    mask_t = (mask_t > 0.5).float()
                    crops, crop_masks, valid = clip_crop_multi(
                        img_t, mask_t, crop_size=self.reference_crop_size,
                    )
                    # Convert to [-1, 1] for pipeline
                    result["reference"] = crops[0] * 2.0 - 1.0
                    result["clip_mask"] = (crop_masks[0] > 0.5).float()
                    result["reference_2"] = crops[1] * 2.0 - 1.0
                    result["clip_mask_2"] = (crop_masks[1] > 0.5).float()
                    result["group_valid"] = torch.tensor(
                        [float(valid[0]), float(valid[1])],
                    )
                else:
                    result["reference"] = _prepare_reference_image(
                        image, mask,
                        mode=self.reference_mode,
                        crop_size=self.reference_crop_size,
                    )

        # Skip samples where mask is empty (tiny anomalies vanish)
        if result["mask"].sum() == 0 and _retries < 10:
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx, _retries=_retries + 1)

        return result


def _prepare_reference_image(
    image: torch.Tensor,
    mask: torch.Tensor,
    mode: str = "full",
    crop_size: int = 224,
) -> torch.Tensor:
    """Prepare reference image for CLIP (non-augmented fallback).

    Args:
        image: [C, H, W] in [-1, 1]
        mask: [1, H, W] in {0, 1}
        mode: 'full' or 'crop'
        crop_size: Target size
    """
    if mode == "crop":
        return _extract_anomaly_crop(image, mask, crop_size)
    else:
        return F.interpolate(
            image.unsqueeze(0), size=(crop_size, crop_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)


def _extract_anomaly_crop(
    image: torch.Tensor,
    mask: torch.Tensor,
    crop_size: int = 224,
) -> torch.Tensor:
    """Extract anomaly bbox crop (non-augmented fallback)."""
    mask_2d = mask.squeeze(0)
    nonzero = torch.nonzero(mask_2d, as_tuple=False)

    if len(nonzero) == 0:
        _, h, w = image.shape
        top = max(0, (h - crop_size) // 2)
        left = max(0, (w - crop_size) // 2)
        crop = image[:, top:top + crop_size, left:left + crop_size]
        return F.interpolate(
            crop.unsqueeze(0), size=(crop_size, crop_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    y_min, x_min = nonzero.min(dim=0).values
    y_max, x_max = nonzero.max(dim=0).values
    _, h, w = image.shape

    bbox_h = y_max - y_min
    bbox_w = x_max - x_min
    pad_y = max(1, int(bbox_h * 0.1))
    pad_x = max(1, int(bbox_w * 0.1))
    y_min = max(0, y_min - pad_y)
    x_min = max(0, x_min - pad_x)
    y_max = min(h, y_max + pad_y + 1)
    x_max = min(w, x_max + pad_x + 1)

    crop_h = y_max - y_min
    crop_w = x_max - x_min
    if crop_h > crop_w:
        diff = crop_h - crop_w
        x_min = max(0, x_min - diff // 2)
        x_max = min(w, x_min + crop_h)
        if x_max - x_min < crop_h:
            x_min = max(0, x_max - crop_h)
    elif crop_w > crop_h:
        diff = crop_w - crop_h
        y_min = max(0, y_min - diff // 2)
        y_max = min(h, y_min + crop_w)
        if y_max - y_min < crop_w:
            y_min = max(0, y_max - crop_w)

    crop = image[:, y_min:y_max, x_min:x_max]
    return F.interpolate(
        crop.unsqueeze(0), size=(crop_size, crop_size),
        mode="bilinear", align_corners=False,
    ).squeeze(0)


# Backward compatibility alias
MultiAnomalyDataset = AnomalyDataset
