"""Mask placement utilities for validation data preparation.

Extracted from viz_mask_placement.py (easy) and redo_hard_placement.py (hard).
Pure numpy/PIL — no torch dependency.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
from PIL import Image


def load_binary_mask(path: Path) -> np.ndarray:
    """Load grayscale mask, binarize at 0.5. Returns float32 {0.0, 1.0}."""
    mask = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0
    return (mask > 0.5).astype(np.float32)


def place_easy_mask(
    anomaly_mask: np.ndarray,
    foreground_mask: np.ndarray,
    max_attempts: int = 500,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    seed: int | None = None,
) -> Optional[np.ndarray]:
    """Place anomaly mask so ALL anomaly pixels are inside foreground.

    Steps: bbox-crop → random flips → random rotation → random scale →
    pick a random FG pixel as centre, check 100% coverage.

    Args:
        anomaly_mask: [H, W] float32 binary mask of the anomaly.
        foreground_mask: [H, W] float32 binary mask of the valid foreground.
        max_attempts: Max random placement attempts.
        scale_range: (lo, hi) uniform scale factor.
        seed: Optional seed for local RNG (deterministic placement).

    Returns:
        Placed mask [H, W] float32 {0, 1}, or None if no valid position found.
    """
    rng = random.Random(seed)
    H, W = foreground_mask.shape

    ys, xs = np.where(anomaly_mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop = anomaly_mask[y0:y1, x0:x1]

    fg_ys, fg_xs = np.where(foreground_mask > 0.5)
    if len(fg_ys) == 0:
        return None

    # Random flips (independent, p=0.5 each)
    if rng.random() < 0.5:
        crop = crop[::-1, :].copy()
    if rng.random() < 0.5:
        crop = crop[:, ::-1].copy()

    # Uniform rotation [0, 360]
    angle = rng.uniform(0, 360)
    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
    rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)

    # Uniform scale
    scale = rng.uniform(*scale_range)
    rw, rh = rotated.size
    new_w, new_h = max(1, int(rw * scale)), max(1, int(rh * scale))
    scaled = rotated.resize((new_w, new_h), Image.NEAREST)
    scaled_arr = (np.array(scaled).astype(np.float32) / 255.0 > 0.5).astype(np.float32)
    if scaled_arr.sum() == 0:
        return None

    sh, sw = scaled_arr.shape
    for _ in range(max_attempts):
        idx = rng.randint(0, len(fg_ys) - 1)
        cy, cx = fg_ys[idx], fg_xs[idx]
        top, left = cy - sh // 2, cx - sw // 2
        if top < 0 or left < 0 or top + sh > H or left + sw > W:
            continue
        placed = np.zeros((H, W), dtype=np.float32)
        placed[top : top + sh, left : left + sw] = scaled_arr
        # ALL anomaly pixels must be within foreground
        if placed.sum() == (placed * foreground_mask).sum():
            return placed
    return None


def place_hard_mask(
    ref_mask: np.ndarray,
    canvas_fg_dict: Dict[str, np.ndarray],
    scale_range: Tuple[float, float] = (0.8, 1.2),
    used: Set[str] | None = None,
    seed: int | None = None,
) -> Optional[Tuple[str, np.ndarray, float, float, float]]:
    """Place mask so entire FG is inside the placed mask (100% coverage).

    Exhaustive search: 4 flip combos × 72 angles × progressive scales.

    Args:
        ref_mask: [H, W] float32 binary mask to place.
        canvas_fg_dict: {canvas_id: fg_mask} dict of candidate canvases.
        scale_range: (lo, hi) initial scale range, widens progressively.
        used: Canvas IDs already used (skip them).
        seed: Optional seed for deterministic ordering.

    Returns:
        (canvas_id, placed_mask, angle, scale, coverage%) or None.
    """
    rng = random.Random(seed)
    if used is None:
        used = set()

    ys, xs = np.where(ref_mask > 0.5)
    if len(ys) == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_raw = ref_mask[y0:y1, x0:x1]

    # Precompute canvas FG centres (skip used)
    fg_info: dict = {}
    for cid, fg in canvas_fg_dict.items():
        if cid in used:
            continue
        fg_ys, fg_xs = np.where(fg > 0.5)
        if len(fg_ys) == 0:
            continue
        fg_info[cid] = (fg, fg.shape, fg_ys.mean(), fg_xs.mean(), fg.sum())

    angles = list(np.linspace(0, 360, 72, endpoint=False))
    flips = [(False, False), (True, False), (False, True), (True, True)]

    max_scale_bump = 5  # up to +0.5 above original max
    for bump in range(max_scale_bump + 1):
        lo = scale_range[0]
        hi = scale_range[1] + bump * 0.1
        scales = list(np.linspace(lo, hi, max(5, int((hi - lo) / 0.05) + 1)))
        rng.shuffle(angles)
        rng.shuffle(scales)

        for fv, fh in flips:
            crop = crop_raw.copy()
            if fv:
                crop = crop[::-1, :].copy()
            if fh:
                crop = crop[:, ::-1].copy()

            for angle in angles:
                crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
                rotated = crop_pil.rotate(angle, expand=True, resample=Image.NEAREST)
                rw, rh = rotated.size

                for scale_try in scales:
                    new_w = max(1, int(rw * scale_try))
                    new_h = max(1, int(rh * scale_try))
                    scaled = rotated.resize((new_w, new_h), Image.NEAREST)
                    scaled_arr = (
                        np.array(scaled).astype(np.float32) / 255.0 > 0.5
                    ).astype(np.float32)
                    if scaled_arr.sum() == 0:
                        continue
                    sh, sw = scaled_arr.shape
                    m_ys, m_xs = np.where(scaled_arr > 0.5)
                    m_cy, m_cx = m_ys.mean(), m_xs.mean()

                    for cid, (fg, (H, W), fg_cy, fg_cx, fg_sum) in fg_info.items():
                        if sh > H or sw > W:
                            continue
                        top = int(fg_cy - m_cy)
                        left = int(fg_cx - m_cx)
                        if top < 0 or left < 0 or top + sh > H or left + sw > W:
                            continue

                        placed = np.zeros((H, W), dtype=np.float32)
                        placed[top : top + sh, left : left + sw] = scaled_arr

                        fg_inside = (fg * placed).sum()
                        if fg_inside >= fg_sum:
                            cov = placed.sum() / (H * W) * 100
                            return (cid, placed, angle, scale_try, cov)

    return None
