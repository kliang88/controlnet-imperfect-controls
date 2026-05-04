"""Pixel-mask IoU between Fill50K-style target and generated filled-circle images."""

from __future__ import annotations

import cv2
import numpy as np


def _corner_background_color(rgb: np.ndarray, border_px: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    k = max(2, min(border_px, h // 4, w // 4))
    patches = (
        rgb[:k, :k],
        rgb[:k, -k:],
        rgb[-k:, :k],
        rgb[-k:, -k:],
    )
    corners = np.concatenate([p.reshape(-1, 3) for p in patches], axis=0).astype(np.float32)
    return np.median(corners, axis=0)


def rgb_to_fill_mask(
    rgb: np.ndarray,
    *,
    border_px: int = 12,
    margin: float = 18.0,
) -> np.ndarray:
    """Rough binary mask of the filled circle (foreground) vs background.

    Background color is estimated from image corners (Fill50K convention).
    """
    bg = _corner_background_color(rgb, border_px)
    rgb_f = rgb.astype(np.float32)
    diff = np.linalg.norm(rgb_f - bg, axis=-1)

    h, w = rgb.shape[:2]
    k = max(2, min(border_px, h // 4, w // 4))
    corner_vecs = np.concatenate(
        [
            rgb[:k, :k].reshape(-1, 3),
            rgb[:k, -k:].reshape(-1, 3),
            rgb[-k:, :k].reshape(-1, 3),
            rgb[-k:, -k:].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)
    corner_diff = np.linalg.norm(corner_vecs - bg, axis=-1)
    thr = float(np.percentile(corner_diff, 99.5) + margin)
    mask = (diff > thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def pixel_mask_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Pixel IoU between two binary masks (same height × width)."""
    if mask_true.shape[:2] != mask_pred.shape[:2]:
        raise ValueError(
            f"Mask shape mismatch: true {mask_true.shape[:2]} vs pred {mask_pred.shape[:2]}"
        )
    true = mask_true > 0
    pred = mask_pred > 0

    inter = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    if mask is None:
        raise TypeError("mask must be a numpy array")
    if mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    # skip background label 0
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = (labels == largest_label).astype(np.uint8) * 255
    return out


def pixel_mask_iou_from_images(
    rgb_true: np.ndarray,
    rgb_pred: np.ndarray,
    *,
    border_px: int = 12,
    margin: float = 18.0,
    keep_largest: bool = True,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Pixel-mask IoU between target filled circle and generated filled circle."""
    true_mask = rgb_to_fill_mask(rgb_true, border_px=border_px, margin=margin)
    pred_mask = rgb_to_fill_mask(rgb_pred, border_px=border_px, margin=margin)

    if keep_largest:
        true_mask = largest_component(true_mask)
        pred_mask = largest_component(pred_mask)

    iou = pixel_mask_iou(true_mask, pred_mask)
    return iou, true_mask, pred_mask
