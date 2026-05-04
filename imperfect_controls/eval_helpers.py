"""Pixel-mask IoU between Fill50K-style target and generated filled-circle images."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from cldm.model import create_model, load_state_dict


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
    margin: float = 8.0,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.95,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """Robust binary mask for a filled object versus background.

    Designed for cases where the object may touch the image border.
    """
    if rgb is None:
        raise TypeError("rgb must be a numpy array")

    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy array, got {type(rgb).__name__}")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {rgb.shape}")

    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        raise ValueError(f"Image is too small for mask extraction: {rgb.shape}")

    if not (0.0 <= min_area_frac <= 1.0):
        raise ValueError("min_area_frac must be between 0 and 1")
    if not (0.0 <= max_area_frac <= 1.0):
        raise ValueError("max_area_frac must be between 0 and 1")
    if min_area_frac > max_area_frac:
        raise ValueError("min_area_frac must be <= max_area_frac")

    if rgb.dtype == np.uint8:
        rgb_for_lab = rgb
    else:
        rgb_for_lab = rgb.astype(np.float32)
        finite = np.isfinite(rgb_for_lab)
        if not np.all(finite):
            raise ValueError("rgb contains NaN or infinite values")
        if rgb_for_lab.max() > 1.5:
            rgb_for_lab = rgb_for_lab / 255.0
        rgb_for_lab = np.clip(rgb_for_lab, 0.0, 1.0)

    lab = cv2.cvtColor(rgb_for_lab, cv2.COLOR_RGB2LAB).astype(np.float32)

    k = max(1, min(int(border_px), h // 4, w // 4))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:k, :] = True
    border_mask[-k:, :] = True
    border_mask[:, :k] = True
    border_mask[:, -k:] = True

    border_pixels = lab[border_mask]
    rough_bg_lab = np.median(border_pixels, axis=0)
    border_dist_to_rough_bg = np.linalg.norm(border_pixels - rough_bg_lab, axis=1)
    trim_cutoff = np.percentile(border_dist_to_rough_bg, 70.0)
    trimmed_border_pixels = border_pixels[border_dist_to_rough_bg <= trim_cutoff]

    if len(trimmed_border_pixels) >= max(16, border_pixels.shape[0] // 20):
        bg_lab = np.median(trimmed_border_pixels, axis=0)
    else:
        bg_lab = rough_bg_lab

    diff = np.linalg.norm(lab - bg_lab, axis=-1)
    border_diff = diff[border_mask]
    thr_border = float(np.percentile(border_diff, 99.0) + margin)

    diff_max = float(diff.max())
    if diff_max <= 1e-6:
        return np.zeros((h, w), dtype=np.uint8)

    diff_u8 = np.clip(diff / diff_max * 255.0, 0, 255).astype(np.uint8)
    thr_otsu_u8, _ = cv2.threshold(
        diff_u8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    thr_otsu = float(thr_otsu_u8) / 255.0 * diff_max
    border_floor = float(np.percentile(border_diff, 95.0))
    thr = max(min(thr_border, thr_otsu), border_floor)

    mask = (diff > thr).astype(np.uint8) * 255
    mask = filter_components_by_area(
        mask,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
    )

    if morph_kernel_size > 1:
        if morph_kernel_size % 2 == 0:
            morph_kernel_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (morph_kernel_size, morph_kernel_size),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = fill_mask_holes(mask)
    mask = filter_components_by_area(
        mask,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
    )
    mask = largest_component(mask)
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
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest_label).astype(np.uint8) * 255


def filter_components_by_area(
    mask: np.ndarray,
    *,
    min_area_frac: float,
    max_area_frac: float,
) -> np.ndarray:
    """Keep connected components whose area fraction is within bounds."""
    h, w = mask.shape[:2]
    total_area = h * w
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        area_frac = area / total_area
        if min_area_frac <= area_frac <= max_area_frac:
            cleaned[labels == label] = 255

    return cleaned


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary uint8 mask."""
    mask = (mask > 0).astype(np.uint8) * 255
    h, w = mask.shape[:2]
    if not np.any(mask):
        return mask

    flood = mask.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        x, y = seed
        if flood[y, x] == 0:
            cv2.floodFill(flood, floodfill_mask, seed, 255)

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def mask_roundness(mask: np.ndarray) -> float:
    """Roundness = 4πA / P² for the largest foreground contour."""
    if mask is None:
        raise TypeError("mask must be a numpy array")
    if mask.size == 0:
        return 0.0

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    if area <= 0.0 or perimeter <= 0.0:
        return 0.0

    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def mask_radius(mask: np.ndarray) -> float:
    """Equivalent radius from largest foreground contour area: r = sqrt(A / pi)."""
    if mask is None:
        raise TypeError("mask must be a numpy array")
    if mask.size == 0:
        return 0.0

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    if area <= 0.0:
        return 0.0

    return float(np.sqrt(area / np.pi))


def mask_center(mask: np.ndarray) -> tuple[float, float]:
    """Centroid (x, y) of largest foreground contour."""
    if mask is None:
        raise TypeError("mask must be a numpy array")
    if mask.size == 0:
        return 0.0, 0.0

    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0

    largest = max(contours, key=cv2.contourArea)
    m = cv2.moments(largest)
    if m["m00"] <= 0.0:
        return 0.0, 0.0

    cx = float(m["m10"] / m["m00"])
    cy = float(m["m01"] / m["m00"])
    return cx, cy


def pixel_mask_iou_from_images(
    rgb_true: np.ndarray,
    rgb_pred: np.ndarray,
    *,
    border_px: int = 12,
    margin: float = 18.0,
    keep_largest: bool = True,
    debug_save_prefix: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Pixel-mask IoU between target filled circle and generated filled circle."""
    true_mask = rgb_to_fill_mask(rgb_true, border_px=border_px, margin=margin)
    pred_mask = rgb_to_fill_mask(rgb_pred, border_px=border_px, margin=margin)

    if keep_largest:
        true_mask = largest_component(true_mask)
        pred_mask = largest_component(pred_mask)

    iou = pixel_mask_iou(true_mask, pred_mask)
    if debug_save_prefix:
        cv2.imwrite(f"{debug_save_prefix}_true_mask.png", true_mask)
        cv2.imwrite(f"{debug_save_prefix}_pred_mask.png", pred_mask)
    return iou, true_mask, pred_mask


def target_to_uint8_rgb(jpg: np.ndarray) -> np.ndarray:
    """Dataset target tensor: float RGB [-1, 1] -> uint8 RGB."""
    return ((np.asarray(jpg) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)


def load_rgb_png(path: Path) -> np.ndarray:
    """RGB uint8 H×W×3."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def load_model(checkpoint: Path, device: torch.device, repo_root: Path):
    model = create_model(str(repo_root / "models/cldm_v15.yaml")).to(device)
    model.load_state_dict(load_state_dict(str(checkpoint), location=str(device)))
    model.sd_locked = True
    model.eval()
    return model


def iter_saved_samples(score_dir: Path) -> Iterator[Tuple[int, Path, Path]]:
    """Yield (dataset_index, target_path, generated_path) from ``test_idx_*`` folders."""
    for folder in sorted(score_dir.glob("test_idx_*")):
        if not folder.is_dir():
            continue
        m = re.match(r"test_idx_(\d+)$", folder.name)
        if not m:
            continue
        dataset_i = int(m.group(1))
        tp = folder / "target.png"
        gp = folder / "generated.png"
        if tp.is_file() and gp.is_file():
            yield dataset_i, tp, gp


def summarize_metrics(
    ious: List[float],
    roundness_deltas: List[float],
    radius_errors: List[float],
    center_errors: List[float],
    circle_color_errors: List[float],
    background_color_errors: List[float],
    total_color_errors: List[float],
) -> Dict[str, Any]:
    if not ious:
        return {
            "count": 0,
            "mask_iou_mean": None,
            "mask_iou_median": None,
            "mask_iou_std": None,
            "roundness_abs_delta_mean": None,
            "roundness_abs_delta_median": None,
            "roundness_abs_delta_std": None,
            "radius_error_mean": None,
            "radius_error_median": None,
            "radius_error_std": None,
            "center_error_mean": None,
            "center_error_median": None,
            "center_error_std": None,
            "circle_color_error_mean": None,
            "circle_color_error_median": None,
            "circle_color_error_std": None,
            "background_color_error_mean": None,
            "background_color_error_median": None,
            "background_color_error_std": None,
            "total_color_error_mean": None,
            "total_color_error_median": None,
            "total_color_error_std": None,
        }
    arr = np.asarray(ious, dtype=np.float64)
    r_arr = np.asarray(roundness_deltas, dtype=np.float64) if roundness_deltas else np.asarray([], dtype=np.float64)
    rad_arr = np.asarray(radius_errors, dtype=np.float64) if radius_errors else np.asarray([], dtype=np.float64)
    ctr_arr = np.asarray(center_errors, dtype=np.float64) if center_errors else np.asarray([], dtype=np.float64)
    circ_col_arr = (
        np.asarray(circle_color_errors, dtype=np.float64) if circle_color_errors else np.asarray([], dtype=np.float64)
    )
    bg_col_arr = (
        np.asarray(background_color_errors, dtype=np.float64)
        if background_color_errors
        else np.asarray([], dtype=np.float64)
    )
    total_col_arr = (
        np.asarray(total_color_errors, dtype=np.float64) if total_color_errors else np.asarray([], dtype=np.float64)
    )
    return {
        "count": len(ious),
        "mask_iou_mean": float(arr.mean()),
        "mask_iou_median": float(np.median(arr)),
        "mask_iou_std": float(arr.std(ddof=0)),
        "roundness_abs_delta_mean": float(r_arr.mean()) if r_arr.size > 0 else None,
        "roundness_abs_delta_median": float(np.median(r_arr)) if r_arr.size > 0 else None,
        "roundness_abs_delta_std": float(r_arr.std(ddof=0)) if r_arr.size > 0 else None,
        "radius_error_mean": float(rad_arr.mean()) if rad_arr.size > 0 else None,
        "radius_error_median": float(np.median(rad_arr)) if rad_arr.size > 0 else None,
        "radius_error_std": float(rad_arr.std(ddof=0)) if rad_arr.size > 0 else None,
        "center_error_mean": float(ctr_arr.mean()) if ctr_arr.size > 0 else None,
        "center_error_median": float(np.median(ctr_arr)) if ctr_arr.size > 0 else None,
        "center_error_std": float(ctr_arr.std(ddof=0)) if ctr_arr.size > 0 else None,
        "circle_color_error_mean": float(circ_col_arr.mean()) if circ_col_arr.size > 0 else None,
        "circle_color_error_median": float(np.median(circ_col_arr)) if circ_col_arr.size > 0 else None,
        "circle_color_error_std": float(circ_col_arr.std(ddof=0)) if circ_col_arr.size > 0 else None,
        "background_color_error_mean": float(bg_col_arr.mean()) if bg_col_arr.size > 0 else None,
        "background_color_error_median": float(np.median(bg_col_arr)) if bg_col_arr.size > 0 else None,
        "background_color_error_std": float(bg_col_arr.std(ddof=0)) if bg_col_arr.size > 0 else None,
        "total_color_error_mean": float(total_col_arr.mean()) if total_col_arr.size > 0 else None,
        "total_color_error_median": float(np.median(total_col_arr)) if total_col_arr.size > 0 else None,
        "total_color_error_std": float(total_col_arr.std(ddof=0)) if total_col_arr.size > 0 else None,
    }


def color_errors_from_true_mask(
    rgb_true: np.ndarray,
    rgb_pred: np.ndarray,
    true_mask: np.ndarray,
) -> Tuple[float, float, float]:
    true_f = rgb_true.astype(np.float32)
    pred_f = rgb_pred.astype(np.float32)
    # Normalize RGB L2 by the maximum possible distance so errors are in [0, 1].
    max_rgb_l2 = np.sqrt(3.0 * (255.0**2))
    per_pixel_l2 = np.linalg.norm(pred_f - true_f, axis=-1) / max_rgb_l2

    circle = true_mask > 0
    background = ~circle

    circle_err = float(np.clip(per_pixel_l2[circle].mean(), 0.0, 1.0)) if np.any(circle) else 0.0
    background_err = float(np.clip(per_pixel_l2[background].mean(), 0.0, 1.0)) if np.any(background) else 0.0
    total_err = 0.5 * circle_err + 0.5 * background_err
    return circle_err, background_err, total_err
