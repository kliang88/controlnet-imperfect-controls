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


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to uint8 RGB."""
    if rgb is None:
        raise TypeError("rgb must be a numpy array")

    if not isinstance(rgb, np.ndarray):
        raise TypeError(f"rgb must be a numpy array, got {type(rgb).__name__}")

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got shape {rgb.shape}")

    if rgb.dtype == np.uint8:
        return rgb

    rgb_f = rgb.astype(np.float32)

    if not np.all(np.isfinite(rgb_f)):
        raise ValueError("rgb contains NaN or infinite values")

    # Support either [0, 1], [-1, 1], or [0, 255].
    if rgb_f.min() >= -1.01 and rgb_f.max() <= 1.01:
        if rgb_f.min() < 0:
            rgb_f = (rgb_f + 1.0) * 127.5
        else:
            rgb_f = rgb_f * 255.0

    return np.clip(rgb_f, 0, 255).astype(np.uint8)


def estimate_background_from_border(
    rgb_u8: np.ndarray,
    *,
    border_px: int = 12,
    quant_bin: int = 8,
) -> np.ndarray:
    """
    Estimate background as the dominant quantized RGB color on the border.
    This works even if the circle touches part of the border.
    """
    h, w = rgb_u8.shape[:2]
    k = max(1, min(int(border_px), h // 4, w // 4))

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:k, :] = True
    border_mask[-k:, :] = True
    border_mask[:, :k] = True
    border_mask[:, -k:] = True

    border_rgb = rgb_u8[border_mask]

    q = border_rgb // quant_bin
    bins, counts = np.unique(q.reshape(-1, 3), axis=0, return_counts=True)
    dominant_bin = bins[int(np.argmax(counts))]

    selected = np.all(q == dominant_bin, axis=1)

    if np.any(selected):
        bg_rgb = np.median(border_rgb[selected], axis=0)
    else:
        bg_rgb = np.median(border_rgb, axis=0)

    return np.clip(bg_rgb, 0, 255).astype(np.uint8)


def rgb_to_fill_mask(
    rgb: np.ndarray,
    *,
    border_px: int = 12,
    margin: float = 8.0,
    min_area_frac: float = 0.001,
    max_area_frac: float = 0.995,
    morph_kernel_size: int = 5,
    debug: bool = False,
) -> np.ndarray:
    """
    Extract foreground mask from a Fill50K-style filled-circle image.

    Important properties:
    - object may touch image border
    - no border-touching component removal
    - dominant border color is treated as background
    - empty-mask failures fall back to positive background-distance pixels

    margin:
        Foreground pixels satisfy Lab-distance ``diff > margin`` (same units as
        ``diff = ||Lab(px) - Lab(bg)||``). When ``margin <= 0``, skip this step and
        use only Otsu on normalized distance.

    Returns a uint8 mask with values 0 and 255.
    """
    rgb_u8 = rgb_to_uint8(rgb)

    h, w = rgb_u8.shape[:2]

    if h < 4 or w < 4:
        raise ValueError(f"Image is too small for mask extraction: {rgb_u8.shape}")

    if not (0.0 <= min_area_frac <= 1.0):
        raise ValueError("min_area_frac must be between 0 and 1")

    if not (0.0 <= max_area_frac <= 1.0):
        raise ValueError("max_area_frac must be between 0 and 1")

    if min_area_frac > max_area_frac:
        raise ValueError("min_area_frac must be <= max_area_frac")

    bg_rgb = estimate_background_from_border(
        rgb_u8,
        border_px=border_px,
        quant_bin=8,
    )

    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(
        bg_rgb.reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(3).astype(np.float32)

    diff = np.linalg.norm(lab - bg_lab, axis=-1)

    diff_max = float(diff.max())
    if debug:
        print("bg_rgb:", bg_rgb.tolist())
        print("diff_min:", float(diff.min()))
        print("diff_max:", diff_max)
        print("unique_rgb_count:", len(np.unique(rgb_u8.reshape(-1, 3), axis=0)))

    mask = np.zeros((h, w), dtype=np.uint8)
    if margin > 0:
        mask = (diff > margin).astype(np.uint8) * 255

    if not np.any(mask) and diff_max > 1e-6:
        diff_u8 = np.clip(diff / diff_max * 255.0, 0, 255).astype(np.uint8)
        otsu_thr, mask = cv2.threshold(
            diff_u8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        if debug:
            print("otsu_fallback; otsu_thr:", float(otsu_thr))

    if debug:
        print("raw_mask_pixels:", int(np.count_nonzero(mask)))

    # Fallback: empty mask — use any pixel measurably separated from bg estimate.
    if not np.any(mask):
        positive = diff > 1e-6
        mask = positive.astype(np.uint8) * 255

        if debug:
            print("fallback_positive_pixels:", int(np.count_nonzero(mask)))

    # Near-full mask: invert only if masked region is less separated from bg than unmasked
    # (avoids flipping legitimate circles that cover almost the entire frame).
    selected_frac = np.count_nonzero(mask) / float(h * w)
    if selected_frac > 0.995:
        fg = mask > 0
        bg = ~fg
        if np.any(fg) and np.any(bg):
            mean_fg = float(diff[fg].mean())
            mean_bg = float(diff[bg].mean())
            if mean_fg < mean_bg:
                inv = cv2.bitwise_not(mask)
                if np.any(inv):
                    mask = inv

                if debug:
                    print(
                        "inverted_large_mask (mean_fg < mean_bg); selected_frac:",
                        selected_frac,
                        "mean_fg:",
                        mean_fg,
                        "mean_bg:",
                        mean_bg,
                    )

    # Area filtering, but fail-soft.
    area_filtered = filter_components_by_area(
        mask,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
    )

    # Do not allow area filtering to erase the mask completely.
    if np.any(area_filtered):
        mask = area_filtered
    else:
        mask = largest_component(mask)

        if debug:
            print("area filter erased mask; using largest raw component")

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
    # Final largest component, but again do not let it erase anything unexpectedly.
    final = largest_component(mask)
    if np.any(final):
        mask = final

    if debug:
        print("final_mask_pixels:", int(np.count_nonzero(mask)))

    return mask


def pixel_mask_iou(mask_true: np.ndarray, mask_pred: np.ndarray) -> float:
    """Pixel IoU between two binary masks (same height × width).

    When both masks are empty, returns 1.0 (vacuous agreement on background).
    """
    if mask_true.shape[:2] != mask_pred.shape[:2]:
        raise ValueError(
            f"Mask shape mismatch: true {mask_true.shape[:2]} vs pred {mask_pred.shape[:2]}"
        )
    true = mask_true > 0
    pred = mask_pred > 0

    inter = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()

    if union == 0:
        return 1.0

    return float(inter / union)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest_label).astype(np.uint8) * 255)


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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        area_frac = area / total_area
        if min_area_frac <= area_frac <= max_area_frac:
            cleaned[labels == label] = 255

    return cleaned


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in a binary mask.

    Safe when foreground touches the image border because the image is padded
    before flood filling the exterior background.
    """
    mask = (mask > 0).astype(np.uint8) * 255

    if not np.any(mask):
        return mask

    h, w = mask.shape[:2]

    padded = cv2.copyMakeBorder(
        mask,
        1,
        1,
        1,
        1,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    flood = padded.copy()
    flood_mask = np.zeros((h + 4, w + 4), dtype=np.uint8)

    # Fill exterior background from the artificial padded corner.
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    # Pixels still equal to 0 after flood fill are enclosed holes.
    holes = cv2.bitwise_not(flood)

    # Add holes back to the original foreground.
    filled_padded = cv2.bitwise_or(padded, holes)

    return filled_padded[1:-1, 1:-1]


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


def _nan_mean_median_std(arr: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return None, None, None
    return (
        float(np.nanmean(arr)),
        float(np.nanmedian(arr)),
        float(np.nanstd(arr, ddof=0)),
    )


def summarize_metrics(
    ious: List[float],
    roundness_deltas: List[float],
    roundness_preds: List[float],
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
            "roundness_pred_mean": None,
            "roundness_pred_median": None,
            "roundness_pred_std": None,
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
    rp_arr = np.asarray(roundness_preds, dtype=np.float64) if roundness_preds else np.asarray([], dtype=np.float64)
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
    r_mean, r_med, r_std = _nan_mean_median_std(rad_arr)
    c_mean, c_med, c_std = _nan_mean_median_std(ctr_arr)
    return {
        "count": len(ious),
        "mask_iou_mean": float(arr.mean()),
        "mask_iou_median": float(np.median(arr)),
        "mask_iou_std": float(arr.std(ddof=0)),
        "roundness_abs_delta_mean": float(r_arr.mean()) if r_arr.size > 0 else None,
        "roundness_abs_delta_median": float(np.median(r_arr)) if r_arr.size > 0 else None,
        "roundness_abs_delta_std": float(r_arr.std(ddof=0)) if r_arr.size > 0 else None,
        "roundness_pred_mean": float(rp_arr.mean()) if rp_arr.size > 0 else None,
        "roundness_pred_median": float(np.median(rp_arr)) if rp_arr.size > 0 else None,
        "roundness_pred_std": float(rp_arr.std(ddof=0)) if rp_arr.size > 0 else None,
        "radius_error_mean": r_mean,
        "radius_error_median": r_med,
        "radius_error_std": r_std,
        "center_error_mean": c_mean,
        "center_error_median": c_med,
        "center_error_std": c_std,
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


def color_errors_from_mask(
    rgb_true: np.ndarray,
    rgb_pred: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Mean per-pixel RGB error (pred vs true) over foreground (mask>0) and background.

    ``mask`` selects which pixels count as circle vs background for aggregation only;
    per-pixel error is always ``||rgb_pred - rgb_true||`` in normalized L2.
    """
    true_f = rgb_true.astype(np.float32)
    pred_f = rgb_pred.astype(np.float32)
    # Normalize RGB L2 by the maximum possible distance so errors are in [0, 1].
    max_rgb_l2 = np.sqrt(3.0 * (255.0**2))
    per_pixel_l2 = np.linalg.norm(pred_f - true_f, axis=-1) / max_rgb_l2

    circle = mask > 0
    background = ~circle

    circle_err = float(np.clip(per_pixel_l2[circle].mean(), 0.0, 1.0)) if np.any(circle) else 0.0
    background_err = float(np.clip(per_pixel_l2[background].mean(), 0.0, 1.0)) if np.any(background) else 0.0
    total_err = 0.5 * circle_err + 0.5 * background_err
    return circle_err, background_err, total_err
