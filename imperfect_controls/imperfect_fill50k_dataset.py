"""Fill50K with a disjoint clean / corrupted split on control (hint) images.

A fraction ``corrupt_fraction`` of samples in each split are assigned to the
corrupted set; the rest are clean. A given dataset index is always corrupted
or always clean — no sample appears both ways.

Default corruption ``edge_segment_remove`` erases random rectangular patches of
edge pixels on the hint (sets them to black). Additional corruption types can be
registered in ``CORRUPTION_FUNCS``.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

# hint: float32 HWC RGB in [0, 1]
CorruptionFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _hint_stroke_mask(hint: np.ndarray) -> np.ndarray:
    """Bool mask of circle stroke / edge pixels (light line on darker canvas).

    Fill50k sources are outlines; strokes are locally brighter than a heavily
    blurred background. We also keep high-luminance pixels (near-white lines).
    """
    h, w = hint.shape[:2]
    gray = hint.mean(axis=2).astype(np.float32)
    g8 = np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)
    k = int(max(15, min(h, w) // 10) | 1)  # odd kernel ~10% of side
    bg = cv2.GaussianBlur(g8, (k, k), 0).astype(np.float32) / 255.0
    local_lift = gray - bg
    # Stroke pops above smoothed plate; threshold is robust across stroke colors.
    by_contrast = local_lift > 0.055
    # Near-white ink (explicit user request); works when line is light gray/white.
    lum_hi = np.percentile(gray, 99.0)
    floor_white = max(0.72, float(lum_hi) * 0.92)
    by_white = gray >= floor_white
    edge = by_contrast | by_white
    if not np.any(edge):
        # Fallback: brightest few percent of pixels (thin strokes still rank high).
        t = float(np.percentile(gray, 94.0))
        edge = gray >= max(t, 0.35)
    return edge


def _count_edge_pixels_removed(hint_before: np.ndarray, out: np.ndarray, edge: np.ndarray) -> int:
    """How many edge-mask pixels went to (near) black in ``out``."""
    before_on_edge = hint_before[edge]
    after_on_edge = out[edge]
    # per-pixel max channel drop to near zero
    dark = after_on_edge.max(axis=1) < 0.08
    was_bright = before_on_edge.max(axis=1) > 0.2
    return int(np.count_nonzero(dark & was_bright))


def _place_box_on_edge(
    out: np.ndarray, edge: np.ndarray, rng: np.random.Generator, h: int, w: int
) -> None:
    """Guarantee at least one box overlapping edge pixels (centered on a random edge pixel)."""
    ys, xs = np.where(edge)
    if len(xs) == 0:
        return
    i = int(rng.integers(0, len(xs)))
    cy, cx = int(ys[i]), int(xs[i])
    # Modest boxes (gentler than large w//5 patches).
    bw = int(rng.integers(max(8, w // 22), max(9, w // 6 + 1)))
    bh = int(rng.integers(max(8, h // 22), max(9, h // 6 + 1)))
    bw = min(bw, w)
    bh = min(bh, h)
    bx = int(np.clip(cx - bw // 2, 0, w - bw))
    by = int(np.clip(cy - bh // 2, 0, h - bh))
    region = np.zeros((h, w), dtype=bool)
    region[by : by + bh, bx : bx + bw] = True
    out[region & edge] = 0.0


def apply_edge_segment_remove(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boxes_min: int = 2,
    n_boxes_max: int = 6,
    min_edge_pixels_removed: int = 70,
    min_edge_fraction: float = 0.0035,
) -> np.ndarray:
    """Zero random rectangles on detected stroke pixels; always removes some real edge ink.

    If random boxes miss the stroke (thin lines), we add targeted boxes until
    ``min_edge_pixels_removed`` (or fraction of edge mask) of stroke pixels are blacked.
    Defaults are just under the original heavy settings (~10–15% gentler).
    """
    out = hint.copy()
    h, w = out.shape[:2]
    edge = _hint_stroke_mask(hint)
    if not np.any(edge):
        return out

    n_edge = int(np.count_nonzero(edge))
    target_removed = min(
        n_edge,
        max(min_edge_pixels_removed, int(min_edge_fraction * n_edge)),
    )

    n_boxes = int(rng.integers(n_boxes_min, n_boxes_max + 1))
    for _ in range(n_boxes):
        # Slightly tighter max box than w//2 (original); min box a bit larger than w//40.
        bw = int(rng.integers(max(1, w // 28), max(2, (w * 9 // 20) + 1)))
        bh = int(rng.integers(max(1, h // 28), max(2, (h * 9 // 20) + 1)))
        bw = min(bw, w)
        bh = min(bh, h)
        bx = int(rng.integers(0, max(1, w - bw + 1)))
        by = int(rng.integers(0, max(1, h - bh + 1)))
        region = np.zeros((h, w), dtype=bool)
        region[by : by + bh, bx : bx + bw] = True
        out[region & edge] = 0.0

    removed = _count_edge_pixels_removed(hint, out, edge)
    max_extra = 20
    tries = 0
    while removed < target_removed and tries < max_extra:
        _place_box_on_edge(out, edge, rng, h, w)
        removed = _count_edge_pixels_removed(hint, out, edge)
        tries += 1

    return out


def apply_hint_gaussian_noise(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma: float = 0.06,
) -> np.ndarray:
    noise = rng.normal(0.0, sigma, hint.shape).astype(np.float32)
    return np.clip(hint + noise, 0.0, 1.0)


def apply_hint_patchy_strong_noise(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches_min: int = 2,
    n_patches_max: int = 6,
    patch_frac_min: float = 0.05,
    patch_frac_max: float = 0.18,
    sigma_strong: float = 0.35,
    sigma_background: float = 0.0,
) -> np.ndarray:
    """Add very strong Gaussian noise in random rectangular patches."""
    out = np.clip(hint, 0.0, 1.0).astype(np.float32, copy=True)
    h, w = out.shape[:2]

    if sigma_background > 0:
        bg = rng.normal(0.0, sigma_background, out.shape).astype(np.float32)
        out = np.clip(out + bg, 0.0, 1.0)

    n = int(rng.integers(n_patches_min, n_patches_max + 1))
    for _ in range(n):
        frac = float(rng.uniform(patch_frac_min, patch_frac_max))
        ph = max(4, int(h * frac))
        pw = max(4, int(w * frac))
        ph = min(ph, h)
        pw = min(pw, w)
        y0 = int(rng.integers(0, max(1, h - ph + 1)))
        x0 = int(rng.integers(0, max(1, w - pw + 1)))
        patch = out[y0 : y0 + ph, x0 : x0 + pw]
        noise = rng.normal(0.0, sigma_strong, patch.shape).astype(np.float32)
        out[y0 : y0 + ph, x0 : x0 + pw] = np.clip(patch + noise, 0.0, 1.0)

    return out


def apply_hint_edge_speckle_noise(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    band_dilate_iters: int = 10,
    band_erode_iters: int = 2,
    # "Displace white pixels" style corruption.
    move_prob: float = 0.55,
    max_displacement_px: int = 18,
    # Radial displacement: shift mostly along inward/outward normal to the stroke,
    # with a per-pixel random mixture so some arcs skew toward the circle interior
    # and others toward the outer side of the noise band.
    inward_bias_prob: float = 0.35,
    outward_bias_prob: float = 0.35,
    # Small tangential jitter (fraction of max_displacement_px) to break uniformity.
    tangential_jitter_frac: float = 0.35,
    # Fraction of detected stroke pixels to erase before scattering speckles.
    # (Previously this only dimmed moved-from pixels; that can leave a bright
    # continuous stroke. Default=1 removes the "center line" entirely.)
    erase_moved_prob: float = 1.0,
    # Optional small local noise to further soften the boundary.
    sigma_edge: float = 0.06,
    white_floor: float = 0.72,
) -> np.ndarray:
    """Corrupt a band around the stroke so the edge is poorly defined.

    Displaces bright stroke pixels within an edge band, biased along the radial
    direction through the stroke centroid (inward vs outward varies per pixel),
    so the "center" of the speckle cloud shifts along the circle.
    """
    base = np.clip(hint, 0.0, 1.0).astype(np.float32, copy=True)
    out = base.copy()
    h, w = out.shape[:2]
    edge = _hint_stroke_mask(base)
    if not np.any(edge):
        return out

    # Create a ring/band around the edge via dilate - erode.
    edge_u8 = edge.astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil = cv2.dilate(edge_u8, k, iterations=int(band_dilate_iters)) > 0
    ero = cv2.erode(edge_u8, k, iterations=int(band_erode_iters)) > 0
    band = dil & (~ero)
    if not np.any(band):
        band = dil

    gray = base.mean(axis=2).astype(np.float32)
    # "White-ish" pixels on/near the stroke within the edge band.
    whiteish = (gray >= float(white_floor)) & band
    ys, xs = np.where(whiteish)
    if len(xs) == 0:
        # Fallback: use edge pixels in the band, even if not super bright.
        ys, xs = np.where(edge & band)
        if len(xs) == 0:
            return out

    keep = rng.random(len(xs)).astype(np.float32) < float(move_prob)
    ys = ys[keep]
    xs = xs[keep]
    if len(xs) == 0:
        return out

    r_max = float(max(1, max_displacement_px))
    # Centroid of stroke mask approximates circle center for radial directions.
    m = cv2.moments(edge_u8, binaryImage=True)
    if m["m00"] <= 1e-6:
        cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    else:
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])

    fx = xs.astype(np.float32) - cx
    fy = ys.astype(np.float32) - cy
    norm = np.sqrt(fx * fx + fy * fy) + 1e-6
    ux = fx / norm
    uy = fy / norm
    # Perpendicular (tangent) unit vectors for small along-edge jitter.
    tx = -uy
    ty = ux

    n = len(xs)
    u = rng.random(n).astype(np.float32)
    p_in = float(inward_bias_prob)
    p_out = float(outward_bias_prob)
    # Remainder: symmetric radial offset (can land on either side).
    sign = np.empty(n, dtype=np.float32)
    sign[u < p_in] = -1.0
    sign[(u >= p_in) & (u < p_in + p_out)] = 1.0
    mask_sym = u >= (p_in + p_out)
    n_sym = int(np.count_nonzero(mask_sym))
    if n_sym > 0:
        sign[mask_sym] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=n_sym)

    # Distance along radial direction: mixture of small vs large moves.
    dist01 = rng.beta(0.65, 1.25, size=n).astype(np.float32)
    dist = dist01 * r_max
    # Occasionally push closer to max displacement for chunkier breaks.
    big = rng.random(n).astype(np.float32) < 0.12
    n_big = int(np.count_nonzero(big))
    if n_big > 0:
        dist[big] = (0.55 + 0.45 * rng.random(n_big).astype(np.float32)) * r_max

    jmag = float(tangential_jitter_frac) * r_max * rng.random(n).astype(np.float32)
    dx = sign * dist * ux + jmag * tx
    dy = sign * dist * uy + jmag * ty
    x2 = np.clip(np.rint(xs.astype(np.float32) + dx).astype(np.int32), 0, w - 1)
    y2 = np.clip(np.rint(ys.astype(np.float32) + dy).astype(np.int32), 0, h - 1)

    # Remove most/all of the original continuous stroke so we don't keep a bright
    # "center line". (Displaced speckles are painted after this, so they can still
    # land on-mask.)
    eys, exs = np.where(edge)
    if len(exs) > 0 and erase_moved_prob > 0:
        clear = rng.random(len(exs)).astype(np.float32) < float(erase_moved_prob)
        if np.any(clear):
            out[eys[clear], exs[clear]] = 0.0

    src = base[ys, xs]  # Nx3
    # Jitter intensity slightly so it becomes white/gray speckles.
    gain = rng.uniform(0.65, 1.15, size=(len(xs), 1)).astype(np.float32)
    src = np.clip(src * gain, 0.0, 1.0)
    # Scatter onto destination using max to preserve bright speckles.
    out[y2, x2] = np.maximum(out[y2, x2], src)

    # Small Gaussian noise inside band to further soften boundary.
    if sigma_edge > 0:
        g = rng.normal(0.0, sigma_edge, out.shape).astype(np.float32)
        out[band] = np.clip(out[band] + g[band], 0.0, 1.0)

    return np.clip(out, 0.0, 1.0)

def apply_hint_gaussian_blur(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    # Larger kernels create a wide, soft falloff (matching "very blurred edge").
    k_choices: Tuple[int, ...] = (31, 41, 51, 61, 71, 81),
    # Thicken the stroke before blurring so the halo is visible.
    dilate_iter_choices: Tuple[int, ...] = (2, 3, 4, 5, 6),
    # Dilating a thin stroke then Gaussian-blurring produces a bright ridge along
    # the stroke skeleton. Subtracting a fraction of the original thin stroke
    # before the big blur removes that sharp "center line" while keeping the halo.
    core_suppress: float = 0.92,
    # Shape the halo so it's mostly gray (not a fat white band).
    halo_strength: float = 0.9,
    halo_gamma: float = 2.2,
) -> np.ndarray:
    # Make blur deterministic and always pick the strongest setting.
    # (We keep the `rng` argument for API compatibility with other corruptions.)
    k = int(max(k_choices))
    base = np.clip(hint, 0.0, 1.0).astype(np.float32, copy=True)
    # Work in single-channel grayscale to avoid any color-channel border artifacts.
    base_g = base.mean(axis=2).astype(np.float32)

    # Halo source (thickened).
    halo_src = base_g
    dilate_iters = int(max(dilate_iter_choices))
    if dilate_iters > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        halo_src = cv2.dilate(
            halo_src,
            dilate_kernel,
            iterations=dilate_iters,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    if core_suppress > 0:
        halo_src = halo_src - float(core_suppress) * base_g
        halo_src = np.clip(halo_src, 0.0, 1.0)

    # Big blur => wide gradient into black.
    # Using sigma proportional to k makes the transition very soft.
    sigma = float(k) / 3.5
    # Use constant-black padding; OpenCV's default reflect padding can "mirror"
    # bright stroke pixels at the image boundary, creating extra bright spots.
    halo = cv2.GaussianBlur(
        halo_src,
        (k, k),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT,
    )
    halo = np.clip(halo, 0.0, 1.0).astype(np.float32)
    halo = np.power(halo, float(halo_gamma)) * float(halo_strength)

    out = np.clip(halo, 0.0, 1.0)
    return np.repeat(out[:, :, None], 3, axis=2)


CORRUPTION_FUNCS: Dict[str, CorruptionFn] = {
    "edge_segment_remove": apply_edge_segment_remove,
    "noise": apply_hint_gaussian_noise,
    "noise_patchy_strong": apply_hint_patchy_strong_noise,
    "noise_edge_speckle": apply_hint_edge_speckle_noise,
    "blur": apply_hint_gaussian_blur,
}


def register_corruption(name: str, fn: CorruptionFn) -> None:
    CORRUPTION_FUNCS[name] = fn


class DisjointCorruptFill50KDataset(Dataset):
    """Fill50K where a fixed fraction of each split has corrupted ``hint`` only."""

    def __init__(
        self,
        split: str = "train",
        *,
        corrupt_fraction: float = 0.2,
        corruption_type: str = "edge_segment_remove",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        partition_seed: Optional[int] = None,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if min(train_ratio, val_ratio, test_ratio) <= 0:
            raise ValueError("train/val/test ratios must all be > 0")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        if not 0.0 <= corrupt_fraction <= 1.0:
            raise ValueError("corrupt_fraction must be in [0, 1]")
        if corruption_type not in CORRUPTION_FUNCS:
            raise ValueError(
                f"unknown corruption_type={corruption_type!r}; "
                f"expected one of: {sorted(CORRUPTION_FUNCS)}"
            )

        base_dir = Path(__file__).resolve().parent
        self.dataset_root = base_dir / "training" / "fill50k"
        prompt_path = self.dataset_root / "prompt.json"
        self.data: List[dict] = []
        with open(prompt_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        total = len(self.data)
        all_indices = np.arange(total)
        rng_split = np.random.default_rng(seed)
        shuffled_indices = rng_split.permutation(all_indices)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        split_to_indices = {
            "train": shuffled_indices[:train_end],
            "val": shuffled_indices[train_end:val_end],
            "test": shuffled_indices[val_end:],
        }
        self.indices = split_to_indices[split]

        n_local = len(self.indices)
        rng_part = np.random.default_rng(
            partition_seed if partition_seed is not None else seed + 10_007
        )
        perm = rng_part.permutation(n_local)
        n_corrupt = int(round(n_local * corrupt_fraction))
        corrupt_locals = set(perm[:n_corrupt].tolist())
        self._corrupt_locals = corrupt_locals
        self._corrupt_fn = CORRUPTION_FUNCS[corruption_type]
        split_id = {"train": 0, "val": 1, "test": 2}[split]
        self._corrupt_sample_seed = int(seed) + 31_337 + split_id

    def __len__(self) -> int:
        return len(self.indices)

    def is_corrupted_index(self, idx: int) -> bool:
        return idx in self._corrupt_locals

    def __getitem__(self, idx: int) -> dict:
        item = self.data[int(self.indices[idx])]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(str(self.dataset_root / source_filename))
        target = cv2.imread(str(self.dataset_root / target_filename))

        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        source = source.astype(np.float32) / 255.0
        target = (target.astype(np.float32) / 127.5) - 1.0

        if idx in self._corrupt_locals:
            global_idx = int(self.indices[idx])
            rng = np.random.default_rng(
                np.random.SeedSequence([self._corrupt_sample_seed, global_idx])
            )
            source = self._corrupt_fn(source, rng)

        return dict(jpg=target, txt=prompt, hint=source)
