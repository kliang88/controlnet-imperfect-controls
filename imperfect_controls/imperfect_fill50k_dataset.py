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


def apply_hint_gaussian_blur(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    k_choices: Tuple[int, ...] = (3, 5, 7, 9),
) -> np.ndarray:
    k = int(rng.choice(k_choices))
    u8 = np.clip(hint * 255.0, 0.0, 255.0).astype(np.uint8)
    blurred = cv2.GaussianBlur(u8, (k, k), 0)
    return blurred.astype(np.float32) / 255.0


CORRUPTION_FUNCS: Dict[str, CorruptionFn] = {
    "edge_segment_remove": apply_edge_segment_remove,
    "noise": apply_hint_gaussian_noise,
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
