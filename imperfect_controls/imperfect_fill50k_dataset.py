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


def apply_edge_segment_remove(
    hint: np.ndarray,
    rng: np.random.Generator,
    *,
    n_boxes_min: int = 2,
    n_boxes_max: int = 7,
    edge_threshold_frac: float = 0.28,
    edge_threshold_floor: float = 0.06,
) -> np.ndarray:
    """Zero random rectangular regions that intersect thresholded edge pixels."""
    out = hint.copy()
    h, w = out.shape[:2]
    gray = out.mean(axis=2)
    thr = float(
        np.clip(
            max(edge_threshold_floor, gray.max() * edge_threshold_frac),
            edge_threshold_floor,
            0.99,
        )
    )
    edge = gray >= thr
    if not np.any(edge):
        return out

    n_boxes = int(rng.integers(n_boxes_min, n_boxes_max + 1))
    for _ in range(n_boxes):
        bw = int(rng.integers(max(1, w // 24), max(2, w // 2 + 1)))
        bh = int(rng.integers(max(1, h // 24), max(2, h // 2 + 1)))
        bw = min(bw, w)
        bh = min(bh, h)
        bx = int(rng.integers(0, max(1, w - bw + 1)))
        by = int(rng.integers(0, max(1, h - bh + 1)))
        region = np.zeros((h, w), dtype=bool)
        region[by : by + bh, bx : bx + bw] = True
        out[region & edge] = 0.0
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
