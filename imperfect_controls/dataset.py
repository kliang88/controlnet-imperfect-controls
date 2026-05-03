import json
import cv2
import numpy as np
from pathlib import Path

from torch.utils.data import Dataset


class Fill50KDataset(Dataset):
    def __init__(
        self,
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    ):
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        if min(train_ratio, val_ratio, test_ratio) <= 0:
            raise ValueError("train/val/test ratios must all be > 0")
        ratio_sum = train_ratio + val_ratio + test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        base_dir = Path(__file__).resolve().parent
        self.dataset_root = base_dir / "training" / "fill50k"
        prompt_path = self.dataset_root / "prompt.json"
        self.data = []
        with open(prompt_path, "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        total = len(self.data)
        all_indices = np.arange(total)
        rng = np.random.default_rng(seed)
        shuffled_indices = rng.permutation(all_indices)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        split_to_indices = {
            "train": shuffled_indices[:train_end],
            "val": shuffled_indices[train_end:val_end],
            "test": shuffled_indices[val_end:],
        }
        self.indices = split_to_indices[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.data[self.indices[idx]]

        source_filename = item["source"]
        target_filename = item["target"]
        prompt = item["prompt"]

        source = cv2.imread(str(self.dataset_root / source_filename))
        target = cv2.imread(str(self.dataset_root / target_filename))

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
