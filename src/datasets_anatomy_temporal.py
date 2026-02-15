from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class TemporalAnatomySample:
    x: torch.Tensor          # (T, 1, H, W) float32
    y: torch.Tensor          # (H, W) int64 class ids {0..3}
    meta: Dict[str, Any]


def _load_grayscale01(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0)


def _load_anatomy_mask_classes(path: Path) -> np.ndarray:
    """
    preprocess.py saved anatomy masks as:
      0=bg, 85=nerve, 170=artery, 255=muscle
    Convert to class ids 0..3 by integer division by 85.
    """
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    cls = (arr // 85).astype(np.int64)  # 0..3
    return cls


def anatomy_temporal_collate(batch: List[TemporalAnatomySample]):
    x = torch.stack([b.x for b in batch], dim=0)  # (B, T, 1, H, W)
    y = torch.stack([b.y for b in batch], dim=0)  # (B, H, W)
    meta = [b.meta for b in batch]
    return x, y, meta


class TemporalAnatomyDataset(Dataset):
    """
    Loads temporal sequences from sequences_anatomy_*.json produced by preprocess.py.

    Each JSON entry looks like:
      {
        "video_id": 0,
        "split": "train",
        "target_frame_id": 29,
        "frames": ["...0_frame_22.jpg", ..., "...0_frame_29.jpg"],
        "target_mask": "...masks\\anatomy\\0_frame_29.png",
        "has_needle": 0/1
      }

    Output:
      x: (T, 1, H, W) float32
      y: (H, W) int64 class ids 0..3
      meta: dict with ids/paths
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_T: Optional[int] = 8,
        normalize: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path.resolve()}")

        self.items: List[Dict[str, Any]] = json.loads(self.manifest_path.read_text())

        if expected_T is not None:
            self.items = [it for it in self.items if len(it.get("frames", [])) == expected_T]

        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TemporalAnatomySample:
        it = self.items[idx]

        frame_paths = [Path(p) for p in it["frames"]]
        mask_path = Path(it["target_mask"])

        frames = [_load_grayscale01(p) for p in frame_paths]  # list of (H,W)
        x = np.stack(frames, axis=0)  # (T,H,W)

        if self.normalize:
            mean = float(x.mean())
            std = float(x.std()) + 1e-6
            x = (x - mean) / std

        x_t = torch.from_numpy(x).unsqueeze(1).float()  # (T,1,H,W)

        y = _load_anatomy_mask_classes(mask_path)        # (H,W) 0..3
        y_t = torch.from_numpy(y).long()

        meta = {
            "video_id": int(it["video_id"]),
            "split": it.get("split", ""),
            "target_frame_id": int(it["target_frame_id"]),
            "frame_paths": [str(p) for p in frame_paths],
            "mask_path": str(mask_path),
        }

        return TemporalAnatomySample(x=x_t, y=y_t, meta=meta)
