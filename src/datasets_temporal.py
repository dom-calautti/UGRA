from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class TemporalSample:
    x: torch.Tensor          # (T, 1, H, W) float32
    y: torch.Tensor          # (1, H, W) float32 (0/1)
    meta: Dict[str, Any]


def _load_grayscale(path: Path) -> np.ndarray:
    # returns float32 array in [0, 1], shape (H, W)
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _load_mask_binary(path: Path) -> np.ndarray:
    # your saved needle masks are 0/255 -> convert to 0/1 float32
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    arr = (arr > 0).astype(np.float32)
    return arr

def temporal_collate(batch: List[TemporalSample]):
    x = torch.stack([b.x for b in batch], dim=0)  # (B, T, 1, H, W)
    y = torch.stack([b.y for b in batch], dim=0)  # (B, 1, H, W)
    meta = [b.meta for b in batch]
    return x, y, meta


class TemporalNeedleDataset(Dataset):
    """
    Loads temporal sequences from sequences_needle_*.json produced by preprocess.py.

    Each JSON entry looks like:
      {
        "video_id": 0,
        "split": "train",
        "target_frame_id": 29,
        "frames": ["...0_frame_22.jpg", ..., "...0_frame_29.jpg"],
        "target_mask": "...masks\\needle\\0_frame_29.png",
        "has_needle": 1
      }

    Output:
      x: (T, 1, H, W) float32
      y: (1, H, W) float32
      meta: dict with ids/paths
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        expected_T: Optional[int] = 8,
        normalize: bool = True,
        keep_all: bool = True,
    ):
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path.resolve()}")

        self.items: List[Dict[str, Any]] = json.loads(self.manifest_path.read_text())

        if expected_T is not None:
            # filter only correct-length sequences
            self.items = [it for it in self.items if len(it.get("frames", [])) == expected_T]

        if not keep_all:
            # usually manifests already contain needle-positive targets only,
            # but if you ever include negatives, you can filter them here.
            self.items = [it for it in self.items if int(it.get("has_needle", 0)) == 1]

        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TemporalSample:
        it = self.items[idx]

        frame_paths = [Path(p) for p in it["frames"]]
        mask_path = Path(it["target_mask"])

        # Load frames
        frames = [_load_grayscale(p) for p in frame_paths]  # list of (H, W)
        x = np.stack(frames, axis=0)  # (T, H, W)

        # Simple per-sequence normalization (safe and usually helpful in ultrasound)
        if self.normalize:
            mean = float(x.mean())
            std = float(x.std()) + 1e-6
            x = (x - mean) / std

        # Add channel dim -> (T, 1, H, W)
        x_t = torch.from_numpy(x).unsqueeze(1).float()

        # Load mask -> (H, W) 0/1
        y = _load_mask_binary(mask_path)
        y_t = torch.from_numpy(y).unsqueeze(0).float()  # (1, H, W)

        meta = {
            "video_id": int(it["video_id"]),
            "split": it.get("split", ""),
            "target_frame_id": int(it["target_frame_id"]),
            "frame_paths": [str(p) for p in frame_paths],
            "mask_path": str(mask_path),
            "has_needle": int(it.get("has_needle", 0)),
        }

        return TemporalSample(x=x_t, y=y_t, meta=meta)
    
   
