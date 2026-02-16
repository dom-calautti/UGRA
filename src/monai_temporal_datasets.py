# src/monai_temporal_datasets.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from monai.data import CacheDataset, Dataset
from monai.transforms import Compose, MapTransform, Transform


# ----------------------------
# Low-level loaders (keep identical semantics)
# ----------------------------
def _load_gray01(path: Path) -> np.ndarray:
    # float32 in [0,1], shape (H,W)
    img = Image.open(path).convert("L")
    return (np.asarray(img, dtype=np.float32) / 255.0)


def _load_needle_mask01(path: Path) -> np.ndarray:
    # needle saved as 0/255 -> 0/1 float32
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    return (arr > 0).astype(np.float32)


def _load_anatomy_mask_classes(path: Path) -> np.ndarray:
    """
    anatomy saved as:
      0=bg, 85=nerve, 170=artery, 255=muscle
    -> class ids 0..3
    """
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.uint8)
    return (arr // 85).astype(np.int64)


# ----------------------------
# MONAI transforms for temporal sequences
# ----------------------------
class LoadTemporalFramesd(MapTransform):
    """
    Expects:
      item["frames"]: list[str] length T
    Produces:
      item["x"]: torch.float32 (T,1,H,W)
      item["meta"]["frame_paths"]
    """
    def __init__(self, keys: Sequence[str], normalize: bool = True):
        super().__init__(keys)
        self.normalize = normalize

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for k in self.keys:
            frame_paths = [Path(p) for p in d[k]]
            frames = [_load_gray01(p) for p in frame_paths]  # list(H,W)
            x = np.stack(frames, axis=0)  # (T,H,W)

            # Same normalization you already used (per-sequence z-score)
            if self.normalize:
                mean = float(x.mean())
                std = float(x.std()) + 1e-6
                x = (x - mean) / std

            # (T,1,H,W)
            d["x"] = torch.from_numpy(x).unsqueeze(1).float()

            meta = d.get("meta", {})
            meta["frame_paths"] = [str(p) for p in frame_paths]
            d["meta"] = meta
        return d


class LoadNeedleMaskd(MapTransform):
    """
    Expects:
      item["target_mask"]: str
    Produces:
      item["y"]: torch.float32 (1,H,W) with {0,1}
      item["meta"]["mask_path"]
    """
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for k in self.keys:
            mp = Path(d[k])
            y = _load_needle_mask01(mp)  # (H,W) float32
            d["y"] = torch.from_numpy(y).unsqueeze(0).float()
            meta = d.get("meta", {})
            meta["mask_path"] = str(mp)
            d["meta"] = meta
        return d


class LoadAnatomyMaskd(MapTransform):
    """
    Expects:
      item["target_mask"]: str
    Produces:
      item["y"]: torch.int64 (H,W) class ids 0..3
      item["meta"]["mask_path"]
    """
    def __init__(self, keys: Sequence[str]):
        super().__init__(keys)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        for k in self.keys:
            mp = Path(d[k])
            y = _load_anatomy_mask_classes(mp)  # (H,W) int64
            d["y"] = torch.from_numpy(y).long()
            meta = d.get("meta", {})
            meta["mask_path"] = str(mp)
            d["meta"] = meta
        return d


class EnsureMetaBasics(Transform):
    """
    Ensures meta fields exist with stable keys.
    """
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(data)
        meta = d.get("meta", {})
        meta["video_id"] = int(d.get("video_id", -1))
        meta["split"] = str(d.get("split", ""))
        meta["target_frame_id"] = int(d.get("target_frame_id", -1))
        meta["has_needle"] = int(d.get("has_needle", 0))
        d["meta"] = meta
        return d


# ----------------------------
# Manifest loader
# ----------------------------
def load_manifest(manifest_path: Path, expected_T: Optional[int] = 8, keep_all: bool = True) -> List[Dict[str, Any]]:
    items = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Manifest JSON must be a list.")

    # filter by T
    if expected_T is not None:
        items = [it for it in items if isinstance(it.get("frames", []), list) and len(it["frames"]) == expected_T]

    # optionally filter needle-positive only
    if not keep_all:
        items = [it for it in items if int(it.get("has_needle", 0)) == 1]

    # ensure required keys exist
    out = []
    for it in items:
        if "frames" not in it or "target_mask" not in it:
            continue
        out.append(it)
    return out


# ----------------------------
# Dataset builders
# ----------------------------
def make_needle_dataset(
    manifest_path: str | Path,
    *,
    expected_T: int = 8,
    normalize: bool = True,
    cache_rate: float = 0.0,
    keep_all: bool = True,
):
    mp = Path(manifest_path)
    items = load_manifest(mp, expected_T=expected_T, keep_all=keep_all)

    xform = Compose([
        EnsureMetaBasics(),
        LoadTemporalFramesd(keys=["frames"], normalize=normalize),
        LoadNeedleMaskd(keys=["target_mask"]),
    ])

    if cache_rate and cache_rate > 0:
        return CacheDataset(data=items, transform=xform, cache_rate=float(cache_rate), num_workers=0)
    return Dataset(data=items, transform=xform)


def make_anatomy_dataset(
    manifest_path: str | Path,
    *,
    expected_T: int = 8,
    normalize: bool = True,
    cache_rate: float = 0.0,
):
    mp = Path(manifest_path)
    items = load_manifest(mp, expected_T=expected_T, keep_all=True)

    xform = Compose([
        EnsureMetaBasics(),
        LoadTemporalFramesd(keys=["frames"], normalize=normalize),
        LoadAnatomyMaskd(keys=["target_mask"]),
    ])

    if cache_rate and cache_rate > 0:
        return CacheDataset(data=items, transform=xform, cache_rate=float(cache_rate), num_workers=0)
    return Dataset(data=items, transform=xform)


# ----------------------------
# Collate helpers (keeps your existing training loops simple)
# ----------------------------
def collate_needle(batch: List[Dict[str, Any]]):
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B,T,1,H,W)
    y = torch.stack([b["y"] for b in batch], dim=0)  # (B,1,H,W)
    meta = [b["meta"] for b in batch]
    return x, y, meta


def collate_anatomy(batch: List[Dict[str, Any]]):
    x = torch.stack([b["x"] for b in batch], dim=0)  # (B,T,1,H,W)
    y = torch.stack([b["y"] for b in batch], dim=0)  # (B,H,W)
    meta = [b["meta"] for b in batch]
    return x, y, meta
