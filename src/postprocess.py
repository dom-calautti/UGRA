from __future__ import annotations
import numpy as np
import cv2

def _component_length_score(coords_xy: np.ndarray) -> float:
    """
    coords_xy: Nx2 array of (x,y) pixel coords
    Returns a 'length-like' score using PCA spread (robust and fast).
    """
    if coords_xy.shape[0] < 10:
        return 0.0
    pts = coords_xy.astype(np.float32)
    mean = pts.mean(axis=0, keepdims=True)
    pts0 = pts - mean
    cov = (pts0.T @ pts0) / max(pts0.shape[0] - 1, 1)
    eigvals, _ = np.linalg.eig(cov)
    eigvals = np.sort(np.real(eigvals))
    # sqrt of largest eigenvalue ~ spread along principal axis
    return float(np.sqrt(max(eigvals[-1], 1e-9)))

def keep_one_component(
    mask_bin: np.ndarray,
    mode: str = "largest",
    min_area: int = 20,
) -> np.ndarray:
    """
    mask_bin: HxW binary {0,1} or {0,255}
    mode:
      - "largest": keep component with largest area
      - "longest": keep component with highest PCA-length score
      - "none": return input unchanged
    min_area: throw away tiny specks
    """
    if mode == "none":
        return (mask_bin > 0).astype(np.uint8)

    m = (mask_bin > 0).astype(np.uint8)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m  # no components

    best_id = -1
    best_score = -1.0

    # components start at 1 (0 is background)
    for cid in range(1, num):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        if mode == "largest":
            score = float(area)
        elif mode == "longest":
            ys, xs = np.where(labels == cid)
            coords = np.stack([xs, ys], axis=1)
            score = _component_length_score(coords)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if score > best_score:
            best_score = score
            best_id = cid

    if best_id == -1:
        return np.zeros_like(m, dtype=np.uint8)

    out = (labels == best_id).astype(np.uint8)
    return out
