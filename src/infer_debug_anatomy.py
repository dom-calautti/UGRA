from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets_anatomy_temporal import TemporalAnatomyDataset
from models_convlstm_unet import ConvLSTMUNet


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def colorize_classes(cls: np.ndarray) -> np.ndarray:
    """
    cls: (H,W) int {0..3}
    colors: bg=black, nerve=yellow, artery=cyan, muscle=magenta (change if you want)
    """
    h, w = cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    # BGR
    out[cls == 1] = (0, 255, 255)   # nerve
    out[cls == 2] = (255, 255, 0)   # artery
    out[cls == 3] = (255, 0, 255)   # muscle
    return out


def overlay(gray: np.ndarray, cls: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    col = colorize_classes(cls)
    m = (cls != 0)
    base[m] = (alpha * col[m] + (1.0 - alpha) * base[m]).astype(np.uint8)
    return base


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/anatomy_convlstm/best.pt")
    ap.add_argument("--meta_dir", default="data_processed/meta")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--thr", type=float, default=0.0)  # unused (symmetry)
    ap.add_argument("--out_dir", default="runs/anatomy_convlstm/debug_preds")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_dir = Path(args.meta_dir)
    json_path = meta_dir / f"sequences_anatomy_{args.split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing: {json_path}")

    ds = TemporalAnatomyDataset(json_path, expected_T=args.T, normalize=True)
    if len(ds) == 0:
        raise RuntimeError(f"Empty dataset for split={args.split}")

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    model = ConvLSTMUNet(in_channels=1, base=args.base, out_channels=4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] split={args.split} samples={len(idxs)} out={out_dir}")

    for k, i in enumerate(idxs):
        sample = ds[i]
        x = sample.x
        y = sample.y
        meta = sample.meta

        xb = x.unsqueeze(0).to(device)     # (1,T,1,H,W)
        logits = model(xb)[0]              # (4,H,W)
        pred = torch.argmax(logits, dim=0).detach().cpu().numpy().astype(np.int64)

        gt = y.detach().cpu().numpy().astype(np.int64)

        gray = read_gray(meta["frame_paths"][-1])

        vis_gt = overlay(gray, gt)
        vis_pred = overlay(gray, pred)

        both = np.concatenate([vis_gt, vis_pred], axis=1)

        vid = meta["video_id"]
        fid = meta["target_frame_id"]
        fn = out_dir / f"{k:03d}_v{vid}_f{fid}.png"
        cv2.imwrite(str(fn), both)

    print("[DONE] Wrote anatomy overlay images.")


if __name__ == "__main__":
    main()
