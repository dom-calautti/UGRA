from __future__ import annotations
import argparse
import os
from pathlib import Path
import random
from postprocess import keep_one_component

import cv2
import numpy as np
import torch

from datasets_temporal import TemporalNeedleDataset, temporal_collate
from models_convlstm_unet import ConvLSTMUNet


def _find_latest_best(default_root: Path) -> Path | None:
    if not default_root.exists():
        return None

    run_dirs = sorted(
        [d for d in default_root.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    for d in run_dirs:
        p = d / "best.pt"
        if p.exists():
            return p

    legacy = default_root / "best.pt"
    if legacy.exists():
        return legacy
    return None


def _resolve_ckpt_path(user_value: str, default_root: Path) -> Path:
    if user_value and str(user_value).strip().lower() != "auto":
        return Path(user_value)

    resolved = _find_latest_best(default_root)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not auto-resolve needle checkpoint from {default_root.resolve()}. Pass --ckpt manually."
        )
    print(f"[INFO] auto-selected needle checkpoint: {resolved}")
    return resolved


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img

def overlay_mask(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    gray: HxW uint8
    mask: HxW {0,1} uint8
    returns: HxWx3 uint8
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # overlay red where mask=1 (no fancy colors, just clear)
    rgb[mask > 0] = (0, 0, 255)
    return rgb

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Save needle GT-vs-pred overlay images for offline debugging.",
        epilog="Output: PNG files in --out_dir/<split>, each image is [GT | prediction].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", default="auto", help='Needle checkpoint path. Use "auto" to pick newest run best.pt.')
    ap.add_argument("--meta_dir", default="data_processed/meta", help="Folder containing needle sequence manifests.")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Split to sample from.")
    ap.add_argument("--T", type=int, default=8, help="Temporal window length (must match checkpoint training T).")
    ap.add_argument("--base", type=int, default=32, help="Needle ConvLSTMUNet base channels.")
    ap.add_argument("--n", type=int, default=12, help="Number of random samples to export.")
    ap.add_argument("--thr", type=float, default=0.5, help="Needle sigmoid threshold.")
    ap.add_argument("--out_dir", default="runs/needle_convlstm/debug_preds", help="Output root directory for overlays.")
    ap.add_argument("--postprocess", choices=["none", "largest", "longest"], default="none", help="Optional binary mask postprocess.")
    ap.add_argument("--min_area", type=int, default=20, help="Min component area used when postprocess is enabled.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_dir = Path(args.meta_dir)
    json_path = meta_dir / f"sequences_needle_{args.split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing: {json_path}")

    ds = TemporalNeedleDataset(json_path, expected_T=args.T, normalize=True)
    if len(ds) == 0:
        raise RuntimeError(f"Empty dataset for split={args.split}")

    # sample indices
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    ckpt_path = _resolve_ckpt_path(args.ckpt, Path("runs") / "needle_convlstm")
    model = ConvLSTMUNet(in_channels=1, base=args.base).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] split={args.split} samples={len(idxs)} out={out_dir}")

    for k, i in enumerate(idxs):
        sample = ds[i]  # TemporalSample
        x = sample.x            # (T,1,H,W)
        y = sample.y            # (1,H,W)
        meta = sample.meta      # dict

        xb = x.unsqueeze(0).to(device)  # (1,T,1,H,W)
        logits = model(xb)              # (1,1,H,W)
        probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        pred = (probs > args.thr).astype(np.uint8)
        pred = keep_one_component(pred, mode=args.postprocess, min_area=args.min_area)

        gt = y[0].cpu().numpy().astype(np.uint8)

        target_frame_path = meta["frame_paths"][-1]
        gray = read_gray(target_frame_path)

        vis_pred = overlay_mask(gray, pred)
        vis_gt = overlay_mask(gray, gt)

        both = np.concatenate([vis_gt, vis_pred], axis=1)

        vid = meta["video_id"]
        fid = meta["target_frame_id"]
        fn = out_dir / f"{k:03d}_v{vid}_f{fid}_thr{args.thr:.2f}.png"
        cv2.imwrite(str(fn), both)


    print("[DONE] Wrote overlay images.")


if __name__ == "__main__":
    main()
