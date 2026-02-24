from __future__ import annotations
import argparse
import json
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


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _extract_ckpt_cfg(ckpt_path: Path):
    cfg = {"args": {}, "model_cfg": {}}

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("args"), dict):
            cfg["args"].update(ckpt["args"])
        if isinstance(ckpt.get("model_cfg"), dict):
            cfg["model_cfg"].update(ckpt["model_cfg"])

    run_cfg = _read_json(ckpt_path.parent / "run_config.json")
    if isinstance(run_cfg, dict):
        if isinstance(run_cfg.get("args"), dict):
            cfg["args"].update(run_cfg["args"])
        if isinstance(run_cfg.get("model_cfg"), dict):
            cfg["model_cfg"].update(run_cfg["model_cfg"])

    return cfg


def _infer_t_from_manifest(json_path: Path) -> int | None:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    frames = first.get("frame_paths")
    if not isinstance(frames, list):
        frames = first.get("frames")
    if not isinstance(frames, list):
        return None
    t = len(frames)
    return int(t) if t > 0 else None


def _resolve_temporal_length(cli_t: int, ckpt_cfg: dict, manifest_json: Path, label: str) -> int:
    if int(cli_t) > 0:
        return int(cli_t)

    t_ckpt = ckpt_cfg.get("args", {}).get("T")
    if t_ckpt is not None and int(t_ckpt) > 0:
        resolved = int(t_ckpt)
        print(f"[INFO] auto-resolved {label} T={resolved} from checkpoint config")
        return resolved

    t_manifest = _infer_t_from_manifest(manifest_json)
    if t_manifest is not None and t_manifest > 0:
        print(f"[INFO] auto-resolved {label} T={t_manifest} from manifest {manifest_json.name}")
        return int(t_manifest)

    raise ValueError(
        f"Could not auto-resolve temporal length for {label}. Pass --T explicitly."
    )


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
    ap.add_argument("--T", type=int, default=0, help="Temporal window length. Use 0 to auto-match checkpoint/manifests.")
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

    ckpt_path = _resolve_ckpt_path(args.ckpt, Path("runs") / "needle_convlstm")
    ckpt_cfg = _extract_ckpt_cfg(ckpt_path)
    resolved_t = _resolve_temporal_length(args.T, ckpt_cfg, json_path, label="needle")
    args.T = int(resolved_t)

    ds = TemporalNeedleDataset(json_path, expected_T=args.T, normalize=True)
    if len(ds) == 0:
        raise RuntimeError(f"Empty dataset for split={args.split}")

    # sample indices
    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

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

        vis_raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis_gt = overlay_mask(gray, gt)
        vis_pred = overlay_mask(gray, pred)

        triptych = np.concatenate([vis_raw, vis_gt, vis_pred], axis=1)

        vid = meta["video_id"]
        fid = meta["target_frame_id"]
        fn = out_dir / f"{k:03d}_v{vid}_f{fid}_thr{args.thr:.2f}.png"
        cv2.imwrite(str(fn), triptych)

    print("[DONE] Wrote overlay images.")


if __name__ == "__main__":
    main()
