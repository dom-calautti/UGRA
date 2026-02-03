# src/live_needle_demo.py
from __future__ import annotations

import argparse
import time
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from models_convlstm_unet import ConvLSTMUNet
from datasets_temporal import TemporalNeedleDataset  # ensures identical preprocessing in manifest mode

FRAME_RE = re.compile(r"^(?P<vid>\d+)_frame_(?P<fid>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


# -----------------------------
# Checkpoint loader (robust)
# -----------------------------
def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    else:
        state = ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print("[WARN] Missing keys (up to 10):", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys (up to 10):", unexpected[:10])


# -----------------------------
# Preprocess parity (raw mode)
# Matches TemporalNeedleDataset(normalize=True):
#  - load grayscale -> [0,1]
#  - stack to (T,H,W)
#  - normalize per-sequence: (x-mean)/std
#  - add channel: (T,1,H,W)
# -----------------------------
def load_gray_01(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(str(path))
    return img.astype(np.float32) / 255.0  # (H,W) in [0,1]


def normalize_sequence(x_thw: np.ndarray) -> np.ndarray:
    # x_thw: (T,H,W)
    mean = float(x_thw.mean())
    std = float(x_thw.std()) + 1e-6
    return (x_thw - mean) / std


# -----------------------------
# Postprocessing
# -----------------------------
def postprocess_mask(bin_mask: np.ndarray, mode: str, min_area: int = 20) -> np.ndarray:
    """
    bin_mask: HxW {0,1}
    """
    if mode == "none":
        return bin_mask.astype(np.uint8)

    if mode == "largest":
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bin_mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 1:
            return bin_mask.astype(np.uint8)
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background at 0
        best_idx = 1 + int(np.argmax(areas))
        out = (labels == best_idx).astype(np.uint8)
        if out.sum() < min_area:
            return np.zeros_like(out, dtype=np.uint8)
        return out

    raise ValueError(f"Unknown postprocess mode: {mode}")


# -----------------------------
# Visualization
# -----------------------------
def overlay_mask_bgr(frame_bgr: np.ndarray, mask01: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = frame_bgr.copy()
    m = (mask01.astype(np.uint8) > 0)
    if not np.any(m):
        return out
    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[:] = color_bgr
    out[m] = (alpha * overlay[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
    return out


def draw_hud(frame_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    out = frame_bgr.copy()
    y = 22
    for s in lines:
        cv2.putText(out, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        y += 20
    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # Choose one mode:
    ap.add_argument("--manifest", type=str, default="", help="Manifest JSON (recommended) e.g. data_processed/meta/sequences_needle_test.json")
    ap.add_argument("--video_id", type=int, default=None, help="Raw mode: scan frames by video id from data_raw/images")

    # Common params
    ap.add_argument("--window", type=int, default=8, help="Temporal window T (must match training, typically 8)")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path e.g. runs/needle_convlstm/best.pt")
    ap.add_argument("--base", type=int, default=32, help="Model base channels (must match training)")
    ap.add_argument("--thr", type=float, default=0.85, help="Probability threshold for binarization")
    ap.add_argument("--postprocess", choices=["none", "largest"], default="largest")
    ap.add_argument("--min_area", type=int, default=20, help="Min area after postprocess")
    ap.add_argument("--fps_cap", type=float, default=30.0, help="Max FPS cap (0 = uncapped, runs as fast as possible)")
    ap.add_argument("--device", type=str, default="cuda")

    # Raw-mode paths
    ap.add_argument("--images_dir", type=str, default=r"data_raw\images", help="Folder with frames e.g. data_raw/images")
    ap.add_argument("--gt_dir", type=str, default="", help="Optional: folder with GT masks for raw mode e.g. data_processed/masks/needle")

    # Viz
    ap.add_argument("--alpha_pred", type=float, default=0.45, help="Pred overlay alpha")
    ap.add_argument("--alpha_gt", type=float, default=0.25, help="GT overlay alpha")

    args = ap.parse_args()

    use_manifest = bool(args.manifest.strip())
    if not use_manifest and args.video_id is None:
        raise ValueError("Provide either --manifest OR --video_id.")

    images_dir = Path(args.images_dir)
    ckpt_path = Path(args.ckpt)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path.resolve()}")
    if not use_manifest and not images_dir.exists():
        raise FileNotFoundError(f"images_dir not found: {images_dir.resolve()}")

    # Device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)
    if device.type == "cuda":
        print("[INFO] gpu:", torch.cuda.get_device_name(0))
        print("[INFO] capability:", torch.cuda.get_device_capability(0))

    # Model
    model = ConvLSTMUNet(in_channels=1, base=args.base).to(device)
    model.eval()
    load_ckpt(model, ckpt_path, device)

    # Toggles (requested)
    show_gt = True   # ON by default
    show_pred = True # ON by default
    paused = False

    # FPS tracking
    last_time = time.time()
    fps_smooth = 0.0

    # Running index
    ds = None
    ds_idx = 0
    raw_frames: list[Path] = []
    raw_idx = 0
    raw_buf_01: list[np.ndarray] = []  # list of (H,W) float32 in [0,1]

    gt_dir = Path(args.gt_dir) if args.gt_dir.strip() else None

    if use_manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {manifest_path.resolve()}")
        # CRITICAL: use exact dataset pipeline (matches infer_debug/sweep/training)
        ds = TemporalNeedleDataset(manifest_path, expected_T=args.window, normalize=True, keep_all=True)
        if len(ds) == 0:
            raise RuntimeError("TemporalNeedleDataset returned 0 samples. Check --window matches manifest.")
        print(f"[INFO] Manifest mode: {manifest_path} samples={len(ds)} T={args.window}")
    else:
        # Raw mode: scan frame filenames for chosen video
        raw_frames = sorted(
            list(images_dir.glob(f"{args.video_id}_frame_*.jpg"))
            + list(images_dir.glob(f"{args.video_id}_frame_*.png"))
            + list(images_dir.glob(f"{args.video_id}_frame_*.jpeg")),
            key=lambda p: int(FRAME_RE.match(p.name).group("fid")) if FRAME_RE.match(p.name) else 10**9,
        )
        if not raw_frames:
            raise RuntimeError(f"No frames found for video_id={args.video_id} in {images_dir.resolve()}")
        print(f"[INFO] Raw mode: video_id={args.video_id} frames={len(raw_frames)} T={args.window}")
        if gt_dir is not None:
            print(f"[INFO] Raw mode GT enabled from: {gt_dir.resolve()}")

    win = "UGRA Needle Live Demo"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Debug cadence
    dbg_every = 15
    step = 0

    while True:
        loop_t0 = time.time()

        if not paused:
            # -----------------------------
            # Get x + display frame + GT
            # -----------------------------
            x = None
            frame_bgr = None
            gt_mask01 = None
            meta_str = ""
            frame_name = ""

            if use_manifest:
                if ds_idx >= len(ds):
                    ds_idx = 0
                sample = ds[ds_idx]
                ds_idx += 1

                # Dataset gives: x (T,1,H,W), y (1,H,W)
                x = sample.x.unsqueeze(0).to(device)  # (1,T,1,H,W)
                gt_mask01 = sample.y[0].detach().cpu().numpy().astype(np.uint8)  # (H,W) 0/1

                # Display last frame from meta
                last_frame_path = Path(sample.meta["frame_paths"][-1])
                frame_bgr = cv2.imread(str(last_frame_path), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue

                frame_name = last_frame_path.name
                meta_str = f"split={sample.meta.get('split')} vid={sample.meta.get('video_id')} tgt={sample.meta.get('target_frame_id')} has={sample.meta.get('has_needle')}"
            else:
                # Raw scan
                if raw_idx >= len(raw_frames):
                    raw_idx = 0
                    raw_buf_01 = []

                p = raw_frames[raw_idx]
                raw_idx += 1

                frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
                frame_name = p.name

                # update rolling buffer (float32 [0,1])
                g01 = load_gray_01(p)
                raw_buf_01.append(g01)
                if len(raw_buf_01) > args.window:
                    raw_buf_01 = raw_buf_01[-args.window:]

                if len(raw_buf_01) == args.window:
                    x_thw = np.stack(raw_buf_01, axis=0)       # (T,H,W) in [0,1]
                    x_thw = normalize_sequence(x_thw)          # match dataset normalize=True
                    x_t = torch.from_numpy(x_thw).unsqueeze(1).float()  # (T,1,H,W)
                    x = x_t.unsqueeze(0).to(device)            # (1,T,1,H,W)

                # Optional GT (raw mode) by stem name
                if gt_dir is not None:
                    stem = Path(p.name).stem  # e.g. 8_frame_26
                    mp = gt_dir / f"{stem}.png"
                    if mp.exists():
                        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            gt_mask01 = (m > 0).astype(np.uint8)

                meta_str = f"raw vid={args.video_id} idx={raw_idx}/{len(raw_frames)} buf={len(raw_buf_01)}/{args.window}"

            H, W = frame_bgr.shape[:2]

            # -----------------------------
            # Inference
            # -----------------------------
            pred_mask01 = np.zeros((H, W), dtype=np.uint8)
            prob_stats = (0.0, 0.0, 0.0)  # min/mean/max
            infer_ms = 0.0

            if x is not None:
                t0 = time.time()
                with torch.no_grad():
                    logits = model(x)              # (1,1,H,W)
                    prob = torch.sigmoid(logits)[0, 0]  # (H,W)
                infer_ms = (time.time() - t0) * 1000.0

                prob_cpu = prob.detach().float().cpu().numpy()
                prob_stats = (float(prob_cpu.min()), float(prob_cpu.mean()), float(prob_cpu.max()))

                pred = (prob_cpu >= args.thr).astype(np.uint8)
                pred = postprocess_mask(pred, args.postprocess, min_area=args.min_area)
                pred_mask01 = pred

            # -----------------------------
            # Overlay
            # -----------------------------
            disp = frame_bgr

            if show_gt and gt_mask01 is not None:
                disp = overlay_mask_bgr(disp, gt_mask01, color_bgr=(0, 255, 0), alpha=args.alpha_gt)

            if show_pred:
                disp = overlay_mask_bgr(disp, pred_mask01, color_bgr=(0, 0, 255), alpha=args.alpha_pred)

            # -----------------------------
            # FPS calc
            # -----------------------------
            now = time.time()
            dt = now - last_time
            last_time = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = (0.9 * fps_smooth + 0.1 * inst_fps) if fps_smooth > 0 else inst_fps

            pred_px = int(pred_mask01.sum())
            gt_px = int(gt_mask01.sum()) if gt_mask01 is not None else None

            # Debug prints (low overhead)
            if (step % dbg_every) == 0 and x is not None:
                x_cpu = x.detach().float().cpu().numpy()
                xmn, xmean, xmx = float(x_cpu.min()), float(x_cpu.mean()), float(x_cpu.max())
                pmin, pmean, pmax = prob_stats
                print(
                    f"[DBG] step={step} thr={args.thr:.2f} "
                    f"x(min/mean/max)={xmn:.3f}/{xmean:.3f}/{xmx:.3f} "
                    f"prob(min/mean/max)={pmin:.4f}/{pmean:.4f}/{pmax:.4f} "
                    f"pred_px={pred_px} gt_px={gt_px} infer_ms={infer_ms:.1f} frame={frame_name}"
                )

            step += 1

            # HUD
            hud = [
                f"{win} | {frame_name}",
                f"{meta_str}",
                f"GPU: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'}",
                f"FPS ~ {fps_smooth:.1f} | infer {infer_ms:.1f} ms | cap {args.fps_cap:.0f}",
                f"thr {args.thr:.2f} | post {args.postprocess} | pred_px {pred_px} | gt_px {gt_px}",
                f"prob(min/mean/max) {prob_stats[0]:.3f}/{prob_stats[1]:.3f}/{prob_stats[2]:.3f}",
                "Keys: q quit | space pause | p pred | g gt | r restart",
            ]
            disp = draw_hud(disp, hud)

            cv2.imshow(win, disp)

        # -----------------------------
        # Key handling
        # -----------------------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("p"):
            show_pred = not show_pred
        if key == ord("g"):
            show_gt = not show_gt
        if key == ord("r"):
            paused = False
            step = 0
            fps_smooth = 0.0
            last_time = time.time()
            if use_manifest:
                ds_idx = 0
            else:
                raw_idx = 0
                raw_buf_01 = []

        # -----------------------------
        # FPS cap (cap only; never force slower than model)
        # If args.fps_cap==0 => uncapped
        # -----------------------------
        if args.fps_cap and args.fps_cap > 0:
            target_dt = 1.0 / float(args.fps_cap)
            elapsed = time.time() - loop_t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cv2.destroyAllWindows()
    print("[DONE] Exited live demo.")


if __name__ == "__main__":
    main()
