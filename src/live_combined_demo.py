from __future__ import annotations

import argparse
import time
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from models_convlstm_unet import ConvLSTMUNet


FRAME_RE = re.compile(r"^(?P<vid>\d+)_frame_(?P<fid>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)


def normalize_window(frames01: np.ndarray) -> np.ndarray:
    # frames01: (T,H,W) float32 in [0,1]
    mean = float(frames01.mean())
    std = float(frames01.std()) + 1e-6
    return (frames01 - mean) / std


def to_tensor_window(frames_gray_u8: list[np.ndarray]) -> torch.Tensor:
    # list length T, each HxW uint8
    x = np.stack([f.astype(np.float32) / 255.0 for f in frames_gray_u8], axis=0)  # (T,H,W)
    x = normalize_window(x)
    t = torch.from_numpy(x).unsqueeze(1).unsqueeze(0).float()  # (1,T,1,H,W)
    return t


def postprocess_largest(bin_mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return bin_mask.astype(np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_idx = 1 + int(np.argmax(areas))
    return (labels == best_idx).astype(np.uint8)


def overlay_red(frame_bgr: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = frame_bgr.copy()
    m = mask01.astype(bool)
    if not np.any(m):
        return out
    red = np.zeros_like(out, dtype=np.uint8)
    red[:, :] = (0, 0, 255)
    out[m] = (alpha * red[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
    return out


def colorize_anat(cls: np.ndarray) -> np.ndarray:
    h, w = cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[cls == 1] = (0, 255, 255)   # nerve
    out[cls == 2] = (255, 255, 0)   # artery
    out[cls == 3] = (255, 0, 255)   # muscle
    return out


def overlay_anat(frame_bgr: np.ndarray, cls: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = frame_bgr.copy()
    m = (cls != 0)
    if not np.any(m):
        return out
    col = colorize_anat(cls)
    out[m] = (alpha * col[m] + (1.0 - alpha) * out[m]).astype(np.uint8)
    return out


def draw_text(img, s, x, y, scale=0.55, thick=1, color=(235, 235, 235)):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_id", type=int, required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--images_dir", type=str, default=r"data_raw\images")
    ap.add_argument("--processed_dir", type=str, default=r"data_processed")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps_cap", type=float, default=0.0, help="0=uncapped (as fast as possible)")

    ap.add_argument("--needle_ckpt", type=str, required=True)
    ap.add_argument("--needle_base", type=int, default=32)
    ap.add_argument("--needle_thr", type=float, default=0.85)
    ap.add_argument("--needle_postprocess", choices=["none", "largest"], default="largest")

    ap.add_argument("--anat_ckpt", type=str, required=True)
    ap.add_argument("--anat_base", type=int, default=32)

    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[INFO] device:", device)

    images_dir = Path(args.images_dir)
    processed_dir = Path(args.processed_dir)
    needle_mask_dir = processed_dir / "masks" / "needle"
    anat_mask_dir = processed_dir / "masks" / "anatomy"

    frames = sorted(
        list(images_dir.glob(f"{args.video_id}_frame_*.jpg"))
        + list(images_dir.glob(f"{args.video_id}_frame_*.png"))
        + list(images_dir.glob(f"{args.video_id}_frame_*.jpeg")),
        key=lambda p: int(FRAME_RE.match(p.name).group("fid")) if FRAME_RE.match(p.name) else 10**9,
    )
    if not frames:
        raise RuntimeError(f"No frames for video_id={args.video_id} in {images_dir.resolve()}")

    needle_model = ConvLSTMUNet(in_channels=1, base=args.needle_base, out_channels=1).to(device).eval()
    load_ckpt(needle_model, Path(args.needle_ckpt), device)

    anat_model = ConvLSTMUNet(in_channels=1, base=args.anat_base, out_channels=4).to(device).eval()
    load_ckpt(anat_model, Path(args.anat_ckpt), device)

    needle_on = True
    anat_on = True
    gt_on = True
    paused = False

    buf: list[np.ndarray] = []
    idx = 0

    win = "UGRA Combined Demo (Needle + Anatomy)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_time = time.time()
    fps_smooth = 0.0

    while True:
        loop_t0 = time.time()

        if not paused:
            p = frames[idx]
            idx = (idx + 1) % len(frames)

            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            buf.append(gray)
            if len(buf) > args.window:
                buf = buf[-args.window:]

            # wait until buffer full
            if len(buf) < args.window:
                cv2.imshow(win, bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
                continue

            x = to_tensor_window(buf).to(device)

            # Predict
            needle_pred = np.zeros(gray.shape, dtype=np.uint8)
            anat_pred = np.zeros(gray.shape, dtype=np.int64)

            infer_t0 = time.time()
            with torch.no_grad():
                # needle
                n_logits = needle_model(x)[0, 0]
                n_prob = torch.sigmoid(n_logits).detach().cpu().numpy()
                n_bin = (n_prob >= args.needle_thr).astype(np.uint8)
                if args.needle_postprocess == "largest":
                    n_bin = postprocess_largest(n_bin)
                needle_pred = n_bin

                # anatomy
                a_logits = anat_model(x)[0]              # (4,H,W)
                a_cls = torch.argmax(a_logits, dim=0).detach().cpu().numpy().astype(np.int64)
                anat_pred = a_cls
            infer_ms = (time.time() - infer_t0) * 1000.0

            # GT masks by filename stem
            stem = p.stem  # "8_frame_26"
            gt_needle = None
            gt_anat = None
            if gt_on:
                npath = needle_mask_dir / f"{stem}.png"
                apath = anat_mask_dir / f"{stem}.png"
                if npath.exists():
                    m = cv2.imread(str(npath), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        gt_needle = (m > 0).astype(np.uint8)
                if apath.exists():
                    m = cv2.imread(str(apath), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        gt_anat = (m // 85).astype(np.int64)

            # Compose overlay
            disp = bgr.copy()

            # anatomy first (so needle sits on top)
            if anat_on:
                disp = overlay_anat(disp, anat_pred, alpha=0.35)
            if gt_on and gt_anat is not None:
                disp = overlay_anat(disp, gt_anat, alpha=0.18)

            if needle_on:
                disp = overlay_red(disp, needle_pred, alpha=0.45)
            if gt_on and gt_needle is not None:
                disp = overlay_red(disp, gt_needle, alpha=0.20)

            # HUD
            now = time.time()
            dt = now - last_time
            last_time = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps

            top_h = 52
            H, W = disp.shape[:2]
            canvas = np.zeros((top_h + H, W, 3), dtype=np.uint8)
            canvas[:top_h, :] = (0, 0, 0)
            canvas[top_h:, :] = disp

            draw_text(canvas, f"{p.name} | vid={args.video_id} | window={args.window}", 10, 20, 0.55, 1)
            draw_text(canvas, f"FPS~{fps_smooth:.1f}  infer={infer_ms:.1f}ms  needle_thr={args.needle_thr:.2f}  needle_px={int(needle_pred.sum())}",
                      10, 42, 0.55, 1)

            cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("p"):
            needle_on = not needle_on
        if key == ord("a"):
            anat_on = not anat_on
        if key == ord("g"):
            gt_on = not gt_on
        if key == ord("["):
            args.needle_thr = max(0.01, args.needle_thr - 0.02)
        if key == ord("]"):
            args.needle_thr = min(0.99, args.needle_thr + 0.02)

        # FPS cap
        if args.fps_cap and args.fps_cap > 0:
            target_dt = 1.0 / args.fps_cap
            elapsed = time.time() - loop_t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cv2.destroyAllWindows()
    print("[DONE] Exited combined demo.")


if __name__ == "__main__":
    main()
