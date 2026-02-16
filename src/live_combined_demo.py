from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import re

import cv2
import numpy as np
import torch

from models_convlstm_unet import ConvLSTMUNet
from models_anatomy_swin_convlstm import AnatomySwinConvLSTMConfig, AnatomySwinEncoderConvLSTM
from postprocess import keep_one_component_per_class


FRAME_RE = re.compile(r"^(?P<vid>\d+)_frame_(?P<fid>\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def load_ckpt(model: torch.nn.Module, ckpt_path: Path, device: torch.device, strict: bool = False):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=strict)
    if missing:
        print("[WARN] Missing keys (up to 10):", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys (up to 10):", unexpected[:10])


@torch.no_grad()
def warmup_anatomy_model(model: torch.nn.Module, device: torch.device, t: int, img_size: int):
    dummy = torch.zeros((1, int(t), 1, int(img_size), int(img_size)), device=device, dtype=torch.float32)
    _ = model(dummy)


def _read_json(path: Path):
    if not path.exists():
        return None


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


def _resolve_ckpt_path(user_value: str, default_root: Path, label: str) -> Path:
    if user_value and str(user_value).strip().lower() != "auto":
        return Path(user_value)

    resolved = _find_latest_best(default_root)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not auto-resolve {label} checkpoint from {default_root.resolve()}. "
            f"Pass --{label}_ckpt manually."
        )
    print(f"[INFO] auto-selected {label} checkpoint: {resolved}")
    return resolved
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_window_scalar(v):
    if isinstance(v, (list, tuple)) and len(v) == 2:
        if int(v[0]) != int(v[1]):
            raise ValueError(f"Non-square window_size not supported in this demo config: {v}")
        return int(v[0])
    return int(v)


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
        else:
            for k, v in run_cfg.items():
                if k != "model_cfg":
                    cfg["args"].setdefault(k, v)
        if isinstance(run_cfg.get("model_cfg"), dict):
            cfg["model_cfg"].update(run_cfg["model_cfg"])

    return cfg


def _validate_compatibility(args, needle_ckpt_path: Path, anat_ckpt_path: Path):
    needle_cfg = _extract_ckpt_cfg(needle_ckpt_path)
    anatomy_cfg = _extract_ckpt_cfg(anat_ckpt_path)

    n_args = needle_cfg["args"]
    a_args = anatomy_cfg["args"]
    a_model_cfg = anatomy_cfg["model_cfg"]

    issues = []

    if "T" in n_args and int(n_args["T"]) != int(args.needle_window):
        issues.append(f"needle window mismatch: ckpt T={n_args['T']} vs --needle_window={args.needle_window}")
    if "base" in n_args and int(n_args["base"]) != int(args.needle_base):
        issues.append(f"needle base mismatch: ckpt base={n_args['base']} vs --needle_base={args.needle_base}")

    if "T" in a_args and int(a_args["T"]) != int(args.anat_window):
        issues.append(f"anatomy window mismatch: ckpt T={a_args['T']} vs --anat_window={args.anat_window}")

    if "arch" in a_args and str(a_args["arch"]) != str(args.anat_arch):
        issues.append(f"anatomy arch mismatch: ckpt arch={a_args['arch']} vs --anat_arch={args.anat_arch}")

    if args.anat_arch == "swin_convlstm":
        if "swin_embed_dim" in a_args and int(a_args["swin_embed_dim"]) != int(args.anat_swin_embed_dim):
            issues.append(
                f"anatomy embed mismatch: ckpt swin_embed_dim={a_args['swin_embed_dim']} vs --anat_swin_embed_dim={args.anat_swin_embed_dim}"
            )

        if "swin_window_size" in a_model_cfg:
            ckpt_ws = _normalize_window_scalar(a_model_cfg["swin_window_size"])
            if ckpt_ws != int(args.anat_swin_window_size):
                issues.append(
                    f"anatomy window-size mismatch: ckpt swin_window_size={ckpt_ws} vs --anat_swin_window_size={args.anat_swin_window_size}"
                )
        elif "swin_window_size" in a_args:
            ckpt_ws = _normalize_window_scalar(a_args["swin_window_size"])
            if ckpt_ws != int(args.anat_swin_window_size):
                issues.append(
                    f"anatomy window-size mismatch: ckpt swin_window_size={ckpt_ws} vs --anat_swin_window_size={args.anat_swin_window_size}"
                )

        if "patch_size" in a_args and int(a_args["patch_size"]) != int(args.anat_patch_size):
            issues.append(f"anatomy patch mismatch: ckpt patch_size={a_args['patch_size']} vs --anat_patch_size={args.anat_patch_size}")

        if "img_size" in a_args and int(a_args["img_size"]) != int(args.anat_img_size):
            issues.append(f"anatomy img_size mismatch: ckpt img_size={a_args['img_size']} vs --anat_img_size={args.anat_img_size}")

        if "use_checkpoint" in a_args and bool(a_args["use_checkpoint"]) != bool(args.anat_use_checkpoint):
            issues.append(
                f"anatomy checkpointing mismatch: ckpt use_checkpoint={a_args['use_checkpoint']} vs --anat_use_checkpoint={args.anat_use_checkpoint}"
            )
    else:
        ckpt_base = None
        if "convlstm_base" in a_args:
            ckpt_base = int(a_args["convlstm_base"])
        elif "base" in a_model_cfg:
            ckpt_base = int(a_model_cfg["base"])
        if ckpt_base is not None and ckpt_base != int(args.anat_convlstm_base):
            issues.append(
                f"anatomy convlstm base mismatch: ckpt base={ckpt_base} vs --anat_convlstm_base={args.anat_convlstm_base}"
            )

    if issues:
        joined = "\n  - " + "\n  - ".join(issues)
        raise ValueError("Incompatible live demo arguments vs checkpoint configuration:" + joined)


def normalize_window(frames01: np.ndarray) -> np.ndarray:
    mean = float(frames01.mean())
    std = float(frames01.std()) + 1e-6
    return (frames01 - mean) / std


def to_tensor_window(frames_gray_u8: list[np.ndarray]) -> torch.Tensor:
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
    ap = argparse.ArgumentParser(
        description="Live-style combined inference for needle + anatomy on looped frame files.",
        epilog="Output: interactive OpenCV window with overlays; press q to exit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--video_id", type=int, default=8, help="Video id to loop from data_raw/images (default: 8).")

    ap.add_argument("--needle_window", type=int, default=8, help="Needle temporal window length (must match needle training T).")
    ap.add_argument("--anat_window", type=int, default=4, help="Anatomy temporal window length (must match anatomy training T).")

    ap.add_argument("--images_dir", type=str, default=r"data_raw\images", help="Input frames folder.")
    ap.add_argument("--processed_dir", type=str, default=r"data_processed", help="Processed folder used for optional GT overlays.")
    ap.add_argument("--device", type=str, default="cuda", help="Inference device: cuda or cpu.")
    ap.add_argument("--fps_cap", type=float, default=0.0, help="FPS cap for display loop (0 = uncapped).")

    ap.add_argument("--needle_ckpt", type=str, default="auto", help='Needle checkpoint path. Use "auto" to pick newest run best.pt.')
    ap.add_argument("--needle_base", type=int, default=32, help="Needle ConvLSTMUNet base channels.")
    ap.add_argument("--needle_thr", type=float, default=0.85, help="Needle sigmoid threshold.")
    ap.add_argument("--needle_postprocess", choices=["none", "largest"], default="largest", help="Needle binary postprocess mode.")

    ap.add_argument("--anat_ckpt", type=str, default="auto", help='Anatomy checkpoint path. Use "auto" to pick newest run best.pt.')
    ap.add_argument("--anat_arch", choices=["swin_convlstm", "convlstm"], default="swin_convlstm", help="Anatomy architecture used by the selected checkpoint.")
    ap.add_argument("--anat_convlstm_base", type=int, default=32, help="Base channels when --anat_arch=convlstm.")
    ap.add_argument("--anat_img_size", type=int, default=256, help="Anatomy model input size used at training time.")
    ap.add_argument("--anat_patch_size", type=int, default=4, help="Anatomy Swin patch size.")
    ap.add_argument("--anat_swin_embed_dim", type=int, default=48, help="Anatomy Swin base embedding dimension.")
    ap.add_argument("--anat_swin_window_size", type=int, default=8, help="Anatomy Swin window size.")
    ap.add_argument(
        "--anat_use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Swin activation checkpointing (default: enabled). Use --no-anat_use_checkpoint to disable.",
    )
    ap.add_argument("--anat_postprocess", choices=["none", "largest_per_class", "longest_per_class"], default="none", help="Optional anatomy class-wise component filtering.")
    ap.add_argument("--anat_min_area_nerve", type=int, default=40, help="Min component area for nerve when anatomy postprocess is enabled.")
    ap.add_argument("--anat_min_area_artery", type=int, default=40, help="Min component area for artery when anatomy postprocess is enabled.")
    ap.add_argument("--anat_min_area_muscle", type=int, default=120, help="Min component area for muscle when anatomy postprocess is enabled.")
    ap.add_argument("--anat_min_conf", type=float, default=0.0, help="Optional confidence gate: set anatomy class to background when max softmax < this value.")

    args = ap.parse_args()

    needle_ckpt_path = _resolve_ckpt_path(args.needle_ckpt, Path("runs") / "needle_convlstm", "needle")
    anat_ckpt_path = _resolve_ckpt_path(args.anat_ckpt, Path("runs") / "anatomy_swin_convlstm", "anat")
    if not needle_ckpt_path.exists():
        raise FileNotFoundError(f"Needle checkpoint not found: {needle_ckpt_path.resolve()}")
    if not anat_ckpt_path.exists():
        raise FileNotFoundError(f"Anatomy checkpoint not found: {anat_ckpt_path.resolve()}")

    _validate_compatibility(args, needle_ckpt_path, anat_ckpt_path)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    print("[INFO] device:", device)
    print("[INFO] needle_window=", args.needle_window, "anat_window=", args.anat_window)

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

    # Needle model (unchanged)
    needle_model = ConvLSTMUNet(in_channels=1, base=args.needle_base, out_channels=1).to(device).eval()
    load_ckpt(needle_model, needle_ckpt_path, device)

    if args.anat_arch == "swin_convlstm":
        anat_cfg = AnatomySwinConvLSTMConfig(
            in_channels=1,
            out_channels=4,
            img_size=args.anat_img_size,
            patch_size=args.anat_patch_size,
            swin_embed_dim=args.anat_swin_embed_dim,
            swin_window_size=args.anat_swin_window_size,
            use_checkpoint=args.anat_use_checkpoint,
        )
        anat_model = AnatomySwinEncoderConvLSTM(anat_cfg).to(device).eval()
        warmup_anatomy_model(anat_model, device, t=args.anat_window, img_size=args.anat_img_size)
        load_ckpt(anat_model, anat_ckpt_path, device, strict=True)
    else:
        anat_model = ConvLSTMUNet(in_channels=1, base=int(args.anat_convlstm_base), out_channels=4).to(device).eval()
        load_ckpt(anat_model, anat_ckpt_path, device, strict=True)

    needle_on = True
    anat_on = True
    gt_on = False
    paused = False

    needle_buf: list[np.ndarray] = []
    anat_buf: list[np.ndarray] = []

    idx = 0

    win = "UGRA Combined Demo (Needle + Anatomy)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("[INFO] keys: q=quit, space=pause, p=toggle needle, a=toggle anatomy, g=toggle GT, [/] needle thr")

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

            # Keep both buffers in sync from the same frame stream
            needle_buf.append(gray)
            anat_buf.append(gray)
            if len(needle_buf) > args.needle_window:
                needle_buf = needle_buf[-args.needle_window:]
            if len(anat_buf) > args.anat_window:
                anat_buf = anat_buf[-args.anat_window:]

            # Build tensors if buffers full
            x_need = to_tensor_window(needle_buf).to(device) if len(needle_buf) == args.needle_window else None
            x_anat = to_tensor_window(anat_buf).to(device) if len(anat_buf) == args.anat_window else None

            needle_pred = np.zeros(gray.shape, dtype=np.uint8)
            anat_pred = np.zeros(gray.shape, dtype=np.int64)

            infer_t0 = time.time()
            with torch.no_grad():
                if x_need is not None:
                    n_logits = needle_model(x_need)[0, 0]
                    n_prob = torch.sigmoid(n_logits).detach().cpu().numpy()
                    n_bin = (n_prob >= args.needle_thr).astype(np.uint8)
                    if args.needle_postprocess == "largest":
                        n_bin = postprocess_largest(n_bin)
                    needle_pred = n_bin

                if x_anat is not None:
                    a_logits = anat_model(x_anat)[0]  # (4,H,W)
                    a_probs = torch.softmax(a_logits, dim=0)
                    a_conf, a_cls_t = torch.max(a_probs, dim=0)
                    a_cls = a_cls_t.detach().cpu().numpy().astype(np.int64)
                    if float(args.anat_min_conf) > 0:
                        a_conf_np = a_conf.detach().cpu().numpy().astype(np.float32)
                        a_cls[a_conf_np < float(args.anat_min_conf)] = 0
                    if args.anat_postprocess != "none":
                        a_cls = keep_one_component_per_class(
                            a_cls,
                            class_ids=(1, 2, 3),
                            mode=("largest" if args.anat_postprocess == "largest_per_class" else "longest"),
                            min_area_by_class={
                                1: int(args.anat_min_area_nerve),
                                2: int(args.anat_min_area_artery),
                                3: int(args.anat_min_area_muscle),
                            },
                        )
                    anat_pred = a_cls

            infer_ms = (time.time() - infer_t0) * 1000.0

            # GT masks by filename stem
            stem = p.stem
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

            disp = bgr.copy()

            if anat_on and x_anat is not None:
                disp = overlay_anat(disp, anat_pred, alpha=0.35)
            if gt_on and gt_anat is not None:
                disp = overlay_anat(disp, gt_anat, alpha=0.18)

            if needle_on and x_need is not None:
                disp = overlay_red(disp, needle_pred, alpha=0.45)
            if gt_on and gt_needle is not None:
                disp = overlay_red(disp, gt_needle, alpha=0.20)

            # HUD
            now = time.time()
            dt = now - last_time
            last_time = now
            inst_fps = 1.0 / max(dt, 1e-6)
            fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth > 0 else inst_fps

            top_h = 58
            H, W = disp.shape[:2]
            canvas = np.zeros((top_h + H, W, 3), dtype=np.uint8)
            canvas[:top_h, :] = (0, 0, 0)
            canvas[top_h:, :] = disp

            draw_text(canvas, f"{p.name} | vid={args.video_id}", 10, 20, 0.55, 1)
            draw_text(
                canvas,
                f"FPS~{fps_smooth:.1f}  infer={infer_ms:.1f}ms  needle(T={args.needle_window},thr={args.needle_thr:.2f},px={int(needle_pred.sum())})  anat(T={args.anat_window})",
                10, 44, 0.55, 1
            )

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

        if args.fps_cap and args.fps_cap > 0:
            target_dt = 1.0 / args.fps_cap
            elapsed = time.time() - loop_t0
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    cv2.destroyAllWindows()
    print("[DONE] Exited combined demo.")


if __name__ == "__main__":
    main()
