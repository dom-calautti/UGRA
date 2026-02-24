from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from datasets_anatomy_temporal import TemporalAnatomyDataset
from models_anatomy_swin_convlstm import AnatomySwinConvLSTMConfig, AnatomySwinEncoderConvLSTM
from models_convlstm_unet import ConvLSTMUNet
from postprocess import keep_one_component_per_class


@torch.no_grad()
def warmup_anatomy_model(model: torch.nn.Module, device: str, t: int, img_size: int):
    dummy = torch.zeros((1, int(t), 1, int(img_size), int(img_size)), device=device, dtype=torch.float32)
    _ = model(dummy)


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
            f"Could not auto-resolve anatomy checkpoint from {default_root.resolve()}. Pass --ckpt manually."
        )
    print(f"[INFO] auto-selected anatomy checkpoint: {resolved}")
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


def colorize_anat(cls: np.ndarray) -> np.ndarray:
    h, w = cls.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[cls == 1] = (0, 255, 255)   # nerve
    out[cls == 2] = (255, 255, 0)   # artery
    out[cls == 3] = (255, 0, 255)   # muscle
    return out


def overlay_anat(gray: np.ndarray, cls: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    m = (cls != 0)
    if not np.any(m):
        return bgr
    col = colorize_anat(cls)
    bgr[m] = (alpha * col[m] + (1.0 - alpha) * bgr[m]).astype(np.uint8)
    return bgr


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(
        description="Save anatomy GT-vs-pred overlay images for offline debugging.",
        epilog="Output: PNG files in --out_dir/<split>, each image is [GT | prediction].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--ckpt", default="auto", help='Anatomy checkpoint path. Use "auto" to pick newest run best.pt.')
    ap.add_argument("--meta_dir", default="data_processed/meta", help="Folder containing anatomy sequence manifests.")
    ap.add_argument("--split", choices=["val", "test"], default="val", help="Split to sample from.")
    ap.add_argument("--T", type=int, default=0, help="Temporal window length. Use 0 to auto-match checkpoint/manifests.")
    ap.add_argument("--n", type=int, default=12, help="Number of random samples to export.")
    ap.add_argument("--out_dir", default="runs/anatomy_swin_convlstm/debug_preds", help="Output root directory for overlays.")
    ap.add_argument("--anat_arch", choices=["swin_convlstm", "convlstm"], default="swin_convlstm", help="Anatomy architecture used by the selected checkpoint.")
    ap.add_argument("--convlstm_base", type=int, default=32, help="Base channels when --anat_arch=convlstm.")

    ap.add_argument("--img_size", type=int, default=256, help="Anatomy model input size used at training.")
    ap.add_argument("--patch_size", type=int, default=4, help="Anatomy Swin patch size.")
    ap.add_argument("--swin_embed_dim", type=int, default=48, help="Anatomy Swin base embedding dimension.")
    ap.add_argument("--swin_window_size", type=int, default=8, help="Anatomy Swin window size.")
    ap.add_argument("--use_checkpoint", action="store_true", help="Use Swin activation checkpointing (match training config).")
    ap.add_argument("--postprocess", choices=["none", "largest_per_class", "longest_per_class"], default="none", help="Optional anatomy class-wise component filtering.")
    ap.add_argument("--min_area_nerve", type=int, default=40, help="Min component area for nerve when postprocess is enabled.")
    ap.add_argument("--min_area_artery", type=int, default=40, help="Min component area for artery when postprocess is enabled.")
    ap.add_argument("--min_area_muscle", type=int, default=120, help="Min component area for muscle when postprocess is enabled.")
    ap.add_argument("--min_conf", type=float, default=0.0, help="Optional confidence gate: set class to bg where max softmax < min_conf.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta_dir = Path(args.meta_dir)
    json_path = meta_dir / f"sequences_anatomy_{args.split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing: {json_path}")

    ckpt_path = _resolve_ckpt_path(args.ckpt, Path("runs") / "anatomy_swin_convlstm")
    ckpt_cfg = _extract_ckpt_cfg(ckpt_path)
    resolved_t = _resolve_temporal_length(args.T, ckpt_cfg, json_path, label="anatomy")
    args.T = int(resolved_t)

    ds = TemporalAnatomyDataset(json_path, expected_T=args.T, normalize=True)
    if len(ds) == 0:
        raise RuntimeError(f"Empty dataset for split={args.split}")

    idxs = list(range(len(ds)))
    random.shuffle(idxs)
    idxs = idxs[: min(args.n, len(idxs))]

    if args.anat_arch == "swin_convlstm":
        cfg = AnatomySwinConvLSTMConfig(
            in_channels=1,
            out_channels=4,
            img_size=args.img_size,
            patch_size=args.patch_size,
            swin_embed_dim=args.swin_embed_dim,
            swin_window_size=args.swin_window_size,
            use_checkpoint=args.use_checkpoint,
        )
        model = AnatomySwinEncoderConvLSTM(cfg).to(device).eval()
        warmup_anatomy_model(model, device, t=args.T, img_size=args.img_size)
    else:
        model = ConvLSTMUNet(in_channels=1, base=int(args.convlstm_base), out_channels=4).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=True)
    if missing:
        print("[WARN] Missing keys (up to 10):", missing[:10])
    if unexpected:
        print("[WARN] Unexpected keys (up to 10):", unexpected[:10])

    out_dir = Path(args.out_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] split={args.split} samples={len(idxs)} out={out_dir}")

    for k, i in enumerate(idxs):
        sample = ds[i]
        x = sample.x.unsqueeze(0).to(device)  # (1,T,1,H,W)
        y = sample.y.cpu().numpy().astype(np.int64)  # (H,W)
        meta = sample.meta

        logits = model(x)[0]  # (4,H,W)
        probs = torch.softmax(logits, dim=0)
        conf, pred_t = torch.max(probs, dim=0)
        pred = pred_t.detach().cpu().numpy().astype(np.int64)
        conf_np = conf.detach().cpu().numpy().astype(np.float32)

        if float(args.min_conf) > 0:
            pred[conf_np < float(args.min_conf)] = 0

        if args.postprocess != "none":
            pred = keep_one_component_per_class(
                pred,
                class_ids=(1, 2, 3),
                mode=("largest" if args.postprocess == "largest_per_class" else "longest"),
                min_area_by_class={
                    1: int(args.min_area_nerve),
                    2: int(args.min_area_artery),
                    3: int(args.min_area_muscle),
                },
            )

        target_frame_path = meta["frame_paths"][-1]
        gray = read_gray(target_frame_path)

        vis_raw = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        vis_gt = overlay_anat(gray, y, alpha=0.55)
        vis_pred = overlay_anat(gray, pred, alpha=0.55)
        triptych = np.concatenate([vis_raw, vis_gt, vis_pred], axis=1)

        vid = meta["video_id"]
        fid = meta["target_frame_id"]
        fn = out_dir / f"{k:03d}_v{vid}_f{fid}.png"
        cv2.imwrite(str(fn), triptych)

    print("[DONE] Wrote overlay images.")


if __name__ == "__main__":
    main()
