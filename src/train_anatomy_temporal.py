# src/train_anatomy_temporal.py
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.losses import DiceCELoss, GeneralizedDiceLoss

from datasets_anatomy_temporal import TemporalAnatomyDataset, anatomy_temporal_collate
from models_anatomy_swin_convlstm import AnatomySwinConvLSTMConfig, AnatomySwinEncoderConvLSTM
from models_convlstm_unet import ConvLSTMUNet


def _write_training_curves_anatomy(history: list[dict], out_path: Path):
    if not history:
        return
    try:
        import matplotlib.pyplot as plt

        epochs = [int(h["epoch"]) for h in history]
        train_loss = [float(h["train_loss"]) for h in history]
        val_fg = [float(h["val_mean_dice_ex_bg"]) for h in history]
        nerve = [float(h["dice_nerve"]) for h in history]
        artery = [float(h["dice_artery"]) for h in history]
        muscle = [float(h["dice_muscle"]) for h in history]

        fp_nerve = [float(h.get("fp_rate_nerve", 0.0)) for h in history]
        fp_artery = [float(h.get("fp_rate_artery", 0.0)) for h in history]
        fp_muscle = [float(h.get("fp_rate_muscle", 0.0)) for h in history]

        iou_nerve = [float(h.get("iou_nerve", 0.0)) for h in history]
        iou_artery = [float(h.get("iou_artery", 0.0)) for h in history]
        iou_muscle = [float(h.get("iou_muscle", 0.0)) for h in history]
        iou_mean = [float(h.get("val_mean_iou_ex_bg", 0.0)) for h in history]

        prec_nerve = [float(h.get("prec_nerve", 0.0)) for h in history]
        prec_artery = [float(h.get("prec_artery", 0.0)) for h in history]
        prec_muscle = [float(h.get("prec_muscle", 0.0)) for h in history]

        rec_nerve = [float(h.get("rec_nerve", 0.0)) for h in history]
        rec_artery = [float(h.get("rec_artery", 0.0)) for h in history]
        rec_muscle = [float(h.get("rec_muscle", 0.0)) for h in history]

        lr = [float(h.get("lr", 0.0)) for h in history]

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        # Row 0: Loss, Dice, IoU
        axes[0, 0].plot(epochs, train_loss, label="train_loss")
        axes[0, 0].set_title("Train Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, val_fg, label="mean_dice_ex_bg", linewidth=2)
        axes[0, 1].plot(epochs, nerve, label="nerve", alpha=0.7)
        axes[0, 1].plot(epochs, artery, label="artery", alpha=0.7)
        axes[0, 1].plot(epochs, muscle, label="muscle", alpha=0.7)
        axes[0, 1].set_title("Validation Dice")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylim(0.0, 1.0)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(epochs, iou_mean, label="mean_iou_ex_bg", linewidth=2)
        axes[0, 2].plot(epochs, iou_nerve, label="nerve", alpha=0.7)
        axes[0, 2].plot(epochs, iou_artery, label="artery", alpha=0.7)
        axes[0, 2].plot(epochs, iou_muscle, label="muscle", alpha=0.7)
        axes[0, 2].set_title("Validation IoU")
        axes[0, 2].set_xlabel("Epoch")
        axes[0, 2].set_ylim(0.0, 1.0)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # Row 1: FP rates, Precision, Recall
        axes[1, 0].plot(epochs, fp_nerve, label="fp_nerve")
        axes[1, 0].plot(epochs, fp_artery, label="fp_artery")
        axes[1, 0].plot(epochs, fp_muscle, label="fp_muscle")
        axes[1, 0].set_title("False Positive Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylim(0.0, 1.0)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, prec_nerve, label="nerve")
        axes[1, 1].plot(epochs, prec_artery, label="artery")
        axes[1, 1].plot(epochs, prec_muscle, label="muscle")
        axes[1, 1].set_title("Precision (per class)")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylim(0.0, 1.0)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(epochs, rec_nerve, label="nerve")
        axes[1, 2].plot(epochs, rec_artery, label="artery")
        axes[1, 2].plot(epochs, rec_muscle, label="muscle")
        axes[1, 2].set_title("Recall (per class)")
        axes[1, 2].set_xlabel("Epoch")
        axes[1, 2].set_ylim(0.0, 1.0)
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # Row 2: Nerve deep-dive, LR curve, dice vs IoU
        axes[2, 0].plot(epochs, nerve, label="dice_nerve")
        axes[2, 0].plot(epochs, fp_nerve, label="fp_nerve", linestyle="--")
        axes[2, 0].plot(epochs, prec_nerve, label="prec_nerve", linestyle=":")
        axes[2, 0].plot(epochs, rec_nerve, label="rec_nerve", linestyle="-.")
        axes[2, 0].set_title("Nerve Deep-Dive")
        axes[2, 0].set_xlabel("Epoch")
        axes[2, 0].set_ylim(0.0, 1.0)
        axes[2, 0].legend(fontsize=8)
        axes[2, 0].grid(True, alpha=0.3)

        axes[2, 1].plot(epochs, lr, label="learning_rate", color="tab:orange")
        axes[2, 1].set_title("Learning Rate")
        axes[2, 1].set_xlabel("Epoch")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        axes[2, 2].plot(epochs, val_fg, label="dice_ex_bg", linewidth=2)
        axes[2, 2].plot(epochs, iou_mean, label="iou_ex_bg", linewidth=2, linestyle="--")
        axes[2, 2].set_title("Dice vs IoU (mean ex bg)")
        axes[2, 2].set_xlabel("Epoch")
        axes[2, 2].set_ylim(0.0, 1.0)
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)

        fig.suptitle("Anatomy Training Report", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not write training curves: {e}")


def _write_confusion_matrix_plot(cm: np.ndarray, out_path: Path):
    try:
        import matplotlib.pyplot as plt

        labels = ["bg", "nerve", "artery", "muscle"]
        row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
        cm_norm = cm.astype(np.float64) / row_sums

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title("Normalized Confusion Matrix (Val)")

        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not write confusion matrix plot: {e}")


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def per_class_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> List[float]:
    eps = 1e-6
    out: List[float] = []
    for c in range(num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        d = (2 * inter + eps) / (denom + eps)
        out.append(float(d.mean().item()))
    return out


@torch.no_grad()
def mean_dice_ex_bg(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> float:
    d = per_class_dice(pred, gt, num_classes=num_classes)
    return float(np.mean(d[1:])) if num_classes > 1 else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 4) -> Tuple[float, List[float], np.ndarray, dict]:
    model.eval()
    fg_scores: List[float] = []
    pc_accum = np.zeros((num_classes,), dtype=np.float64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    n = 0

    for x, y, _meta in loader:
        x = x.to(device)          # (B,T,1,H,W)
        y = y.to(device).long()   # (B,H,W)

        logits = model(x)         # (B,C,H,W)
        pred = torch.argmax(logits, dim=1)

        fg_scores.append(mean_dice_ex_bg(pred, y, num_classes=num_classes))
        pc = per_class_dice(pred, y, num_classes=num_classes)
        pc_accum += np.asarray(pc, dtype=np.float64)

        yt = y.view(-1).detach().cpu().numpy().astype(np.int64)
        pt = pred.view(-1).detach().cpu().numpy().astype(np.int64)
        binc = np.bincount(num_classes * yt + pt, minlength=num_classes * num_classes)
        cm += binc.reshape(num_classes, num_classes)
        n += 1

    if n == 0:
        return 0.0, [0.0] * num_classes, cm, {}

    fp_rates = {}
    total = int(cm.sum())
    for c in range(num_classes):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = total - tp - fp - fn
        fpr = float(fp / max(fp + tn, 1))
        fp_rates[c] = fpr

    return float(np.mean(fg_scores)), (pc_accum / n).tolist(), cm, fp_rates


# -------------------------
# Class weights helpers
# -------------------------
def parse_class_weights(s: str, num_classes: int = 4) -> Optional[torch.Tensor]:
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"--class_weights needs {num_classes} comma values, got {len(parts)}: {parts}")
    return torch.tensor([float(x) for x in parts], dtype=torch.float32)


@torch.no_grad()
def estimate_pixel_counts(ds: TemporalAnatomyDataset, num_classes: int = 4, max_samples: int = 2000) -> np.ndarray:
    n = min(len(ds), max_samples)
    counts = np.zeros((num_classes,), dtype=np.int64)

    for i in range(n):
        sample = ds[i]
        y = sample.y
        y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)

        if y_np.ndim == 3 and y_np.shape[0] == 1:
            y_np = y_np[0]

        for c in range(num_classes):
            counts[c] += int((y_np == c).sum())

    return counts


def suggest_inv_freq_weights(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float64)
    total = max(counts.sum(), 1.0)
    freq = counts / total
    w = 1.0 / np.maximum(freq, 1e-12)
    w = w / np.mean(w)
    return w.astype(np.float32)


# -------------------------
# Small helpers (Swin args)
# -------------------------
def _as_2tuple(v: int) -> Tuple[int, int]:
    # MONAI Swin expects iterable window_size for 2D: (wh, ww)
    return (int(v), int(v))


def _warn_if_not_divisible(name: str, a: int, b: int):
    # Not always fatal, but a good early warning.
    if b > 0 and (a % b) != 0:
        print(f"[WARN] {name}: {a} not divisible by {b}. If Swin errors later, fix this first.")


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


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train anatomy temporal segmentation (Swin+ConvLSTM or ConvLSTM-only).",
        epilog="Output: writes best.pt, last.pt, and run_config.json to --out_dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--meta_dir", default="data_processed/meta", help="Folder containing sequences_anatomy_*.json manifests.")
    ap.add_argument("--out_dir", default="runs/anatomy_swin_convlstm", help="Output folder for checkpoints and run_config.json.")

    ap.add_argument("--train_split", choices=["train", "val", "test"], default="train", help="Manifest split used for training.")
    ap.add_argument("--val_split", choices=["train", "val", "test"], default="val", help="Manifest split used for validation.")

    ap.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW.")

    ap.add_argument("--T", type=int, default=0, help="Temporal window length. Use 0 to auto-match preprocess manifest sequence length.")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    ap.add_argument("--device", default="cuda", help="Training device: cuda or cpu.")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")

    ap.add_argument("--class_weights", default="", help='Optional CE weights as "bg,nerve,artery,muscle".')

    ap.add_argument(
        "--arch",
        choices=["swin_convlstm", "convlstm"],
        default="swin_convlstm",
        help="Model architecture: Swin encoder + ConvLSTM (default) or ConvLSTM-only baseline.",
    )
    ap.add_argument("--convlstm_base", type=int, default=32, help="Base channels for ConvLSTM-only anatomy model.")

    ap.add_argument("--loss", choices=["ce", "dicece", "gdice", "gdicece"], default="ce", help="Loss type: CE, DiceCE, GeneralizedDice, or GeneralizedDice+CE.")
    ap.add_argument("--lambda_dice", type=float, default=1.0, help="Dice term weight when --loss=dicece.")
    ap.add_argument("--lambda_ce", type=float, default=1.0, help="CE term weight when --loss=dicece.")

    ap.add_argument("--calc_class_weights", action="store_true", help="Only estimate class weights from training targets and exit.")
    ap.add_argument("--calc_max_samples", type=int, default=2000, help="Sample cap used by --calc_class_weights.")

    # Swin encoder knobs
    ap.add_argument("--img_size", type=int, default=256, help="Expected image size used by Swin encoder init.")
    ap.add_argument("--patch_size", type=int, default=4, help="Swin patch size (used when --arch=swin_convlstm).")
    ap.add_argument("--swin_embed_dim", type=int, default=48, help="Swin base embedding dimension.")
    ap.add_argument("--swin_window_size", type=int, default=7, help="Swin window size (converted to 2D tuple).")
    ap.add_argument("--use_checkpoint", action="store_true", help="Enable Swin activation checkpointing to reduce VRAM.")

    ap.add_argument("--scheduler", choices=["none", "cosine"], default="none", help="LR schedule: none (constant) or cosine annealing.")
    ap.add_argument("--warmup_epochs", type=int, default=5, help="Linear warmup epochs before cosine decay (only used with --scheduler cosine).")

    ap.add_argument("--seed", type=int, default=0, help="Random seed for NumPy and PyTorch.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    meta_dir = Path(args.meta_dir)
    train_json = meta_dir / f"sequences_anatomy_{args.train_split}.json"
    val_json = meta_dir / f"sequences_anatomy_{args.val_split}.json"

    if not train_json.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_json.resolve()}")
    if not val_json.exists():
        raise FileNotFoundError(f"Missing val manifest: {val_json.resolve()}")

    if int(args.T) <= 0:
        inferred_t = _infer_t_from_manifest(train_json)
        if inferred_t is None:
            raise ValueError("Could not auto-infer T from train manifest. Pass --T explicitly.")
        args.T = int(inferred_t)
        print(f"[INFO] auto-resolved T={args.T} from {train_json.name}")

    ds_train = TemporalAnatomyDataset(train_json, expected_T=args.T, normalize=True)
    ds_val = TemporalAnatomyDataset(val_json, expected_T=args.T, normalize=True)

    if len(ds_train) == 0:
        raise RuntimeError(f"Empty train dataset: {train_json}")
    if len(ds_val) == 0:
        raise RuntimeError(f"Empty val dataset: {val_json}")

    if args.calc_class_weights:
        counts = estimate_pixel_counts(ds_train, num_classes=4, max_samples=args.calc_max_samples)
        w = suggest_inv_freq_weights(counts)
        ratios = (counts / max(counts.sum(), 1)).astype(np.float64)
        print("[INFO] estimated pixel counts (bg,nerve,artery,muscle):", counts.tolist())
        print("[INFO] ratios:", [float(x) for x in ratios.tolist()])
        print("[INFO] suggested --class_weights:", ",".join([f"{x:.4f}" for x in w.tolist()]))
        return

    cw = parse_class_weights(args.class_weights, num_classes=4)
    if cw is not None:
        cw = cw.to(device)

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=anatomy_temporal_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=anatomy_temporal_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_tag}_{args.arch}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (out_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    model_cfg = {}
    if args.arch == "swin_convlstm":
        swin_window = _as_2tuple(args.swin_window_size)

        # Helpful early warnings (won't stop training)
        _warn_if_not_divisible("img_size vs patch_size", args.img_size, args.patch_size)
        _warn_if_not_divisible("img_size vs window_size", args.img_size, args.swin_window_size)

        cfg = AnatomySwinConvLSTMConfig(
            in_channels=1,
            out_channels=4,
            img_size=args.img_size,
            patch_size=args.patch_size,
            swin_embed_dim=args.swin_embed_dim,
            swin_window_size=swin_window,
            use_checkpoint=args.use_checkpoint,
        )
        model = AnatomySwinEncoderConvLSTM(cfg).to(device)
        model_cfg = {
            **cfg.__dict__,
            "swin_window_size": list(cfg.swin_window_size) if isinstance(cfg.swin_window_size, tuple) else cfg.swin_window_size,
        }
    else:
        model = ConvLSTMUNet(in_channels=1, base=int(args.convlstm_base), out_channels=4).to(device)
        model_cfg = {
            "arch": "convlstm",
            "base": int(args.convlstm_base),
            "in_channels": 1,
            "out_channels": 4,
        }

    # Save config for reproducibility
    run_config = {
        "args": vars(args),
        "model_cfg": model_cfg,
        "train_manifest": str(train_json),
        "val_manifest": str(val_json),
        "device": str(device),
        "class_weights": (cw.detach().cpu().tolist() if cw is not None else None),
        "run_dir": str(run_dir),
        "out_root": str(out_root),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(
            run_config,
            indent=2,
        )
    )
    command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command_line + "\n", encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # --- LR scheduler ---
    scheduler = None
    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(opt, start_factor=0.01, total_iters=max(args.warmup_epochs, 1))
        cosine = CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
        print(f"[INFO] scheduler=cosine warmup={args.warmup_epochs} epochs")

    crit = nn.CrossEntropyLoss(weight=cw)
    try:
        dicece = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            ce_weight=cw,
            lambda_dice=float(args.lambda_dice),
            lambda_ce=float(args.lambda_ce),
        )
    except TypeError:
        dicece = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            weight=cw,
            lambda_dice=float(args.lambda_dice),
            lambda_ce=float(args.lambda_ce),
        )
    gdice = GeneralizedDiceLoss(
        to_onehot_y=True,
        softmax=True,
    )
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best = -1.0
    best_epoch = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    metrics_csv = run_dir / "metrics.csv"
    history: list[dict] = []

    print(f"[INFO] device={device} amp={bool(scaler)}")
    print(f"[INFO] train={len(ds_train)} ({args.train_split}) val={len(ds_val)} ({args.val_split}) T={args.T}")
    print(f"[INFO] out_root={out_root.resolve()}")
    print(f"[INFO] run_dir={run_dir.resolve()}")
    print(f"[INFO] arch={args.arch}")
    print("[INFO] CE weights:", (cw.detach().cpu().tolist() if cw is not None else "unweighted"))
    print(f"[INFO] loss={args.loss} lambda_dice={args.lambda_dice} lambda_ce={args.lambda_ce}")
    print(f"[INFO] scheduler={args.scheduler}")
    if args.arch == "swin_convlstm":
        print("[INFO] Swin: embed_dim=", args.swin_embed_dim, "window=", _as_2tuple(args.swin_window_size), "patch=", args.patch_size, "ckpt=", args.use_checkpoint)
    else:
        print("[INFO] ConvLSTM: base=", args.convlstm_base)

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for step, (x, y, _meta) in enumerate(train_loader, start=1):
            x = x.to(device)          # (B,T,1,H,W)
            y = y.to(device).long()   # (B,H,W)

            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    if args.loss == "dicece":
                        loss = dicece(logits, y.unsqueeze(1))
                    elif args.loss == "gdice":
                        loss = gdice(logits, y.unsqueeze(1))
                    elif args.loss == "gdicece":
                        loss = float(args.lambda_dice) * gdice(logits, y.unsqueeze(1)) + float(args.lambda_ce) * crit(logits, y)
                    else:
                        loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                if args.loss == "dicece":
                    loss = dicece(logits, y.unsqueeze(1))
                elif args.loss == "gdice":
                    loss = gdice(logits, y.unsqueeze(1))
                elif args.loss == "gdicece":
                    loss = float(args.lambda_dice) * gdice(logits, y.unsqueeze(1)) + float(args.lambda_ce) * crit(logits, y)
                else:
                    loss = crit(logits, y)
                loss.backward()
                opt.step()

            losses.append(float(loss.item()))
            if step == 1 or (step % 20) == 0:
                print(f"[E{ep:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f}")

        avg_loss = float(np.mean(losses)) if losses else 0.0
        val_fg, val_pc, val_cm, val_fp = evaluate(model, val_loader, device, num_classes=4)

        # --- derive IoU / precision / recall from confusion matrix ---
        class_names = ["bg", "nerve", "artery", "muscle"]
        iou_vals, prec_vals, rec_vals = {}, {}, {}
        for c in range(4):
            tp = int(val_cm[c, c])
            fp = int(val_cm[:, c].sum() - tp)
            fn = int(val_cm[c, :].sum() - tp)
            iou_vals[c] = float(tp / max(tp + fp + fn, 1))
            prec_vals[c] = float(tp / max(tp + fp, 1))
            rec_vals[c] = float(tp / max(tp + fn, 1))
        mean_iou_ex_bg = float(np.mean([iou_vals[c] for c in range(1, 4)]))

        cur_lr = float(opt.param_groups[0]["lr"])
        print(
            f"[E{ep:03d}] avg_train_loss={avg_loss:.4f} "
            f"val_mean_dice(ex_bg)={val_fg:.4f} val_mean_iou(ex_bg)={mean_iou_ex_bg:.4f} "
            f"val_dice(bg/nerve/artery/muscle)={val_pc[0]:.3f}/{val_pc[1]:.3f}/{val_pc[2]:.3f}/{val_pc[3]:.3f} "
            f"lr={cur_lr:.2e} time={time.time()-t0:.1f}s"
        )

        # step LR scheduler
        if scheduler is not None:
            scheduler.step()

        # save last
        last_obj = {"model": model.state_dict(), "epoch": ep, "val_mean_dice_ex_bg": val_fg, "args": vars(args), "model_cfg": model_cfg}
        torch.save(last_obj, last_path)

        row = {
            "epoch": ep,
            "train_loss": avg_loss,
            "val_mean_dice_ex_bg": float(val_fg),
            "val_mean_iou_ex_bg": mean_iou_ex_bg,
            "dice_bg": float(val_pc[0]),
            "dice_nerve": float(val_pc[1]),
            "dice_artery": float(val_pc[2]),
            "dice_muscle": float(val_pc[3]),
            "iou_nerve": iou_vals[1],
            "iou_artery": iou_vals[2],
            "iou_muscle": iou_vals[3],
            "prec_nerve": prec_vals[1],
            "prec_artery": prec_vals[2],
            "prec_muscle": prec_vals[3],
            "rec_nerve": rec_vals[1],
            "rec_artery": rec_vals[2],
            "rec_muscle": rec_vals[3],
            "fp_rate_nerve": float(val_fp.get(1, 0.0)),
            "fp_rate_artery": float(val_fp.get(2, 0.0)),
            "fp_rate_muscle": float(val_fp.get(3, 0.0)),
            "lr": cur_lr,
        }
        history.append(row)

        # save best
        if val_fg > best:
            best = val_fg
            best_epoch = ep
            best_obj = {"model": model.state_dict(), "epoch": ep, "val_mean_dice_ex_bg": best, "args": vars(args), "model_cfg": model_cfg}
            torch.save(best_obj, best_path)
            print(f"[SAVE] best checkpoint -> {best_path}")

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    # --- write per-class summary from best-epoch row ---
    best_row = next((r for r in history if r["epoch"] == best_epoch), {})
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_mean_dice_ex_bg": float(best),
        "best_val_mean_iou_ex_bg": float(best_row.get("val_mean_iou_ex_bg", 0.0)),
        "best_dice_nerve": float(best_row.get("dice_nerve", 0.0)),
        "best_dice_artery": float(best_row.get("dice_artery", 0.0)),
        "best_dice_muscle": float(best_row.get("dice_muscle", 0.0)),
        "best_iou_nerve": float(best_row.get("iou_nerve", 0.0)),
        "best_iou_artery": float(best_row.get("iou_artery", 0.0)),
        "best_iou_muscle": float(best_row.get("iou_muscle", 0.0)),
        "best_prec_nerve": float(best_row.get("prec_nerve", 0.0)),
        "best_rec_nerve": float(best_row.get("rec_nerve", 0.0)),
        "epochs": int(args.epochs),
        "arch": args.arch,
        "loss": args.loss,
        "scheduler": args.scheduler,
        "run_dir": str(run_dir),
        "out_root": str(out_root),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_training_curves_anatomy(history, run_dir / "training_curves.jpg")
    if history:
        _, _, cm_final, _ = evaluate(model, val_loader, device, num_classes=4)
        np.savetxt(run_dir / "confusion_matrix.csv", cm_final, fmt="%d", delimiter=",")
        _write_confusion_matrix_plot(cm_final, run_dir / "confusion_matrix.jpg")

    print("[DONE] Anatomy training complete.")


if __name__ == "__main__":
    main()
