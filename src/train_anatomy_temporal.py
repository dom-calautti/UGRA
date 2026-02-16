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
from monai.losses import DiceCELoss

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

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss, label="train_loss")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_fg, label="val_mean_dice_ex_bg")
        axes[1].plot(epochs, nerve, label="nerve")
        axes[1].plot(epochs, artery, label="artery")
        axes[1].plot(epochs, muscle, label="muscle")
        axes[1].set_title("Validation Dice")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not write training curves: {e}")


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
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int = 4) -> Tuple[float, List[float]]:
    model.eval()
    fg_scores: List[float] = []
    pc_accum = np.zeros((num_classes,), dtype=np.float64)
    n = 0

    for x, y, _meta in loader:
        x = x.to(device)          # (B,T,1,H,W)
        y = y.to(device).long()   # (B,H,W)

        logits = model(x)         # (B,C,H,W)
        pred = torch.argmax(logits, dim=1)

        fg_scores.append(mean_dice_ex_bg(pred, y, num_classes=num_classes))
        pc = per_class_dice(pred, y, num_classes=num_classes)
        pc_accum += np.asarray(pc, dtype=np.float64)
        n += 1

    if n == 0:
        return 0.0, [0.0] * num_classes

    return float(np.mean(fg_scores)), (pc_accum / n).tolist()


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

    ap.add_argument("--T", type=int, default=4, help="Temporal window length (must match manifest sequence length).")
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

    ap.add_argument("--loss", choices=["ce", "dicece"], default="ce", help="Loss type: CE or MONAI DiceCE.")
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
    (out_root / "run_config.json").write_text(json.dumps(run_config, indent=2))
    command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command_line + "\n", encoding="utf-8")
    (out_root / "latest_command.txt").write_text(command_line + "\n", encoding="utf-8")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best = -1.0
    best_epoch = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    root_best_path = out_root / "best.pt"
    root_last_path = out_root / "last.pt"
    metrics_csv = run_dir / "metrics.csv"
    history: list[dict] = []

    print(f"[INFO] device={device} amp={bool(scaler)}")
    print(f"[INFO] train={len(ds_train)} ({args.train_split}) val={len(ds_val)} ({args.val_split}) T={args.T}")
    print(f"[INFO] out_root={out_root.resolve()}")
    print(f"[INFO] run_dir={run_dir.resolve()}")
    print(f"[INFO] arch={args.arch}")
    print("[INFO] CE weights:", (cw.detach().cpu().tolist() if cw is not None else "unweighted"))
    print(f"[INFO] loss={args.loss} lambda_dice={args.lambda_dice} lambda_ce={args.lambda_ce}")
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
                    else:
                        loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                if args.loss == "dicece":
                    loss = dicece(logits, y.unsqueeze(1))
                else:
                    loss = crit(logits, y)
                loss.backward()
                opt.step()

            losses.append(float(loss.item()))
            if step == 1 or (step % 20) == 0:
                print(f"[E{ep:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f}")

        avg_loss = float(np.mean(losses)) if losses else 0.0
        val_fg, val_pc = evaluate(model, val_loader, device, num_classes=4)

        print(
            f"[E{ep:03d}] avg_train_loss={avg_loss:.4f} "
            f"val_mean_dice(ex_bg)={val_fg:.4f} "
            f"val_dice(bg/nerve/artery/muscle)={val_pc[0]:.3f}/{val_pc[1]:.3f}/{val_pc[2]:.3f}/{val_pc[3]:.3f} "
            f"time={time.time()-t0:.1f}s"
        )

        # save last
        last_obj = {"model": model.state_dict(), "epoch": ep, "val_mean_dice_ex_bg": val_fg, "args": vars(args), "model_cfg": model_cfg}
        torch.save(last_obj, last_path)
        torch.save(last_obj, root_last_path)

        row = {
            "epoch": ep,
            "train_loss": avg_loss,
            "val_mean_dice_ex_bg": float(val_fg),
            "dice_bg": float(val_pc[0]),
            "dice_nerve": float(val_pc[1]),
            "dice_artery": float(val_pc[2]),
            "dice_muscle": float(val_pc[3]),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        # save best
        if val_fg > best:
            best = val_fg
            best_epoch = ep
            best_obj = {"model": model.state_dict(), "epoch": ep, "val_mean_dice_ex_bg": best, "args": vars(args), "model_cfg": model_cfg}
            torch.save(best_obj, best_path)
            torch.save(best_obj, root_best_path)
            (out_root / "latest_best.txt").write_text(str(best_path), encoding="utf-8")
            print(f"[SAVE] best checkpoint -> {best_path}")

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_mean_dice_ex_bg": float(best),
        "epochs": int(args.epochs),
        "arch": args.arch,
        "run_dir": str(run_dir),
        "out_root": str(out_root),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_training_curves_anatomy(history, run_dir / "training_curves.jpg")

    print("[DONE] Anatomy training complete.")


if __name__ == "__main__":
    main()
