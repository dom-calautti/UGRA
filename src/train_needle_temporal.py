# src/train_needle_temporal.py
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from monai_temporal_datasets import make_needle_dataset, collate_needle
from models_convlstm_unet import ConvLSTMUNet


def _write_training_curves_needle(history: list[dict], out_path: Path):
    if not history:
        return
    try:
        import matplotlib.pyplot as plt

        epochs = [int(h["epoch"]) for h in history]
        train_loss = [float(h["train_loss"]) for h in history]
        val_dice = [float(h["val_dice"]) if h["val_dice"] is not None else np.nan for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, train_loss, label="train_loss")
        axes[0].set_title("Train Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, val_dice, label="val_dice")
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


@torch.no_grad()
def evaluate(model, loader, device, thr: float) -> float:
    model.eval()
    dice = DiceMetric(include_background=False, reduction="mean")

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)          # (B,T,1,H,W)
        y = y.to(device, non_blocking=True)          # (B,1,H,W)

        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs >= thr).float()

        dice(preds, y)

    score = float(dice.aggregate().item())
    dice.reset()
    return score


def main():
    ap = argparse.ArgumentParser(
        description="Train temporal needle segmentation model (ConvLSTMUNet).",
        epilog="Output: writes best.pt, last.pt, and run_config.json to --save_dir.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--meta_dir", default="data_processed/meta", help="Folder containing sequences_needle_*.json manifests.")
    ap.add_argument("--save_dir", default="runs/needle_convlstm", help="Output folder for checkpoints and run_config.json.")

    ap.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    ap.add_argument("--batch_size", type=int, default=1, help="Training batch size.")
    ap.add_argument("--lr", type=float, default=2e-4, help="Learning rate for AdamW.")

    ap.add_argument("--T", type=int, default=8, help="Temporal window length.")
    ap.add_argument("--base", type=int, default=32, help="ConvLSTMUNet base channels.")

    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count.")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    ap.add_argument("--device", default="cuda", help="Training device: cuda or cpu.")

    # MONAI-ish knobs (keep simple)
    ap.add_argument("--cache_rate", type=float, default=0.0, help="MONAI cache fraction for train dataset (0 disables cache).")
    ap.add_argument("--val_thr", type=float, default=0.5, help="Threshold used for validation Dice reporting.")

    ap.add_argument("--seed", type=int, default=0, help="Random seed for NumPy and PyTorch.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    meta_dir = Path(args.meta_dir)
    train_json = meta_dir / "sequences_needle_train.json"
    val_json = meta_dir / "sequences_needle_val.json"

    if not train_json.exists():
        raise FileNotFoundError(f"Missing train manifest: {train_json.resolve()}")

    ds_train = make_needle_dataset(train_json, expected_T=args.T, normalize=True, cache_rate=args.cache_rate, keep_all=True)

    ds_val = None
    if val_json.exists():
        ds_val = make_needle_dataset(val_json, expected_T=args.T, normalize=True, cache_rate=0.0, keep_all=True)
        if len(ds_val) == 0:
            ds_val = None

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_needle,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = None
    if ds_val is not None:
        val_loader = DataLoader(
            ds_val,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_needle,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)
    run_tag = time.strftime("%Y%m%d_%H%M%S")
    run_dir = save_root / f"run_{run_tag}_needle"
    run_dir.mkdir(parents=True, exist_ok=True)
    (save_root / "latest_run.txt").write_text(str(run_dir), encoding="utf-8")

    run_config = {
        "args": vars(args),
        "run_dir": str(run_dir),
        "save_root": str(save_root),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (save_root / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    command_line = subprocess.list2cmdline([sys.executable, *sys.argv])
    (run_dir / "command.txt").write_text(command_line + "\n", encoding="utf-8")
    (save_root / "latest_command.txt").write_text(command_line + "\n", encoding="utf-8")

    model = ConvLSTMUNet(in_channels=1, base=args.base, out_channels=1).to(device)

    # Stable needle loss: Dice + CE (sigmoid)
    loss_fn = DiceCELoss(
        include_background=False,
        sigmoid=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val = -1.0
    best_epoch = 0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    root_best_path = save_root / "best.pt"
    root_last_path = save_root / "last.pt"
    metrics_csv = run_dir / "metrics.csv"
    history: list[dict] = []

    print(f"[INFO] device={device} amp={bool(scaler)}")
    print(f"[INFO] train={len(ds_train)} T={args.T} base={args.base} cache_rate={args.cache_rate}")
    if ds_val is not None:
        print(f"[INFO] val={len(ds_val)} val_thr={args.val_thr}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, (x, y, _meta) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item())
            if step == 1 or (step % 10) == 0:
                print(f"[E{epoch:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f}")

        avg_loss = running / max(1, len(train_loader))
        print(f"[E{epoch:03d}] avg_train_loss={avg_loss:.4f}")

        # save last
        last_obj = {"model": model.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(last_obj, last_path)
        torch.save(last_obj, root_last_path)

        val_dice = None

        # val
        if val_loader is not None:
            val_dice = evaluate(model, val_loader, device, thr=float(args.val_thr))
            print(f"[E{epoch:03d}] val_dice@thr{args.val_thr:.2f}={val_dice:.4f}")

            if val_dice > best_val:
                best_val = val_dice
                best_epoch = epoch
                best_obj = {"model": model.state_dict(), "epoch": epoch, "val_dice": best_val, "args": vars(args)}
                torch.save(best_obj, best_path)
                torch.save(best_obj, root_best_path)
                (save_root / "latest_best.txt").write_text(str(best_path), encoding="utf-8")
                print(f"[SAVE] best checkpoint -> {best_path}")

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(avg_loss),
                "val_dice": (float(val_dice) if val_dice is not None else None),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_dice": float(best_val),
        "epochs": int(args.epochs),
        "run_dir": str(run_dir),
        "save_root": str(save_root),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_training_curves_needle(history, run_dir / "training_curves.jpg")

    print("[DONE] Needle training complete.")


if __name__ == "__main__":
    main()
