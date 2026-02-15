# src/train_anatomy_temporal.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_anatomy_temporal import TemporalAnatomyDataset, anatomy_temporal_collate
from models_convlstm_unet import ConvLSTMUNet


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def per_class_dice(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> List[float]:
    """
    pred: (B,H,W) int64
    gt:   (B,H,W) int64
    returns dice per class (0..C-1)
    """
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
    """
    "0.05,6,4,1" -> tensor([0.05, 6, 4, 1])
    Order: (bg, nerve, artery, muscle)
    """
    s = (s or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"--class_weights needs {num_classes} comma values, got {len(parts)}: {parts}")
    return torch.tensor([float(x) for x in parts], dtype=torch.float32)


@torch.no_grad()
def estimate_pixel_counts(ds: TemporalAnatomyDataset, num_classes: int = 4, max_samples: int = 2000) -> np.ndarray:
    """
    Estimates class pixel counts from dataset targets.
    This is just to help you pick reasonable weights.
    """
    n = min(len(ds), max_samples)
    counts = np.zeros((num_classes,), dtype=np.int64)

    for i in range(n):
        sample = ds[i]
        y = sample.y
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)

        # accept either (H,W) or (1,H,W)
        if y_np.ndim == 3 and y_np.shape[0] == 1:
            y_np = y_np[0]

        for c in range(num_classes):
            counts[c] += int((y_np == c).sum())

    return counts


def suggest_inv_freq_weights(counts: np.ndarray) -> np.ndarray:
    """
    Classic inverse-frequency weights, normalized so mean(weight)=1.
    """
    counts = counts.astype(np.float64)
    total = max(counts.sum(), 1.0)
    freq = counts / total
    w = 1.0 / np.maximum(freq, 1e-12)
    w = w / np.mean(w)
    return w.astype(np.float32)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--meta_dir", default="data_processed/meta")
    ap.add_argument("--out_dir", default="runs/anatomy_convlstm")

    # which manifests to use (so you can train on whatever split you decide)
    ap.add_argument("--train_split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--val_split", choices=["train", "val", "test"], default="val")

    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)

    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)

    # Optional: weighted CE for class imbalance
    ap.add_argument(
        "--class_weights",
        default="",
        help='Optional CE weights (bg,nerve,artery,muscle), e.g. "0.05,6,4,1". Default = unweighted.',
    )

    # Helper: print suggested weights and exit (so you stay in control)
    ap.add_argument("--calc_class_weights", action="store_true")
    ap.add_argument("--calc_max_samples", type=int, default=2000)

    ap.add_argument("--seed", type=int, default=0)
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=anatomy_temporal_collate,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save run config so you can reproduce runs later
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "train_manifest": str(train_json),
                "val_manifest": str(val_json),
                "device": str(device),
                "class_weights": (cw.detach().cpu().tolist() if cw is not None else None),
            },
            indent=2,
        )
    )

    model = ConvLSTMUNet(in_channels=1, base=args.base, out_channels=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(weight=cw)  # logits (B,C,H,W) vs target (B,H,W)

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best = -1.0
    best_path = out_dir / "best.pt"

    print(f"[INFO] device={device} amp={bool(scaler)}")
    print(f"[INFO] train={len(ds_train)} ({args.train_split}) val={len(ds_val)} ({args.val_split}) T={args.T} base={args.base}")
    print(f"[INFO] out_dir={out_dir.resolve()}")
    print("[INFO] CE weights:", (cw.detach().cpu().tolist() if cw is not None else "unweighted"))

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
                    logits = model(x)     # (B,4,H,W)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
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

        if val_fg > best:
            best = val_fg
            torch.save({"model": model.state_dict(), "epoch": ep, "val_mean_dice_ex_bg": best, "args": vars(args)}, best_path)
            print(f"[SAVE] best checkpoint -> {best_path}")

    print("[DONE] Anatomy training complete.")


if __name__ == "__main__":
    main()
