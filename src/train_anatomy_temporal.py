from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets_anatomy_temporal import TemporalAnatomyDataset, anatomy_temporal_collate
from models_convlstm_unet import ConvLSTMUNet


def mean_dice_excluding_bg(pred: torch.Tensor, gt: torch.Tensor, num_classes: int = 4) -> float:
    """
    pred: (B,H,W) int64
    gt:   (B,H,W) int64
    returns mean dice over classes 1..num_classes-1
    """
    eps = 1e-6
    dices = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        g = (gt == c).float()
        inter = (p * g).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + g.sum(dim=(1, 2))
        d = (2 * inter + eps) / (denom + eps)
        dices.append(d.mean().item())
    return float(np.mean(dices)) if dices else 0.0


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    scores = []
    for x, y, _meta in loader:
        x = x.to(device)              # (B,T,1,H,W)
        y = y.to(device)              # (B,H,W)
        logits = model(x)             # (B,4,H,W)
        pred = torch.argmax(logits, dim=1)  # (B,H,W)
        scores.append(mean_dice_excluding_bg(pred, y, num_classes=4))
    return float(np.mean(scores)) if scores else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default="data_processed/meta")
    ap.add_argument("--out_dir", default="runs/anatomy_convlstm")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = Path(args.meta_dir)
    train_json = meta_dir / "sequences_anatomy_train.json"
    val_json = meta_dir / "sequences_anatomy_val.json"

    ds_train = TemporalAnatomyDataset(train_json, expected_T=args.T, normalize=True)
    ds_val = TemporalAnatomyDataset(val_json, expected_T=args.T, normalize=True)

    if len(ds_train) == 0:
        raise RuntimeError(f"Empty train dataset: {train_json}")
    if len(ds_val) == 0:
        raise RuntimeError(f"Empty val dataset: {val_json}")

    train_loader = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=anatomy_temporal_collate
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=anatomy_temporal_collate
    )

    model = ConvLSTMUNet(in_channels=1, base=args.base, out_channels=4).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()  # expects (B,C,H,W) vs (B,H,W)

    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    best = -1.0
    best_path = out_dir / "best.pt"

    print(f"[INFO] device={device} amp={bool(scaler)}")
    print(f"[INFO] train={len(ds_train)} val={len(ds_val)} T={args.T} base={args.base}")
    print(f"[INFO] out_dir={out_dir.resolve()}")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        t0 = time.time()

        for step, (x, y, _meta) in enumerate(train_loader, start=1):
            x = x.to(device)  # (B,T,1,H,W)
            y = y.to(device)  # (B,H,W)

            opt.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(x)          # (B,4,H,W)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()

            losses.append(loss.item())
            if step == 1 or (step % 20) == 0:
                print(f"[E{ep:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f}")

        avg_loss = float(np.mean(losses)) if losses else 0.0
        val_dice = evaluate(model, val_loader, device)

        print(f"[E{ep:03d}] avg_train_loss={avg_loss:.4f} val_mean_dice(ex_bg)={val_dice:.4f} time={time.time()-t0:.1f}s")

        # save best
        if val_dice > best:
            best = val_dice
            torch.save(
                {"model": model.state_dict(), "epoch": ep, "val_dice": best, "args": vars(args)},
                best_path
            )
            print(f"[SAVE] best checkpoint -> {best_path}")

    print("[DONE] Anatomy training complete.")


if __name__ == "__main__":
    main()
