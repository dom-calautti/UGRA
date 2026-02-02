from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric

from datasets_temporal import TemporalNeedleDataset, temporal_collate
from models_convlstm_unet import ConvLSTMUNet


def make_loaders(meta_dir: Path, batch_size: int, T: int, num_workers: int):
    train = TemporalNeedleDataset(meta_dir / "sequences_needle_train.json", expected_T=T, normalize=True)
    val_path = meta_dir / "sequences_needle_val.json"
    val = TemporalNeedleDataset(val_path, expected_T=T, normalize=True) if val_path.exists() else None

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=temporal_collate, pin_memory=True
    )
    val_loader = None
    if val is not None and len(val) > 0:
        val_loader = DataLoader(
            val, batch_size=1, shuffle=False, num_workers=num_workers,
            collate_fn=temporal_collate, pin_memory=True
        )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    dice = DiceMetric(include_background=False, reduction="mean")

    for x, y, _meta in loader:
        x = x.to(device, non_blocking=True)          # (B,T,1,H,W)
        y = y.to(device, non_blocking=True)          # (B,1,H,W)

        logits = model(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # DiceMetric expects (B,C,...) and y same shape
        dice(preds, y)

    score = dice.aggregate().item()
    dice.reset()
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_dir", default="data_processed/meta")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=1)  # start conservative
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--save_dir", default="runs/needle_convlstm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta_dir = Path(args.meta_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(meta_dir, args.batch_size, args.T, args.num_workers)

    model = ConvLSTMUNet(in_channels=1, base=args.base).to(device)

    # Needle-friendly loss: Dice + CE (sigmoid)
    loss_fn = DiceCELoss(
        include_background=False,
        sigmoid=True,
        lambda_dice=1.0,
        lambda_ce=1.0
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for step, (x, y, _meta) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()
            if step % 10 == 0 or step == 1:
                print(f"[E{epoch:03d}] step {step:04d}/{len(train_loader):04d} loss={loss.item():.4f}")

        avg_loss = running / max(1, len(train_loader))
        print(f"[E{epoch:03d}] avg_train_loss={avg_loss:.4f}")

        # quick val (if available)
        if val_loader is not None:
            val_dice = evaluate(model, val_loader, device)
            print(f"[E{epoch:03d}] val_dice={val_dice:.4f}")

            if val_dice > best_val:
                best_val = val_dice
                ckpt = save_dir / "best.pt"
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": val_dice}, ckpt)
                print(f"[SAVE] best checkpoint -> {ckpt}")

        # always save last
        torch.save({"model": model.state_dict(), "epoch": epoch}, save_dir / "last.pt")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
