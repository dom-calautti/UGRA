from __future__ import annotations
import argparse
import numpy as np
import torch

from datasets_temporal import TemporalNeedleDataset
from models_convlstm_unet import ConvLSTMUNet
from postprocess import keep_one_component

def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    inter = float((pred & gt).sum())
    return (2.0 * inter + eps) / (float(pred.sum() + gt.sum()) + eps)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/needle_convlstm/best.pt")
    ap.add_argument("--meta_dir", default="data_processed/meta")
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--postprocess", choices=["none", "largest", "longest"], default="none")
    ap.add_argument("--min_area", type=int, default=20)
    ap.add_argument("--thr_min", type=float, default=0.05)
    ap.add_argument("--thr_max", type=float, default=0.95)
    ap.add_argument("--thr_step", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    json_path = f"{args.meta_dir}/sequences_needle_{args.split}.json"
    ds = TemporalNeedleDataset(json_path, expected_T=args.T, normalize=True)

    model = ConvLSTMUNet(in_channels=1, base=args.base).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    thresholds = np.arange(args.thr_min, args.thr_max + 1e-9, args.thr_step)
    best = (-1.0, None)

    for thr in thresholds:
        scores = []
        for i in range(len(ds)):
            sample = ds[i]
            x = sample.x.unsqueeze(0).to(device)          # (1,T,1,H,W)
            y = sample.y[0].cpu().numpy().astype(np.uint8) # (H,W)

            probs = torch.sigmoid(model(x))[0, 0].detach().cpu().numpy()
            pred = (probs > thr).astype(np.uint8)

            pred = keep_one_component(pred, mode=args.postprocess, min_area=args.min_area)
            scores.append(dice(pred, y))

        mean_d = float(np.mean(scores)) if scores else 0.0
        print(f"thr={thr:.2f}  mean_dice={mean_d:.4f}  post={args.postprocess}")

        if mean_d > best[0]:
            best = (mean_d, thr)

    print(f"\n[BEST] mean_dice={best[0]:.4f} at thr={best[1]:.2f} with post={args.postprocess}")

if __name__ == "__main__":
    main()
