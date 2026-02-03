import json
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from models_convlstm_unet import ConvLSTMUNet

def load_ckpt(model, ckpt_path, device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k.replace("module.",""): v for k,v in state.items()}
    model.load_state_dict(state, strict=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--idx", type=int, default=0)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    seqs = json.loads(Path(args.manifest).read_text())
    s = seqs[args.idx]
    frames = s["frames"][-args.window:]
    mask_path = s["target_mask"]

    buf=[]
    for fp in frames:
        g = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        assert g is not None, fp
        t = torch.from_numpy((g.astype(np.float32)/255.0)).unsqueeze(0)  # (1,H,W)
        buf.append(t)
    x = torch.stack(buf, dim=0).unsqueeze(0)  # (1,T,1,H,W)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvLSTMUNet(in_channels=1, base=args.base).to(device).eval()
    load_ckpt(model, Path(args.ckpt), device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x.to(device)))[0,0].float().cpu().numpy()

    print("[DBG] prob stats:",
          "min", float(prob.min()),
          "mean", float(prob.mean()),
          "max", float(prob.max()),
          "p99", float(np.quantile(prob,0.99)),
          "p999", float(np.quantile(prob,0.999)))

    pred = (prob >= args.thr).astype(np.uint8)
    print("[DBG] pred_px @thr", args.thr, "=", int(pred.sum()))

    gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt = (gt>0).astype(np.uint8)
    print("[DBG] gt_px =", int(gt.sum()), "mask_path =", mask_path)

if __name__ == "__main__":
    main()
