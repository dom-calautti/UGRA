from pathlib import Path
from torch.utils.data import DataLoader

from datasets_temporal import TemporalNeedleDataset, temporal_collate

def main():
    root = Path("data_processed/meta")
    train_manifest = root / "sequences_needle_train.json"

    ds = TemporalNeedleDataset(train_manifest, expected_T=8, normalize=True)
    print("[OK] Train samples:", len(ds))

    loader = DataLoader(
        ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=temporal_collate,
    )

    x, y, meta = next(iter(loader))
    print("[OK] x shape:", tuple(x.shape))   # expected (B, T, 1, H, W)
    print("[OK] y shape:", tuple(y.shape))   # expected (B, 1, H, W)
    print("[OK] first meta:", meta[0])
    print("[OK] y unique (first):", y[0].unique())

if __name__ == "__main__":
    main()
