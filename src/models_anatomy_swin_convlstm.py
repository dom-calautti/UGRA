# src/models_anatomy_swin_convlstm.py
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.nets.swin_unetr import SwinTransformer

from models_convlstm_unet import ConvLSTMCell, Up


@dataclass
class AnatomySwinConvLSTMConfig:
    """
    2D Swin encoder + ConvLSTM temporal bottleneck + UNet-ish decoder.
    Predicts LAST frame only (live-friendly).
    """
    in_channels: int = 1
    out_channels: int = 4

    img_size: int = 256
    patch_size: int = 4

    swin_embed_dim: int = 48
    swin_depths: Sequence[int] = (2, 2, 6, 2)
    swin_num_heads: Sequence[int] = (3, 6, 12, 24)
    swin_window_size: int | Sequence[int] = 7

    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    dropout_path_rate: float = 0.0
    use_checkpoint: bool = False

    convlstm_kernel: int = 3


class AnatomySwinEncoderConvLSTM(nn.Module):
    """
    Input:  x (B, T, 1, H, W)
    Output: logits (B, 4, H, W)

    per-frame Swin encoder -> ConvLSTM over time on deepest feature -> decode with last-frame skips.
    """
    def __init__(self, cfg: AnatomySwinConvLSTMConfig):
        super().__init__()
        self.cfg = cfg

        swin_window_size = self._as_2tuple(cfg.swin_window_size)
        swin_patch_size = self._as_2tuple(cfg.patch_size)

        # IMPORTANT: force 2D Swin (otherwise MONAI expects patch_size/window_size length 3).
        self.encoder = SwinTransformer(
            in_chans=cfg.in_channels,
            embed_dim=cfg.swin_embed_dim,
            window_size=swin_window_size,
            patch_size=swin_patch_size,
            depths=tuple(cfg.swin_depths),
            num_heads=tuple(cfg.swin_num_heads),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=cfg.drop_rate,
            attn_drop_rate=cfg.attn_drop_rate,
            drop_path_rate=cfg.dropout_path_rate,
            use_checkpoint=cfg.use_checkpoint,
            spatial_dims=2,
        )

        # Build these lazily from real Swin feature channel sizes.
        self.convlstm: Optional[ConvLSTMCell] = None
        self.up1: Optional[Up] = None
        self.up2: Optional[Up] = None
        self.up3: Optional[Up] = None
        self.outc: Optional[nn.Conv2d] = None

        self._built = False

    @staticmethod
    def _as_2tuple(v: int | Sequence[int]) -> Tuple[int, int]:
        if isinstance(v, (list, tuple)):
            if len(v) != 2:
                raise ValueError(f"Expected 2 values for 2D size, got {v}")
            return (int(v[0]), int(v[1]))
        return (int(v), int(v))

    def _encode_frame(self, xt: torch.Tensor) -> List[torch.Tensor]:
        """
        xt: (B,1,H,W)
        returns [s0, s1, s2, s3] each (B,C,H',W')
        """
        feats = self.encoder(xt)

        # Some MONAI versions return extra features; keep the last 4 hierarchy stages.
        if isinstance(feats, (list, tuple)) and len(feats) > 4:
            feats = feats[-4:]
        feats = list(feats)

        if len(feats) != 4:
            raise RuntimeError(f"Expected 4 Swin stage features, got {len(feats)}")

        # Ensure channel-first (should already be for spatial_dims=2).
        out = []
        for f in feats:
            if f.dim() != 4:
                raise RuntimeError(f"Unexpected feature shape: {tuple(f.shape)}")
            out.append(f)
        return out

    def _lazy_build(self, s0: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor):
        """
        Build ConvLSTM + decoder to match actual Swin channels.
        """
        c0 = int(s0.shape[1])
        c1 = int(s1.shape[1])
        c2 = int(s2.shape[1])
        c3 = int(s3.shape[1])

        self.convlstm = ConvLSTMCell(input_dim=c3, hidden_dim=c3, kernel_size=self.cfg.convlstm_kernel)

        # Decode: stage3 -> stage2 -> stage1 -> stage0 -> logits
        # Keep out_ch equal to skip channels for stability / simplicity.
        self.up1 = Up(in_ch_up=c3, in_ch_skip=c2, out_ch=c2)
        self.up2 = Up(in_ch_up=c2, in_ch_skip=c1, out_ch=c1)
        self.up3 = Up(in_ch_up=c1, in_ch_skip=c0, out_ch=c0)

        self.outc = nn.Conv2d(c0, self.cfg.out_channels, kernel_size=1)

        target_device = s3.device
        self.convlstm = self.convlstm.to(device=target_device)
        self.up1 = self.up1.to(device=target_device)
        self.up2 = self.up2.to(device=target_device)
        self.up3 = self.up3.to(device=target_device)
        self.outc = self.outc.to(device=target_device)

        self._built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"Expected (B,T,C,H,W), got {tuple(x.shape)}"
        B, T, C, H, W = x.shape
        device = x.device

        h = c = None
        skip0 = skip1 = skip2 = None

        for t in range(T):
            xt = x[:, t]  # (B,1,H,W)
            s0, s1, s2, s3 = self._encode_frame(xt)

            if not self._built:
                self._lazy_build(s0, s1, s2, s3)

            assert self.convlstm is not None
            assert self.up1 is not None and self.up2 is not None and self.up3 is not None
            assert self.outc is not None

            if h is None:
                _, c3, h3, w3 = s3.shape
                h, c = self.convlstm.init_state(B, (h3, w3), device=device)

            amp_off = torch.amp.autocast("cuda", enabled=False) if device.type == "cuda" else nullcontext()
            with amp_off:
                h, c = self.convlstm(s3.float(), (h.float(), c.float()))

            if t == T - 1:
                skip0, skip1, skip2 = s0, s1, s2

        y = self.up1(h, skip2)
        y = self.up2(y, skip1)
        y = self.up3(y, skip0)
        logits = self.outc(y)

        if logits.shape[-2:] != (H, W):
            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

        return logits
