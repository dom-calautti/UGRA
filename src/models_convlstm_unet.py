from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch_up: int, in_ch_skip: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch_up, in_ch_up, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch_up + in_ch_skip, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM cell:
      input:  (B, C_in, H, W)
      state:  h,c each (B, C_hidden, H, W)
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)  # (B, C_in + C_hid, H, W)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch_size: int, spatial_size, device):
        H, W = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=device)
        return h, c


class ConvLSTMUNet(nn.Module):
    """
    Input:  x (B, T, 1, H, W)
    Output:
      - needle: logits (B, 1, H, W)
      - anatomy: logits (B, 4, H, W) where classes 0..3
    """
    def __init__(self, in_channels: int = 1, base: int = 32, out_channels: int = 1):
        super().__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)

        self.bot_conv = DoubleConv(base * 8, base * 8)

        self.convlstm = ConvLSTMCell(input_dim=base * 8, hidden_dim=base * 8, kernel_size=3)

        # Decoder
        self.up1 = Up(base * 8, base * 4, base * 4)
        self.up2 = Up(base * 4, base * 2, base * 2)
        self.up3 = Up(base * 2, base, base)

        self.outc = nn.Conv2d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, f"Expected (B,T,C,H,W), got {tuple(x.shape)}"
        B, T, C, H, W = x.shape
        device = x.device

        skip1 = skip2 = skip3 = None
        h, c = self.convlstm.init_state(B, (H // 8, W // 8), device=device)

        for t in range(T):
            xt = x[:, t]

            x1 = self.inc(xt)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x4 = self.bot_conv(x4)

            h, c = self.convlstm(x4, (h, c))

            if t == T - 1:
                skip1, skip2, skip3 = x1, x2, x3

        x = self.up1(h, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        logits = self.outc(x)
        return logits
