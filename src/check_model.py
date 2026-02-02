import torch
from models_convlstm_unet import ConvLSTMUNet

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvLSTMUNet(in_channels=1, base=32).to(device)
    x = torch.randn(2, 8, 1, 640, 640, device=device)  # (B,T,C,H,W)
    with torch.no_grad():
        y = model(x)
    print("out shape:", y.shape)  # expect (2,1,640,640)

if __name__ == "__main__":
    main()
