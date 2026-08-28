"""
TrackNet-style encoder-decoder (U-Net with skip connections) for ball heatmap
regression, following the standard TrackNetV2 "N-in-N-out" design: stack N
consecutive RGB frames on the channel axis, predict one heatmap per frame.

Input:  (B, 3*seq_len, H, W)  float32, RGB frames stacked on channel dim, [0, 1]
Output: (B, seq_len, H, W)    float32, per-frame ball heatmaps, sigmoid in [0, 1]
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, n=2):
    layers = []
    for i in range(n):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    def __init__(self, seq_len=3, base_ch=32):
        super().__init__()
        self.seq_len = seq_len
        in_ch = 3 * seq_len
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = conv_block(in_ch, c1, n=2)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(c1, c2, n=2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(c2, c3, n=3)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(c3, c4, n=3)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = conv_block(c4 + c3, c3, n=3)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = conv_block(c3 + c2, c2, n=2)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = conv_block(c2 + c1, c1, n=2)

        self.head = nn.Conv2d(c1, seq_len, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.head(d1))


if __name__ == "__main__":
    m = TrackNet(seq_len=3, base_ch=32)
    n_params = sum(p.numel() for p in m.parameters())
    print(f"TrackNet params: {n_params / 1e6:.2f}M")
    x = torch.randn(2, 9, 288, 512)
    y = m(x)
    print("input:", x.shape, "-> output:", y.shape)
