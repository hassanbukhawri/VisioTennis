"""
Sequence classifier for VisioTennis Model 2: stroke classification.

Input:  (B, 60, 99) float32 -- 60-frame MediaPipe Pose keypoint sequences.
Output: (B, 5) logits over the collapsed classes (see dataset.py COLLAPSED_CLASSES).

Bidirectional GRU over the frame sequence (cheap, well-suited to short fixed-length
keypoint sequences on a 3060 -- no need for the two-stream video fallback given this
data is already clean pose sequences), mean+max pooled over time, small FC head.
"""
import torch
import torch.nn as nn


class StrokeGRU(nn.Module):
    def __init__(self, input_dim=99, hidden_dim=128, num_layers=2, num_classes=5, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        feat_dim = hidden_dim * 2  # bidirectional
        self.head = nn.Sequential(
            nn.Linear(feat_dim * 2, 128),  # mean-pool + max-pool concatenated
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        out, _ = self.gru(x)          # (B, T, 2*hidden_dim)
        mean_pool = out.mean(dim=1)
        max_pool, _ = out.max(dim=1)
        feat = torch.cat([mean_pool, max_pool], dim=1)
        return self.head(feat)


if __name__ == "__main__":
    m = StrokeGRU()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"StrokeGRU params: {n_params / 1e6:.2f}M")
    x = torch.randn(4, 60, 99)
    y = m(x)
    print("input:", x.shape, "-> output:", y.shape)
