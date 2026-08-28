"""
Lightweight direct-regression court keypoint model: ImageNet-pretrained ResNet18
backbone (truncated, fine-tuned) + small FC head predicting 14 (x, y) pairs.

Chose direct regression over a heatmap decoder because every image in this dataset
contains exactly one fully-visible court (confirmed during data inspection: all
sampled keypoints are within image bounds, no occlusion/visibility field in the
schema) -- a well-conditioned single-instance regression problem, so the extra
complexity of a heatmap decoder isn't buying much here.

Input:  (B, 3, S, S) ImageNet-normalized RGB
Output: (B, 2*N_KEYPOINTS) sigmoid, normalized (x, y) pairs in [0, 1]
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models

N_KEYPOINTS = 14


class CourtKeypointNet(nn.Module):
    def __init__(self, n_keypoints=N_KEYPOINTS, pretrained=True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2 * n_keypoints),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)


if __name__ == "__main__":
    m = CourtKeypointNet()
    n_params = sum(p.numel() for p in m.parameters())
    print(f"CourtKeypointNet params: {n_params / 1e6:.2f}M")
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    print("input:", x.shape, "-> output:", y.shape)
