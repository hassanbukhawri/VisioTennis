"""
Dataset for VisioTennis Model 2: stroke classification.

Source: data/VisioTennis/Dataset/StrokeClassification/processed/{split}/X_{split}.npy
+ labels_{split}.csv. Confirmed by inspection (see chat): X_*.npy is
(n_samples, 60, 99) float32 -- 60-frame sequences of 99-dim MediaPipe Pose
landmarks (33 keypoints x (x, y, z)), already normalized, no NaNs. This is the
public THETIS dataset's 12 native stroke classes (confirmed via labels_*.csv
filenames, e.g. "p47_serflat_s3.avi", "p20_serkick_s3.avi").

The FYP proposal targets 5 categories (forehand, backhand, serve, volley,
smash) with F1>82%, not the native 12. NATIVE_TO_COLLAPSED below is the
user-confirmed mapping (decided in chat, collapse-at-training, not
collapse-at-reporting): every native class maps to exactly one of the 5
proposal categories, with volley pulled out as its own category (forehand_volley
+ backhand_volley) rather than folded into forehand/backhand.

Collapsed class distribution (title -> clip count, all splits): forehand=495,
backhand=495, serve=495, volley=330, smash=165 -- smash is a real 3:1 minority
class relative to the three largest categories; train.py handles this via a
class-weighted loss, not by resampling/dropping data.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_ROOT = os.path.join("data", "VisioTennis", "Dataset", "StrokeClassification", "processed")

COLLAPSED_CLASSES = ["forehand", "backhand", "serve", "volley", "smash"]

# native label_idx (0-11, per class_map.csv) -> collapsed class index (0-4)
NATIVE_TO_COLLAPSED = {
    0: 1,   # backhand            -> backhand
    1: 1,   # backhand2hands      -> backhand
    2: 1,   # backhand_slice      -> backhand
    3: 3,   # backhand_volley     -> volley
    4: 2,   # flat_service        -> serve
    5: 0,   # forehand_flat       -> forehand
    6: 0,   # forehand_openstands -> forehand
    7: 0,   # forehand_slice      -> forehand
    8: 3,   # forehand_volley     -> volley
    9: 2,   # kick_service        -> serve
    10: 2,  # slice_service       -> serve
    11: 4,  # smash               -> smash
}

N_KEYPOINT_FEATURES = 99
SEQ_LEN = 60


class StrokeDataset(Dataset):
    def __init__(self, split, data_root=DATA_ROOT, max_samples=None):
        assert split in ("train", "val", "test")
        x_path = os.path.join(data_root, split, f"X_{split}.npy")
        lbl_path = os.path.join(data_root, split, f"labels_{split}.csv")

        X = np.load(x_path)
        labels = pd.read_csv(lbl_path)
        assert X.shape[0] == len(labels), f"{x_path} has {X.shape[0]} rows but {lbl_path} has {len(labels)}"
        assert X.shape[1:] == (SEQ_LEN, N_KEYPOINT_FEATURES), f"unexpected feature shape {X.shape[1:]}"

        collapsed = labels["label_idx"].map(NATIVE_TO_COLLAPSED)
        assert collapsed.isna().sum() == 0, "label_idx values outside the known 0-11 native class range"

        if max_samples is not None:
            X = X[:max_samples]
            collapsed = collapsed.iloc[:max_samples]

        self.X = torch.from_numpy(X).float()
        self.y = torch.tensor(collapsed.values, dtype=torch.long)
        self.native_label_idx = torch.tensor(labels["label_idx"].values[:len(self.y)], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def class_counts(self):
        counts = torch.zeros(len(COLLAPSED_CLASSES), dtype=torch.long)
        for c in self.y:
            counts[c] += 1
        return counts


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = StrokeDataset(split)
        print(f"{split}: {len(ds)} samples, X shape {tuple(ds.X.shape)}, class_counts={ds.class_counts().tolist()}")
