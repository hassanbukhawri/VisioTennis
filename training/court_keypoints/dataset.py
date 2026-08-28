"""
Dataset loader for VisioTennis Model 1B (court keypoint regression).

data_train.json / data_val.json schema (confirmed by inspection before writing this
file -- see chat log):
    [
      {"id": "PuXlxKdUIes_2450", "metric": 0.283, "kps": [[x1,y1], [x2,y2], ..., [x14,y14]]},
      ...
    ]
- Always exactly 14 (x, y) keypoints per entry, absolute pixel coordinates.
- Images are at data/images/{id}.png, all confirmed 1280x720, all referenced ids exist
  (checked missing-image count = 0 on a 200-entry sample per split).
- `metric` is an annotation-quality score in the source data; unused for training here.
- data_train.json: 6630 entries, data_val.json: 2211 entries. NOTE: this is a frame-level
  split, not a video-level one -- 434 of 442 val "videos" (id prefix before the trailing
  frame number) also appear in train, so val metrics will run somewhat optimistic. Used
  as shipped per user decision (matches the public TennisCourtDetector dataset convention
  this resembles); not re-split here.
- This is NOT YOLO-pose format (no bbox/visibility fields) -- confirmed before writing,
  hence the custom direct-regression model in model.py rather than YOLOv8-pose.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

KP_ROOT = os.path.join("data", "VisioTennis", "Dataset", "Court_Detector_Keypoints", "data")
NATIVE_W, NATIVE_H = 1280, 720
N_KEYPOINTS = 14

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_entries(json_path):
    with open(json_path) as f:
        entries = json.load(f)
    for e in entries:
        assert len(e["kps"]) == N_KEYPOINTS, f"expected {N_KEYPOINTS} kps, got {len(e['kps'])} for id={e['id']}"
    return entries


class CourtKeypointsDataset(Dataset):
    """
    Returns:
        image:  (3, out_size, out_size) float32, ImageNet-normalized RGB
        kps:    (2*N_KEYPOINTS,) float32, (x, y) pairs flattened, normalized to [0, 1]
                by dividing by native (W, H) -- independent of any resize/crop since we
                do a plain resize (no cropping) to a square input.
    """

    def __init__(self, json_path, images_dir=None, out_size=224, max_samples=None):
        self.entries = load_entries(json_path)
        self.images_dir = images_dir or os.path.join(KP_ROOT, "images")
        self.out_size = out_size

        if max_samples is not None and max_samples < len(self.entries):
            import random
            rng = random.Random(0)
            self.entries = rng.sample(self.entries, max_samples)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        img_path = os.path.join(self.images_dir, e["id"] + ".png")
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"cv2 failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = np.transpose(img, (2, 0, 1))  # (3, out_size, out_size)

        kps = np.asarray(e["kps"], dtype=np.float32)  # (14, 2)
        kps_norm = kps.copy()
        kps_norm[:, 0] /= NATIVE_W
        kps_norm[:, 1] /= NATIVE_H
        kps_norm = kps_norm.reshape(-1)  # (28,)

        return (
            torch.from_numpy(np.ascontiguousarray(img)).float(),
            torch.from_numpy(kps_norm).float(),
        )
