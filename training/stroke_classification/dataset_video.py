"""
Video-clip Dataset for Path B (r2plus1d_18 fine-tuning), reading directly
from the raw THETIS .avi files listed in the player-grouped manifests
written by prepare_video_split.py (training/stroke_classification/
video_splits/{train,val,test}.csv) -- no re-encoding, no intermediate
tensor cache, decoded on the fly via OpenCV.

Preprocessing matches torchvision's own R2Plus1D_18_Weights.DEFAULT.transforms()
recipe exactly (checked directly via `.transforms()`, not assumed from
memory): resize to 128(H)x171(W) -- which happens to be the same 4:3 aspect
ratio as THETIS's native 640x480 frames, so this is a direct resize, no
letterboxing needed -- then a 112x112 crop, normalized with Kinetics-400
mean/std. Train uses a random temporal window + random spatial crop + random
horizontal flip (same crop/flip decision applied to all 16 frames of a clip,
not per-frame, since a clip must stay spatially/temporally coherent). Val/
test use a deterministic center window + center crop, no flip.

Every THETIS clip has >=24 frames (confirmed by inspection across all 1980
files before writing this), comfortably above CLIP_LEN=16, so the random/
center window always has a valid range -- the short-clip pad-by-repeat path
below is a defensive fallback (e.g. a corrupt read), not an expected case.
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SPLITS_DIR = os.path.join(SCRIPT_DIR, "video_splits")

COLLAPSED_CLASSES = ["forehand", "backhand", "serve", "volley", "smash"]

CLIP_LEN = 16
RESIZE_HW = (128, 171)  # (H, W) -- matches R2Plus1D_18_Weights.DEFAULT.transforms().resize_size
CROP_HW = (112, 112)    # matches .crop_size
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def _read_clip_frames(filepath, start, clip_len):
    """Reads clip_len contiguous frames starting at `start`, BGR uint8 (T, H, W, 3).
    Pads by repeating the last frame if the video runs out early (defensive only,
    see module docstring -- not expected for THETIS's real frame counts)."""
    cap = cv2.VideoCapture(filepath)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(clip_len):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"could not read any frames from {filepath} (start={start})")
    while len(frames) < clip_len:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


class VideoClipDataset(Dataset):
    def __init__(self, split, splits_dir=SPLITS_DIR, clip_len=CLIP_LEN, max_samples=None, seed=0):
        assert split in ("train", "val", "test")
        self.split = split
        self.train_mode = split == "train"
        self.clip_len = clip_len
        self.rng = np.random.default_rng(seed)

        manifest_path = os.path.join(splits_dir, f"{split}.csv")
        df = pd.read_csv(manifest_path)
        if max_samples is not None:
            # Stratified, not df.iloc[:max_samples] -- the manifest is grouped by
            # native class in alphabetical order (all "backhand"-collapsed rows
            # first), so a plain head-slice produced a degenerate single-class
            # smoke subset (found by inspection of a real smoke-test run: class
            # weights came out [inf, w, inf, inf, inf], macro-F1 pinned at 0.2).
            # Round-robin across collapsed classes instead, so even a tiny smoke
            # subset exercises real multi-class supervision.
            per_class = max(1, max_samples // len(COLLAPSED_CLASSES))
            parts = [group.head(per_class) for _, group in df.groupby("collapsed_idx", sort=True)]
            df = pd.concat(parts, ignore_index=True).head(max_samples).reset_index(drop=True)
        self.df = df

    def __len__(self):
        return len(self.df)

    def class_counts(self):
        counts = torch.zeros(len(COLLAPSED_CLASSES), dtype=torch.long)
        for idx in self.df["collapsed_idx"]:
            counts[idx] += 1
        return counts

    def _sample_window(self, n_frames):
        max_start = max(n_frames - self.clip_len, 0)
        if self.train_mode:
            return int(self.rng.integers(0, max_start + 1))
        return max_start // 2  # deterministic center window

    def _sample_crop_offset(self):
        max_y = RESIZE_HW[0] - CROP_HW[0]
        max_x = RESIZE_HW[1] - CROP_HW[1]
        if self.train_mode:
            return int(self.rng.integers(0, max_y + 1)), int(self.rng.integers(0, max_x + 1))
        return max_y // 2, max_x // 2  # deterministic center crop

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"]
        label = int(row["collapsed_idx"])

        cap = cv2.VideoCapture(filepath)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        start = self._sample_window(n_frames)
        frames = _read_clip_frames(filepath, start, self.clip_len)  # (T, H, W, 3) BGR uint8

        y0, x0 = self._sample_crop_offset()
        flip = self.train_mode and self.rng.random() < 0.5

        processed = np.empty((self.clip_len, CROP_HW[0], CROP_HW[1], 3), dtype=np.float32)
        for t in range(self.clip_len):
            frame = cv2.cvtColor(frames[t], cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (RESIZE_HW[1], RESIZE_HW[0]), interpolation=cv2.INTER_LINEAR)
            frame = frame[y0:y0 + CROP_HW[0], x0:x0 + CROP_HW[1], :]
            if flip:
                frame = frame[:, ::-1, :]
            processed[t] = frame.astype(np.float32) / 255.0

        processed = (processed - KINETICS_MEAN) / KINETICS_STD
        # (T, H, W, C) -> (C, T, H, W), the standard torchvision video model input layout
        clip = torch.from_numpy(processed).permute(3, 0, 1, 2).float()
        return clip, label


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = VideoClipDataset(split)
        x, y = ds[0]
        print(f"{split}: {len(ds)} samples, class_counts={ds.class_counts().tolist()}, "
              f"sample clip shape={tuple(x.shape)} label={y}")
