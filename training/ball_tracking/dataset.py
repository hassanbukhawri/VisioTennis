"""
Dataset loader for VisioTennis Model 1A (TrackNet-style ball heatmap regression).

Label.csv schema (confirmed by inspecting game1/Clip1, game5/Clip8, game10/Clip3,
plus a full scan of all 95 clips / 19,835 rows before writing this file):

    file name,visibility,x-coordinate,y-coordinate,status
    0000.jpg,1,599,423,0

- visibility: 0 = ball not visible, 1 = clearly visible, 2 = occluded, 3 = motion-blurred.
  Full-dataset counts: {1: 17632, 2: 1392, 0: 729, 3: 82}.
- status is blank iff visibility == 0 (verified with 0 mismatches across all 19,835 rows);
  otherwise status in {0, 1, 2} = {normal flight, hit, bounce}. Model 1A's TrackNet
  training (train.py) does NOT use `status` -- it's a per-frame ball-EVENT label, a
  different task than TrackNet's position-heatmap regression, and was intentionally
  left out of BallTrackingDataset's training tuples to avoid changing the already-
  validated training pipeline's interface. It IS now parsed into ClipRecord.status
  (see IR population code, ir/populate.py) since it's real ground-truth information
  that shouldn't be silently discarded when building IR records from these labeled
  clips. Distribution across a 30-clip sample (4959 frames): status=0 (flight) 92.0%,
  status=1 (hit) 2.8%, status=2 (bounce) 2.7%; NaN count matched visibility==0 count
  exactly (120/120) -- clean, complete labeling with no unexplained gaps.
- Every Label.csv row count matches the number of .jpg files in its own clip directory
  (verified on a sample of clips), and every referenced frame file exists on disk.
- All sampled clips are native 1280x720 RGB (checked across all 95 clips' first frame:
  100% are 1280x720).
- 10 games, 95 clips total, 19,835 frames total; per-game frame counts range ~1570-2240,
  so a game-level train/val split is reasonably balanced without any special weighting.

Ground truth (x, y) is only meaningful when visibility != 0; when visibility == 0 the
Gaussian heatmap target is all-zero (the model should learn to predict "no ball").
"""
import csv
import glob
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

BALL_ROOT = os.path.join("data", "VisioTennis", "Dataset", "Ball Tracker+Bounce")
NATIVE_W, NATIVE_H = 1280, 720


@dataclass
class ClipRecord:
    game: str
    clip: str
    frame_files: list = field(default_factory=list)   # sorted list of image paths, len L
    visibility: np.ndarray = None                       # (L,) int64, raw code 0..3
    xy: np.ndarray = None                                # (L, 2) float32 native px, NaN if invisible
    status: np.ndarray = None                            # (L,) float32, {0,1,2}={flight,hit,bounce}, NaN iff visibility==0


def _read_label_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def discover_clips(root=BALL_ROOT):
    """Scan the dataset root and build one ClipRecord per game*/Clip* folder.

    Raises if a Label.csv references a frame file that doesn't exist on disk --
    we do not silently skip missing frames.
    """
    clips = []
    label_paths = sorted(glob.glob(os.path.join(root, "game*", "Clip*", "Label.csv")))
    if not label_paths:
        raise FileNotFoundError(f"No Label.csv files found under {root!r}")

    for label_path in label_paths:
        clip_dir = os.path.dirname(label_path)
        game = os.path.basename(os.path.dirname(clip_dir))
        clip = os.path.basename(clip_dir)
        rows = _read_label_csv(label_path)

        frame_files, vis, xy, status = [], [], [], []
        for r in rows:
            fp = os.path.join(clip_dir, r["file name"])
            if not os.path.isfile(fp):
                raise FileNotFoundError(f"Label.csv references missing frame: {fp}")
            frame_files.append(fp)
            v = int(r["visibility"])
            vis.append(v)
            if v == 0:
                xy.append((np.nan, np.nan))
            else:
                xy.append((float(r["x-coordinate"]), float(r["y-coordinate"])))
            s = r.get("status", "")
            status.append(float(s) if s not in ("", None) else np.nan)

        clips.append(ClipRecord(
            game=game, clip=clip,
            frame_files=frame_files,
            visibility=np.asarray(vis, dtype=np.int64),
            xy=np.asarray(xy, dtype=np.float32),
            status=np.asarray(status, dtype=np.float32),
        ))
    return clips


def split_by_game(clips, val_games=None, val_fraction=0.2, seed=42):
    """Split clips into train/val by GAME, never by frame -- avoids leaking frames
    from the same rally/game between train and val (they're highly correlated)."""
    games = sorted({c.game for c in clips}, key=lambda g: int(g.replace("game", "")))
    if val_games is None:
        rng = random.Random(seed)
        n_val = max(1, round(len(games) * val_fraction))
        val_games = set(rng.sample(games, n_val))
    else:
        val_games = set(val_games)
    train_clips = [c for c in clips if c.game not in val_games]
    val_clips = [c for c in clips if c.game in val_games]
    if not val_clips:
        raise ValueError("val split produced 0 clips -- check val_games/val_fraction")
    return train_clips, val_clips, sorted(val_games, key=lambda g: int(g.replace("game", "")))


def make_gaussian_heatmap(x, y, out_w, out_h, sigma=3.0):
    """2D Gaussian heatmap peaked at (x, y), already in OUTPUT-resolution coords.
    Returns an all-zero heatmap if x or y is NaN (ball not visible in this frame)."""
    if np.isnan(x) or np.isnan(y):
        return np.zeros((out_h, out_w), dtype=np.float32)
    xs = np.arange(out_w, dtype=np.float32)
    ys = np.arange(out_h, dtype=np.float32)
    gx = np.exp(-((xs - x) ** 2) / (2 * sigma ** 2))
    gy = np.exp(-((ys - y) ** 2) / (2 * sigma ** 2))
    return np.outer(gy, gx).astype(np.float32)


class BallTrackingDataset(Dataset):
    """
    Sliding window of `seq_len` consecutive frames, never crossing a clip boundary.

    Returns:
        frame_stack:   (3*seq_len, H, W) float32 in [0, 1], channel-stacked RGB frames
        heatmap_stack: (seq_len, H, W)   float32 in [0, 1], one Gaussian heatmap per frame
        visibility:    (seq_len,)        int64, raw visibility code (0..3) per frame
    """

    def __init__(self, clips, seq_len=3, out_size=(512, 288), sigma=3.0, stride=1,
                 max_samples=None):
        assert len(clips) > 0, "BallTrackingDataset got an empty clip list"
        self.seq_len = seq_len
        self.out_w, self.out_h = out_size
        self.sigma = sigma

        self.index = []  # list of (ClipRecord, start_idx)
        for c in clips:
            L = len(c.frame_files)
            for start in range(0, L - seq_len + 1, stride):
                self.index.append((c, start))

        if not self.index:
            raise ValueError(
                f"No windows of length {seq_len} could be built (shortest clip too short)."
            )

        if max_samples is not None and max_samples < len(self.index):
            rng = random.Random(0)
            self.index = rng.sample(self.index, max_samples)

    def __len__(self):
        return len(self.index)

    def _load_frame(self, path):
        img = cv2.imread(path)
        if img is None:
            raise IOError(f"cv2 failed to read frame: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0  # (H, W, 3)

    def __getitem__(self, idx):
        clip, start = self.index[idx]
        sx = self.out_w / NATIVE_W
        sy = self.out_h / NATIVE_H

        frames, heatmaps, vis_flags = [], [], []
        for i in range(start, start + self.seq_len):
            frames.append(self._load_frame(clip.frame_files[i]))
            x, y = clip.xy[i]
            heatmaps.append(make_gaussian_heatmap(x * sx, y * sy, self.out_w, self.out_h, self.sigma))
            vis_flags.append(int(clip.visibility[i]))

        frame_stack = np.concatenate(frames, axis=-1)        # (H, W, 3*seq_len)
        frame_stack = np.transpose(frame_stack, (2, 0, 1))   # (3*seq_len, H, W)
        heatmap_stack = np.stack(heatmaps, axis=0)            # (seq_len, H, W)

        return (
            torch.from_numpy(np.ascontiguousarray(frame_stack)).float(),
            torch.from_numpy(heatmap_stack).float(),
            torch.tensor(vis_flags, dtype=torch.int64),
        )
