"""
Task A (data augmentation) -- builds an augmented copy of the clean-split
(player-disjoint) TRAINING set only, simulating the diagnosed real-world
domain-shift gap (PROJECT_LOG.md, 2026-08-12: THETIS was recorded at fixed
~1.5m indoor range vs. this project's real footage, which is further/
variable-distance, natural outdoor lighting, consumer/phone camera).

Augments ONLY train (val/test stay on the original clean-split data, copied
unchanged) -- augmented synthetic data has no business being in a held-out
evaluation split. Runs each raw THETIS video (data/.../raw_rgb/<native_class>/
<file>.avi) through the SAME decode_and_resample_video + run_pose_extraction
functions backend/app.py uses on a real upload, so the augmented pose
features are extracted by the identical code path production runs on real
footage -- not a separately-maintained extraction pipeline that could drift
from what's actually deployed.

Augmentations applied, chosen to simulate the SPECIFIC diagnosed gap (not a
generic "augmentation library defaults" approach):
  - zoom_out: shrinks the frame content and letterboxes it back to full
    size on a neutral gray canvas -- simulates a subject occupying less of
    the frame at greater camera distance (the single biggest diagnosed
    factor). Scale is randomized per clip within [0.45, 0.75], consistent
    across all 60 frames of that clip (a real camera doesn't randomly
    re-zoom mid-swing).
  - color_jitter: per-clip brightness/contrast/saturation shift -- outdoor
    daylight vs. indoor Kinect capture.
  - motion_blur: mild per-clip linear blur kernel -- handheld/consumer
    camera vs. THETIS's fixed static rig.
  - noise: light additive Gaussian noise -- consumer camera sensor noise /
    compression, vs. Kinect's clean RGB stream.
  - background_clutter (explicitly a PROXY, not a real simulation of
    people moving in the background -- documented honestly, not oversold):
    2-3 semi-transparent blobs on a smooth per-clip random-walk trajectory,
    confined to the outer thirds of the frame so they don't occlude the
    subject's own pose. A real second person walking through frame was not
    attempted -- no such background-actor footage/asset exists in this
    repo to composite from, and synthesizing a convincing one is out of
    scope for this investigation pass.

Usage:
    python training/stroke_classification/build_augmented_train_set.py [--n-aug-per-clip 1]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from backend.app import decode_and_resample_video, run_pose_extraction, POSE_TASK_PATH, SEQ_LEN

CLEAN_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
RAW_RGB_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "raw_rgb")
OUT_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean_augmented")


def zoom_out(frame, scale, fill_value=128):
    h, w = frame.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(frame, fill_value)
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = small
    return canvas


def color_jitter(frame, brightness_delta, contrast_factor, saturation_factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    frame = frame.astype(np.float32) * contrast_factor + brightness_delta
    return np.clip(frame, 0, 255).astype(np.uint8)


def motion_blur(frame, kernel_size, angle_deg):
    if kernel_size <= 1:
        return frame
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    center = (kernel_size / 2 - 0.5, kernel_size / 2 - 0.5)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (kernel_size, kernel_size))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(frame, -1, kernel)


def add_noise(frame, sigma, rng):
    noise = rng.normal(0, sigma, frame.shape)
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def make_blob_trajectories(n_frames, h, w, rng, n_blobs=3):
    trajectories = []
    for _ in range(n_blobs):
        edge = rng.choice(["left", "right"])
        x = rng.uniform(0, w * 0.15) if edge == "left" else rng.uniform(w * 0.85, w)
        y = rng.uniform(h * 0.2, h * 0.8)
        vx = rng.uniform(-1.5, 1.5)
        vy = rng.uniform(-0.5, 0.5)
        radius = rng.uniform(15, 35)
        color = rng.integers(40, 200, size=3)
        alpha = rng.uniform(0.15, 0.35)
        pts = []
        for _ in range(n_frames):
            pts.append((x, y))
            x += vx
            y += vy
        trajectories.append((pts, radius, color, alpha))
    return trajectories


def apply_background_clutter(frames, rng):
    h, w = frames[0].shape[:2]
    trajectories = make_blob_trajectories(len(frames), h, w, rng)
    out = []
    for i, frame in enumerate(frames):
        overlay = frame.copy()
        for pts, radius, color, alpha in trajectories:
            x, y = pts[i]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(overlay, (int(x), int(y)), int(radius), tuple(int(c) for c in color), -1)
        out.append(cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0))
    return out


def augment_clip(frames, rng):
    """One coherent augmentation applied consistently across all frames of a clip."""
    scale = rng.uniform(0.45, 0.75)
    brightness_delta = rng.uniform(-35, 35)
    contrast_factor = rng.uniform(0.8, 1.25)
    saturation_factor = rng.uniform(0.6, 1.3)
    blur_kernel = rng.choice([1, 1, 3, 5])  # often none/light, occasionally more
    blur_angle = rng.uniform(0, 180)
    noise_sigma = rng.uniform(3, 10)

    frames = apply_background_clutter(frames, rng)
    out = []
    for frame in frames:
        f = zoom_out(np.array(frame), scale)
        f = color_jitter(f, brightness_delta, contrast_factor, saturation_factor)
        f = motion_blur(f, blur_kernel, blur_angle)
        f = add_noise(f, noise_sigma, rng)
        out.append(f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-aug-per-clip", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    labels = pd.read_csv(os.path.join(CLEAN_ROOT, "train", "labels_train.csv"))
    print(f"[setup] {len(labels)} train clips to augment, {args.n_aug_per_clip}x each")

    print("[setup] loading MediaPipe Pose landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    rng = np.random.default_rng(args.seed)
    aug_X, aug_rows = [], []
    n_pose_fail = 0
    t_start = time.time()

    for i, row in labels.iterrows():
        raw_path = os.path.join(RAW_RGB_ROOT, row["label"], row["filename"])
        if not os.path.exists(raw_path):
            print(f"[skip] missing raw video: {raw_path}")
            continue
        try:
            frames, fps, n_real = decode_and_resample_video(raw_path)
        except ValueError as e:
            print(f"[skip] decode failed for {row['filename']}: {e}")
            continue

        for k in range(args.n_aug_per_clip):
            aug_frames = augment_clip(frames, rng)
            feats, n_detected = run_pose_extraction(aug_frames, landmarker)
            if n_detected < SEQ_LEN * 0.5:  # augmentation degraded detection past usefulness
                n_pose_fail += 1
                continue
            aug_X.append(feats)
            aug_rows.append({
                "filename": f"aug{k}_{row['filename']}",
                "label": row["label"],
                "label_idx": row["label_idx"],
            })

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(labels) - i - 1) / rate
            print(f"[progress] {i+1}/{len(labels)} clips, {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s, {n_pose_fail} augmented instances dropped "
                  f"(pose detection <50% after augmentation)", flush=True)

    aug_X = np.stack(aug_X, axis=0)
    aug_labels = pd.DataFrame(aug_rows)
    print(f"\n[done extracting] {aug_X.shape[0]} augmented clips built "
          f"({n_pose_fail} dropped for degraded pose detection)")

    # Combine with the original clean-split train set (originals + augmented copies).
    orig_X = np.load(os.path.join(CLEAN_ROOT, "train", "X_train.npy"))
    orig_labels = pd.read_csv(os.path.join(CLEAN_ROOT, "train", "labels_train.csv"))

    combined_X = np.concatenate([orig_X, aug_X], axis=0)
    combined_labels = pd.concat([orig_labels, aug_labels], ignore_index=True)

    os.makedirs(os.path.join(OUT_ROOT, "train"), exist_ok=True)
    np.save(os.path.join(OUT_ROOT, "train", "X_train.npy"), combined_X)
    combined_labels.to_csv(os.path.join(OUT_ROOT, "train", "labels_train.csv"), index=False)
    print(f"[write] train: {combined_X.shape[0]} clips "
          f"({orig_X.shape[0]} original + {aug_X.shape[0]} augmented) -> {OUT_ROOT}/train")

    # val/test: copy unchanged from processed_clean -- augmented data must never appear
    # in a held-out evaluation split.
    for split in ("val", "test"):
        src_dir = os.path.join(CLEAN_ROOT, split)
        dst_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(dst_dir, exist_ok=True)
        X = np.load(os.path.join(src_dir, f"X_{split}.npy"))
        lbl = pd.read_csv(os.path.join(src_dir, f"labels_{split}.csv"))
        np.save(os.path.join(dst_dir, f"X_{split}.npy"), X)
        lbl.to_csv(os.path.join(dst_dir, f"labels_{split}.csv"), index=False)
        print(f"[copy] {split}: {X.shape[0]} clips (unchanged from processed_clean) -> {dst_dir}")

    print("\n[class balance] combined train set, collapsed-native label counts:")
    print(combined_labels["label_idx"].value_counts().sort_index())


if __name__ == "__main__":
    main()
