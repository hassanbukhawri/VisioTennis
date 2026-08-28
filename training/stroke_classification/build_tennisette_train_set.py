"""
Task A+B combined: extracts pose features from the real TenniSet broadcast
clips (data/TenniSet/extracted/manifest.csv, built by the TenniSet
extraction pass) and folds them into a training set alongside the existing
THETIS-derived clean-split (+ THETIS-augmented) train data -- the actual
test of this session's hypothesis: does real broadcast footage, not just
augmented THETIS, fix the serve failure.

Applies the SAME augmentation pipeline built for Task A
(augment_clip from build_augmented_train_set.py) to the REAL TenniSet clips
too, not just THETIS -- one augmented copy per real clip, doubling the real
data's effective volume the same way THETIS's train set was doubled, so any
improvement can't be attributed to "more real data" alone vs. "the
augmentation approach also generalizes to real footage."

TenniSet's classes (forehand/backhand/serve only, volley/smash excluded
per the investigation's own finding) are 5-class-native already -- StrokeDataset
expects a 12-class "native" label_idx it collapses via NATIVE_TO_COLLAPSED
(dataset.py), so each TenniSet row is tagged with ONE representative native
THETIS label string/idx that collapses to the correct 5-class target
(cosmetic bookkeeping only, not real provenance -- documented here so it's
never mistaken for an actual THETIS clip):
    forehand -> "forehand_flat" (native idx 5 -> collapsed 0)
    backhand -> "backhand"      (native idx 0 -> collapsed 1)
    serve    -> "flat_service"  (native idx 4 -> collapsed 2)

Usage:
    python training/stroke_classification/build_tennisette_train_set.py
"""
import os
import sys
import time

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
from build_augmented_train_set import augment_clip

MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "TenniSet", "extracted", "manifest.csv")
AUGMENTED_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean_augmented")
OUT_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean_augmented_tennisette")

CLASS_TO_NATIVE = {
    "forehand": ("forehand_flat", 5),
    "backhand": ("backhand", 0),
    "serve": ("flat_service", 4),
}


def main():
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"manifest not found at {MANIFEST_PATH} -- run the TenniSet extraction first")

    manifest = pd.read_csv(MANIFEST_PATH)
    train_rows = manifest[manifest["split"] == "train"].reset_index(drop=True)
    print(f"[setup] {len(train_rows)} TenniSet train clips to process "
          f"(classes: {train_rows['label'].value_counts().to_dict()})")

    print("[setup] loading MediaPipe Pose landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    rng = np.random.default_rng(43)  # different seed than THETIS augmentation, real independence
    real_X, aug_X, rows = [], [], []
    n_decode_fail, n_pose_fail, n_aug_pose_fail = 0, 0, 0
    t_start = time.time()

    for i, row in train_rows.iterrows():
        clip_path = row["filepath"]
        if not os.path.isabs(clip_path):
            clip_path = os.path.join(REPO_ROOT, "data", "TenniSet", clip_path)
        if not os.path.exists(clip_path):
            print(f"[skip] missing clip: {clip_path}")
            continue

        try:
            frames, fps, n_real = decode_and_resample_video(clip_path)
        except ValueError as e:
            n_decode_fail += 1
            print(f"[skip] decode failed for {clip_path}: {e}")
            continue

        native_label, native_idx = CLASS_TO_NATIVE[row["label"]]

        # Real (unaugmented) clip.
        feats, n_detected = run_pose_extraction(frames, landmarker)
        if n_detected < SEQ_LEN * 0.5:
            n_pose_fail += 1
        else:
            real_X.append(feats)
            rows.append({"filename": f"tennisette_{row['match']}_{row['event_index']}.mp4",
                         "label": native_label, "label_idx": native_idx, "source": "tennisette_real"})

        # Augmented copy of the real clip -- same pipeline as THETIS, applied to real footage.
        aug_frames = augment_clip(frames, rng)
        aug_feats, n_aug_detected = run_pose_extraction(aug_frames, landmarker)
        if n_aug_detected < SEQ_LEN * 0.5:
            n_aug_pose_fail += 1
        else:
            aug_X.append(aug_feats)
            rows.append({"filename": f"tennisette_aug_{row['match']}_{row['event_index']}.mp4",
                         "label": native_label, "label_idx": native_idx, "source": "tennisette_augmented"})

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            print(f"[progress] {i+1}/{len(train_rows)} clips, {elapsed:.0f}s elapsed", flush=True)

    all_X = real_X + aug_X
    combined_new_X = np.stack(all_X, axis=0)
    combined_new_labels = pd.DataFrame(rows)
    print(f"\n[done extracting] {len(real_X)} real + {len(aug_X)} augmented = {combined_new_X.shape[0]} "
          f"new TenniSet-derived train clips "
          f"(decode failures={n_decode_fail}, real pose failures={n_pose_fail}, "
          f"augmented pose failures={n_aug_pose_fail})")
    print(combined_new_labels.groupby(["source", "label"]).size())

    # Fold into the existing THETIS clean-split + augmented train set (Task A's output).
    base_X = np.load(os.path.join(AUGMENTED_ROOT, "train", "X_train.npy"))
    base_labels = pd.read_csv(os.path.join(AUGMENTED_ROOT, "train", "labels_train.csv"))
    if "source" not in base_labels.columns:
        base_labels["source"] = "thetis"

    final_X = np.concatenate([base_X, combined_new_X], axis=0)
    final_labels = pd.concat([base_labels, combined_new_labels], ignore_index=True)

    os.makedirs(os.path.join(OUT_ROOT, "train"), exist_ok=True)
    np.save(os.path.join(OUT_ROOT, "train", "X_train.npy"), final_X)
    final_labels.to_csv(os.path.join(OUT_ROOT, "train", "labels_train.csv"), index=False)
    print(f"\n[write] final train set: {final_X.shape[0]} clips "
          f"({base_X.shape[0]} THETIS(+aug) + {combined_new_X.shape[0]} TenniSet(+aug)) -> {OUT_ROOT}/train")

    # val/test: unchanged from processed_clean (THETIS's own held-out clean-split) -- the
    # THETIS-distribution val is still used for checkpoint selection during fine-tuning; the
    # REAL validation of this whole hypothesis is the external 14-instance real-footage set,
    # evaluated separately after training, not folded in here.
    clean_root = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed_clean")
    for split in ("val", "test"):
        X = np.load(os.path.join(clean_root, split, f"X_{split}.npy"))
        lbl = pd.read_csv(os.path.join(clean_root, split, f"labels_{split}.csv"))
        dst_dir = os.path.join(OUT_ROOT, split)
        os.makedirs(dst_dir, exist_ok=True)
        np.save(os.path.join(dst_dir, f"X_{split}.npy"), X)
        lbl.to_csv(os.path.join(dst_dir, f"labels_{split}.csv"), index=False)
        print(f"[copy] {split}: {X.shape[0]} clips (unchanged THETIS clean-split) -> {dst_dir}")


if __name__ == "__main__":
    main()
