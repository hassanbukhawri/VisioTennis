"""
Classifier diagnostic: predicted vs. actual stroke type on real, non-THETIS
footage.

This is diagnosis only -- no fix is attempted here. It runs the existing,
unmodified StrokeGRU classifier (training/stroke_classification/checkpoints/
stroke_gru.onnx) against every real-stroke candidate window found by
ir.stroke_segmentation.detect_candidate_strokes on 5 real clips:

  - clip1_serve_practice.mp4 / clip2_teen_practice.mp4 / clip3_group_distant.mp4
    (mvp/real_footage_test/ -- same 3 clips used for the stroke-segmentation
    validation; clip3 is known-invalid for pose tracking, see
    ir/example_output/stroke_segmentation_trial/, kept here anyway since a
    wrong-subject pose sequence is itself a data point about what breaks the
    pipeline, just not a classifier-accuracy data point).
  - Two new clips pulled from data/VisioTennis/Dataset/Ball Tracker+Bounce/
    (frame-sequence broadcast footage, originally collected for ball
    tracking, reused here because they're the only other real -- non-THETIS
    -- tennis video in the repo): game4/Clip1 (close baseline camera, clay,
    ITF-level singles) and game6/Clip1 (medium-distance TV broadcast, clay,
    ATP-level). Chosen from a 81-clip visual scan specifically because their
    player scale looked closer to usable than the broadcast norm (most of
    that dataset is full-court wide shots, same distant-camera problem that
    already invalidated clip3 -- not worth re-demonstrating 81 times).

For every surviving candidate this script:
  1. classifies the window with the unmodified StrokeGRU ONNX model
  2. saves a full-res crop of the contact frame for hand-verification of the
     ACTUAL stroke type (filled in by eye afterwards, not automated)
  3. computes a candidate cheap fallback signal: whether the dominant wrist
     is above the nose at the contact frame (image y grows downward, so
     "above" means wrist_y < nose_y) -- using only landmarks already
     extracted, no new feature engineering in ir/biomechanics.py itself.

Usage:
    python ir/run_classifier_diagnostic.py

Writes to ir/example_output/classifier_diagnostic/<clip_stem>/:
    candidate_<i>_frame<f>.png  -- full-res contact-frame crop
    result.json                 -- candidates + predicted class + wrist/nose check
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort

from backend.app import run_pose_extraction, run_stroke_classification, POSE_TASK_PATH, STROKE_GRU_ONNX_PATH
from ir.stroke_segmentation import detect_candidate_strokes, WINDOW_LEN
from ir.biomechanics import _valid_frame_mask, determine_dominant_wrist, LEFT_WRIST, RIGHT_WRIST, NOSE, _xy

CLIPS = [
    dict(name="clip1_serve_practice", kind="video",
         path=os.path.join(REPO_ROOT, "mvp", "real_footage_test", "clip1_serve_practice.mp4")),
    dict(name="clip2_teen_practice", kind="video",
         path=os.path.join(REPO_ROOT, "mvp", "real_footage_test", "clip2_teen_practice.mp4")),
    dict(name="clip3_group_distant", kind="video",
         path=os.path.join(REPO_ROOT, "mvp", "real_footage_test", "clip3_group_distant.mp4")),
    dict(name="game4_clip1_broadcast_close", kind="frame_dir",
         path=os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "Ball Tracker+Bounce", "game4", "Clip1")),
    dict(name="game6_clip1_broadcast_medium", kind="frame_dir",
         path=os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "Ball Tracker+Bounce", "game6", "Clip1")),
]
ASSUMED_FPS = 25.0
OUT_ROOT = os.path.join(REPO_ROOT, "ir", "example_output", "classifier_diagnostic")


def decode_video(path, max_frames=2000):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def decode_frame_dir(path, max_frames=2000):
    files = sorted(glob.glob(os.path.join(path, "*.jpg")))[:max_frames]
    frames = []
    for f in files:
        img = cv2.imread(f)
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


if __name__ == "__main__":
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    session = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])

    all_results = {}
    for clip in CLIPS:
        name, kind, path = clip["name"], clip["kind"], clip["path"]
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {name} ({kind}: {path}) ===")

        frames = decode_video(path) if kind == "video" else decode_frame_dir(path)
        print(f"  {len(frames)} frames")

        pose_seq, n_detected = run_pose_extraction(frames, landmarker)
        print(f"  pose detected {n_detected}/{len(frames)} ({100 * n_detected / len(frames):.1f}%)")

        valid_mask = _valid_frame_mask(pose_seq)
        if not valid_mask.any():
            print("  no valid pose frames, skipping")
            continue
        dom_side = determine_dominant_wrist(pose_seq, valid_mask)
        dom_idx = LEFT_WRIST if dom_side == "left" else RIGHT_WRIST

        candidates = detect_candidate_strokes(pose_seq, dominant_wrist_idx=dom_idx, use_rotation_filter=True)
        print(f"  {len(candidates)} candidates")

        for i, c in enumerate(candidates):
            window = pose_seq[c["window_start"]:c["window_end"]]
            if window.shape[0] != WINDOW_LEN:
                pad = np.zeros((WINDOW_LEN - window.shape[0], window.shape[1]), dtype=window.dtype)
                window = np.concatenate([window, pad], axis=0)
            predicted_class, class_probabilities = run_stroke_classification(window, session)
            c["predicted_class"] = predicted_class
            c["confidence"] = class_probabilities[predicted_class]
            c["class_probabilities"] = class_probabilities

            cf = pose_seq[c["contact_frame"]]
            wrist_y = _xy(cf, dom_idx)[1]
            nose_y = _xy(cf, NOSE)[1]
            c["wrist_above_nose_at_contact"] = bool(wrist_y < nose_y)
            c["wrist_minus_nose_y"] = float(wrist_y - nose_y)  # negative = wrist above nose

            # Same check but scanned across the whole classification window,
            # not just the single peak-wrist-speed contact frame -- the peak-
            # speed instant can land after the wrist has already dropped past
            # its highest point (see PROJECT_LOG.md), so also report whether
            # the wrist is EVER above the nose anywhere in the window.
            w_start, w_end = c["window_start"], c["window_end"]
            window_valid = valid_mask[w_start:w_end]
            window_diffs = [
                float(_xy(pose_seq[t], dom_idx)[1] - _xy(pose_seq[t], NOSE)[1])
                for t in range(w_start, w_end) if valid_mask[t]
            ]
            c["min_wrist_minus_nose_y_in_window"] = min(window_diffs) if window_diffs else None
            c["wrist_ever_above_nose_in_window"] = bool(window_diffs and min(window_diffs) < 0)

            frame_bgr = cv2.cvtColor(frames[c["contact_frame"]], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"candidate_{i}_frame{c['contact_frame']}.png"), frame_bgr)

            print(f"    frame {c['contact_frame']:4d} speed={c['speed']:.4f} "
                  f"rotation={c['trunk_rotation_peak']} -> {predicted_class} ({c['confidence']:.2f}) "
                  f"wrist_above_nose={c['wrist_above_nose_at_contact']}")

        result = dict(
            clip_name=name, source_path=path, kind=kind, n_frames=len(frames),
            n_pose_detected=n_detected, pose_detection_rate=n_detected / len(frames),
            dominant_wrist=dom_side, n_candidates=len(candidates), candidates=candidates,
        )
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        all_results[name] = result

    print(f"\n[done] wrote per-clip results to {OUT_ROOT}")
