"""
The pose-based (StrokeGRU) analog of evaluate_real_footage.py -- that script
validates Path B's video model against the 14 hand-labeled real-footage
instances; this one validates the actually-deployed pose-based classifier
against the SAME 14 instances, so "does augmentation help" has a real,
reusable before/after baseline on real footage (not just the THETIS-derived
test split).

Ground-truth instances and window boundaries are NOT re-derived here --
copied from two already-verified, on-record sources:
  - INSTANCES' (filename, true_label, contact_frame) tuples: identical to
    evaluate_real_footage.py's own list (same hand-verified labels/frames,
    see that file's docstring for full provenance).
  - Each contact_frame's exact (window_start, window_end): read directly
    from ir/example_output/classifier_diagnostic/<clip>/result.json, which
    ir/run_classifier_diagnostic.py already computed via
    ir.stroke_segmentation.detect_candidate_strokes (window_radius=30 ->
    60-frame windows, matching StrokeGRU's fixed SEQ_LEN exactly). Re-using
    these exact boundaries rather than recomputing keeps this evaluation
    tied to the same windows that were hand-verified against the real
    footage, not a fresh (and possibly slightly different) auto-detection.

For clip4/5/6 (whole-clip, single-stroke recordings), reuses the same
"treat as one production upload" convention as
ir/run_new_clips_pipeline_check.py: decode_and_resample_video ->
run_pose_extraction, no windowing needed.

Accepts --onnx-path / --temperature-path so this can be pointed at any
candidate checkpoint (deployed clean-split model, or an augmented-retrain
candidate) without needing a separate script per model.

Usage:
    python training/stroke_classification/evaluate_real_footage_pose_based.py \\
        [--onnx-path PATH] [--temperature-path PATH] [--label LABEL]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort

from backend.app import (
    decode_and_resample_video, run_pose_extraction, run_stroke_classification,
    POSE_TASK_PATH, STROKE_CLASSES,
)

FOOTAGE_DIR = os.path.join(REPO_ROOT, "mvp", "real_footage_test")
DIAG_DIR = os.path.join(REPO_ROOT, "ir", "example_output", "classifier_diagnostic")

# (filename, true_label, contact_frame_or_None) -- identical list to
# evaluate_real_footage.py; contact_frame=None means whole-clip treatment.
INSTANCES = [
    ("clip1_serve_practice.mp4", "serve", 155),
    ("clip1_serve_practice.mp4", "serve", 235),
    ("clip2_teen_practice.mp4", "backhand", 34),
    ("clip2_teen_practice.mp4", "backhand", 142),
    ("clip2_teen_practice.mp4", "backhand", 213),
    ("clip2_teen_practice.mp4", "backhand", 259),
    ("clip2_teen_practice.mp4", "backhand", 340),
    ("clip2_teen_practice.mp4", "backhand", 385),
    ("clip2_teen_practice.mp4", "backhand", 439),
    ("clip2_teen_practice.mp4", "backhand", 493),
    ("clip2_teen_practice.mp4", "backhand", 574),
    ("clip4_game_rally.mp4", "backhand", None),
    ("clip5_serve_setup.mp4", "serve", None),
    ("clip6_solo_practice.mp4", "forehand", None),
]


def load_window_bounds(clip_stem):
    path = os.path.join(DIAG_DIR, clip_stem, "result.json")
    with open(path) as f:
        d = json.load(f)
    return {c["contact_frame"]: (c["window_start"], c["window_end"]) for c in d["candidates"]}


def decode_all_frames(path, max_frames=2000):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-path", default=None, help="defaults to backend.app.STROKE_GRU_ONNX_PATH (the deployed model)")
    p.add_argument("--temperature-path", default=None, help="defaults to backend.app.STROKE_TEMPERATURE_PATH")
    p.add_argument("--label", default="deployed model", help="human-readable label for this run, shown in output")
    args = p.parse_args()

    from backend import app as backend_app
    onnx_path = args.onnx_path or backend_app.STROKE_GRU_ONNX_PATH
    temperature_path = args.temperature_path or backend_app.STROKE_TEMPERATURE_PATH

    temperature = 1.0
    if os.path.exists(temperature_path):
        with open(temperature_path) as f:
            temperature = json.load(f)["temperature"]

    print(f"[setup] evaluating: {args.label}")
    print(f"[setup] onnx_path={onnx_path}")
    print(f"[setup] temperature={temperature:.4f} (from {temperature_path if os.path.exists(temperature_path) else 'default, no file found'})")

    stroke_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    # Cache full-clip pose sequences for clip1/clip2 (multiple instances share one clip).
    full_pose_cache = {}

    results = []
    for fname, true_label, contact_frame in INSTANCES:
        clip_path = os.path.join(FOOTAGE_DIR, fname)
        clip_stem = fname.replace(".mp4", "")

        if contact_frame is None:
            frames, fps, n_real = decode_and_resample_video(clip_path)
            pose_seq, n_detected = run_pose_extraction(frames, landmarker)
            window_valid_fraction = n_detected / len(pose_seq)
        else:
            if clip_stem not in full_pose_cache:
                all_frames = decode_all_frames(clip_path)
                full_seq, n_det_full = run_pose_extraction(all_frames, landmarker)
                full_pose_cache[clip_stem] = full_seq
            full_seq = full_pose_cache[clip_stem]
            bounds = load_window_bounds(clip_stem)
            window_start, window_end = bounds[contact_frame]
            pose_seq = full_seq[window_start:window_end]
            window_valid_fraction = float((pose_seq != 0).any(axis=1).mean())

        predicted, class_probs = run_stroke_classification(pose_seq, stroke_session, temperature=temperature)
        correct = predicted == true_label
        results.append(dict(
            filename=fname, true_label=true_label, contact_frame=contact_frame,
            predicted=predicted, confidence=class_probs[predicted],
            window_valid_fraction=window_valid_fraction, correct=correct,
        ))
        loc = f"contact_frame={contact_frame}" if contact_frame is not None else "whole-clip"
        print(f"  [{fname:28s} {loc:22s}] true={true_label:10s} -> pred={predicted:10s} "
              f"conf={class_probs[predicted]:.1%} {'OK' if correct else '** WRONG **'}")

    print(f"\n{'=' * 70}\nPER-CLASS BREAKDOWN -- {args.label}\n{'=' * 70}")
    for cls in STROKE_CLASSES:
        cls_results = [r for r in results if r["true_label"] == cls]
        if not cls_results:
            print(f"  {cls:10s}: 0 instances -- no coverage")
            continue
        n_correct = sum(r["correct"] for r in cls_results)
        print(f"  {cls:10s}: {n_correct}/{len(cls_results)} correct")

    total_correct = sum(r["correct"] for r in results)
    print(f"\n[overall] {total_correct}/{len(results)} correct ({total_correct/len(results):.1%}) -- {args.label}")
    return results


if __name__ == "__main__":
    main()
