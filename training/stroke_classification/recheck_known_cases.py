"""
Re-runs the two real-footage cases referenced in the calibration task through
the real pipeline path (decode -> pose -> StrokeGRU), reporting raw softmax
confidence vs. temperature-scaled confidence side by side, using the T fitted
in calibrate_temperature.py (checkpoints/temperature.json).

Case 1: clip5_serve_setup.mp4 -- true label serve, predicted backhand at
99.3% raw confidence (PROJECT_LOG.md, 2026-08-12 entry) -- a confident,
decisive WRONG answer, the clearest documented real-footage miss.

Case 2: clip1_serve_practice.mp4 -- whole-clip upload produced volley/72%
confidence in two separate on-device verification passes (PROJECT_LOG.md,
lines ~2148 and ~2314). NOTE: neither log entry actually grades this
prediction against a ground-truth label -- both were incidental readouts
during unrelated UI-failure-mode tests, not an accuracy check. Reported
here as-is; flagged plainly rather than assumed correct.

Usage:
    python training/stroke_classification/recheck_known_cases.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort
import numpy as np

from backend.app import (
    decode_and_resample_video, run_pose_extraction,
    POSE_TASK_PATH, STROKE_GRU_ONNX_PATH, MIN_POSE_DETECTION_RATE, STROKE_CLASSES,
)

TEMPERATURE_PATH = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/temperature.json")

CASES = [
    ("clip5_serve_setup.mp4", "serve", "confirmed via ball-toss + trophy position, unambiguous"),
    ("clip1_serve_practice.mp4", None, "whole-clip upload -- no independently graded ground truth on record for this exact window"),
]


def classify_with_and_without_temperature(pose_seq, session, temperature):
    ins = session.get_inputs()[0]
    x = pose_seq.astype(np.float32)[None]
    logits = session.run(None, {ins.name: x})[0][0]

    raw_probs = np.exp(logits - logits.max())
    raw_probs = raw_probs / raw_probs.sum()

    scaled_logits = logits / temperature
    cal_probs = np.exp(scaled_logits - scaled_logits.max())
    cal_probs = cal_probs / cal_probs.sum()

    raw_dict = {c: float(p) for c, p in zip(STROKE_CLASSES, raw_probs)}
    cal_dict = {c: float(p) for c, p in zip(STROKE_CLASSES, cal_probs)}
    predicted = max(raw_dict, key=raw_dict.get)  # argmax unchanged by temperature
    return predicted, raw_dict, cal_dict


def main():
    with open(TEMPERATURE_PATH) as f:
        T = json.load(f)["temperature"]
    print(f"[setup] loaded fitted temperature T={T:.4f} from {TEMPERATURE_PATH}")

    print("[setup] loading StrokeGRU ONNX session...")
    stroke_session = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])
    print("[setup] loading MediaPipe Pose landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    for fname, true_label, note in CASES:
        clip_path = os.path.join(REPO_ROOT, "mvp", "real_footage_test", fname)
        print(f"\n{'=' * 78}\n[{fname}]  true_label={true_label or 'UNGRADED'}\n  note: {note}\n{'=' * 78}")
        if not os.path.exists(clip_path):
            print("  SKIPPED -- file not found")
            continue

        frames, fps, n_real_frames = decode_and_resample_video(clip_path)
        pose_seq, n_detected = run_pose_extraction(frames, landmarker)
        detection_rate = n_detected / len(pose_seq)
        print(f"  pose detection: {n_detected}/{len(pose_seq)} ({detection_rate:.1%})")

        predicted, raw_dict, cal_dict = classify_with_and_without_temperature(pose_seq, stroke_session, T)
        correct_str = ("CORRECT" if predicted == true_label else "WRONG") if true_label else "n/a (ungraded)"
        print(f"  predicted: {predicted}  [{correct_str}]")
        print(f"\n  {'class':>10}  {'raw (T=1)':>10}  {'calibrated':>10}")
        for c in STROKE_CLASSES:
            print(f"  {c:>10}  {raw_dict[c]:>9.1%}  {cal_dict[c]:>9.1%}")
        print(f"\n  top-1 confidence: raw={raw_dict[predicted]:.1%}  ->  calibrated={cal_dict[predicted]:.1%}")


if __name__ == "__main__":
    main()
