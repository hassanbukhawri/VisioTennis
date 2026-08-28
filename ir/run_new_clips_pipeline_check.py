"""
Runs 3 newly-added real validation clips (clip4/5/6, mvp/real_footage_test/)
through the exact same pipeline path the real backend uses for a single
uploaded clip: decode_and_resample_video() -> run_pose_extraction() ->
run_stroke_classification(). No segmentation/candidate-window step -- each
clip is treated as one upload, same as production.

Deliberately does NOT assume a true label from filename/content -- reports
only the model's real output (predicted class, per-class probabilities,
pose detection rate), for the user to grade by eye afterwards.

Usage:
    python ir/run_new_clips_pipeline_check.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort

from backend.app import (
    decode_and_resample_video, run_pose_extraction, run_stroke_classification,
    POSE_TASK_PATH, STROKE_GRU_ONNX_PATH, MIN_POSE_DETECTION_RATE,
)

CLIPS = [
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip4_game_rally.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip5_serve_setup.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip6_solo_practice.mp4"),
]

if __name__ == "__main__":
    print("[setup] loading StrokeGRU ONNX session...")
    stroke_session = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])
    print("[setup] loading MediaPipe Pose landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    for clip_path in CLIPS:
        name = os.path.basename(clip_path)
        print(f"\n{'=' * 70}\n[{name}]\n{'=' * 70}")
        if not os.path.exists(clip_path):
            print("  SKIPPED -- file not found")
            continue

        try:
            frames, fps, n_real_frames = decode_and_resample_video(clip_path)
        except ValueError as e:
            print(f"  REJECTED at decode stage (bad_video_file): {e}")
            continue
        print(f"  real frames read: {n_real_frames}, fps={fps}, resampled to {len(frames)}")

        pose_seq, n_detected = run_pose_extraction(frames, landmarker)
        detection_rate = n_detected / len(pose_seq)
        trustworthy = detection_rate >= MIN_POSE_DETECTION_RATE
        print(f"  pose detection: {n_detected}/{len(pose_seq)} frames ({detection_rate:.1%}) "
              f"-- {'OK, meets' if trustworthy else 'BELOW'} the {MIN_POSE_DETECTION_RATE:.0%} "
              f"production trust threshold")

        if n_detected == 0:
            print("  no player detected at all -- classification skipped (would be rejected in production)")
            continue

        predicted_class, class_probabilities = run_stroke_classification(pose_seq, stroke_session)
        sorted_probs = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
        print(f"  PREDICTED: {predicted_class} ({class_probabilities[predicted_class]:.1%} confidence)")
        print("  full distribution:")
        for cls, p in sorted_probs:
            print(f"    {cls}: {p:.1%}")
        if not trustworthy:
            print("  ** FLAG: pose detection rate below production trust threshold -- "
                  "treat this prediction with caution, not clean validation data **")
