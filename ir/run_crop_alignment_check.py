"""
One-off diagnostic (NOT a production change): tests the crop-alignment
hypothesis from PROJECT_LOG.md's 2026-08-02 classifier diagnostic entry --
that `clip1_serve_practice.mp4`'s real serve (contact_frame=235) is
misclassified as volley because the fixed +/-30-frame window centered on the
peak-wrist-speed contact frame (205-265) is NOT how THETIS training clips
were framed (tightly trimmed prep-through-follow-through, little dead time).

Manually re-crops the SAME clip's SAME serve to span the actual visible
swing (backswing start -> follow-through end, identified by eye from
mvp/real_footage_test/clip1_serve_practice.mp4 frames -- see montage built
during this diagnostic), resamples to exactly 60 frames via the same
np.linspace technique backend.app.decode_and_resample_video already uses for
any real clip (this repo's one established way to turn an arbitrary-length
real clip into the model's fixed 60-frame input, the closest available
proxy for "how THETIS clips were framed" since the original THETIS
preprocessing script isn't in this repo -- only its already-processed
output, training/stroke_classification/dataset.py's docstring), and runs it
through the completely unmodified StrokeGRU ONNX classifier.

Does NOT touch ir/stroke_segmentation.py's windowing logic -- diagnostic
only, per explicit instruction.

Usage:
    python ir/run_crop_alignment_check.py
"""
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

CLIP_PATH = os.path.join(REPO_ROOT, "mvp", "real_footage_test", "clip1_serve_practice.mp4")
WINDOW_LEN = 60

# Original fixed +/-30-frame window from ir/run_classifier_diagnostic.py's
# real run (ir/example_output/classifier_diagnostic/clip1_serve_practice/
# result.json candidate 3): contact_frame=235, window_start=205, window_end=265.
ORIGINAL_WINDOW = (205, 265)

# Manually identified by eye from individual decoded frames (every 2 frames,
# 188-256) -- NOT from the segmentation algorithm's own logic:
#   frames <=196: idle/rest stance, racket trailing low, no swing motion yet
#   ~198-210: backswing/loading visibly begins and builds (racket rising,
#     body coiling into the loaded "trophy" position)
#   ~232-238: uncoil + jump + contact (racket at/near full extension, ball
#     visible near racket around 236-238 -- close to but not identical to
#     the peak-wrist-speed proxy's frame 235)
#   ~240-248: follow-through (arm swinging down/across, landing leg)
#   >=250: walking/recovery, no longer part of the stroke
# Chosen with a couple frames of margin on each side of the visible motion
# rather than cutting exactly at the first/last moving frame.
MANUAL_WINDOW = (196, 250)

# Additional candidate crops, tried so the conclusion doesn't rest on a
# single manual guess -- all still hand-picked from the same visual read of
# the frames (no algorithm, no threshold), just different amounts of margin
# around the same identified backswing/contact/follow-through boundaries.
EXTRA_WINDOWS = {
    "tighter (load-contact-follow-through only, no backswing head start)": (208, 248),
    "very tight (uncoil-contact-follow-through only)": (218, 244),
    "wider margin": (190, 252),
}


def decode_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open {path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def resample_to_window_len(window: np.ndarray, target_len: int = WINDOW_LEN) -> np.ndarray:
    """Same technique as backend.app.decode_and_resample_video -- uniform
    np.linspace resampling, not truncation/padding. Applied here because the
    manual crop (54 raw frames) isn't exactly WINDOW_LEN, and StrokeGRU's
    ONNX graph hard-codes seq_len=60 (ir/verify_onnx_exports.py)."""
    n = window.shape[0]
    if n == target_len:
        return window
    idx = np.linspace(0, n - 1, target_len).round().astype(int)
    return window[idx]


def classify_window(pose_seq, start, end, session, label):
    raw = pose_seq[start:end]
    resampled = resample_to_window_len(raw)
    predicted_class, class_probabilities = run_stroke_classification(resampled, session)
    print(f"\n[{label}] window=({start},{end}) raw_len={raw.shape[0]} -> resampled to {resampled.shape[0]} frames")
    print(f"  predicted: {predicted_class} (confidence={class_probabilities[predicted_class]:.4f})")
    print(f"  full distribution: " + ", ".join(f"{k}={v:.4f}" for k, v in class_probabilities.items()))
    return predicted_class, class_probabilities


if __name__ == "__main__":
    print(f"[setup] decoding {CLIP_PATH}")
    frames = decode_video(CLIP_PATH)
    print(f"[setup] {len(frames)} frames decoded")

    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    print("[setup] running pose extraction (unmodified MediaPipe Pose, same as production)")
    pose_seq, n_detected = run_pose_extraction(frames, landmarker)
    print(f"[setup] pose detected {n_detected}/{len(frames)} ({100 * n_detected / len(frames):.1f}%)")

    session = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])

    print("\n" + "=" * 70)
    print("SANITY CHECK -- reproduce the documented original-window result")
    print("(expected: volley, ~0.907 confidence, per PROJECT_LOG.md 2026-08-02"
          " and ir/example_output/classifier_diagnostic/clip1_serve_practice/result.json)")
    print("=" * 70)
    orig_class, orig_probs = classify_window(pose_seq, *ORIGINAL_WINDOW, session, "ORIGINAL +/-30 window")

    print("\n" + "=" * 70)
    print("CROP-ALIGNMENT TEST -- manually re-cropped window (THETIS-style tight trim)")
    print("=" * 70)
    new_class, new_probs = classify_window(pose_seq, *MANUAL_WINDOW, session, "MANUAL re-cropped window")

    print("\n" + "=" * 70)
    print("ADDITIONAL CROP VARIANTS -- so the conclusion doesn't rest on one guess")
    print("=" * 70)
    extra_results = {}
    for label, win in EXTRA_WINDOWS.items():
        c, p = classify_window(pose_seq, *win, session, label)
        extra_results[label] = (win, c, p)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  ground truth:      serve (jump serve, full extension -- hand-verified unambiguous)")
    print(f"  original window:   {ORIGINAL_WINDOW} -> {orig_class} ({orig_probs[orig_class]:.4f}) -- {'CORRECT' if orig_class == 'serve' else 'WRONG'}")
    print(f"  manual crop:       {MANUAL_WINDOW} -> {new_class} ({new_probs[new_class]:.4f}) -- {'CORRECT' if new_class == 'serve' else 'WRONG'}")
    for label, (win, c, p) in extra_results.items():
        print(f"  {label}:")
        print(f"    {win} -> {c} ({p[c]:.4f}) -- {'CORRECT' if c == 'serve' else 'WRONG'}")

    any_correct = new_class == "serve" or any(c == "serve" for _, c, _ in extra_results.values())
    if any_correct:
        print("\n  -> Crop-alignment hypothesis CONFIRMED on this instance for at least one manual")
        print("     crop: the same unmodified classifier correctly predicts serve once the window")
        print("     is trimmed to match how THETIS clips were framed, with no model/threshold changes.")
    else:
        print("\n  -> Crop-alignment hypothesis NOT confirmed on this instance: no manually")
        print("     re-cropped window (4 tried, different amounts of margin around the same")
        print("     hand-identified backswing/contact/follow-through boundaries) produced a")
        print("     correct serve prediction. The problem is deeper than window alignment alone.")
