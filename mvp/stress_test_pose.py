"""
MVP viability test: does MediaPipe Pose work on real, non-dataset phone-style
footage WITHOUT the throwaway red-shirt-crop hack ir/populate_e2e.py needed
for wide broadcast footage? Genuinely real footage was searched for on this
machine first (none found -- no video files exist anywhere in this repo or
common user directories); per explicit user instruction, 3 free stock clips
were fetched as real-footage stand-ins, chosen specifically because they are
casual/amateur practice footage (not curated dataset clips, not broadcast TV),
and are flagged as such below, not presented as literally self-shot.

Clips used (all downloaded from Pexels, free/no-attribution license):
  clip1_serve_practice.mp4  -- solo player, full-body, close-ish framing,
                                bright sun, outdoor clay court
                                https://www.pexels.com/video/a-tennis-player-practicing-his-serve-4902139/
  clip2_teen_practice.mp4   -- solo player, medium framing, OVERCAST/hazy
                                lighting, outdoor hard court
                                https://www.pexels.com/video/a-teenage-girl-practising-tennis-11154706/
  clip3_group_distant.mp4   -- multiple people, more distant/wide framing,
                                hazy lighting, real amateur group lesson
                                https://www.pexels.com/video/a-group-of-children-practising-tennis-with-their-teachers-11154490/

Tests, per clip, on RAW frames -- NO crop, NO red-shirt hack:
  1. Detection rate: % of sampled frames with a valid MediaPipe Pose result
     (same category as the zero-padded-frame issue already found in the
     biomechanics work).
  2. Whether raw-frame detection works at all, or the red-shirt-tracker
     problem recurs (i.e. 0% detection without a workaround).
  3. Qualitative joint-angle sanity check: runs the REAL, already-validated
     ir/biomechanics.py:extract_biomechanics_features() on the extracted
     landmark sequence and prints the same joint-angle values spot-checked
     in the earlier biomechanics validation (knee flex, elbow angle,
     shoulder-hip separation) -- reusing that exact code, not reinventing
     a parallel check.

Usage:
    python mvp/stress_test_pose.py
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

from ir.biomechanics import extract_biomechanics_features, _valid_frame_mask, _xy, LEFT_HIP, RIGHT_HIP  # noqa: E402

IDENTITY_JUMP_THRESHOLD = 0.15  # normalized-coordinate hip-center displacement; calibrated against the
# solo clips' own frame-to-frame noise floor (max observed: 0.011-0.017) -- anything an order of
# magnitude past that is a real subject switch, not natural motion

POSE_TASK_PATH = os.path.join(REPO_ROOT, "training/stroke_classification/pose_landmarker_full.task")
CLIPS_DIR = os.path.join(REPO_ROOT, "mvp", "real_footage_test")

CLIPS = [
    dict(fname="clip1_serve_practice.mp4", label="close/full-body, bright sun",
         start_frame=0, n_frames=90),
    dict(fname="clip2_teen_practice.mp4", label="medium framing, overcast/hazy",
         start_frame=60, n_frames=90),
    dict(fname="clip3_group_distant.mp4", label="distant, multi-person, hazy (amateur group lesson)",
         start_frame=150, n_frames=90),
]


def extract_frames(path, start_frame, n_frames):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run_pose_raw(frames, landmarker):
    """NO crop, NO hack -- feeds each raw frame directly to MediaPipe Pose,
    exactly testing whether real (non-broadcast, non-pre-cropped) footage
    needs a workaround the way ir/populate_e2e.py's broadcast clip did.
    Also tracks the hip-center position per frame to check SUBJECT
    CONTINUITY, not just per-frame detection -- MediaPipe's num_poses=1
    picks whichever detection is most confident independently each frame,
    with no identity/re-identification logic, so in a multi-person scene
    it can silently switch which person it's tracking mid-sequence while
    still reporting "a pose detected" on every single frame."""
    n_landmarks = 33
    feats = np.zeros((len(frames), n_landmarks * 3), dtype=np.float32)
    n_detected = 0
    hip_centers = [None] * len(frames)
    for i, img_rgb in enumerate(frames):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            arr = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)
            feats[i] = arr
            n_detected += 1
            hip_centers[i] = (_xy(arr, LEFT_HIP) + _xy(arr, RIGHT_HIP)) / 2

    jumps = [
        float(np.linalg.norm(hip_centers[i] - hip_centers[i - 1]))
        for i in range(1, len(hip_centers))
        if hip_centers[i] is not None and hip_centers[i - 1] is not None
    ]
    n_identity_jumps = sum(1 for j in jumps if j > IDENTITY_JUMP_THRESHOLD)
    return feats, n_detected, n_identity_jumps, len(jumps)


if __name__ == "__main__":
    if not os.path.isfile(POSE_TASK_PATH):
        raise SystemExit(f"pose task not found at {POSE_TASK_PATH} -- run ir/populate_e2e.py once first "
                          f"(it downloads this file)")

    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    print("=" * 78)
    print("MVP viability test: MediaPipe Pose on real (non-dataset) footage, NO crop hack")
    print("=" * 78)

    results_summary = []
    for clip in CLIPS:
        path = os.path.join(CLIPS_DIR, clip["fname"])
        print(f"\n--- {clip['fname']} ({clip['label']}) ---")
        frames = extract_frames(path, clip["start_frame"], clip["n_frames"])
        print(f"  extracted {len(frames)} real frames (starting at frame {clip['start_frame']})")

        feats, n_detected, n_identity_jumps, n_transitions = run_pose_raw(frames, landmarker)
        detection_rate = n_detected / len(frames) if frames else 0.0
        print(f"  RAW-FRAME detection (no crop): {n_detected}/{len(frames)} frames "
              f"({detection_rate*100:.1f}%)")
        print(f"  subject continuity (hip-center jump > {IDENTITY_JUMP_THRESHOLD}): "
              f"{n_identity_jumps}/{n_transitions} frame-transitions -- "
              f"{'SAME PERSON tracked throughout' if n_identity_jumps == 0 else 'LIKELY IDENTITY SWITCH(ES) mid-clip'}")
        results_summary.append((clip["fname"], clip["label"], n_detected, len(frames), detection_rate, n_identity_jumps))

        if n_detected == 0:
            print("  [FAIL] zero detections on raw frames -- red-shirt-tracker-style problem RECURS, "
                  "a crop/workaround would be required")
            continue

        valid_mask = _valid_frame_mask(feats)
        print(f"  valid (non-zero) frames available for biomechanics extraction: {int(valid_mask.sum())}")

        try:
            joint_features = extract_biomechanics_features(feats)
        except Exception as e:
            print(f"  [ERROR] biomechanics extraction failed even with {n_detected} valid frames: {e}")
            continue

        print(f"  contact_frame={joint_features['contact_frame']} "
              f"({joint_features['contact_frame_source']}), "
              f"dominant_wrist={joint_features['dominant_wrist']}")
        print(f"  QUALITATIVE JOINT-ANGLE SANITY CHECK (same fields spot-checked in the "
              f"biomechanics validation):")
        print(f"    knee_flex_left_deg   = {joint_features['knee_flex_left_deg']:7.2f}  "
              f"(anatomically plausible range: ~0-180, straight leg ~180)")
        print(f"    knee_flex_right_deg  = {joint_features['knee_flex_right_deg']:7.2f}")
        print(f"    elbow_angle_deg      = {joint_features['elbow_angle_deg']:7.2f}  "
              f"(0-180, straight arm ~180)")
        print(f"    shoulder_hip_separation_deg = {joint_features['shoulder_hip_separation_deg']:6.2f}  "
              f"(bounded 0-90 by construction)")
        print(f"    stance_width_ratio   = {joint_features['stance_width_ratio']:7.3f}")
        plausible = (
            0 <= joint_features['knee_flex_left_deg'] <= 180 and
            0 <= joint_features['knee_flex_right_deg'] <= 180 and
            0 <= joint_features['elbow_angle_deg'] <= 180 and
            0 <= joint_features['shoulder_hip_separation_deg'] <= 90 and
            joint_features['stance_width_ratio'] > 0
        )
        print(f"  [{'OK' if plausible else 'SUSPECT'}] all values within their known-valid mathematical "
              f"ranges (this confirms no NaN/degenerate collapse, not that the specific numbers are "
              f"'correct' for this specific real stroke -- there's no ground truth to compare against here)")

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    for fname, label, n_det, n_total, rate, n_jumps in results_summary:
        jump_flag = "" if n_jumps == 0 else f"  [{n_jumps} likely identity switch(es)]"
        print(f"  {fname:32s} {label:50s} {n_det:3d}/{n_total:3d} ({rate*100:5.1f}%){jump_flag}")
