"""
Step 4 of the biomechanics proposal (ir/DESIGN.md, sign-off received
2026-07-19): run real feature extraction (ir/biomechanics.py) on real
StrokeClassification (THETIS) pose sequences, construct real
BiomechanicsInfo instances (schema-validated, not hand-typed), and print
the actual computed values -- so results can be sanity-checked before being
trusted, per the user's explicit instruction, not assumed correct because
the code runs without error.

Also runs the one contact-frame-proxy validation check the design allows:
game1/Clip5 (ir/populate_e2e.py's true end-to-end example) is the only
place in this project with both real pose data AND a real TrackNet-labeled
hit frame. Re-extracts that clip's pose sequence and compares the proxy's
picked frame against the real hit frame(s) -- see the printed caveat about
that window containing a multi-hit rally segment, not one isolated stroke.

Usage:
    python ir/populate_biomechanics.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.schema import BiomechanicsInfo, ModelProvenance  # noqa: E402
from ir.biomechanics import extract_biomechanics_features, _valid_frame_mask  # noqa: E402

DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed")
POSE_TASK_REL = "training/stroke_classification/pose_landmarker_full.task"

NATIVE_TO_COLLAPSED = {
    0: "backhand", 1: "backhand", 2: "backhand", 3: "volley", 4: "serve",
    5: "forehand", 6: "forehand", 7: "forehand", 8: "volley", 9: "serve",
    10: "serve", 11: "smash",
}


def load_one_example_per_class(split="test"):
    """Picks the first clip in `split` for each of the 5 collapsed classes,
    so what's printed below is a real, not cherry-picked, sample."""
    X = np.load(os.path.join(DATA_ROOT, split, f"X_{split}.npy"))
    labels = pd.read_csv(os.path.join(DATA_ROOT, split, f"labels_{split}.csv"))
    labels["collapsed"] = labels["label_idx"].map(NATIVE_TO_COLLAPSED)

    picked = {}
    for collapsed_class in ("forehand", "backhand", "serve", "volley", "smash"):
        rows = labels.index[labels["collapsed"] == collapsed_class]
        if len(rows) == 0:
            continue
        i = rows[0]
        picked[collapsed_class] = (X[i], labels.loc[i, "filename"], labels.loc[i, "label"])
    return picked


def build_biomechanics_info(pose_seq, source_file) -> BiomechanicsInfo:
    features = extract_biomechanics_features(pose_seq)
    return BiomechanicsInfo(
        **features,
        source_model=ModelProvenance(
            component="biomechanics",
            checkpoint="pose_landmarker_full.task (upstream MediaPipe Pose landmarks; geometric derivation itself has no learned weights)",
            metric_name="n/a",
            metric_value=None,
            source_file=source_file,
        ),
    )


def print_result(collapsed_class, native_label, filename, info: BiomechanicsInfo, n_valid, seq_len):
    print(f"\n=== {collapsed_class} (native label: {native_label}, file: {filename}) ===")
    if n_valid < seq_len:
        print(f"  [data note] {seq_len - n_valid}/{seq_len} frames had no detected pose "
              f"(zero-padded in the source .npy) -- excluded from contact-frame/dominant-wrist selection")
    print(f"  dominant_wrist            = {info.dominant_wrist} ({info.dominant_wrist_source})")
    print(f"  contact_frame             = {info.contact_frame} ({info.contact_frame_source})")
    print(f"  wrist_height_ratio        = {info.wrist_height_ratio:+.3f}")
    print(f"  shoulder_rotation_deg     = {info.shoulder_rotation_deg:+7.2f}")
    print(f"  hip_rotation_deg          = {info.hip_rotation_deg:+7.2f}")
    print(f"  shoulder_hip_separation   = {info.shoulder_hip_separation_deg:7.2f}")
    print(f"  knee_flex_left_deg        = {info.knee_flex_left_deg:7.2f}")
    print(f"  knee_flex_right_deg       = {info.knee_flex_right_deg:7.2f}")
    print(f"  stance_width_ratio        = {info.stance_width_ratio:7.3f}")
    print(f"  weight_transfer_ratio     = {info.weight_transfer_ratio:+.3f}")
    print(f"  follow_through_ext_ratio  = {info.follow_through_extension_ratio:7.3f}")
    print(f"  forearm_orientation_deg   = {info.forearm_orientation_deg:+7.2f}")
    print(f"  elbow_angle_deg           = {info.elbow_angle_deg:7.2f}")


def validate_contact_frame_proxy_on_e2e_clip():
    """The one real ground-truth check available in this project (see
    ir/DESIGN.md open question #2): re-extracts game1/Clip5's pose sequence
    (same window ir/populate_e2e.py used) and compares the peak-wrist-speed
    proxy's picked frame against the real TrackNet-labeled hit frames."""
    print("\n" + "=" * 70)
    print("Contact-frame proxy validation against the one real ground-truth clip")
    print("=" * 70)

    real_hit_frames_local = [10, 11, 30, 53, 54]  # from ir/example_output/stroke_event_end_to_end.json
    print(f"Real TrackNet-labeled 'hit' frames (local, 0-59) in this window: {real_hit_frames_local}")
    print("NOTE: this window is a multi-shot rally segment (5 hit events), not one isolated "
          "stroke like the THETIS clips above -- StrokeGRU makes a single class prediction "
          "for the whole window with no per-frame localization, so there is no single "
          "unambiguous 'the' contact frame to validate against. Treating the hit frame "
          "closest to the window's temporal center as the best available single reference, "
          "since that is the shot the window is centered on -- this is a rough sanity check, "
          "not a precise ground-truth comparison, and is reported as such.")

    try:
        import cv2  # noqa: F401
        import torch  # noqa: F401
        import mediapipe as mp  # noqa: F401
    except ImportError as e:
        print(f"\n[skipped] required extraction dependencies not available in this environment ({e}); "
              f"re-run in the environment used for ir/populate_e2e.py to perform this check.")
        return

    from ir.populate_e2e import (  # noqa: E402
        GAME, CLIP, WINDOW_START, WINDOW_END, POSE_TASK_PATH, ensure_pose_task, extract_pose_sequence,
    )
    from ir.populate import ball_dataset_mod  # noqa: E402
    from mediapipe.tasks import python as mp_python  # noqa: E402
    from mediapipe.tasks.python import vision  # noqa: E402

    ensure_pose_task()
    clips = ball_dataset_mod.discover_clips()
    clip = next(c for c in clips if c.game == GAME and c.clip == CLIP)
    frame_paths = clip.frame_files[WINDOW_START:WINDOW_END]

    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    pose_seq = extract_pose_sequence(frame_paths, landmarker)

    features = extract_biomechanics_features(pose_seq)
    proxy_frame = features["contact_frame"]
    window_center = (len(pose_seq) - 1) / 2.0
    nearest_real_hit = min(real_hit_frames_local, key=lambda f: abs(f - window_center))

    print(f"\nProxy-picked contact frame (peak {features['dominant_wrist']}-wrist speed): {proxy_frame}")
    print(f"Nearest-to-center real hit frame (best available single reference): {nearest_real_hit}")
    print(f"Distance: {abs(proxy_frame - nearest_real_hit)} frames (window is 60 frames / ~2s @ 30fps)")


if __name__ == "__main__":
    print("Real StrokeClassification (THETIS) examples, one per collapsed class, first match in `test` split:")
    examples = load_one_example_per_class()
    for collapsed_class, (pose_seq, filename, native_label) in examples.items():
        info = build_biomechanics_info(pose_seq, source_file=f"StrokeClassification/processed/test/{filename}")
        n_valid = int(_valid_frame_mask(pose_seq).sum())
        print_result(collapsed_class, native_label, filename, info, n_valid, len(pose_seq))

    validate_contact_frame_proxy_on_e2e_clip()
