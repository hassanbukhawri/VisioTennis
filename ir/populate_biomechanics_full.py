"""
Full-dataset run of ir/biomechanics.py across all ~1980 real
StrokeClassification (THETIS) clips (all three splits combined) -- not the
5 hand-picked sanity examples in ir/populate_biomechanics.py. Purpose:
(1) find out how many clips extract cleanly vs. fail/get excluded, bucketed
by why; (2) check whether the three bugs fixed on the 5-example pass
(zero-padded frames, signed-axis angle, single-frame shoulder-width
collapse) recur at scale, now that fixes are in place -- not assumed fixed
just because 5 examples looked right; (3) produce the real per-class
feature distributions that percentile ranking (ir/biomechanics_percentiles.py)
is built against, per ir/DESIGN.md's decision to use population percentiles
within our own data.

Two exclusion buckets (clip's biomechanics are computed but NOT included in
the percentile-reference population, because the numbers are judged
unreliable):
  - insufficient_valid_frames: fewer than MIN_VALID_FRAMES real (non-zero)
    pose frames in the 60-frame window -- see ir/biomechanics.py's
    _valid_frame_mask docstring for why zero frames exist at all.
  - degenerate_camera_angle: the CONTACT FRAME ITSELF (not the window
    average -- window-averaging already protects the ratio features, see
    the bug-3 fix in ir/DESIGN.md, but shoulder_rotation_deg/
    hip_rotation_deg/shoulder_hip_separation_deg are computed directly from
    the single contact frame's landmarks, not smoothed) has a near-collapsed
    shoulder line, making the angle numerically unstable regardless of the
    ratio fix.

Usage:
    python ir/populate_biomechanics_full.py
"""
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics import (  # noqa: E402
    extract_biomechanics_features, _valid_frame_mask, shoulder_width, _xy, _dist,
    LEFT_WRIST, RIGHT_WRIST, LEFT_SHOULDER, RIGHT_SHOULDER,
)

DATA_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "processed")

NATIVE_TO_COLLAPSED = {
    0: "backhand", 1: "backhand", 2: "backhand", 3: "volley", 4: "serve",
    5: "forehand", 6: "forehand", 7: "forehand", 8: "volley", 9: "serve",
    10: "serve", 11: "smash",
}

FEATURE_COLS = [
    "wrist_height_ratio", "shoulder_rotation_deg", "hip_rotation_deg",
    "shoulder_hip_separation_deg", "knee_flex_left_deg", "knee_flex_right_deg",
    "stance_width_ratio", "weight_transfer_ratio", "follow_through_extension_ratio",
    "forearm_orientation_deg", "elbow_angle_deg",
]

# Data-driven thresholds -- chosen by inspecting this project's own
# distributions (see PROJECT_LOG.md 2026-07-19 entry), not guessed blind:
# n_valid ranges 24-60 across all 1980 clips with a long tail below 45 and
# only 6 clips (0.3%) below 30 (half the window) -- 30 is the natural
# "less than half the window is real" cutoff.
MIN_VALID_FRAMES = 30
# Window-averaged shoulder width across all 1980 clips ranges 0.032-0.181,
# no near-zero cluster (the earlier single-frame collapse of 0.0048 does
# NOT recur in the window-averaged statistic at scale) -- but
# shoulder_rotation_deg/hip_rotation_deg/shoulder_hip_separation_deg are
# computed directly from the single CONTACT FRAME's landmarks, not
# smoothed by the window-averaging fix, so they remain exposed to the same
# collapse bug 3 fixed for the ratio features. Checked, not assumed: the
# contact-frame shoulder-width distribution has a long left tail down to
# ~0.0002 (near-total collapse), and clips in its bottom 10%
# (sw <= ~0.0194) show a mean shoulder_hip_separation_deg of 41.7 deg vs.
# 19.5 deg overall (roughly double) -- a real, verified correlation between
# low contact-frame shoulder width and unstable angle output, not an
# assumption. Threshold set at that empirically-justified 10th-percentile
# point, rounded to a clean number.
DEGENERATE_CONTACT_SW_THRESHOLD = 0.02

# A 4th instance of the same root-cause pattern (dividing by a quantity
# that can collapse near zero on a single frame), found only at
# full-dataset scale, not on the 5-example pass: follow_through_extension_ratio
# divides by the dominant wrist-to-shoulder distance AT THE CONTACT FRAME,
# which is small whenever the arm is tucked close to the body at that
# instant (a folded/compact arm position, or a landmark-noise artifact).
# Checked, not assumed: of the 116 clips with follow_through_extension_ratio
# > 5 across the whole dataset, 89 (77%) fall in the bottom 5% of this
# contact-arm distance -- the same "verify before excluding" standard
# applied to the camera-angle threshold above. Threshold set at that
# empirically-justified 5th-percentile point.
DEGENERATE_ARM_DISTANCE_THRESHOLD = 0.034


def load_all_clips():
    frames = []
    for split in ("train", "val", "test"):
        X = np.load(os.path.join(DATA_ROOT, split, f"X_{split}.npy"))
        labels = pd.read_csv(os.path.join(DATA_ROOT, split, f"labels_{split}.csv"))
        labels["collapsed"] = labels["label_idx"].map(NATIVE_TO_COLLAPSED)
        labels["split"] = split
        for i, row in labels.iterrows():
            frames.append((row["filename"], row["label"], row["collapsed"], row["split"], X[i]))
    return frames


def run():
    clips = load_all_clips()
    print(f"Loaded {len(clips)} real StrokeClassification clips (all 3 splits combined).")

    rows = []
    bucket_counts = {
        "no_valid_frames": 0, "insufficient_valid_frames": 0,
        "degenerate_camera_angle": 0, "degenerate_arm_extension": 0, "clean": 0,
    }

    for filename, native_label, collapsed, split, pose_seq in clips:
        valid_mask = _valid_frame_mask(pose_seq)
        n_valid = int(valid_mask.sum())

        if n_valid == 0:
            bucket_counts["no_valid_frames"] += 1
            rows.append(dict(filename=filename, collapsed=collapsed, split=split,
                              bucket="no_valid_frames", n_valid=n_valid))
            continue
        if n_valid < MIN_VALID_FRAMES:
            bucket_counts["insufficient_valid_frames"] += 1
            rows.append(dict(filename=filename, collapsed=collapsed, split=split,
                              bucket="insufficient_valid_frames", n_valid=n_valid))
            continue

        features = extract_biomechanics_features(pose_seq)
        cf = pose_seq[features["contact_frame"]]
        contact_sw = shoulder_width(cf)
        wrist_idx = LEFT_WRIST if features["dominant_wrist"] == "left" else RIGHT_WRIST
        shoulder_idx = LEFT_SHOULDER if features["dominant_wrist"] == "left" else RIGHT_SHOULDER
        contact_arm_dist = _dist(_xy(cf, wrist_idx), _xy(cf, shoulder_idx))

        if contact_sw < DEGENERATE_CONTACT_SW_THRESHOLD:
            bucket = "degenerate_camera_angle"
        elif contact_arm_dist < DEGENERATE_ARM_DISTANCE_THRESHOLD:
            bucket = "degenerate_arm_extension"
        else:
            bucket = "clean"
        bucket_counts[bucket] += 1

        row = dict(filename=filename, collapsed=collapsed, split=split, bucket=bucket,
                    n_valid=n_valid, contact_frame_sw=contact_sw, contact_arm_dist=contact_arm_dist)
        row.update(features)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df, bucket_counts


if __name__ == "__main__":
    df, bucket_counts = run()

    total = len(df)
    print("\n" + "=" * 70)
    print("Extraction outcome, all clips")
    print("=" * 70)
    for bucket, count in bucket_counts.items():
        print(f"  {bucket:28s} {count:5d}  ({100*count/total:.1f}%)")

    print("\nBreakdown of excluded/flagged clips by collapsed class:")
    excl = df[df["bucket"] != "clean"]
    print(pd.crosstab(excl["collapsed"], excl["bucket"]))

    clean = df[df["bucket"] == "clean"].copy()
    print(f"\nClean population for percentile reference: {len(clean)}/{total} clips")
    print(clean["collapsed"].value_counts())

    print("\nContact-frame shoulder-width distribution (all clips reaching that stage, "
          "i.e. clean + degenerate_camera_angle):")
    sw_all = df.loc[df["bucket"].isin(["clean", "degenerate_camera_angle"]), "contact_frame_sw"]
    print(sw_all.describe())
    print(sw_all.quantile([0.001, 0.005, 0.01, 0.02, 0.05, 0.1]))

    print("\n" + "=" * 70)
    print("Per-class, per-feature distribution (clean population only)")
    print("=" * 70)
    for feat in FEATURE_COLS:
        print(f"\n--- {feat} ---")
        print(clean.groupby("collapsed")[feat].agg(["min", "max", "mean", "std", "count"]).round(3))

    out_path = os.path.join(REPO_ROOT, "ir", "example_output", "biomechanics_full_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[done] wrote full per-clip results (all buckets) to {out_path}")
