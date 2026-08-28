"""
First real-footage validation of ir.stroke_segmentation.detect_candidate_strokes.

Runs the full pipeline -- decode every frame of a real (non-synthetic,
non-THETIS) video, MediaPipe pose extraction, candidate stroke detection,
StrokeGRU classification per candidate window -- on one of the 3 real
footage clips in mvp/real_footage_test/ (mvp/stress_test_pose.py; Pexels
stock clips, chosen there for MediaPipe detection-rate testing, reused
here since they're the only real non-dataset footage in the repo).

This is a detection-accuracy check, not a coaching check: it reports
detected-vs-actual stroke count and characterizes errors (false positives
= non-stroke wrist-speed peaks detected as strokes; false negatives = real
strokes the peak search missed), against a HAND count from the emitted
contact-sheet images -- not tuned to fit this one clip. See
PROJECT_LOG.md for the reported numbers and the honest error breakdown.

Also runs detection WITHOUT the trunk-rotation filter (wrist-speed-only,
the original first-pass behavior) alongside the default (filter on), and
reports both candidate sets -- so the filter's actual effect on this clip
is visible, not just asserted.

Usage:
    python ir/run_stroke_segmentation_trial.py [clip_name]
        clip_name defaults to clip1_serve_practice.mp4

Writes to ir/example_output/stroke_segmentation_trial/<clip_stem>/:
    speed_plot.png            -- wrist-speed + trunk-rotation signals, both candidate sets marked
    contact_sheet_*.png       -- tiled, frame-index-labeled thumbnails for hand-counting
    candidate_<i>_frame<f>.png -- full-res frame at each filter-ON candidate's contact frame
    result.json                -- both candidate sets + classifications + run metadata
"""
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import onnxruntime as ort

from backend.app import run_pose_extraction, run_stroke_classification, POSE_TASK_PATH, STROKE_GRU_ONNX_PATH
from ir.stroke_segmentation import detect_candidate_strokes, compute_wrist_speed_series, compute_trunk_rotation_series, WINDOW_LEN
from ir.biomechanics import _valid_frame_mask, determine_dominant_wrist, LEFT_WRIST, RIGHT_WRIST

CLIPS_DIR = os.path.join(REPO_ROOT, "mvp", "real_footage_test")


def decode_all_frames(video_path, max_frames=2000):
    """RAW, unresampled decode -- every readable frame, unlike
    backend.app.decode_and_resample_video which deliberately resamples down
    to 60 frames (single-stroke MVP assumption this trial exists to move
    past)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def make_contact_sheets(frames, out_dir, stem, sample_every=4, segment_seconds=8.0, fps=25.0):
    """Tiled, frame-index-labeled thumbnails for HAND counting real strokes
    -- split into multi-second segments so each thumbnail stays legible
    instead of cramming an entire clip into one tiny grid."""
    frames_per_segment = max(1, int(segment_seconds * fps))
    n = len(frames)
    paths = []
    seg_idx = 0
    for seg_start in range(0, n, frames_per_segment):
        seg_end = min(n, seg_start + frames_per_segment)
        idxs = list(range(seg_start, seg_end, sample_every))
        if not idxs:
            continue
        thumbs = []
        for i in idxs:
            th = cv2.resize(frames[i], (220, 124))
            th = cv2.cvtColor(th, cv2.COLOR_RGB2BGR)
            cv2.putText(th, str(i), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            thumbs.append(th)
        ncols = 8
        nrows = int(np.ceil(len(thumbs) / ncols))
        sheet = np.zeros((nrows * 124, ncols * 220, 3), dtype=np.uint8)
        for k, th in enumerate(thumbs):
            r, c = divmod(k, ncols)
            sheet[r * 124:(r + 1) * 124, c * 220:(c + 1) * 220] = th
        out_path = os.path.join(out_dir, f"contact_sheet_{seg_idx}_frames{seg_start}-{seg_end}.png")
        cv2.imwrite(out_path, sheet)
        paths.append(out_path)
        seg_idx += 1
    return paths


if __name__ == "__main__":
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "clip1_serve_practice.mp4"
    video_path = os.path.join(CLIPS_DIR, clip_name)
    stem = os.path.splitext(clip_name)[0]
    out_dir = os.path.join(REPO_ROOT, "ir", "example_output", "stroke_segmentation_trial", stem)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/5] decoding {video_path} (raw, unresampled)...")
    frames, fps = decode_all_frames(video_path)
    print(f"  {len(frames)} frames @ {fps:.2f} fps ({len(frames) / fps:.1f}s)")

    print("[2/5] running MediaPipe pose extraction on every frame...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    pose_seq, n_detected = run_pose_extraction(frames, landmarker)
    print(f"  pose detected on {n_detected}/{len(frames)} frames ({100 * n_detected / len(frames):.1f}%)")

    valid_mask = _valid_frame_mask(pose_seq)
    dom_side = determine_dominant_wrist(pose_seq, valid_mask)
    dom_idx = LEFT_WRIST if dom_side == "left" else RIGHT_WRIST
    print(f"  dominant wrist: {dom_side}")

    print("[3/6] detecting candidate strokes (with and without the rotation filter)...")
    candidates_no_rotation = detect_candidate_strokes(pose_seq, dominant_wrist_idx=dom_idx, use_rotation_filter=False)
    candidates = detect_candidate_strokes(pose_seq, dominant_wrist_idx=dom_idx, use_rotation_filter=True)
    rejected_by_rotation = [c for c in candidates_no_rotation
                             if c["contact_frame"] not in {x["contact_frame"] for x in candidates}]
    print(f"  wrist-speed only (filter OFF): {len(candidates_no_rotation)} candidates")
    print(f"  wrist-speed + trunk-rotation (filter ON): {len(candidates)} candidates")
    print(f"  rotation filter removed {len(rejected_by_rotation)} candidate(s) at "
          f"frame(s) {[c['contact_frame'] for c in rejected_by_rotation]}")
    speeds = compute_wrist_speed_series(pose_seq, dom_idx, valid_mask)
    rotation = compute_trunk_rotation_series(pose_seq, valid_mask)

    print("[4/6] classifying each filter-ON candidate window with StrokeGRU...")
    session = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])
    for c in candidates:
        window = pose_seq[c["window_start"]:c["window_end"]]
        assert window.shape[0] == WINDOW_LEN, window.shape
        predicted_class, class_probabilities = run_stroke_classification(window, session)
        c["predicted_class"] = predicted_class
        c["confidence"] = class_probabilities[predicted_class]
        c["timestamp_s"] = c["contact_frame"] / fps
        print(f"    frame {c['contact_frame']:4d} (t={c['timestamp_s']:5.2f}s) speed={c['speed']:.4f} "
              f"rotation_peak={c['trunk_rotation_peak']:.2f} valid_frac={c['window_valid_fraction']:.2f} "
              f"-> {predicted_class} ({c['confidence']:.2f})")
    for c in rejected_by_rotation:
        c["timestamp_s"] = c["contact_frame"] / fps

    print("[5/6] writing plot, contact sheets, and per-candidate frame crops...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    t_axis = np.arange(len(speeds)) / fps

    ax1.plot(t_axis, speeds, color="steelblue", linewidth=0.8, label="dominant-wrist speed")
    for c in candidates:
        ax1.axvline(c["timestamp_s"], color="crimson", linestyle="--", linewidth=1)
        ax1.annotate(f"{c['predicted_class']}\nf{c['contact_frame']}", (c["timestamp_s"], c["speed"]),
                     fontsize=7, color="crimson", ha="center", va="bottom")
    for c in rejected_by_rotation:
        ax1.axvline(c["timestamp_s"], color="grey", linestyle=":", linewidth=1.2)
        ax1.annotate(f"rejected\nf{c['contact_frame']}", (c["timestamp_s"], c["speed"]),
                     fontsize=7, color="grey", ha="center", va="bottom")
    invalid_t = t_axis[~valid_mask]
    for it in invalid_t:
        ax1.axvspan(it, it + 1 / fps, color="grey", alpha=0.15)
    ax1.set_ylabel("wrist displacement\nper frame (uncalibrated)")
    ax1.set_title(f"{clip_name}: {len(candidates)} candidate(s) survive both filters (red), "
                  f"{len(rejected_by_rotation)} rejected by rotation filter alone (grey, dotted)")
    ax1.legend(loc="upper right")

    ax2.plot(t_axis, rotation, color="darkorange", linewidth=0.8, label="trunk (shoulder/hip) rotational speed")
    for c in candidates:
        ax2.axvline(c["timestamp_s"], color="crimson", linestyle="--", linewidth=1)
    for c in rejected_by_rotation:
        ax2.axvline(c["timestamp_s"], color="grey", linestyle=":", linewidth=1.2)
    for it in invalid_t:
        ax2.axvspan(it, it + 1 / fps, color="grey", alpha=0.15)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("trunk rotation\n(deg/frame)")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plot_path = os.path.join(out_dir, "speed_plot.png")
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)

    sheet_paths = make_contact_sheets(frames, out_dir, stem, fps=fps)

    for i, c in enumerate(candidates):
        frame_bgr = cv2.cvtColor(frames[c["contact_frame"]], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"candidate_{i}_frame{c['contact_frame']}.png"), frame_bgr)
    for i, c in enumerate(rejected_by_rotation):
        frame_bgr = cv2.cvtColor(frames[c["contact_frame"]], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, f"rejected_{i}_frame{c['contact_frame']}.png"), frame_bgr)

    print("[6/6] writing result.json...")
    result = dict(
        clip_name=clip_name, n_frames=len(frames), fps=fps, duration_s=len(frames) / fps,
        n_pose_detected=n_detected, pose_detection_rate=n_detected / len(frames),
        dominant_wrist=dom_side,
        n_candidates_filter_off=len(candidates_no_rotation), candidates_filter_off=candidates_no_rotation,
        n_candidates_filter_on=len(candidates), candidates_filter_on=candidates,
        rejected_by_rotation=rejected_by_rotation,
        plot_path=plot_path, contact_sheet_paths=sheet_paths,
    )
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[done] wrote results to {out_dir}")
    print(f"  -> hand-count strokes from {sheet_paths}")
    print(f"  -> compare against {len(candidates)} filter-ON candidates in speed_plot.png / result.json")
