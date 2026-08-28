"""
Evaluates a simple-averaging ensemble of the clean-split baseline checkpoint
and the THETIS-augmentation checkpoint (currently deployed) against the same
14-instance real-footage validation set used throughout this investigation --
same ground-truth instances, same hand-verified window boundaries, reused
directly from evaluate_real_footage_pose_based.py rather than redefined, so
results are directly comparable to every number already on record.

Ensemble rule: run both ONNX models on the identical pose-sequence input,
average their raw (T=1) softmax class_probabilities elementwise, argmax the
average. No temperature applied at this stage -- calibration (if the
ensemble is deployed) is fit AFTER this comparison, same order of
operations as every other checkpoint this session (evaluate raw
correctness first, calibrate confidence honesty second -- they're
independent axes).

Usage:
    python training/stroke_classification/evaluate_ensemble_real_footage.py
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from backend.app import decode_and_resample_video, run_pose_extraction, POSE_TASK_PATH, STROKE_CLASSES
from evaluate_real_footage_pose_based import INSTANCES, load_window_bounds, decode_all_frames, FOOTAGE_DIR

CLEAN_SPLIT_ONNX = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/clean_split/stroke_gru_clean.onnx")
AUGMENTED_ONNX = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/augmented/stroke_gru_augmented.onnx")


def raw_softmax_probs(pose_seq, session):
    ins = session.get_inputs()[0]
    x = pose_seq.astype(np.float32)[None]
    logits = session.run(None, {ins.name: x})[0][0]
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum()
    return {c: float(p) for c, p in zip(STROKE_CLASSES, probs)}


def main():
    print(f"[setup] clean-split model: {CLEAN_SPLIT_ONNX}")
    print(f"[setup] augmented model:   {AUGMENTED_ONNX}")
    clean_session = ort.InferenceSession(CLEAN_SPLIT_ONNX, providers=["CPUExecutionProvider"])
    aug_session = ort.InferenceSession(AUGMENTED_ONNX, providers=["CPUExecutionProvider"])

    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    full_pose_cache = {}
    results = []

    for fname, true_label, contact_frame in INSTANCES:
        clip_path = os.path.join(FOOTAGE_DIR, fname)
        clip_stem = fname.replace(".mp4", "")

        if contact_frame is None:
            frames, fps, n_real = decode_and_resample_video(clip_path)
            pose_seq, n_detected = run_pose_extraction(frames, landmarker)
        else:
            if clip_stem not in full_pose_cache:
                all_frames = decode_all_frames(clip_path)
                full_seq, n_det_full = run_pose_extraction(all_frames, landmarker)
                full_pose_cache[clip_stem] = full_seq
            full_seq = full_pose_cache[clip_stem]
            bounds = load_window_bounds(clip_stem)
            window_start, window_end = bounds[contact_frame]
            pose_seq = full_seq[window_start:window_end]

        clean_probs = raw_softmax_probs(pose_seq, clean_session)
        aug_probs = raw_softmax_probs(pose_seq, aug_session)
        ensemble_probs = {c: (clean_probs[c] + aug_probs[c]) / 2 for c in STROKE_CLASSES}
        predicted = max(ensemble_probs, key=ensemble_probs.get)
        correct = predicted == true_label

        clean_pred = max(clean_probs, key=clean_probs.get)
        aug_pred = max(aug_probs, key=aug_probs.get)

        results.append(dict(
            filename=fname, true_label=true_label, contact_frame=contact_frame,
            predicted=predicted, confidence=ensemble_probs[predicted], correct=correct,
            clean_pred=clean_pred, clean_conf=clean_probs[clean_pred],
            aug_pred=aug_pred, aug_conf=aug_probs[aug_pred],
        ))
        loc = f"contact_frame={contact_frame}" if contact_frame is not None else "whole-clip"
        print(f"  [{fname:28s} {loc:22s}] true={true_label:10s} | "
              f"clean={clean_pred:10s}({clean_probs[clean_pred]:.0%}) "
              f"aug={aug_pred:10s}({aug_probs[aug_pred]:.0%}) -> "
              f"ENSEMBLE={predicted:10s}({ensemble_probs[predicted]:.0%}) "
              f"{'OK' if correct else '** WRONG **'}")

    print(f"\n{'=' * 70}\nPER-CLASS BREAKDOWN -- ensemble (clean-split + augmented, simple avg)\n{'=' * 70}")
    for cls in STROKE_CLASSES:
        cls_results = [r for r in results if r["true_label"] == cls]
        if not cls_results:
            print(f"  {cls:10s}: 0 instances -- no coverage")
            continue
        n_correct = sum(r["correct"] for r in cls_results)
        print(f"  {cls:10s}: {n_correct}/{len(cls_results)} correct")

    total_correct = sum(r["correct"] for r in results)
    print(f"\n[overall] ensemble: {total_correct}/{len(results)} correct ({total_correct/len(results):.1%})")
    print(f"[reference] augmented alone (currently deployed): 8/14 (57.1%)")
    print(f"[reference] clean-split alone: 5/14 (35.7%)")

    with open(os.path.join(REPO_ROOT, "training/stroke_classification/ensemble_eval_result.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
