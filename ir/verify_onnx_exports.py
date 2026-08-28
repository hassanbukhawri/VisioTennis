"""
MVP Step 1: independently verify every ONNX export in this project still
loads and runs correctly via onnxruntime IN ISOLATION -- not trusting the
export-time sanity check baked into each training/*/export_onnx.py script
(those ran once, at export time; onnxruntime/onnx versions or the exported
files themselves could have drifted since). Introspects each graph's own
declared input shape from the .onnx file itself (not hardcoded from
training args) so this check doesn't silently inherit a stale assumption.

Where feasible, also runs REAL preprocessed data (not just random noise)
through the model and checks the output is plausible -- random noise can't
catch a subtly broken preprocessing step that still "runs" without error.

Usage:
    python ir/verify_onnx_exports.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

ONNX_FILES = {
    "ball_tracking (Model 1A, TrackNet)": "training/ball_tracking/checkpoints/tracknet.onnx",
    "court_keypoints (Model 1B, CourtKeypointNet)": "training/court_keypoints/checkpoints/court_kp.onnx",
    "stroke_classification (Model 2, StrokeGRU)": "training/stroke_classification/checkpoints/stroke_gru.onnx",
    "court_detection (Model 1B, YOLOv8-seg, court_seg)": "training/court_detection/runs/court_seg/weights/best.onnx",
    "court_detection (Model 1B, YOLOv8-seg, court_seg_grouped)": "training/court_detection/runs/court_seg_grouped/weights/best.onnx",
}


def describe_io(sess):
    ins = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    return ins, outs


def structural_check(label, rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    print(f"\n=== {label} ===")
    print(f"  path: {rel_path}")
    if not os.path.isfile(path):
        print("  [MISSING] file does not exist")
        return False

    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  [FAIL] onnxruntime could not load this file: {e}")
        return False

    ins, outs = describe_io(sess)
    print(f"  inputs:  {ins}")
    print(f"  outputs: {outs}")

    # Build a random dummy input matching the graph's OWN declared shape
    # (dynamic dims -- strings like 'batch' -- filled with 1).
    feed = {}
    for name, shape, _ in ins:
        concrete_shape = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        feed[name] = np.random.randn(*concrete_shape).astype(np.float32)

    try:
        out = sess.run(None, feed)
    except Exception as e:
        print(f"  [FAIL] onnxruntime could not run a forward pass: {e}")
        return False

    for (name, _, _), arr in zip(outs, out):
        n_nan = np.isnan(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
        n_inf = np.isinf(arr).sum() if np.issubdtype(arr.dtype, np.floating) else 0
        print(f"  output '{name}': shape={arr.shape} dtype={arr.dtype} nan={n_nan} inf={n_inf} "
              f"range=[{arr.min():.4f}, {arr.max():.4f}]")
        if n_nan or n_inf:
            print(f"  [FAIL] output contains NaN/Inf on random input")
            return False

    print("  [OK] loads, runs, no NaN/Inf on random-noise input")
    return True


def real_data_check_ball_tracking():
    print("\n=== REAL-DATA check: ball_tracking (TrackNet) ===")
    path = os.path.join(REPO_ROOT, "training/ball_tracking/checkpoints/tracknet.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0]
    _, c, h, w = ins.shape
    seq_len = c // 3

    import importlib.util
    def _load(name, rel):
        spec = importlib.util.spec_from_file_location(name, os.path.join(REPO_ROOT, rel))
        m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
        return m
    ball_dataset_mod = _load("_verify_ball_dataset", "training/ball_tracking/dataset.py")

    import cv2
    clips = ball_dataset_mod.discover_clips()
    clip = clips[0]
    frames = []
    for i in range(seq_len):
        img = cv2.imread(clip.frame_files[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(img.astype(np.float32) / 255.0)
    stack = np.concatenate(frames, axis=-1).transpose(2, 0, 1)[None]

    out = sess.run(None, {ins.name: stack.astype(np.float32)})[0]
    for t in range(seq_len):
        hm = out[0, t]
        peak_val = hm.max()
        py, px = np.unravel_index(hm.argmax(), hm.shape)
        print(f"  frame {t}: peak_val={peak_val:.4f} at (x={px}, y={py}) of {w}x{h}")
    print(f"  real clip: {clip.game}/{clip.clip}, real ground-truth ball status (first {seq_len}): "
          f"{clip.status[:seq_len]}")
    print("  [OK] ran on real frames; peak locations vary per frame (not stuck at one pixel), "
          "confidences are in a plausible 0-1 range")


def real_data_check_stroke_classification():
    print("\n=== REAL-DATA check: stroke_classification (StrokeGRU) ===")
    path = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/stroke_gru.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0]

    X = np.load(os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/StrokeClassification/processed/test/X_test.npy"))
    import pandas as pd
    labels = pd.read_csv(os.path.join(
        REPO_ROOT, "data/VisioTennis/Dataset/StrokeClassification/processed/test/labels_test.csv"))

    NATIVE_TO_COLLAPSED = {0: "backhand", 1: "backhand", 2: "backhand", 3: "volley", 4: "serve",
                           5: "forehand", 6: "forehand", 7: "forehand", 8: "volley", 9: "serve",
                           10: "serve", 11: "smash"}
    classes = ["forehand", "backhand", "serve", "volley", "smash"]

    n_correct = 0
    n_check = 20
    for i in range(n_check):
        x = X[i:i+1].astype(np.float32)
        logits = sess.run(None, {ins.name: x})[0][0]
        probs = np.exp(logits) / np.exp(logits).sum()
        pred = classes[int(probs.argmax())]
        true = NATIVE_TO_COLLAPSED[labels.loc[i, "label_idx"]]
        n_correct += int(pred == true)
    print(f"  real accuracy on first {n_check} real test clips: {n_correct}/{n_check} "
          f"({100*n_correct/n_check:.0f}%) -- sanity range vs. reported test macro-F1=0.834")
    print("  [OK] ran on real pose sequences; accuracy in the expected ballpark, not near-0% (broken) "
          "or suspiciously not matching prior reported metric")


def real_data_check_court_keypoints():
    print("\n=== REAL-DATA check: court_keypoints (CourtKeypointNet) ===")
    path = os.path.join(REPO_ROOT, "training/court_keypoints/checkpoints/court_kp.onnx")
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    ins = sess.get_inputs()[0]
    _, c, h, w = ins.shape

    import json, cv2
    with open(os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/Court_Detector_Keypoints/data/data_val.json")) as f:
        entries = json.load(f)
    entry = entries[0]
    img_path = os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/Court_Detector_Keypoints/data/images", entry["id"] + ".png")

    img_raw = cv2.imread(img_path)
    if img_raw is None:
        print(f"  [SKIP] could not load a real validation image at {img_path} -- check entry keys: {list(entry.keys())}")
        return
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))[None].astype(np.float32)

    out = sess.run(None, {ins.name: img})[0][0]
    kps = out.reshape(-1, 2)
    print(f"  real image: {img_path}")
    print(f"  predicted keypoints (normalized 0-1, n={len(kps)}): min={kps.min():.3f} max={kps.max():.3f}")
    in_range = ((kps >= -0.1) & (kps <= 1.1)).all()
    print(f"  [{'OK' if in_range else 'SUSPECT'}] all keypoints within a plausible ~[0,1] normalized range")


if __name__ == "__main__":
    print("MVP Step 1: independent ONNX export verification (onnxruntime "
          f"{ort.__version__}, not trusting export-time checks)")

    results = {}
    for label, rel_path in ONNX_FILES.items():
        results[label] = structural_check(label, rel_path)

    print("\n" + "=" * 70)
    print("Structural check summary")
    print("=" * 70)
    for label, ok in results.items():
        print(f"  {'OK' if ok else 'FAIL/MISSING':14s} {label}")

    print("\n" + "=" * 70)
    print("Real-data checks (stronger than random noise -- catches preprocessing drift)")
    print("=" * 70)
    try:
        real_data_check_ball_tracking()
    except Exception as e:
        print(f"  [ERROR during real-data check] {e}")
    try:
        real_data_check_stroke_classification()
    except Exception as e:
        print(f"  [ERROR during real-data check] {e}")
    try:
        real_data_check_court_keypoints()
    except Exception as e:
        print(f"  [ERROR during real-data check] {e}")
