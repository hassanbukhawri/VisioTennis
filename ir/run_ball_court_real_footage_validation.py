"""
Ball-tracking (TrackNet) + court-homography (CourtKeypointNet + ir/geometry.py)
validation on real, out-of-dataset footage.

GAP THIS FILLS: neither model has ever been run against the 3 real
non-dataset clips (mvp/real_footage_test/) that every other pipeline
component (MediaPipe Pose, StrokeGRU, stroke segmentation, the tennis-content
classifier) has been extensively validated against -- confirmed by searching
PROJECT_LOG.md before writing this. Both models have only ever run against
(a) their own training dataset's held-out split, and (b) ONE cross-domain
spot-check (CourtKeypointNet on a single frame from the UNRELATED
ball-tracking dataset, ir/populate.py's "true end-to-end" example,
reprojection error 7.37cm). This is the first time either model sees footage
from a different camera/source than anything in their own training data,
run at each clip's OWN real native resolution (not the training datasets'
fixed 1280x720 -- see the scaling note in each function below).

Per this project's established discipline (e.g. the keypoint-order
spot-check in the IR layer follow-up entry), this script also writes
annotated debug frames (predicted court keypoints / ball position drawn on
the real frame) so results can be visually confirmed, not just trusted as a
number.

Usage:
    python ir/run_ball_court_real_footage_validation.py
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.geometry import compute_homography  # noqa: E402


def _load_module(unique_name, rel_path):
    path = os.path.join(REPO_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


ball_model_mod = _load_module("_val_ball_model", "training/ball_tracking/model.py")
court_model_mod = _load_module("_val_court_model", "training/court_keypoints/model.py")
court_dataset_mod = _load_module("_val_court_dataset", "training/court_keypoints/dataset.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BALL_CKPT = os.path.join(REPO_ROOT, "training/ball_tracking/checkpoints/tracknet_best.pt")
COURT_CKPT = os.path.join(REPO_ROOT, "training/court_keypoints/checkpoints/court_kp_best.pt")

REAL_CLIPS = [
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip1_serve_practice.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip2_teen_practice.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip3_group_distant.mp4"),
]

DEBUG_OUT_DIR = os.path.join(REPO_ROOT, "ir/example_output/ball_court_real_footage_validation")

KEYPOINT_CONNECTIONS = [
    (0, 1), (2, 3), (0, 2), (1, 3),  # doubles baselines/sidelines
    (4, 6), (5, 7), (4, 5), (6, 7),  # singles sidelines
    (8, 9), (10, 11), (12, 13),      # service lines
]


def read_real_frames(video_path):
    """Reads every frame of a real clip at its OWN native resolution --
    deliberately not resized to the training datasets' fixed 1280x720
    (Ball Tracker+Bounce / Court_Detector_Keypoints both assume that), since
    an arbitrary real phone/stock clip has no reason to share it."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, w, h


def run_tracknet_on_clip(frames, native_w, native_h, n_windows=20, conf_thresh=0.5):
    """Runs TrackNet over up to n_windows non-overlapping seq_len=3 windows
    spread across the clip (not just the first N frames -- a real amateur
    clip's ball may not be near the camera/in a clean position early on),
    scaling output-space peaks to THIS clip's real native resolution, not
    the ball-tracking dataset's fixed 1280x720 (see module docstring)."""
    ckpt = torch.load(BALL_CKPT, map_location=DEVICE, weights_only=False)
    seq_len, out_w, out_h = ckpt["args"]["seq_len"], ckpt["args"]["out_w"], ckpt["args"]["out_h"]
    model = ball_model_mod.TrackNet(seq_len=seq_len, base_ch=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sx, sy = native_w / out_w, native_h / out_h
    n_total_windows = len(frames) // seq_len
    window_starts = np.linspace(0, (n_total_windows - 1) * seq_len, min(n_windows, n_total_windows))
    window_starts = sorted(set(int(s) - int(s) % seq_len for s in window_starts))

    detections = []
    with torch.no_grad():
        for start in window_starts:
            stack = []
            for i in range(start, start + seq_len):
                img = cv2.resize(frames[i], (out_w, out_h), interpolation=cv2.INTER_AREA)
                stack.append(img.astype(np.float32) / 255.0)
            x = np.concatenate(stack, axis=-1).transpose(2, 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(DEVICE)
            heatmaps = model(x)[0]
            for t in range(seq_len):
                hm = heatmaps[t]
                peak_val, peak_idx = hm.view(-1).max(dim=0)
                py, px = int(peak_idx.item() // out_w), int(peak_idx.item() % out_w)
                detections.append({
                    "frame": start + t,
                    "x_px": px * sx, "y_px": py * sy,
                    "confidence": float(peak_val.item()),
                    "visible": float(peak_val.item()) >= conf_thresh,
                })
    return detections, ckpt.get("val_pck")


def run_court_keypoints_on_frame(frame_rgb, native_w, native_h):
    """Runs CourtKeypointNet on one real frame, scaling predictions to THIS
    frame's real native resolution (not the court dataset's fixed
    1280x720)."""
    ckpt = torch.load(COURT_CKPT, map_location=DEVICE, weights_only=False)
    out_size = ckpt["args"]["out_size"]
    model = court_model_mod.CourtKeypointNet(n_keypoints=14, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img = cv2.resize(frame_rgb, (out_size, out_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - court_dataset_mod.IMAGENET_MEAN) / court_dataset_mod.IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()

    kps_norm = pred.reshape(14, 2)
    kps_native = kps_norm * np.array([native_w, native_h])
    return kps_native, ckpt.get("val_dist")


def draw_debug_frame(frame_rgb, keypoints_native=None, ball_detections=None, out_path=None):
    img = cv2.cvtColor(frame_rgb.copy(), cv2.COLOR_RGB2BGR)
    if keypoints_native is not None:
        for i, (x, y) in enumerate(keypoints_native):
            cv2.circle(img, (int(x), int(y)), 6, (0, 255, 255), -1)
            cv2.putText(img, str(i), (int(x) + 8, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for a, b in KEYPOINT_CONNECTIONS:
            pa, pb = keypoints_native[a], keypoints_native[b]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 0), 1)
    if ball_detections is not None:
        for d in ball_detections:
            color = (0, 0, 255) if d["visible"] else (128, 128, 128)
            cv2.circle(img, (int(d["x_px"]), int(d["y_px"])), 4, color, -1)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
    print(f"[setup] device={DEVICE}")

    for clip_path in REAL_CLIPS:
        name = os.path.splitext(os.path.basename(clip_path))[0]
        print(f"\n{'=' * 70}\n[{name}]\n{'=' * 70}")
        if not os.path.exists(clip_path):
            print("  SKIPPED -- file not found")
            continue
        frames, native_w, native_h = read_real_frames(clip_path)
        print(f"  {len(frames)} real frames, native resolution {native_w}x{native_h}")

        # -- Ball tracking --
        detections, ball_val_pck = run_tracknet_on_clip(frames, native_w, native_h)
        n_visible = sum(1 for d in detections if d["visible"])
        confidences = [d["confidence"] for d in detections]
        print(f"  [ball] TrackNet val_pck@4px (training-set metric)={ball_val_pck:.4f}")
        print(f"  [ball] {n_visible}/{len(detections)} frames scored >= conf_thresh on THIS clip "
              f"({n_visible / len(detections):.0%} 'visible' rate)")
        print(f"  [ball] confidence: min={min(confidences):.3f} max={max(confidences):.3f} "
              f"mean={np.mean(confidences):.3f}")
        mid_detections = [d for d in detections if abs(d["frame"] - len(frames) // 2) < 30]
        debug_ball_path = os.path.join(DEBUG_OUT_DIR, f"{name}_ball_debug.png")
        if mid_detections:
            draw_debug_frame(frames[mid_detections[len(mid_detections) // 2]["frame"]],
                              ball_detections=mid_detections, out_path=debug_ball_path)
            print(f"  [ball] debug overlay written to {debug_ball_path}")

        # -- Court homography --
        mid_frame = frames[len(frames) // 2]
        kps_native, court_val_dist = run_court_keypoints_on_frame(mid_frame, native_w, native_h)
        print(f"  [court] CourtKeypointNet val_dist_px (training-set metric)={court_val_dist:.2f}")
        try:
            H, reproj_err_m = compute_homography(kps_native.tolist())
            print(f"  [court] homography_reprojection_error_m={reproj_err_m:.4f} "
                  f"(compare: 2.2cm on ground-truth court-dataset keypoints, "
                  f"3.6-10.5cm on real predicted court-dataset keypoints, per PROJECT_LOG.md)")
        except ValueError as e:
            print(f"  [court] homography FAILED: {e}")
        debug_court_path = os.path.join(DEBUG_OUT_DIR, f"{name}_court_debug.png")
        draw_debug_frame(mid_frame, keypoints_native=kps_native, out_path=debug_court_path)
        print(f"  [court] debug overlay (predicted keypoints on real frame) written to {debug_court_path}")

    print(f"\n[done] Inspect the debug PNGs in {DEBUG_OUT_DIR} before trusting any number above -- "
          f"a plausible reprojection error does not by itself confirm the keypoints landed on the "
          f"actual court lines (same lesson as the IR layer's original keypoint-order spot-check).")
