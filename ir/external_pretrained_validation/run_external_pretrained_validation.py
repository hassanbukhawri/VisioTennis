"""
Validates two externally-sourced pretrained checkpoints -- yastrebksv/TrackNet
(ball tracking) and yastrebksv/TennisCourtDetector (court keypoints) -- against
the SAME 3 real out-of-dataset clips and the SAME reprojection-error measurement
(ir/geometry.py's compute_homography) used in ir/run_ball_court_real_footage_validation.py
to validate this project's own from-scratch models (2026-08-12 entry,
PROJECT_LOG.md: 1.22m-6.27m reprojection error, confirmed generalization
failure). Purpose: cheap, no-training-required check of whether externally
pretrained checkpoints (trained on real broadcast footage, not this project's
narrower curated set) generalize better to real amateur footage.

LICENSE NOTE (checked before downloading, not glossed over): neither
yastrebksv/TrackNet nor yastrebksv/TennisCourtDetector ships a LICENSE file or
any license statement in its README -- confirmed via the GitHub API (license:
None on both) and by listing repo contents directly. This script and the
checkpoints are used here for local validation only, not redistributed or
shipped. If these numbers turn out to justify unblocking the heatmap feature,
shipping them in the product would still need the author's explicit permission
first, or a differently-licensed replacement -- NOT assumed clear by this test.

IMPORTANT ARCHITECTURE/PREPROCESSING NOTES (both external models, confirmed by
reading their source, not assumed):
- Both expect BGR input (trained directly off cv2.VideoCapture frames, no
  BGR->RGB conversion anywhere in their own code) -- this script deliberately
  does NOT convert to RGB, unlike this project's own models/validation script,
  to stay faithful to how these checkpoints were actually trained.
- Both normalize with a plain /255.0 -- no ImageNet mean/std subtraction
  (their CourtKeypointNet-equivalent, unlike ours, is NOT a pretrained ResNet
  backbone, so it never needed that normalization).
- Both were trained on inputs fixed-resized to 640x360 regardless of source
  resolution. Their own reference inference scripts hardcode a "scale=2" when
  mapping predictions back to real pixels, silently assuming native resolution
  is exactly 1280x720. That assumption is WRONG for these 3 real clips
  (1920x1080 native) and would already bias the comparison in the external
  models' favor by re-using stale hardcoded logic -- so, matching this
  project's own validation script's discipline, this script computes real
  sx/sy scale factors per clip from each clip's actual native resolution.
- TrackNet (ball): 9-channel input (3 stacked RGB[here: BGR]-frame windows,
  current frame first: [frame_t, frame_t-1, frame_t-2], per the reference
  infer_on_video.py's concatenation order), 256-channel per-pixel
  classification output (argmax gives a 0-255 "intensity" value per pixel --
  effectively a discretized heatmap regression, not a true probability). This
  script reuses this project's own methodology exactly: take the global peak
  pixel of the argmax intensity map, confidence = peak_intensity/255,
  visible = confidence >= 0.5. Deliberately skips the reference repo's own
  Hough-circle postprocessing so the visibility/confidence DEFINITION is
  identical to the one already used for this project's own TrackNet numbers --
  an apples-to-apples comparison, not two different criteria.
- TennisCourtDetector (court): 3-channel input (single BGR frame), 15-channel
  sigmoid heatmap output (14 real keypoints + 1 unused "center" point, per
  their own README -- only channels 0-13 used here). This script takes each
  channel's own peak-heatmap pixel (argmax), scaled to native resolution --
  again skipping their reference repo's Hough-circle + classical-CV
  refine_kps() stage, since this project's own CourtKeypointNet score is also
  a raw base-model number with no such refinement. Also skips their optional
  homography-based keypoint correction entirely for validity: that stage
  warps points onto the exact reference template, which would make a
  reprojection-error test circular.
- CONFIRMED (by reading court_reference.py's key_points construction and
  checking real-world ratios) that this repo's 14-keypoint order is IDENTICAL
  to ir/geometry.py's COURT_KEYPOINTS_REAL_WORLD_M order (both: far-baseline
  doubles L/R, near-baseline doubles L/R, far-baseline singles L/R,
  near-baseline singles L/R, far/near service-line singles L/R, far/near
  service-line center) -- so the 14 predicted keypoints plug directly into
  ir.geometry.compute_homography() with no reordering.

Usage:
    python ir/external_pretrained_validation/run_external_pretrained_validation.py
"""
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from ir.geometry import compute_homography  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_module(unique_name, rel_path):
    path = os.path.join(THIS_DIR, rel_path)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = module
    spec.loader.exec_module(module)
    return module


ext_ball_mod = _load_module("_ext_ball_model", "tracknet_src/model.py")
ext_court_mod = _load_module("_ext_court_model", "court_det_src/tracknet.py")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXT_BALL_CKPT = os.path.join(THIS_DIR, "checkpoints/tracknet_model_best.pt")
EXT_COURT_CKPT = os.path.join(THIS_DIR, "checkpoints/court_det_model.pt")

REAL_CLIPS = [
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip1_serve_practice.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip2_teen_practice.mp4"),
    os.path.join(REPO_ROOT, "mvp/real_footage_test/clip3_group_distant.mp4"),
]

DEBUG_OUT_DIR = os.path.join(THIS_DIR, "example_output")

KEYPOINT_CONNECTIONS = [
    (0, 1), (2, 3), (0, 2), (1, 3),
    (4, 6), (5, 7), (4, 5), (6, 7),
    (8, 9), (10, 11), (12, 13),
]

BALL_OUT_W, BALL_OUT_H = 640, 360
COURT_OUT_W, COURT_OUT_H = 640, 360


def read_real_frames_bgr(video_path):
    """Reads every frame at native resolution, BGR (no color conversion --
    both external models were trained on raw cv2.VideoCapture BGR frames)."""
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, w, h


def load_ext_ball_model():
    model = ext_ball_mod.BallTrackerNet(out_channels=256).to(DEVICE)
    state = torch.load(EXT_BALL_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def load_ext_court_model():
    model = ext_court_mod.BallTrackerNet(out_channels=15).to(DEVICE)
    state = torch.load(EXT_COURT_CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def run_ext_tracknet_on_clip(model, frames_bgr, native_w, native_h, n_windows=60, conf_thresh=0.5):
    """Same scaling/visibility-definition discipline as this project's own
    run_ball_court_real_footage_validation.py, applied to the external
    checkpoint -- so the confidence/visibility numbers are directly
    comparable. ONE disclosed difference: this project's own TrackNet outputs
    seq_len=3 heatmaps per forward pass (one per input frame), so 20 windows
    gives 60 detections; the external architecture outputs exactly ONE
    heatmap per 3-frame input (for the newest frame only), so n_windows=60
    here (not 20) to land on the same ~60-detection sample size per clip for
    a fair visibility-rate comparison."""
    seq_len = 3
    sx, sy = native_w / BALL_OUT_W, native_h / BALL_OUT_H
    n_total_windows = len(frames_bgr) // seq_len
    window_starts = np.linspace(0, (n_total_windows - 1) * seq_len, min(n_windows, n_total_windows))
    window_starts = sorted(set(int(s) - int(s) % seq_len for s in window_starts))

    detections = []
    with torch.no_grad():
        for start in window_starts:
            # reference repo's own concat order: current frame first, then t-1, t-2
            img_t = cv2.resize(frames_bgr[start + 2], (BALL_OUT_W, BALL_OUT_H), interpolation=cv2.INTER_AREA)
            img_t1 = cv2.resize(frames_bgr[start + 1], (BALL_OUT_W, BALL_OUT_H), interpolation=cv2.INTER_AREA)
            img_t2 = cv2.resize(frames_bgr[start + 0], (BALL_OUT_W, BALL_OUT_H), interpolation=cv2.INTER_AREA)
            x = np.concatenate((img_t, img_t1, img_t2), axis=2).astype(np.float32) / 255.0
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float().unsqueeze(0).to(DEVICE)

            logits = model(x)  # (1, 256, H*W)
            intensity = torch.argmax(logits[0], dim=0).float()  # (H*W,) argmax intensity 0-255
            peak_val, peak_idx = intensity.max(dim=0)
            py, px = int(peak_idx.item() // BALL_OUT_W), int(peak_idx.item() % BALL_OUT_W)
            confidence = float(peak_val.item()) / 255.0
            detections.append({
                "frame": start + 2,
                "x_px": px * sx, "y_px": py * sy,
                "confidence": confidence,
                "visible": confidence >= conf_thresh,
            })
    return detections


def run_ext_court_on_frame(model, frame_bgr, native_w, native_h):
    sx, sy = native_w / COURT_OUT_W, native_h / COURT_OUT_H
    img = cv2.resize(frame_bgr, (COURT_OUT_W, COURT_OUT_H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)[0]  # (15, 360, 640)
        heatmaps = torch.sigmoid(out).cpu().numpy()

    kps_native = []
    for k in range(14):
        hm = heatmaps[k]
        py, px = np.unravel_index(np.argmax(hm), hm.shape)
        kps_native.append((px * sx, py * sy))
    return np.array(kps_native)


def draw_debug_frame(frame_bgr, keypoints_native=None, ball_detections=None, out_path=None):
    img = frame_bgr.copy()
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

    ball_model = load_ext_ball_model()
    court_model = load_ext_court_model()

    results = []
    for clip_path in REAL_CLIPS:
        name = os.path.splitext(os.path.basename(clip_path))[0]
        print(f"\n{'=' * 70}\n[{name}]\n{'=' * 70}")
        if not os.path.exists(clip_path):
            print("  SKIPPED -- file not found")
            continue
        frames, native_w, native_h = read_real_frames_bgr(clip_path)
        print(f"  {len(frames)} real frames, native resolution {native_w}x{native_h}")

        # -- Ball tracking (external yastrebksv/TrackNet checkpoint) --
        detections = run_ext_tracknet_on_clip(ball_model, frames, native_w, native_h)
        n_visible = sum(1 for d in detections if d["visible"])
        confidences = [d["confidence"] for d in detections]
        print(f"  [ball-ext] {n_visible}/{len(detections)} frames scored >= 0.5 conf "
              f"({n_visible / len(detections):.0%} 'visible' rate)")
        print(f"  [ball-ext] confidence: min={min(confidences):.3f} max={max(confidences):.3f} "
              f"mean={np.mean(confidences):.3f}")
        mid_detections = [d for d in detections if abs(d["frame"] - len(frames) // 2) < 30]
        debug_ball_path = os.path.join(DEBUG_OUT_DIR, f"{name}_ball_ext_debug.png")
        if mid_detections:
            draw_debug_frame(frames[mid_detections[len(mid_detections) // 2]["frame"]],
                              ball_detections=mid_detections, out_path=debug_ball_path)
            print(f"  [ball-ext] debug overlay written to {debug_ball_path}")

        # -- Court homography (external yastrebksv/TennisCourtDetector checkpoint) --
        mid_frame = frames[len(frames) // 2]
        kps_native = run_ext_court_on_frame(court_model, mid_frame, native_w, native_h)
        reproj_err_m = None
        try:
            H, reproj_err_m = compute_homography(kps_native.tolist())
            print(f"  [court-ext] homography_reprojection_error_m={reproj_err_m:.4f} "
                  f"(compare: this project's own model on the same clip/method, per PROJECT_LOG.md 2026-08-12)")
        except ValueError as e:
            print(f"  [court-ext] homography FAILED: {e}")
        debug_court_path = os.path.join(DEBUG_OUT_DIR, f"{name}_court_ext_debug.png")
        draw_debug_frame(mid_frame, keypoints_native=kps_native, out_path=debug_court_path)
        print(f"  [court-ext] debug overlay written to {debug_court_path}")

        results.append({
            "clip": name,
            "ball_visible_rate": n_visible / len(detections),
            "ball_confidence_mean": float(np.mean(confidences)),
            "court_reprojection_error_m": reproj_err_m,
        })

    print(f"\n{'=' * 70}\n[summary]\n{'=' * 70}")
    for r in results:
        err_str = f"{r['court_reprojection_error_m']:.4f}m" if r["court_reprojection_error_m"] is not None else "FAILED"
        print(f"  {r['clip']}: ball_visible={r['ball_visible_rate']:.0%} "
              f"ball_conf_mean={r['ball_confidence_mean']:.3f} court_reproj_err={err_str}")

    print(f"\n[done] Inspect the debug PNGs in {DEBUG_OUT_DIR} before trusting any number above -- "
          f"same lesson as this project's own validation pass: a plausible reprojection error does "
          f"not by itself confirm the keypoints landed on the actual court lines.")
