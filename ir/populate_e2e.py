"""
True single-video end-to-end IR example (as opposed to the three-way
composites in ir/populate.py's __main__, which stitch together three
independently-real but unrelated source clips -- see that file's module
docstring). This script runs real TrackNet + real CourtKeypointNet + a real
MediaPipe Pose extraction -> StrokeGRU pipeline on ONE genuine rally clip:
Ball Tracker+Bounce game1/Clip5, frames 29-88 (60 frames, ~2s at 30fps,
Djokovic vs Federer, Shanghai 2014). game1 is in TrackNet's held-out
VALIDATION split (see ball_tracking/dataset.py:split_by_game), so this is
also a genuine generalization check for Model 1A, not a memorized clip.

Why MediaPipe Pose extraction is needed here (and wasn't needed in
populate.py): StrokeGRU consumes 60-frame x 99-dim MediaPipe Pose landmark
sequences (33 landmarks x (x, y, z)). The THETIS dataset that trained it
ships only pre-extracted features (X_test.npy) -- no raw video exists in this
repo -- so populate.py's composite examples reuse a pre-extracted THETIS test
sample verbatim. To get a stroke prediction on OUR OWN video instead of
THETIS's, pose landmarks must be extracted from these frames ourselves. Done
here via mediapipe's PoseLandmarker Tasks API (mediapipe==0.10.35 -- the
legacy `mp.solutions.pose` API that whatever pipeline originally produced
X_*.npy likely used was removed in this mediapipe version; PoseLandmarker is
the same underlying BlazePose 33-point model family, standard/interchangeable
output format, but not bit-identical to the original extractor, which is
unknown/undocumented -- see caveats in DESIGN.md).

Player localization caveat: THETIS clips are already tightly cropped to a
single player filling most of the frame. This Ball-Tracker frame is a wide
broadcast shot where the player occupies a small fraction of the image --
PoseLandmarker on the RAW full frame detects 0/60 frames (tested). A simple
HSV red-shirt color-blob tracker (this player, Federer per the on-screen
scoreboard, wears a red shirt in this rally) crops a generous box around him
per frame before pose detection. This is a crude, clip-specific hack, NOT a
general player-tracking solution -- multi-player detection/tracking remains
explicit future work (DESIGN.md decision #1 / "Not yet built").

Usage:
    python ir/populate_e2e.py
"""
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.schema import (  # noqa: E402
    StrokeEvent, BallInfo, BallFramePosition, PixelPoint, CourtPoint,
    StrokeInfo, ModelProvenance, FrameRange, STROKE_CLASSES,
)
from ir.geometry import compute_homography, pixel_to_court  # noqa: E402
from ir.populate import (  # noqa: E402
    ball_dataset_mod, ball_model_mod, court_dataset_mod, court_model_mod,
    stroke_model_mod, DEVICE, BALL_NATIVE_W, BALL_NATIVE_H, COURT_NATIVE_W, COURT_NATIVE_H,
)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

GAME, CLIP = "game1", "Clip5"
WINDOW_START, WINDOW_END = 29, 89  # 60 frames, local frame 0..59
POSE_TASK_PATH = os.path.join(REPO_ROOT, "training/stroke_classification/pose_landmarker_full.task")
POSE_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                  "pose_landmarker_full/float16/latest/pose_landmarker_full.task")


def ensure_pose_task():
    if not os.path.isfile(POSE_TASK_PATH):
        import urllib.request
        print(f"[setup] downloading {POSE_TASK_URL} -> {POSE_TASK_PATH}")
        urllib.request.urlretrieve(POSE_TASK_URL, POSE_TASK_PATH)


# ---------------------------------------------------------------------------
# NOT A PIPELINE COMPONENT -- ONE-OFF DEMO WORKAROUND, DO NOT REUSE.
#
# This function exists ONLY to make ir/populate_e2e.py's single demo clip
# (game1/Clip5, this player wears a red shirt) producible without a real
# player-detection model, which this project does not have. It is not
# player tracking: it hardcodes a red-shirt HSV color range that happens to
# match this one player in this one clip, has no identity/re-identification
# logic, and would silently fail or lock onto the wrong thing on any other
# clip (different kit color, a red ad board, a ball kid's uniform, etc.).
# Multi-player detection/tracking is explicit future work (see
# ir/DESIGN.md decision #1 / "Not yet built") -- this is not a first draft
# of that, it is scaffolding for one illustrative script and should be
# deleted, not extended, when real player detection is eventually built.
# ---------------------------------------------------------------------------
def _demo_only_find_player_box_by_red_shirt(img, margin=90):
    """Crude HSV red-shirt blob tracker for ONE specific clip's ONE player --
    see the banner comment above and ir/DESIGN.md's true-e2e section. Not a
    general player-tracking solution; do not call this from anywhere other
    than this demo script."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    m2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = m1 | m2
    mask[:150, :] = 0  # exclude crowd/scoreboard/ballkid-uniform area above the court
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    largest = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = max(0, cx - margin), max(0, cy - margin - 30)
    x1, y1 = min(img.shape[1], cx + margin), min(img.shape[0], cy + margin + 60)
    return x0, y0, x1, y1


def extract_pose_sequence(frame_paths, landmarker):
    """Returns (seq_len, 99) float32 array: 33 MediaPipe landmarks x (x, y, z)
    per frame, x/y normalized to the per-frame player crop (matching how
    THETIS's tightly-cropped clips implicitly normalize pose to the frame)."""
    n_landmarks = 33
    feats = np.zeros((len(frame_paths), n_landmarks * 3), dtype=np.float32)
    n_missing = 0
    for i, path in enumerate(frame_paths):
        img = cv2.imread(path)
        box = _demo_only_find_player_box_by_red_shirt(img)
        if box is None:
            n_missing += 1
            continue
        x0, y0, x1, y1 = box
        crop = img[y0:y1, x0:x1]
        crop_up = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)
        if not result.pose_landmarks:
            n_missing += 1
            continue
        lm = result.pose_landmarks[0]
        feats[i] = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)
    if n_missing:
        print(f"[pose] WARNING: {n_missing}/{len(frame_paths)} frames had no detected pose "
              f"(left as zeros -- see caveat in module docstring)")
    return feats


def populate_ball_info_window(clip, checkpoint_path, start, end):
    """Same TrackNet inference as ir/populate.py:populate_ball_info, but over
    an explicit [start, end) frame window instead of the clip's first N frames
    -- lets ball/court/stroke all reference the SAME real time window of the
    SAME real clip."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    seq_len = ckpt["args"]["seq_len"]
    out_w, out_h = ckpt["args"]["out_w"], ckpt["args"]["out_h"]
    model = ball_model_mod.TrackNet(seq_len=seq_len, base_ch=32).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    sx_native = BALL_NATIVE_W / out_w
    sy_native = BALL_NATIVE_H / out_h
    n_frames = end - start
    assert n_frames % seq_len == 0, f"window length {n_frames} not divisible by seq_len {seq_len}"

    trajectory = []
    with torch.no_grad():
        for wstart in range(0, n_frames, seq_len):
            frames = []
            for t in range(seq_len):
                img = cv2.imread(clip.frame_files[start + wstart + t])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
                frames.append(img.astype(np.float32) / 255.0)
            stack = np.concatenate(frames, axis=-1).transpose(2, 0, 1)
            x = torch.from_numpy(np.ascontiguousarray(stack)).float().unsqueeze(0).to(DEVICE)
            heatmaps = model(x)[0]
            for t in range(seq_len):
                hm = heatmaps[t]
                peak_val, peak_idx = hm.view(-1).max(dim=0)
                py = int(peak_idx.item() // out_w)
                px = int(peak_idx.item() % out_w)
                local_frame = wstart + t
                global_frame = start + local_frame
                raw_status = clip.status[global_frame]
                event_type = None
                if not np.isnan(raw_status):
                    event_type = {0.0: "flight", 1.0: "hit", 2.0: "bounce"}[float(raw_status)]
                trajectory.append(BallFramePosition(
                    frame=local_frame,
                    pixel=PixelPoint(x_px=px * sx_native, y_px=py * sy_native),
                    court=None,
                    visible=float(peak_val.item()) >= 0.5,
                    confidence=min(max(float(peak_val.item()), 0.0), 1.0),
                    event_type=event_type,
                ))

    hit_frames = [p.frame for p in trajectory if p.event_type == "hit"]
    bounce_frames = [p.frame for p in trajectory if p.event_type == "bounce"]
    return BallInfo(
        trajectory=trajectory,
        contact_frame=hit_frames[0] if hit_frames else None,
        bounce_frame=bounce_frames[0] if bounce_frames else None,
        source_model=ModelProvenance(
            component="ball", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_pck@4px", metric_value=ckpt.get("val_pck", float("nan")),
            source_file=f"{clip.game}/{clip.clip} frames[{start}:{end}]",
        ),
    ), trajectory, hit_frames, bounce_frames


def populate_court_info_from_frame(img_path, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    out_size = ckpt["args"]["out_size"]
    model = court_model_mod.CourtKeypointNet(n_keypoints=14, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_raw = cv2.imread(img_path)
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - court_dataset_mod.IMAGENET_MEAN) / court_dataset_mod.IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = model(x)[0].cpu().numpy()

    kps_norm = pred.reshape(14, 2)
    kps_native = kps_norm * np.array([COURT_NATIVE_W, COURT_NATIVE_H])
    keypoints_px = [PixelPoint(x_px=float(x_), y_px=float(y_)) for x_, y_ in kps_native]
    H, reproj_err = compute_homography(kps_native.tolist())

    from ir.schema import CourtInfo
    court_info = CourtInfo(
        keypoints_px=keypoints_px,
        homography_matrix=H.tolist(),
        homography_reprojection_error_m=reproj_err,
        source_model=ModelProvenance(
            component="court", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_dist_px", metric_value=ckpt.get("val_dist", float("nan")),
            source_file=os.path.basename(img_path),
        ),
    )
    return court_info, H, reproj_err


def populate_stroke_info_from_pose(pose_seq, checkpoint_path, source_file):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = stroke_model_mod.StrokeGRU(
        input_dim=99, hidden_dim=ckpt["args"]["hidden_dim"], num_layers=ckpt["args"]["num_layers"],
        num_classes=len(STROKE_CLASSES), dropout=ckpt["args"]["dropout"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(pose_seq).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    class_probabilities = {c: float(p) for c, p in zip(STROKE_CLASSES, probs)}
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    return StrokeInfo(
        predicted_class=predicted_class,
        confidence=class_probabilities[predicted_class],
        class_probabilities=class_probabilities,
        source_model=ModelProvenance(
            component="stroke", checkpoint=os.path.basename(checkpoint_path),
            metric_name="val_macro_f1", metric_value=ckpt.get("val_macro_f1", float("nan")),
            source_file=source_file,
        ),
    )


if __name__ == "__main__":
    ball_ckpt = os.path.join(REPO_ROOT, "training/ball_tracking/checkpoints/tracknet_best.pt")
    court_ckpt = os.path.join(REPO_ROOT, "training/court_keypoints/checkpoints/court_kp_best.pt")
    stroke_ckpt = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/stroke_gru_best.pt")

    print(f"[setup] device={DEVICE}, clip={GAME}/{CLIP} frames[{WINDOW_START}:{WINDOW_END}]")

    clips = ball_dataset_mod.discover_clips()
    clip = next(c for c in clips if c.game == GAME and c.clip == CLIP)

    print("[1/3] running real TrackNet inference over the window...")
    ball_info, trajectory, hit_frames, bounce_frames = populate_ball_info_window(
        clip, ball_ckpt, WINDOW_START, WINDOW_END)
    print(f"      contact_frame(local)={ball_info.contact_frame} bounce_frame(local)={ball_info.bounce_frame}")
    print(f"      all hit frames(local)={hit_frames} bounce frames(local)={bounce_frames}")

    mid_frame_global = WINDOW_START + (WINDOW_END - WINDOW_START) // 2
    court_frame_path = clip.frame_files[mid_frame_global]
    print(f"[2/3] running real CourtKeypointNet on frame {mid_frame_global} of the SAME clip "
          f"({os.path.basename(court_frame_path)})...")
    court_info, H, reproj_err = populate_court_info_from_frame(court_frame_path, court_ckpt)
    print(f"      homography_reprojection_error_m={reproj_err:.4f}")

    for i, p in enumerate(ball_info.trajectory):
        x_m, y_m = pixel_to_court(H, p.pixel.x_px, p.pixel.y_px)
        ball_info.trajectory[i] = p.model_copy(update={"court": CourtPoint(x_m=x_m, y_m=y_m)})

    print("[3/3] extracting real MediaPipe Pose sequence + running real StrokeGRU inference...")
    ensure_pose_task()
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(pose_options)
    frame_paths = clip.frame_files[WINDOW_START:WINDOW_END]
    pose_seq = extract_pose_sequence(frame_paths, landmarker)

    stroke_info = populate_stroke_info_from_pose(
        pose_seq, stroke_ckpt,
        source_file=f"{GAME}/{CLIP} frames[{WINDOW_START}:{WINDOW_END}] (real MediaPipe Pose extraction, "
                    f"no ground-truth stroke label -- this is real broadcast footage, not a THETIS clip)",
    )
    print(f"      predicted_class={stroke_info.predicted_class} confidence={stroke_info.confidence:.3f}")
    print(f"      class_probabilities={ {k: round(v,4) for k,v in stroke_info.class_probabilities.items()} }")

    event = StrokeEvent(
        player_id="federer_e2e_demo",
        player_side="near",
        source_video=f"{GAME}/{CLIP} frames[{WINDOW_START}:{WINDOW_END}] "
                      f"(TRUE END-TO-END: ball, court, and stroke all from this ONE real clip -- "
                      f"contrast with ir/example_output/stroke_event_*.json, which are composites)",
        frame_range=FrameRange(start_frame=0, end_frame=len(ball_info.trajectory) - 1, fps=30.0),
        ball=ball_info,
        court=court_info,
        stroke=stroke_info,
    )

    out_path = os.path.join(REPO_ROOT, "ir", "example_output", "stroke_event_end_to_end.json")
    with open(out_path, "w") as f:
        f.write(event.model_dump_json(indent=2))
    print(f"\n[done] wrote {out_path}")
