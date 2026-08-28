"""
Step 5 of Path B: evaluate the trained r2plus1d_18 video model
(checkpoints/video/r2plus1d_best.pt) against the real, hand-verified
non-THETIS footage in mvp/real_footage_test/ -- the actual point of this
whole exercise, since the pose-based StrokeGRU classifier has a
repeatedly-confirmed 0/3 real-footage serve failure (PROJECT_LOG.md,
2026-08-02 and 2026-08-12 entries) that this fine-tune exists to test.

Ground-truth instances used here are NOT re-derived or guessed -- they are
the exact hand-verified labels and frame locations already on record in
PROJECT_LOG.md from prior diagnostic work on this same footage:

  - clip1_serve_practice.mp4: 2 confirmed real serves (2026-08-02 entry),
    candidate windows from ir/example_output/classifier_diagnostic/
    clip1_serve_practice/result.json -- contact_frame=155 (window 125-185,
    toss-initiation moment) and contact_frame=235 (window 205-265, jump-serve
    contact point, unambiguous). The same result.json also has 2 confirmed
    NON-stroke false positives (frame 48, frame 108 -- walking with racket)
    which are excluded here: a 5-way stroke classifier has no correct answer
    for "not a stroke," so scoring them would be meaningless, not informative.
  - clip2_teen_practice.mp4: 9 confirmed backhand instances (2026-08-02
    stroke-segmentation validation), contact frames 34/142/213/259/340/385/
    439/493/574.
  - clip3_group_distant.mp4: EXCLUDED -- known pose-tracking failure on this
    clip (2026-08-02 diagnostic), never treated as valid for classification-
    accuracy conclusions anywhere in this project.
  - clip4_game_rally.mp4: whole-clip true label backhand (2026-08-12 entry,
    confirmed by eye, two-handed grip).
  - clip5_serve_setup.mp4: whole-clip true label serve (2026-08-12 entry,
    confirmed via ball-toss frame + full trophy-position swing frame --
    the THIRD independently-confirmed real-footage serve, and the one the
    pose-based model called "backhand" at 99.3% confidence).
  - clip6_solo_practice.mp4: whole-clip true label forehand (2026-08-12
    entry, confirmed by eye, one-handed swing prep).

For clip1/clip2 (candidate-window instances), a 16-frame clip is taken
centered on the hand-verified contact_frame (contact-8 : contact+8,
clamped to valid range) -- consistent with this project's CLIP_LEN=16 and
with dataset_video.py's own deterministic-center-window convention for
val/test. For clip4/5/6 (whole-clip, single-stroke recordings, following
the same treat-as-one-upload convention run_new_clips_pipeline_check.py
uses for these exact clips), the 16-frame clip is centered on the whole
raw video.

This gives 14 hand-labeled real-footage instances across 3 classes
(serve=3, backhand=10, forehand=1) -- KNOWN THIN, most classes have zero
real-footage coverage (no real volley or smash instance exists anywhere in
this project), and forehand has exactly one instance. Per explicit
instruction: report the numbers plainly, flag this thinness in the same
breath, do not declare success or failure from this set alone.

Usage:
    python training/stroke_classification/evaluate_real_footage.py
"""
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from dataset_video import CLIP_LEN, RESIZE_HW, CROP_HW, KINETICS_MEAN, KINETICS_STD, COLLAPSED_CLASSES
from train_video import build_model

FOOTAGE_DIR = os.path.join(REPO_ROOT, "mvp", "real_footage_test")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "video", "r2plus1d_best.pt")

# (filename, true_label, contact_frame_or_None)
# contact_frame=None means "whole clip, single stroke recording" -> center window.
INSTANCES = [
    ("clip1_serve_practice.mp4", "serve", 155),
    ("clip1_serve_practice.mp4", "serve", 235),
    ("clip2_teen_practice.mp4", "backhand", 34),
    ("clip2_teen_practice.mp4", "backhand", 142),
    ("clip2_teen_practice.mp4", "backhand", 213),
    ("clip2_teen_practice.mp4", "backhand", 259),
    ("clip2_teen_practice.mp4", "backhand", 340),
    ("clip2_teen_practice.mp4", "backhand", 385),
    ("clip2_teen_practice.mp4", "backhand", 439),
    ("clip2_teen_practice.mp4", "backhand", 493),
    ("clip2_teen_practice.mp4", "backhand", 574),
    ("clip4_game_rally.mp4", "backhand", None),
    ("clip5_serve_setup.mp4", "serve", None),
    ("clip6_solo_practice.mp4", "forehand", None),
]


def load_clip_frames(filepath, contact_frame):
    cap = cv2.VideoCapture(filepath)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if contact_frame is None:
        max_start = max(n_frames - CLIP_LEN, 0)
        start = max_start // 2
    else:
        start = contact_frame - CLIP_LEN // 2
        start = max(0, min(start, max(n_frames - CLIP_LEN, 0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(CLIP_LEN):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    while len(frames) < CLIP_LEN and frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0), n_frames, start


def preprocess(frames_bgr):
    """Deterministic center-crop path, identical to VideoClipDataset's val/test branch."""
    y0 = (RESIZE_HW[0] - CROP_HW[0]) // 2
    x0 = (RESIZE_HW[1] - CROP_HW[1]) // 2
    out = np.empty((CLIP_LEN, CROP_HW[0], CROP_HW[1], 3), dtype=np.float32)
    for t in range(CLIP_LEN):
        frame = cv2.cvtColor(frames_bgr[t], cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (RESIZE_HW[1], RESIZE_HW[0]), interpolation=cv2.INTER_LINEAR)
        frame = frame[y0:y0 + CROP_HW[0], x0:x0 + CROP_HW[1], :]
        out[t] = frame.astype(np.float32) / 255.0
    out = (out - KINETICS_MEAN) / KINETICS_STD
    clip = torch.from_numpy(out).permute(3, 0, 1, 2).float()
    return clip


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device={device}")
    print(f"[setup] loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"[setup] checkpoint from epoch={ckpt['epoch']} val_macroF1={ckpt['val_macro_f1']:.3f}")

    model = build_model(len(COLLAPSED_CLASSES), device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    results = []
    with torch.no_grad():
        for fname, true_label, contact_frame in INSTANCES:
            filepath = os.path.join(FOOTAGE_DIR, fname)
            frames_bgr, n_frames, start = load_clip_frames(filepath, contact_frame)
            clip = preprocess(frames_bgr).unsqueeze(0).to(device)

            logits = model(clip)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())
            pred_label = COLLAPSED_CLASSES[pred_idx]
            correct = pred_label == true_label

            results.append(dict(
                filename=fname, true_label=true_label, contact_frame=contact_frame,
                window_start=start, n_frames=n_frames, pred_label=pred_label,
                confidence=float(probs[pred_idx]), probs=probs, correct=correct,
            ))

            loc = f"contact_frame={contact_frame}" if contact_frame is not None else "whole-clip center"
            print(f"[{fname:28s}] true={true_label:10s} ({loc:28s}) -> "
                  f"pred={pred_label:10s} conf={probs[pred_idx]:.3f} "
                  f"{'OK' if correct else '** WRONG **'}")

    print("\n" + "=" * 78)
    print("PER-CLASS BREAKDOWN (real footage, hand-verified labels)")
    print("=" * 78)
    for cls in COLLAPSED_CLASSES:
        cls_results = [r for r in results if r["true_label"] == cls]
        if not cls_results:
            print(f"  {cls:10s}: 0 real-footage instances -- NO COVERAGE in this validation set")
            continue
        n_correct = sum(r["correct"] for r in cls_results)
        print(f"  {cls:10s}: {n_correct}/{len(cls_results)} correct")
        for r in cls_results:
            tag = f"frame {r['contact_frame']}" if r["contact_frame"] is not None else "whole-clip"
            print(f"      {r['filename']:28s} {tag:14s} -> {r['pred_label']:10s} "
                  f"({r['confidence']:.1%}) {'OK' if r['correct'] else 'WRONG'}")

    total_correct = sum(r["correct"] for r in results)
    print(f"\n[overall] {total_correct}/{len(results)} correct across all labeled real-footage instances")
    print("\n[caveat] This is a 14-instance validation set covering only 3 of 5 classes "
          "(no real volley or smash instance exists anywhere in this project); forehand "
          "has exactly 1 instance. Numbers above are reported as-is, not as a generalization "
          "claim -- see PROJECT_LOG.md for the full caveat.")


if __name__ == "__main__":
    main()
