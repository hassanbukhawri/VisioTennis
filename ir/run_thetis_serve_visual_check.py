"""
Cheap visual diagnostic: are THETIS's own serve clips visually distinctive
from its own backhand/forehand clips, even to a human, in the training data
itself? If not, that's evidence the dataset -- not either model architecture
-- is the ceiling on the serve class specifically. Pulls 3 frames per clip
(30%/55%/80% through) from a handful of serve clips and a handful of
backhand/forehand clips, same manual-verification spirit as the real-footage
frame grabs done earlier this session, and tiles them into one contact-sheet
image for direct visual inspection (not automated -- graded by eye after).

Usage:
    python ir/run_thetis_serve_visual_check.py
"""
import os

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_ROOT = os.path.join(REPO_ROOT, "data", "VisioTennis", "Dataset", "StrokeClassification", "raw_rgb")
OUT_PATH = os.path.join(REPO_ROOT, "ir", "example_output", "thetis_serve_visual_check.png")

# (native_class, filename) -- 2 players x 3 serve types, 1-2 players each for backhand/forehand family
CLIPS = [
    ("flat_service", "p3_serflat_s1.avi"),
    ("flat_service", "p20_serflat_s1.avi"),
    ("kick_service", "p3_serkick_s1.avi"),
    ("kick_service", "p20_serkick_s1.avi"),
    ("slice_service", "p3_serslice_s1.avi"),
    ("slice_service", "p20_serslice_s1.avi"),
    ("backhand", "p3_backhand_s1.avi"),
    ("backhand", "p20_backhand_s1.avi"),
    ("backhand2hands", "p3_backhand2h_s1.avi"),
    ("forehand_flat", "p3_foreflat_s1.avi"),
    ("forehand_flat", "p20_foreflat_s1.avi"),
    ("forehand_openstands", "p3_foreopen_s1.avi"),
]

FRAC_POINTS = (0.30, 0.55, 0.80)
THUMB_W, THUMB_H = 220, 165
LABEL_H = 22


def grab_frames(path, fracs):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for f in fracs:
        idx = min(int(n * f), n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        frames.append(frame if ok else np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8))
    cap.release()
    return frames, n


def main():
    rows = []
    for native_class, fname in CLIPS:
        path = os.path.join(RAW_ROOT, native_class, fname)
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            continue
        frames, n = grab_frames(path, FRAC_POINTS)
        thumbs = [cv2.resize(f, (THUMB_W, THUMB_H)) for f in frames]
        row_img = np.concatenate(thumbs, axis=1)

        label_bar = np.full((LABEL_H, row_img.shape[1], 3), 255, dtype=np.uint8)
        label = f"{native_class} / {fname} (n={n})"
        cv2.putText(label_bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        rows.append(np.concatenate([label_bar, row_img], axis=0))
        print(f"[grabbed] {native_class}/{fname} n_frames={n} fracs={FRAC_POINTS}")

    sheet = np.concatenate(rows, axis=0)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, sheet)
    print(f"\n[done] wrote {OUT_PATH} ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
