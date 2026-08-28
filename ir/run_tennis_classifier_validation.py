"""
Validation harness for ir/tennis_classifier.py's zero-shot tennis-content
gate. Decodes each clip with the SAME resampling backend/app.py's
analyze-stroke endpoint uses (decode_and_resample_video, 60 frames), so this
validates against the exact input shape the real pipeline will feed the
classifier -- not a hand-picked still image.

DEFAULT_THRESHOLD was LOCKED at 0.5 (2026-08-12) based on the results this
script produces -- see ir/tennis_classifier.py's module docstring for the
full validated-against / explicitly-not-validated-against breakdown before
trusting this classifier against content materially different from what's
covered here (real physical-activity footage; NOT adversarial/edited/
produced content like a movie premiere).

Usage:
    python ir/run_tennis_classifier_validation.py [extra_clip_path ...]

Any extra paths passed on the command line are scored too, labeled
"unlabeled" -- lets this same script be re-run once the two blocked clips
are located, without editing the script.
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.app import decode_and_resample_video  # noqa: E402
from ir.tennis_classifier import DEFAULT_THRESHOLD, classify_frames  # noqa: E402

REAL_TENNIS_CLIPS = [
    ("clip1_serve_practice", os.path.join(REPO_ROOT, "mvp/real_footage_test/clip1_serve_practice.mp4"), True),
    ("clip2_teen_practice", os.path.join(REPO_ROOT, "mvp/real_footage_test/clip2_teen_practice.mp4"), True),
    ("clip3_group_distant", os.path.join(REPO_ROOT, "mvp/real_footage_test/clip3_group_distant.mp4"), True),
]

# Real, already-in-repo non-tennis footage gathered as additional negatives
# (per explicit instruction: "any other negatives you gather"), separate from
# and NOT a substitute for the two specifically-referenced clips (Spider-Man
# premiere, ball-over-net) that remain unlocated. Penn Action
# (data/VisioTennis/Dataset/Penn_Action) is a real multi-sport action dataset
# already present in this repo -- picked a mix of harder negatives (swing
# motions, the closest visual analog to a tennis stroke) and easier ones.
PENN_ACTION_DIR = os.path.join(REPO_ROOT, "data/VisioTennis/Dataset/Penn_Action/frames")
PENN_ACTION_NEGATIVES = [
    ("penn_baseball_swing_0168", "0168"),
    ("penn_golf_swing_0789", "0789"),
    ("penn_bowl_0481", "0481"),
    ("penn_strum_guitar_1890", "1890"),
    ("penn_jumping_jacks_1037", "1037"),
]


def _load_frame_folder(folder: str, n: int = 20) -> list[np.ndarray]:
    """Loads up to n evenly-spaced JPEG frames from a Penn Action-style
    frames/<clip_id>/ folder as RGB uint8 arrays, matching
    decode_and_resample_video()'s output format."""
    paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if len(paths) > n:
        idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
        paths = [paths[i] for i in idx]
    return [np.array(Image.open(p).convert("RGB")) for p in paths]


if __name__ == "__main__":
    extra_paths = sys.argv[1:]
    cases = list(REAL_TENNIS_CLIPS) + [(os.path.basename(p), p, None) for p in extra_paths]

    print(f"[setup] DEFAULT_THRESHOLD = {DEFAULT_THRESHOLD} (PROVISIONAL -- see module docstring)")
    print(f"[setup] {len(cases)} video clip(s) + {len(PENN_ACTION_NEGATIVES)} Penn Action negative(s) to score\n")

    results = []
    for name, path, expected_is_tennis in cases:
        if not os.path.exists(path):
            print(f"[{name}] SKIPPED -- file not found at {path}")
            continue
        try:
            frames, fps, n_real_frames = decode_and_resample_video(path)
        except ValueError as e:
            # A pre-existing pipeline constraint (bad_video_file, e.g. this
            # clip's real fps pushes it over MAX_RAW_FRAMES) rejects this clip
            # before it would ever reach the tennis classifier in production --
            # not a failure of THIS check, just untestable via the real
            # pipeline path. Reported, not swallowed silently.
            print(f"[{name}] SKIPPED -- rejected by decode_and_resample_video() before reaching the "
                  f"classifier (pre-existing pipeline constraint): {e}")
            continue
        result = classify_frames(frames)
        label = "expected TENNIS" if expected_is_tennis is True else \
                ("expected NOT-TENNIS" if expected_is_tennis is False else "unlabeled")
        ok = (expected_is_tennis is None) or (result["is_tennis"] == expected_is_tennis)
        print(f"[{name}] ({label}) tennis_score={result['tennis_score']:.4f} "
              f"threshold={result['threshold']} is_tennis={result['is_tennis']} "
              f"n_frames_sampled={result['n_frames_sampled']} "
              f"{'OK' if ok else 'MISMATCH -- WOULD BE WRONGLY CLASSIFIED'}")
        results.append((name, expected_is_tennis, result, ok))

    print()
    for name, clip_id in PENN_ACTION_NEGATIVES:
        folder = os.path.join(PENN_ACTION_DIR, clip_id)
        if not os.path.isdir(folder):
            print(f"[{name}] SKIPPED -- folder not found at {folder}")
            continue
        frames = _load_frame_folder(folder)
        result = classify_frames(frames)
        ok = not result["is_tennis"]
        print(f"[{name}] (expected NOT-TENNIS, real Penn Action footage) tennis_score={result['tennis_score']:.4f} "
              f"threshold={result['threshold']} is_tennis={result['is_tennis']} "
              f"n_frames_sampled={result['n_frames_sampled']} "
              f"{'OK' if ok else 'MISMATCH -- WOULD BE WRONGLY CLASSIFIED'}")
        results.append((name, False, result, ok))

    n_labeled = sum(1 for _, exp, _, _ in results if exp is not None)
    n_ok = sum(1 for _, exp, _, ok in results if exp is not None and ok)
    print(f"\n[summary] {n_ok}/{n_labeled} labeled clips classified as expected at threshold={DEFAULT_THRESHOLD}.")
    print("[summary] Threshold LOCKED at this value (2026-08-12) based on this validation set -- "
          "real physical-activity footage only. NOT validated against adversarial/edited/produced "
          "content (e.g. a movie premiere) -- see ir/tennis_classifier.py's module docstring.")
