"""
Multi-stroke candidate detection over a full (unwindowed) pose sequence --
finds candidate contact points as local maxima in dominant-wrist speed, so a
longer video (multiple strokes, walking between them) can be broken into
StrokeGRU-sized windows instead of assuming the whole clip is one stroke.
That whole-clip assumption is explicit and deliberate elsewhere --
ir.biomechanics.find_peak_wrist_speed_frame collapses to one global argmax,
and backend.app.decode_and_resample_video's docstring calls "finding a
stroke inside a longer video" an explicit non-goal for the MVP. This module
is the fill-in for that gap.

No new model, no new training data: reuses the same wrist-speed signal and
valid-frame handling as ir.biomechanics (dominant wrist, zero-padded-frame
exclusion), and windows around each candidate at the existing StrokeGRU
input length (60 frames, training/stroke_classification/model.py) so the
classifier runs unmodified.

Wrist-speed magnitude alone under-suppresses: the first real-footage pass
(ir/run_stroke_segmentation_trial.py, clip1_serve_practice.mp4) found that
walking with the racket swinging at the player's side produces wrist-speed
peaks on the same order as -- in that case exceeding -- genuine strokes.
The kinetic-chain filter below (compute_trunk_rotation_series /
min_rotation_percentile) adds a second, independent signal: a real stroke
turns the trunk into the shot (shoulders and hips rotate rapidly around
contact), while a walking arm-swing is a pendulum at the shoulder joint
with the torso continuing to face roughly forward. Reuses
ir.biomechanics._angle_vs_horizontal_deg -- the exact same formula already
used once per clip to compute shoulder_rotation_deg/hip_rotation_deg at
the contact frame -- applied per-frame across the whole sequence instead;
still no new model, no new landmarks.

See ir/run_stroke_segmentation_trial.py for real-footage validation
across all 3 available real clips and its honest false-positive/false-
negative characterization per clip -- these suppression knobs were NOT
tuned to fit any one clip; they're an analytically-reasonable first guess,
reported against real footage, not backfitted to it. If a threshold that
works on one clip fails on another, that's reported as a limitation, not
smoothed over by picking a compromise value.
"""
import numpy as np
from scipy.signal import find_peaks

from ir.biomechanics import (
    LEFT_SHOULDER, LEFT_WRIST, RIGHT_SHOULDER, RIGHT_WRIST,
    _angle_vs_horizontal_deg, _dist, _valid_frame_mask, _xy,
    determine_dominant_wrist,
)

WINDOW_LEN = 60  # StrokeGRU input length (training/stroke_classification/model.py) -- fixed, not tunable here.


def compute_wrist_speed_series(pose_seq: np.ndarray, dominant_wrist_idx: int,
                                valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-frame dominant-wrist displacement from the previous frame, for
    every frame in a full (potentially multi-stroke) sequence. Same
    per-transition validity rule as ir.biomechanics.find_peak_wrist_speed_frame
    (a transition touching a zero/missing-pose frame never contributes a
    speed value), but returns the whole array instead of collapsing to a
    single global argmax -- multi-stroke detection needs every local peak,
    not just the biggest one."""
    if valid_mask is None:
        valid_mask = _valid_frame_mask(pose_seq)
    speeds = np.zeros(len(pose_seq))
    for t in range(1, len(pose_seq)):
        if valid_mask[t] and valid_mask[t - 1]:
            speeds[t] = _dist(_xy(pose_seq[t - 1], dominant_wrist_idx), _xy(pose_seq[t], dominant_wrist_idx))
    return speeds


def _angular_speed_series(angles: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Per-frame |change in angle| from the previous frame, in degrees,
    wrapped correctly across the +/-180 boundary (a line's orientation angle
    can flip sign between two frames that are physically almost identical --
    e.g. 179 -> -179 is a 2-degree change, not 358). Zero on frame 0 and on
    any transition touching an invalid frame, same convention as
    compute_wrist_speed_series."""
    speed = np.zeros(len(angles))
    for t in range(1, len(angles)):
        if valid_mask[t] and valid_mask[t - 1]:
            d = angles[t] - angles[t - 1]
            d = (d + 180) % 360 - 180
            speed[t] = abs(d)
    return speed


def _foreshortening_stable_mask(pose_seq: np.ndarray, valid_mask: np.ndarray,
                                 left_idx: int, right_idx: int, min_fraction: float = 0.35) -> np.ndarray:
    """A line's 2D-projected angle is only a reliable signal when the line
    itself isn't nearly degenerate (near-zero length) in that frame's
    projection -- a real, physical finding on clip1_serve_practice.mp4, not
    a hypothetical: the false-positive candidate at frame 108 (walking with
    the racket, not a stroke) had shoulder_width=0.0045 vs. 0.0387 at the
    true-positive serve contact frame 235 -- an 8.6x difference -- and its
    projected shoulder angle swung 77 -> 113 -> 115 degrees across 3
    consecutive frames purely from landmark-detection jitter on that
    near-degenerate line, producing the single HIGHEST rotation reading in
    the whole clip despite not being a real rotation at all. This is the
    exact same foreshortening problem ir.biomechanics.extract_biomechanics_features
    already works around by averaging shoulder width over the whole window
    rather than trusting one frame -- applied here per-frame instead, since
    a rotation-speed series needs a per-frame stability judgement, not one
    window-level average. A frame's line length is trusted only if it's at
    least min_fraction of the clip's own median (valid-frame) line length --
    relative, not absolute, for the same no-camera-calibration reason every
    other threshold in this module is relative to its own clip."""
    n = len(pose_seq)
    lengths = np.zeros(n)
    for t in range(n):
        if valid_mask[t]:
            lengths[t] = _dist(_xy(pose_seq[t], left_idx), _xy(pose_seq[t], right_idx))
    valid_lengths = lengths[valid_mask]
    positive_lengths = valid_lengths[valid_lengths > 0]
    if len(positive_lengths) == 0:
        return np.zeros(n, dtype=bool)
    reference = float(np.median(positive_lengths))
    return valid_mask & (lengths >= min_fraction * reference)


def compute_trunk_rotation_series(pose_seq: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-frame trunk rotational speed (deg/frame) -- the kinetic-chain
    signal that separates a real stroke (shoulders turn rapidly as the body
    rotates into the shot) from a walking arm-swing (the torso keeps facing
    roughly forward; only the arm swings like a pendulum). Computed from
    the SAME shoulder-line angle formula ir.biomechanics uses once per clip
    at the contact frame (shoulder_rotation_deg), evaluated every frame and
    turned into a rate of change, gated by _foreshortening_stable_mask so a
    transition is only trusted where both endpoint frames have a
    non-degenerate projected shoulder line.

    SHOULDER LINE ONLY -- hip was tried and dropped. Hip landmarks sit
    naturally closer together than shoulder landmarks in a 2D projection
    (a shorter baseline), so the same per-landmark pixel jitter from pose
    tracking produces proportionally more angular noise on the hip line
    than the shoulder line. Confirmed on clip1_serve_practice.mp4's frame
    108 false positive: at frame 105 the hip line's length was 42% of the
    clip's own median hip length -- comfortably past this module's 35%
    stability cutoff -- yet the hip angle still swung wildly frame-to-frame
    from tracking jitter, producing a 166.94 deg/frame spurious reading
    that was the single highest rotation value in the entire clip (versus
    the true-positive serve's ~39-88 deg/frame). The shoulder line's larger
    natural baseline meant the identical relative threshold correctly
    excluded the noisy frames in that same window. This is a geometric
    argument (shorter baseline -> more noise-sensitive angle for the same
    tracking jitter), not a threshold picked to make this one clip's number
    come out right -- but it IS only confirmed on one clip so far; if
    shoulder-only later proves too fragile on real footage where the
    shoulders (not hips) are the noisy line, that would be a genuine
    limitation to report, not a case for re-adding hip via a differently-
    tuned threshold chosen to fix that one failure."""
    if valid_mask is None:
        valid_mask = _valid_frame_mask(pose_seq)
    n = len(pose_seq)
    shoulder_angles = np.zeros(n)
    for t in range(n):
        if valid_mask[t]:
            shoulder_angles[t] = _angle_vs_horizontal_deg(_xy(pose_seq[t], LEFT_SHOULDER), _xy(pose_seq[t], RIGHT_SHOULDER))
    shoulder_stable = _foreshortening_stable_mask(pose_seq, valid_mask, LEFT_SHOULDER, RIGHT_SHOULDER)
    return _angular_speed_series(shoulder_angles, shoulder_stable)


def detect_candidate_strokes(
    pose_seq: np.ndarray,
    min_separation_frames: int = 45,
    height_percentile: float = 75.0,
    min_window_valid_fraction: float = 0.8,
    window_radius: int = 30,
    dominant_wrist_idx: int | None = None,
    use_rotation_filter: bool = True,
    rotation_percentile: float = 75.0,
    rotation_window_radius: int = 10,
) -> list[dict]:
    """Find candidate stroke windows in a full multi-stroke pose sequence.

    Over-detection suppression -- walking to pick up a ball, practice
    swings, and shadow swings all produce wrist-speed peaks too, so a naive
    "every local max is a stroke" search over-detects badly:
      - min_separation_frames: no two candidates closer than this
        (scipy.signal.find_peaks' `distance`) -- real strokes in a rally or
        practice session are seconds apart, not frames apart.
      - height_percentile: a peak must exceed this percentile of the
        clip's OWN valid-frame speed distribution, not a fixed absolute
        speed -- wrist-speed units here are uncalibrated pixel-ish
        displacement (see ir/biomechanics.py docstring: no camera
        calibration exists), so a fixed absolute threshold picked on one
        clip's framing/distance wouldn't transfer to another.
      - min_window_valid_fraction: the +/-window_radius window around a
        candidate must have at least this fraction of frames with a real
        detected pose -- reuses the MIN_POSE_DETECTION_RATE idea from
        backend/app.py (there applied whole-clip, here per-window),
        rejecting candidates sitting in a patch of missed pose detection
        where the "peak" may just be tracking noise/jitter rather than a
        real fast movement.
      - use_rotation_filter / rotation_percentile / rotation_window_radius:
        a wrist-speed peak only survives if the trunk (shoulders or hips)
        also rotates fast enough somewhere in a +/-rotation_window_radius
        window around it -- the kinetic-chain signal (see
        compute_trunk_rotation_series). rotation_percentile is, like
        height_percentile, relative to the clip's OWN rotation-speed
        distribution, not an absolute angle -- same reasoning: no camera
        calibration, so an absolute deg/frame threshold picked on one
        clip's distance/framing wouldn't transfer to another. The window
        (rather than requiring the rotation peak to land on the exact same
        frame as the wrist-speed peak) allows for the real kinetic-chain
        lag -- hips and shoulders start turning before the wrist, the last
        link in the chain, reaches its own peak speed.

    These are exactly the knobs that need tuning against real footage, not
    values picked analytically -- see ir/run_stroke_segmentation_trial.py,
    which validates against 3 real clips and reports per-clip, not a
    single number tuned to fit all of them.

    Returns a list of dicts ordered by frame index, one per surviving
    candidate: {contact_frame, window_start, window_end, speed,
    window_valid_fraction, trunk_rotation_peak}. window_end - window_start
    is always exactly WINDOW_LEN (a window that would run off either end of
    the sequence is shifted to stay inside it, not zero-padded -- the
    classifier should always see WINDOW_LEN real frames, never fabricated
    ones).
    """
    n = len(pose_seq)
    valid_mask = _valid_frame_mask(pose_seq)
    if not valid_mask.any() or n < WINDOW_LEN:
        return []
    if dominant_wrist_idx is None:
        dominant_wrist_idx = (
            LEFT_WRIST if determine_dominant_wrist(pose_seq, valid_mask) == "left" else RIGHT_WRIST
        )

    speeds = compute_wrist_speed_series(pose_seq, dominant_wrist_idx, valid_mask)
    valid_speeds = speeds[valid_mask]
    positive_speeds = valid_speeds[valid_speeds > 0]
    if len(positive_speeds) == 0:
        return []
    height = float(np.percentile(positive_speeds, height_percentile))

    peak_indices, _ = find_peaks(speeds, height=height, distance=max(1, min_separation_frames))

    rotation = None
    rotation_threshold = None
    if use_rotation_filter:
        rotation = compute_trunk_rotation_series(pose_seq, valid_mask)
        valid_rotation = rotation[valid_mask]
        positive_rotation = valid_rotation[valid_rotation > 0]
        rotation_threshold = float(np.percentile(positive_rotation, rotation_percentile)) if len(positive_rotation) else 0.0

    candidates = []
    for p in peak_indices:
        start = p - window_radius
        end = start + WINDOW_LEN
        if start < 0:
            start, end = 0, WINDOW_LEN
        elif end > n:
            end, start = n, n - WINDOW_LEN
        window_valid_fraction = float(valid_mask[start:end].mean())
        if window_valid_fraction < min_window_valid_fraction:
            continue
        trunk_rotation_peak = None
        if use_rotation_filter:
            r_start, r_end = max(0, p - rotation_window_radius), min(n, p + rotation_window_radius + 1)
            trunk_rotation_peak = float(rotation[r_start:r_end].max())
            if trunk_rotation_peak < rotation_threshold:
                continue
        candidates.append(dict(
            contact_frame=int(p), window_start=int(start), window_end=int(end),
            speed=float(speeds[p]), window_valid_fraction=window_valid_fraction,
            trunk_rotation_peak=trunk_rotation_peak,
        ))
    return candidates
