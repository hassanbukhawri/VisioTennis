"""
Stroke-level biomechanical feature extraction from MediaPipe Pose landmark
sequences -- see ir/DESIGN.md ("Stroke-level biomechanical features") for
the full design review this implements, including the two flagged
methodological caveats (dominant-wrist heuristic, contact-frame proxy) and
why every field is a dimensionless ratio/degree rather than a real-world
distance (no camera calibration exists for this pose data).

Input format: (seq_len, 99) float32 arrays -- 33 MediaPipe BlazePose
landmarks x (x, y, z) per frame, exactly what
training/stroke_classification/dataset.py's X_*.npy and
ir/populate_e2e.py:extract_pose_sequence() both produce. z is never used
below (unvalidated depth estimate, see DESIGN.md) -- only the frame-plane
(x, y) projection.
"""
import numpy as np

# Standard MediaPipe BlazePose 33-landmark indices (same topology used by
# both mp.solutions.pose and the PoseLandmarker Tasks API).
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def _xy(frame_landmarks: np.ndarray, idx: int) -> np.ndarray:
    """frame_landmarks: (99,) flat array for one frame. Returns (x, y)."""
    return frame_landmarks[idx * 3: idx * 3 + 2]


def _valid_frame_mask(pose_seq: np.ndarray) -> np.ndarray:
    """True for frames with a real detected pose. Some StrokeClassification
    (THETIS) clips have trailing all-zero frames -- the same "no pose
    detected, left as zeros" fallback documented in
    ir/populate_e2e.py:extract_pose_sequence's docstring, present here too
    (confirmed by inspection: e.g. p53_foreopen_s2.avi has 46 real frames
    followed by 14 all-zero frames). A zero frame must never be selected as
    a contact frame or treated as a real position -- a transition from a
    real position to the origin looks like a huge, fake "wrist speed" spike
    to a naive peak-speed search, which is exactly what happened before this
    fix was added (see ir/DESIGN.md)."""
    return ~np.all(pose_seq == 0, axis=1)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _angle_vs_horizontal_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of vector a->b vs. the frame's +x axis, in degrees, sign-flipped
    on y since image y increases downward (so a positive angle here means
    "upward-and-across" the way it would on a normal x/y plot)."""
    dx, dy = b[0] - a[0], -(b[1] - a[1])
    return float(np.degrees(np.arctan2(dy, dx)))


def _interior_angle_deg(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
    """Interior angle at `vertex` formed by vertex->p1 and vertex->p2."""
    v1, v2 = p1 - vertex, p2 - vertex
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def shoulder_width(frame_landmarks: np.ndarray) -> float:
    return _dist(_xy(frame_landmarks, LEFT_SHOULDER), _xy(frame_landmarks, RIGHT_SHOULDER))


def determine_dominant_wrist(pose_seq: np.ndarray, valid_mask: np.ndarray | None = None) -> str:
    """Heuristic (DESIGN.md open question #1): whichever wrist has greater
    total frame-to-frame displacement across the window is "dominant" for
    this clip. Applied uniformly, including to two-handed backhand clips,
    where it's a known simplification (both hands grip the racket there).
    Transitions touching an invalid (zero/missing-pose) frame are excluded
    -- see _valid_frame_mask."""
    if valid_mask is None:
        valid_mask = _valid_frame_mask(pose_seq)
    left_disp = sum(
        _dist(_xy(pose_seq[t], LEFT_WRIST), _xy(pose_seq[t + 1], LEFT_WRIST))
        for t in range(len(pose_seq) - 1) if valid_mask[t] and valid_mask[t + 1]
    )
    right_disp = sum(
        _dist(_xy(pose_seq[t], RIGHT_WRIST), _xy(pose_seq[t + 1], RIGHT_WRIST))
        for t in range(len(pose_seq) - 1) if valid_mask[t] and valid_mask[t + 1]
    )
    return "left" if left_disp >= right_disp else "right"


def find_peak_wrist_speed_frame(pose_seq: np.ndarray, dominant_wrist_idx: int, valid_mask: np.ndarray | None = None) -> int:
    """Contact-frame proxy (DESIGN.md open question #2): the frame of maximum
    dominant-wrist speed. Speed at frame t is the displacement from t-1 to t;
    a transition touching an invalid (zero/missing-pose) frame is never
    considered, so a jump-to-origin can't masquerade as a real fast movement
    -- see _valid_frame_mask. Only valid frames can be returned."""
    if valid_mask is None:
        valid_mask = _valid_frame_mask(pose_seq)
    speeds = np.zeros(len(pose_seq))
    for t in range(1, len(pose_seq)):
        if valid_mask[t] and valid_mask[t - 1]:
            speeds[t] = _dist(_xy(pose_seq[t - 1], dominant_wrist_idx), _xy(pose_seq[t], dominant_wrist_idx))
    speeds[~valid_mask] = -1.0  # never pick an invalid frame as the peak itself
    return int(np.argmax(speeds))


def extract_biomechanics_features(
    pose_seq: np.ndarray,
    contact_frame: int | None = None,
    dominant_wrist: str | None = None,
) -> dict:
    """Computes the 10 BiomechanicsInfo fields (plus contact_frame/
    dominant_wrist and their *_source fields) from a (seq_len, 99) pose
    sequence. Pass contact_frame explicitly only when a real
    ball-tracking hit frame exists in the same window (see DESIGN.md);
    otherwise it's derived via the peak-wrist-speed proxy. Same for
    dominant_wrist: pass it explicitly only if externally known (never
    true today -- neither dataset labels handedness)."""
    seq_len = pose_seq.shape[0]
    assert pose_seq.shape[1] == 99, f"expected (seq_len, 99), got {pose_seq.shape}"

    valid_mask = _valid_frame_mask(pose_seq)
    if not valid_mask.any():
        raise ValueError("pose_seq has no valid (non-zero) frames -- pose detection failed on every frame")
    valid_indices = np.nonzero(valid_mask)[0]
    first_valid, last_valid = int(valid_indices[0]), int(valid_indices[-1])

    if dominant_wrist is not None:
        dominant_wrist_source = "externally_supplied"
    else:
        dominant_wrist = determine_dominant_wrist(pose_seq, valid_mask)
        dominant_wrist_source = "max_displacement_heuristic"
    wrist_idx = LEFT_WRIST if dominant_wrist == "left" else RIGHT_WRIST
    elbow_idx = LEFT_ELBOW if dominant_wrist == "left" else RIGHT_ELBOW
    shoulder_idx = LEFT_SHOULDER if dominant_wrist == "left" else RIGHT_SHOULDER

    if contact_frame is not None:
        contact_frame_source = "ball_tracking_hit_frame"
        contact_frame = max(0, min(contact_frame, seq_len - 1))
        if not valid_mask[contact_frame]:
            raise ValueError(f"supplied contact_frame={contact_frame} has no detected pose (zero landmarks)")
    else:
        contact_frame = find_peak_wrist_speed_frame(pose_seq, wrist_idx, valid_mask)
        contact_frame_source = "peak_wrist_speed_proxy"

    cf = pose_seq[contact_frame]
    # Scale reference: the WINDOW-AVERAGED shoulder width across all valid
    # frames, not the single contact-frame value. A single frame's 2D-
    # projected shoulder width can collapse toward zero when the player is
    # side-on to the camera at that instant -- common right at contact for
    # serves and one-handed strokes, where the body naturally rotates
    # side-on into the shot -- which would otherwise blow up every ratio
    # feature that divides by it (observed on a real example: shoulder
    # width of 0.0048 normalized units at one clip's contact frame, vs.
    # ~0.03-0.15 in most other frames of the same clip). Averaging over the
    # window smooths out this single-instant foreshortening; it does not
    # fix a clip where the player stays side-on for the whole window (a
    # residual, undisturbed limitation of using 2D-only pose -- see
    # ir/DESIGN.md).
    sw = float(np.mean([shoulder_width(pose_seq[t]) for t in valid_indices]))
    if sw < 1e-6:
        sw = 1e-6  # every valid frame degenerate; avoid div-by-zero, features will just be extreme

    l_sh, r_sh = _xy(cf, LEFT_SHOULDER), _xy(cf, RIGHT_SHOULDER)
    l_hip, r_hip = _xy(cf, LEFT_HIP), _xy(cf, RIGHT_HIP)
    hip_mid = (l_hip + r_hip) / 2.0
    dom_wrist_xy = _xy(cf, wrist_idx)
    dom_elbow_xy = _xy(cf, elbow_idx)
    dom_shoulder_xy = _xy(cf, shoulder_idx)

    wrist_height_ratio = float((hip_mid[1] - dom_wrist_xy[1]) / sw)  # image y down -> flip sign implicitly via (hip - wrist)

    shoulder_rotation_deg = _angle_vs_horizontal_deg(l_sh, r_sh)
    hip_rotation_deg = _angle_vs_horizontal_deg(l_hip, r_hip)
    # Separation between the shoulder AXIS and hip AXIS, not between two
    # directed vectors: a line has no inherent direction, so "left shoulder
    # -> right shoulder" pointing screen-left vs. "left hip -> right hip"
    # pointing screen-right is the SAME physical axis alignment, not a
    # 180-degree twist. Reduce the raw vector-angle difference mod 180 and
    # fold anything past 90 back down, so the result is the true minimal
    # angle between the two axes (anatomically bounded well under 90 in
    # practice; a torso cannot physically twist 167 degrees, which an
    # earlier signed-vector version of this calc produced on a real
    # backhand2hands example -- see ir/DESIGN.md).
    raw_diff = abs(shoulder_rotation_deg - hip_rotation_deg) % 180
    shoulder_hip_separation_deg = float(min(raw_diff, 180 - raw_diff))

    knee_flex_left_deg = _interior_angle_deg(_xy(cf, LEFT_HIP), _xy(cf, LEFT_KNEE), _xy(cf, LEFT_ANKLE))
    knee_flex_right_deg = _interior_angle_deg(_xy(cf, RIGHT_HIP), _xy(cf, RIGHT_KNEE), _xy(cf, RIGHT_ANKLE))

    stance_width_ratio = float(_dist(_xy(cf, LEFT_ANKLE), _xy(cf, RIGHT_ANKLE)) / sw)

    # Use the first/last VALID frame as the window's start/end reference,
    # not literally pose_seq[0]/[-1] -- some clips have trailing (or
    # leading) zero/missing-pose frames, see _valid_frame_mask.
    start_frame, end_frame = pose_seq[first_valid], pose_seq[last_valid]

    start_hip_mid = (_xy(start_frame, LEFT_HIP) + _xy(start_frame, RIGHT_HIP)) / 2.0
    weight_transfer_ratio = float((hip_mid[0] - start_hip_mid[0]) / sw)

    end_wrist_xy = _xy(end_frame, wrist_idx)
    end_shoulder_xy = _xy(end_frame, shoulder_idx)
    dist_at_contact = _dist(dom_wrist_xy, dom_shoulder_xy)
    dist_at_end = _dist(end_wrist_xy, end_shoulder_xy)
    follow_through_extension_ratio = float(dist_at_end / (dist_at_contact + 1e-9))

    forearm_orientation_deg = _angle_vs_horizontal_deg(dom_elbow_xy, dom_wrist_xy)
    elbow_angle_deg = _interior_angle_deg(dom_shoulder_xy, dom_elbow_xy, dom_wrist_xy)

    return dict(
        contact_frame=contact_frame,
        contact_frame_source=contact_frame_source,
        dominant_wrist=dominant_wrist,
        dominant_wrist_source=dominant_wrist_source,
        wrist_height_ratio=wrist_height_ratio,
        shoulder_rotation_deg=shoulder_rotation_deg,
        hip_rotation_deg=hip_rotation_deg,
        shoulder_hip_separation_deg=shoulder_hip_separation_deg,
        knee_flex_left_deg=knee_flex_left_deg,
        knee_flex_right_deg=knee_flex_right_deg,
        stance_width_ratio=stance_width_ratio,
        weight_transfer_ratio=weight_transfer_ratio,
        follow_through_extension_ratio=follow_through_extension_ratio,
        forearm_orientation_deg=forearm_orientation_deg,
        elbow_angle_deg=elbow_angle_deg,
    )
