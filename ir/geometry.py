"""
Pixel -> court-relative-coordinate homography for VisioTennis IR population.

Court keypoint semantics (14 points, matching CourtKeypointNet's output order)
are NOT documented anywhere in the source export -- no keypoint-name file
ships with Court_Detector_Keypoints/data/data_train.json, just an ordered
[x, y] list per entry. The order below was INFERRED from the raw pixel
geometry, not read from an authoritative source: paired top/bottom rows at
consistent y-values, with doubles-corner x-spacing consistently ~1.333x the
singles-corner x-spacing (matches the real ITF ratio 10.97m/8.23m = 1.333
to within 0.3%, checked across 4 independent sample frames -- see
ir/DESIGN.md "Homography implementation" for the verification). Flagged as an
inferred-not-authoritative assumption; revisit if this is ever cross-checked
against the dataset's original public source.

Real-world court coordinate system: origin at the far-baseline doubles-left
corner, +x along the baseline toward doubles-right, +y along the sideline
toward the near baseline (i.e. toward the camera in a typical broadcast
elevated-behind-baseline view). Standard ITF doubles court: 23.77m x 10.97m;
singles sidelines inset (10.97-8.23)/2 = 1.37m from each doubles sideline;
service line 5.485m from each baseline (net at the 11.885m midpoint, service
line 6.4m from net); center service line at x=5.485 (midpoint of court width).
"""
import numpy as np
import cv2

COURT_LENGTH_M = 23.77
DOUBLES_WIDTH_M = 10.97
SINGLES_WIDTH_M = 8.23
SINGLES_INSET_M = (DOUBLES_WIDTH_M - SINGLES_WIDTH_M) / 2  # 1.37
SERVICE_LINE_FROM_BASELINE_M = COURT_LENGTH_M / 2 - 6.40   # 5.485
CENTER_X_M = DOUBLES_WIDTH_M / 2                            # 5.485

# Real-world (x_m, y_m) for keypoints_px[0..13], in the CourtKeypointNet output order.
COURT_KEYPOINTS_REAL_WORLD_M = [
    (0.0, 0.0),                                              # 0: far baseline, doubles-left
    (DOUBLES_WIDTH_M, 0.0),                                   # 1: far baseline, doubles-right
    (0.0, COURT_LENGTH_M),                                    # 2: near baseline, doubles-left
    (DOUBLES_WIDTH_M, COURT_LENGTH_M),                        # 3: near baseline, doubles-right
    (SINGLES_INSET_M, 0.0),                                   # 4: far baseline, singles-left
    (SINGLES_INSET_M, COURT_LENGTH_M),                        # 5: near baseline, singles-left
    (SINGLES_INSET_M + SINGLES_WIDTH_M, 0.0),                 # 6: far baseline, singles-right
    (SINGLES_INSET_M + SINGLES_WIDTH_M, COURT_LENGTH_M),      # 7: near baseline, singles-right
    (SINGLES_INSET_M, SERVICE_LINE_FROM_BASELINE_M),          # 8: far service line, singles-left
    (SINGLES_INSET_M + SINGLES_WIDTH_M, SERVICE_LINE_FROM_BASELINE_M),               # 9: far service line, singles-right
    (SINGLES_INSET_M, COURT_LENGTH_M - SERVICE_LINE_FROM_BASELINE_M),                # 10: near service line, singles-left
    (SINGLES_INSET_M + SINGLES_WIDTH_M, COURT_LENGTH_M - SERVICE_LINE_FROM_BASELINE_M),  # 11: near service line, singles-right
    (CENTER_X_M, SERVICE_LINE_FROM_BASELINE_M),               # 12: far service line, center (T)
    (CENTER_X_M, COURT_LENGTH_M - SERVICE_LINE_FROM_BASELINE_M),  # 13: near service line, center (T)
]


def compute_homography(keypoints_px):
    """keypoints_px: list of 14 (x_px, y_px) pairs, in CourtKeypointNet's output
    order. Returns (H, reprojection_error_m) where H is a 3x3 pixel->court-meters
    homography (cv2.findHomography, RANSAC over all 14 points -- more robust to
    any single noisy keypoint than a plain 4-point DLT) and reprojection_error_m
    is the mean Euclidean error (in meters) reprojecting the 14 known real-world
    points back through H, as a fit-quality signal callers should check."""
    if len(keypoints_px) != 14:
        raise ValueError(f"expected exactly 14 keypoints, got {len(keypoints_px)}")

    src = np.asarray(keypoints_px, dtype=np.float64)
    dst = np.asarray(COURT_KEYPOINTS_REAL_WORLD_M, dtype=np.float64)

    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=0.5)
    if H is None:
        raise ValueError("cv2.findHomography failed to compute a homography (degenerate keypoints?)")

    reprojected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    reprojection_error_m = float(np.mean(np.linalg.norm(reprojected - dst, axis=1)))
    return H, reprojection_error_m


def pixel_to_court(H, x_px, y_px):
    """Apply a homography (from compute_homography) to a single pixel point.
    Returns (x_m, y_m) in court-relative coordinates.

    IMPORTANT: a homography maps points on a single PLANE (the court surface,
    z=0). It is exact for court keypoints (which are painted on the court) but
    only APPROXIMATE for the ball whenever the ball is airborne (z > 0) -- see
    ir/DESIGN.md "Homography and the ball's z=0 assumption" for the caveat this
    implies for ball.trajectory court-position accuracy."""
    pt = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


if __name__ == "__main__":
    # Shape/sanity check only, not real detections -- uses one real sample
    # frame's keypoints (read directly from the dataset) to confirm the
    # homography reprojects with low error on genuine data.
    import json
    with open("data/VisioTennis/Dataset/Court_Detector_Keypoints/data/data_train.json") as f:
        data = json.load(f)
    kps = data[0]["kps"]
    H, err = compute_homography(kps)
    print(f"sample entry: {data[0]['id']}")
    print(f"reprojection error: {err:.4f} m")
    print("far baseline doubles-left (kp 0) ->", pixel_to_court(H, *kps[0]), "expected (0.0, 0.0)")
    print("near baseline doubles-right (kp 3) ->", pixel_to_court(H, *kps[3]),
          f"expected ({DOUBLES_WIDTH_M}, {COURT_LENGTH_M})")
