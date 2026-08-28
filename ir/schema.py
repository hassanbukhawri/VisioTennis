"""
VisioTennis Intermediate Representation (IR) schema.

DESIGN ARTIFACT -- see DESIGN.md for rationale and open questions, including
the six scope/design decisions this version implements (player_id externally
supplied; pre-segmented-clip scope for v1; ball event_type surfaced from raw
labels; homography now implemented; SessionRecord split out for Model 4;
position fields normalized to court-relative meters). The population code
that constructs real StrokeEvent/SessionRecord instances from real model
output lives in ir/populate.py.

"100% schema compliance" (proposal objective 2) is enforced structurally:
every field below has type/range/enum constraints, so an instance that
constructs successfully IS valid by construction -- there is no separate
compliance check to run later. Invalid data raises pydantic.ValidationError
at construction time.
"""
from datetime import date, datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, confloat, conlist, field_validator, model_validator

SCHEMA_VERSION = "0.2.0"

# Must match training/stroke_classification/dataset.py:COLLAPSED_CLASSES exactly.
STROKE_CLASSES = ("forehand", "backhand", "serve", "volley", "smash")
StrokeClass = Literal["forehand", "backhand", "serve", "volley", "smash"]

# Must match training/anomaly_detection/train.py:FEATURE_COLUMNS exactly.
ANOMALY_FEATURE_COLUMNS = (
    "first_serve_pct", "first_serve_points_won_pct", "second_serve_points_won_pct",
    "ace_rate", "double_fault_rate", "break_points_saved_pct", "unforced_error_rate",
    "winners_rate", "avg_rally_length", "court_coverage_speed_mps",
    "avg_reaction_time_ms", "total_distance_covered_m",
)

Confidence = confloat(ge=0.0, le=1.0)
MomentumScore = confloat(ge=-1.0, le=1.0)


class ModelProvenance(BaseModel):
    """Which model/checkpoint produced a given field group, how it was
    validated, and which real input file it ran on -- lets a consumer trace
    any field back to PROJECT_LOG.md and the exact source data without
    cross-referencing separately. source_file matters especially for
    composite/illustrative records (see ir/populate.py module docstring) where
    different field groups come from different, unrelated source clips."""
    component: str  # e.g. "ball", "court", "stroke"
    checkpoint: str  # e.g. "tracknet_best.pt"
    metric_name: str  # e.g. "val_pck@4px"
    # Made Optional 2026-07-19 (found via a pydantic dependency upgrade,
    # 2.5.3 -> 2.13.4, pulled in by installing fastapi for the MVP backend):
    # ir/populate_biomechanics*.py used float("nan") as a "no metric
    # applies" sentinel (biomechanics is deterministic geometry, not a
    # learned model with an accuracy metric) -- model_dump_json() has
    # always serialized that nan as JSON `null`, but re-validating that
    # same `null` back into a plain (non-Optional) float field, which the
    # older pydantic silently accepted, the newer pydantic correctly
    # rejects (`null` is not a float). NaN-as-missing-value was a fragile
    # sentinel either way; None is the honest, explicit representation of
    # "no metric applies here" and doesn't depend on any particular
    # pydantic version's leniency.
    metric_value: float | None = None
    source_file: str | None = None  # e.g. "game3/Clip7" or "PuXlxKdUIes_2450.png"


class PixelPoint(BaseModel):
    """Raw model-output coordinate, in whatever pixel space that model
    natively operates in (see DESIGN.md -- different models use different
    resolutions, e.g. TrackNet's 512x288 output-space vs CourtKeypointNet's
    224x224 input-space; this is NOT the source video's native resolution).
    Kept for traceability/debugging alongside the canonical court-relative
    CourtPoint -- see DESIGN.md decision #6."""
    x_px: float
    y_px: float


class CourtPoint(BaseModel):
    """Court-relative coordinate in meters, derived via homography (see
    ir/geometry.py). Origin = far-baseline doubles-left corner, +x along the
    baseline, +y along the sideline toward the near baseline. This is the
    CANONICAL position representation for downstream consumers (decision #6)."""
    x_m: float
    y_m: float


class FrameRange(BaseModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    fps: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def check_order(self):
        if self.end_frame < self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) < start_frame ({self.start_frame})")
        return self


class BallFramePosition(BaseModel):
    """One TrackNet-derived observation. visible=False means the model
    predicted no ball present (heatmap peak below confidence threshold) --
    pixel/court are still populated (heatmap argmax) but should be ignored by
    consumers when visible=False, matching ball_tracking/train.py's
    no_ball_acc metric semantics.

    court is None whenever no valid homography was available for this event
    (e.g. court detection failed for the source clip) -- see DESIGN.md for
    the more fundamental caveat that even a valid homography only gives an
    EXACT court position when the ball is on the ground plane (z=0); for
    airborne frames (most of them) it's a ground-plane projection, not the
    ball's true 3D position -- reliable primarily at/near contact_frame and
    bounce_frame.

    event_type is sourced from ground-truth Label.csv for IR population v1
    (pre-segmented clips from our own labeled dataset only) -- it is NOT
    predicted by TrackNet at inference time on new/unlabeled footage. See
    DESIGN.md "Ball event_type: ground truth now, model later" for the
    reliability assessment and fast-follow recommendation."""
    frame: int = Field(ge=0)
    pixel: PixelPoint
    court: CourtPoint | None = None
    visible: bool
    confidence: Confidence
    event_type: Literal["flight", "hit", "bounce"] | None = None


class BallInfo(BaseModel):
    trajectory: list[BallFramePosition] = Field(min_length=1)
    # Derived from trajectory[i].event_type == "hit"/"bounce" when available
    # (ground-truth-sourced in v1, see BallFramePosition.event_type docstring).
    contact_frame: int | None = None
    bounce_frame: int | None = None
    source_model: ModelProvenance


class CourtInfo(BaseModel):
    # Exactly 14 points, matching CourtKeypointNet's N_KEYPOINTS. Deliberately
    # kept in pixel space, NOT court-relative -- these points are the
    # calibration input used to COMPUTE the homography in ir/geometry.py, so
    # expressing them in the coordinate system the homography itself produces
    # would be circular. See DESIGN.md decision #6.
    keypoints_px: conlist(PixelPoint, min_length=14, max_length=14)
    # Not populated by CourtKeypointNet (direct regression, no per-point confidence
    # output) -- see DESIGN.md.
    keypoints_confidence: list[Confidence] | None = None
    homography_matrix: conlist(conlist(float, min_length=3, max_length=3), min_length=3, max_length=3) | None = None
    # Mean reprojection error (meters) of the 14 known real-world court points
    # through homography_matrix -- a fit-quality signal; see ir/geometry.py.
    homography_reprojection_error_m: float | None = Field(default=None, ge=0)
    source_model: ModelProvenance

    @field_validator("keypoints_confidence")
    @classmethod
    def check_confidence_length(cls, v):
        if v is not None and len(v) != 14:
            raise ValueError(f"keypoints_confidence must have exactly 14 entries, got {len(v)}")
        return v


class StrokeInfo(BaseModel):
    predicted_class: StrokeClass
    confidence: Confidence
    class_probabilities: dict[StrokeClass, Confidence]
    source_model: ModelProvenance

    @field_validator("class_probabilities")
    @classmethod
    def check_probabilities(cls, v):
        if set(v.keys()) != set(STROKE_CLASSES):
            raise ValueError(f"class_probabilities keys must be exactly {STROKE_CLASSES}, got {sorted(v.keys())}")
        total = sum(v.values())
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"class_probabilities must sum to ~1.0, got {total:.4f}")
        return v

    @model_validator(mode="after")
    def check_predicted_matches_argmax(self):
        argmax_class = max(self.class_probabilities, key=self.class_probabilities.get)
        if argmax_class != self.predicted_class:
            raise ValueError(
                f"predicted_class={self.predicted_class!r} does not match "
                f"argmax of class_probabilities ({argmax_class!r})"
            )
        return self


class BiomechanicsInfo(BaseModel):
    """Stroke-level biomechanical features derived purely from MediaPipe
    Pose body-landmark geometry (33 3D landmarks per frame over a
    StrokeEvent's pose sequence). NO RACKET DATA EXISTS ANYWHERE IN THIS
    PROJECT (confirmed by repo-wide search, see DESIGN.md) -- every field
    here is a body pose measurement, or an explicitly-named body-pose PROXY
    for a racket-related concept (forearm_orientation_deg is NOT racket
    angle), never a renamed approximation pretending to be the racket
    measurement itself. Proposed and reviewed in ir/DESIGN.md ("Stroke-level
    biomechanical features") before this schema was added -- see that
    section for the full feature-by-feature rationale and the two flagged
    methodological caveats (dominant-wrist heuristic, contact-frame proxy).

    Coordinate caveat: no camera calibration exists for the pose data this
    is derived from (unlike CourtPoint, which has real homography). Every
    length/height field is a DIMENSIONLESS RATIO against a body-scale
    reference (shoulder width), not an absolute distance -- comparable
    across clips, NOT convertible to real-world centimeters. Angles are
    true degrees computed from the frame-plane (x, y) projection only;
    MediaPipe's z is unvalidated in this project and is not used here, so
    out-of-plane (toward/away from camera) rotation is not captured.

    Deliberately excludes percentile/benchmark fields (see DESIGN.md) --
    this holds only raw, always-computable per-stroke geometry; percentile
    ranking against the wider population is a separate downstream utility,
    analogous to explain_anomaly() but for biomechanics, not yet built."""

    contact_frame: int = Field(ge=0)
    contact_frame_source: Literal["ball_tracking_hit_frame", "peak_wrist_speed_proxy"]
    # "ball_tracking_hit_frame": a real event_type=="hit" frame exists in
    # this same StrokeEvent's ball.trajectory (true today only for the
    # game1/Clip5 end-to-end example). "peak_wrist_speed_proxy": no
    # ball-tracking data coexists with this pose sequence (true for every
    # StrokeClassification/THETIS-sourced clip) -- contact is approximated
    # as the frame of maximum dominant-wrist speed in the window.

    dominant_wrist: Literal["left", "right"]
    dominant_wrist_source: Literal["externally_supplied", "max_displacement_heuristic"]
    # Neither pose dataset labels player handedness. Applied uniformly even
    # to two-handed backhand clips (both hands grip the racket there, so
    # "one dominant wrist" is a known simplification for that subclass --
    # see DESIGN.md), rather than special-casing it.

    wrist_height_ratio: float
    shoulder_rotation_deg: float
    hip_rotation_deg: float
    shoulder_hip_separation_deg: float = Field(ge=0, le=180)
    knee_flex_left_deg: float = Field(ge=0, le=180)
    knee_flex_right_deg: float = Field(ge=0, le=180)
    stance_width_ratio: float = Field(ge=0)
    weight_transfer_ratio: float
    follow_through_extension_ratio: float = Field(ge=0)
    forearm_orientation_deg: float
    elbow_angle_deg: float = Field(ge=0, le=180)

    source_model: ModelProvenance
    # component="biomechanics". The derivation itself is deterministic
    # geometry (arctan2/vector math over landmark coordinates), not a
    # learned model with its own accuracy metric -- source_model instead
    # names the upstream MediaPipe Pose landmarker that produced the
    # landmarks this is computed from.


class MomentumInfo(BaseModel):
    """PLACEHOLDER -- Model 3 has no trained weights (see
    momentum_tracking/DATA_ACQUISITION_CHECKLIST.md, including the unresolved
    momentum-label-definition question). This type exists so StrokeEvent's
    shape doesn't change once Model 3 is eventually trained; StrokeEvent.momentum
    should stay None until then."""
    momentum_score: MomentumScore
    source_model: ModelProvenance


class StrokeEvent(BaseModel):
    """One identified stroke within a pre-segmented clip (IR population v1
    scope -- see DESIGN.md decision #2; continuous-match-video ingestion,
    including stroke-boundary segmentation, is out of scope for v1).

    player_id/player_side are EXTERNALLY SUPPLIED, not inferred by any model
    (decision #1) -- populating code should pass these through from whatever
    external source labeled the clip (e.g. the source dataset's filename
    convention), never guess or default them. match_id/point_id have no
    current producer at all (multi-player tracking / point-boundary detection
    are both explicit future work, not built) and should be left None."""

    schema_version: str = SCHEMA_VERSION
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    match_id: str | None = None
    point_id: str | None = None
    player_id: str | None = None  # externally supplied -- see class docstring
    player_side: Literal["near", "far"] | None = None
    source_video: str | None = None
    frame_range: FrameRange

    # ball/court made OPTIONAL 2026-07-19 for the MVP slice (mvp/DESIGN.md):
    # the MVP backend deliberately does not run Model 1A (TrackNet) or
    # Model 1B (CourtKeypointNet) -- the stroke-level coaching path
    # (rag/stroke_prompt.py) never references ball/court at all, only
    # biomechanics + stroke.predicted_class, and running them would add
    # real generalization risk (TrackNet was never validated on phone-shot
    # footage) for no benefit to that specific output. Every StrokeEvent
    # built before this change still populates both (composites in
    # ir/populate.py, the true end-to-end example, the biomechanics
    # examples) -- this is a backward-compatible relaxation, not a removal;
    # nothing existing relied on them being required.
    ball: BallInfo | None = None
    court: CourtInfo | None = None
    stroke: StrokeInfo
    momentum: MomentumInfo | None = None  # always None until Model 3 exists
    biomechanics: BiomechanicsInfo | None = None  # None until real pose data is available

    provenance: list[ModelProvenance] = Field(default_factory=list)


class SessionRecord(BaseModel):
    """One player-session anomaly-detection record, from Model 4. Kept as a
    SEPARATE top-level entity from StrokeEvent (decision #5) -- Model 4
    operates on session-level aggregate performance vectors (one row per
    player per session), not per-stroke, so forcing its output onto every
    StrokeEvent in a session would either duplicate a session-level flag
    across many events (misleading) or leave it null on most of them."""

    schema_version: str = SCHEMA_VERSION
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    player_id: str  # externally supplied, same as StrokeEvent -- required here
    session_date: date | None = None

    features: dict[str, float]
    anomaly_flag: Literal[-1, 1]
    anomaly_score: float
    explanation: str  # human-readable top-k deviated features, e.g. anomaly_detection/train.py:explain_anomaly()
    source_model: ModelProvenance

    @field_validator("features")
    @classmethod
    def check_feature_columns(cls, v):
        if set(v.keys()) != set(ANOMALY_FEATURE_COLUMNS):
            missing = set(ANOMALY_FEATURE_COLUMNS) - set(v.keys())
            extra = set(v.keys()) - set(ANOMALY_FEATURE_COLUMNS)
            raise ValueError(f"features keys mismatch -- missing={sorted(missing)} extra={sorted(extra)}")
        return v


class SessionDigest(BaseModel):
    """Cross-session progress digest (2026-08-12, "Option C" per PROJECT_LOG.md
    -- see that entry for the full design writeup). Fresh-computed from real
    stored session rows every time one is requested (ir/session_digest.py) --
    NEVER cached, incrementally updated, or chained from a prior digest. There
    is no persisted SessionDigest table anywhere; record_id is minted fresh
    per computation (same pattern as StrokeEvent.event_id), never reused
    across calls. source_session_ids is the audit trail -- exactly which real
    StrokeEvent.event_id values this digest's numbers were computed from, so
    any generated claim citing this record can in principle be independently
    re-derived from the raw session data, not just trusted.

    Every numeric field here is a PERCENTILE RANK (0-100), not a raw
    biomechanics unit. This is deliberate, not incidental: a raw feature's
    typical value differs by stroke class (e.g. wrist_height_ratio's mean is
    far apart for forehand vs. serve, per ir/biomechanics_percentiles.py's own
    demo output), so averaging RAW values across a player's mixed-stroke-type
    session history would silently conflate classes and produce a number with
    no real meaning. Percentile rank is already class-normalized (computed
    per-session against that session's own predicted_class via
    BiomechanicsPercentileBenchmark), so it's the only value that stays
    meaningful when averaged across sessions of different stroke types.

    Only covers ir/biomechanics_percentiles.py:RANKED_FEATURES -- deliberately
    excludes the same 3 fields that module excludes from percentile ranking
    at all (shoulder_rotation_deg, hip_rotation_deg, forearm_orientation_deg):
    camera-facing-direction-dependent raw angles, not comparable even within
    one session, let alone averaged across sessions potentially filmed on
    different days with different camera setups -- that problem only gets
    worse across sessions, not better, so the same exclusion applies with more
    force here, not less."""
    schema_version: str = SCHEMA_VERSION
    record_id: str = Field(default_factory=lambda: str(uuid4()))
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    player_id: str
    lookback_k: int = Field(ge=1)
    # Sessions actually used in the trailing-average window -- <= lookback_k,
    # since a player with fewer than k usable sessions still gets a digest,
    # just over whatever's available (graceful degradation, not withheld).
    session_count: int = Field(ge=0)
    # Total usable sessions considered across BOTH windows (recent + prior) --
    # lets a consumer tell "digest computed from thin data" apart from
    # "digest computed from a deep history," even though session_count alone
    # is capped at lookback_k either way.
    total_sessions_considered: int = Field(ge=0)
    # Most common predicted stroke class among the recent window's sessions --
    # informational context only (which stroke the player has mostly been
    # practicing lately), NOT used to filter which sessions feed the
    # percentile averages above (see class docstring on why percentile
    # normalization makes that filtering unnecessary).
    stroke_type_mode: StrokeClass | None = None
    # feature -> average percentile rank (0-100) over the recent window.
    trailing_averages: dict[str, float]
    # feature -> percentile-POINT change (recent window avg - prior window
    # avg). Empty dict (not omitted, not null) when no prior window exists
    # yet (fewer than 2*lookback_k usable sessions) -- an explicit "no trend
    # available" signal a consumer must check for, not a silently-missing key.
    trend_deltas: dict[str, float]
    # How many sessions actually fed the PRIOR half of trend_deltas -- may be
    # less than lookback_k (e.g. a player with 7 usable sessions gets a real,
    # non-empty trend_deltas compared against only 2 prior sessions, not a
    # full k). 0 when trend_deltas is empty. Exists so a thin trend
    # comparison can be honestly flagged rather than presented with the same
    # confidence as a full-k one -- the same "low_confidence" discipline
    # ir/biomechanics_percentiles.py already applies to thin percentile tails.
    trend_prior_window_size: int = Field(ge=0)
    source_session_ids: list[str] = Field(min_length=1)

    @field_validator("trailing_averages", "trend_deltas")
    @classmethod
    def check_feature_keys(cls, v, info):
        from ir.biomechanics_percentiles import RANKED_FEATURES
        if info.field_name == "trend_deltas" and not v:
            return v  # explicitly valid -- see field docstring
        if set(v.keys()) != set(RANKED_FEATURES):
            missing = set(RANKED_FEATURES) - set(v.keys())
            extra = set(v.keys()) - set(RANKED_FEATURES)
            raise ValueError(
                f"{info.field_name} keys mismatch -- missing={sorted(missing)} extra={sorted(extra)}")
        return v
