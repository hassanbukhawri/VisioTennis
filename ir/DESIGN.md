# Intermediate Representation (IR) — design document

**Status: v2, decisions made, population code written (`ir/populate.py`), see
`ir/example_output/` for real populated records.** v1 of this document (see
git history / PROJECT_LOG.md) listed six open questions; this revision
records the decisions made on each, with rationale, and updates the schema
(`ir/schema.py`) to match.

## Purpose

Objective 2 of the proposal calls for a schema-validated JSON layer bridging
CV model output (Models 1A/1B/2, eventually 3/4) with downstream ML/RAG
components, with a stated requirement of **100% schema compliance**. This
document designs that layer: what a record contains, why it's shaped this
way, and what's explicitly still out of scope.

## Design principle: validate at construction, not after

Unchanged from v1: `schema.py` uses Pydantic v2 models with field-level
constraints so that **it is not possible to construct an invalid record in
the first place** — `pydantic.ValidationError` is raised at construction
time. `StrokeEvent.model_json_schema()` / `SessionRecord.model_json_schema()`
export standard JSON Schema for non-Python consumers.

## Two top-level entities

`StrokeEvent` (one record per identified stroke) and, new in this revision,
`SessionRecord` (one record per player-session anomaly-detection result) —
see decision #5.

---

## The six decisions

### 1. `player_id` is externally supplied, not inferred

No model in this project does player detection, tracking, or re-identification.
`StrokeEvent.player_id` / `player_side` are typed fields that populating code
must pass through from an external source (e.g. the source dataset's own
per-clip labeling, like THETIS's `p{N}_...` filename convention) — never
guessed, never defaulted to a placeholder. **Multi-player tracking
(detecting and re-identifying players across frames in raw match video) is
explicit future work, not built and not attempted here.** This keeps the
schema honest: a populated `player_id` means "some external process labeled
this," not "the pipeline figured out who this is."

### 2. IR population v1 targets pre-segmented per-stroke clips

Model 2 (`StrokeGRU`) was trained on THETIS's pre-segmented 60-frame clips —
it has no concept of finding stroke boundaries in continuous footage. Rather
than pretend the IR layer solves a problem none of the four trained models
solve, **v1 population code (`ir/populate.py`) takes a single pre-segmented
clip as input** (matching Model 2's own training assumption) and produces one
`StrokeEvent`. **Continuous match-video ingestion — segmenting raw footage
into per-stroke windows, tracking players across cuts, assigning point/rally
boundaries — is a separate, larger future component,** not attempted here.
This is the same boundary as decision #1: the IR schema anticipates these
fields (`point_id`, multi-event ingestion) but the population code doesn't
pretend to produce them yet.

### 3. Ball `event_type` (hit/bounce) is surfaced from the raw label, not a new model

Re-inspected `ball_tracking/dataset.py`: it parsed `visibility` and
`x/y-coordinate` from `Label.csv` but silently discarded the `status` column
(0=flight, 1=hit, 2=bounce). This is real ground-truth information that
existed all along — **`ClipRecord` now exposes it as `status`** (see
`ball_tracking/dataset.py`'s updated docstring), and `ir/populate.py` maps it
into `BallFramePosition.event_type` for our own labeled clips.

**Reliability assessment** (per a 30-clip / 4959-frame sample, checked before
deciding this was good enough to surface as-is): every `NaN` status value
corresponds exactly to `visibility == 0` (120/120 match — status is simply
inapplicable when there's no ball to have a status, not a labeling gap).
Status distribution (flight 92.0% / hit 2.8% / bounce 2.7%) is consistent
across clips and matches plausible event rates for rally play. **This looks
like clean, complete, deliberately-structured labeling — reliable as ground
truth for our own labeled dataset.**

**But this is not a permanent solution for new footage.** The raw `status`
column only exists because these clips were manually/deliberately annotated
as part of the `Ball Tracker+Bounce` dataset. TrackNet itself only regresses
ball position (a heatmap) — it has no output head for event classification,
so **on genuinely new, unlabeled match video, there is no `status` to read**.
Recommendation: a lightweight trained hit/bounce classifier (e.g. a small
1D-CNN or rule-based detector over the ball trajectory's velocity/acceleration
profile, which should show a sharp direction change at contact/bounce) should
be a **fast-follow**, not deferred indefinitely — v1's ground-truth-replay
approach only works for clips we already have labels for.

### 4. Homography is implemented now

`ir/geometry.py` computes a pixel→court-relative-meters homography via
`cv2.findHomography` (RANSAC over all 14 keypoints) from `CourtKeypointNet`'s
output, and applies it to project ball/court positions into court space.

**Keypoint semantics had to be inferred, not read from a source file** — no
keypoint-name file ships with `Court_Detector_Keypoints/data/data_train.json`,
just an ordered `[x, y]` list per entry. Order was inferred from raw pixel
geometry: paired top/bottom rows at consistent y-values, with
doubles-corner x-spacing consistently ~1.333x the singles-corner x-spacing —
matching the real ITF ratio (10.97m / 8.23m = 1.333) to within 0.3% across 4
independently-checked sample frames. This gave a complete, self-consistent
14-point layout (4 baseline-doubles corners, 4 baseline-singles corners, 4
service-line-singles corners, 2 service-line center marks) — **flagged as an
inferred, not authoritative, assumption; worth verifying against the
dataset's original public source if this is ever cross-referenced.**

**Verification:** reprojecting the 14 known real-world court points through
the fitted homography, on real (ground-truth) keypoints, gives a mean error
of **2.2cm** on the sample checked (`ir/geometry.py:__main__`) — a strong
signal the keypoint-order inference and homography implementation are both
correct. Real-world coordinate system: origin at the far-baseline
doubles-left corner, +x along the baseline, +y along the sideline toward the
near baseline; standard ITF doubles court (23.77m x 10.97m), singles inset
1.37m, service line 5.485m from each baseline.

**Direct visual spot-check (2026-07-17, requested as a second, independent
check beyond the aggregate-proportion/reprojection reasoning above):** ran
real `CourtKeypointNet` inference on val entry `S0Gn0tZQ-jg_900` (mean error
vs ground truth 1.59px, the strongest prediction found in the first 60 val
entries scanned), drew the 14 predicted points labeled with their inferred
names directly on the source frame, and manually checked each one against
what's visually at that pixel location — see
`ir/example_output/keypoint_spotcheck.png`. **Result: CONFIRMS the inferred
ordering.** All 14 points land exactly where their label says: kp0/1
(far-baseline doubles corners) at the top-left/top-right of the court in the
image, kp2/3 (near-baseline doubles corners) at the bottom corners, kp4-7
(singles corners) sit inset from the doubles corners at both baselines,
kp8-11 (singles service-line corners) sit on the service line at the singles
sidelines, and kp12/13 (service-line center "T" marks) sit exactly at the
center-line/service-line intersections, far and near. No point is
mislabeled or swapped. Combined with the 2.2cm ground-truth reprojection
error and the 3.6–10.5cm error range on real model predictions (see
"True end-to-end example" below for a fourth data point in that range), this
closes out the keypoint-order inference as verified by three independent
methods (proportion-matching, reprojection-error, and now direct visual
inspection) rather than resting on inference alone.

**Homography and the ball's z=0 assumption (a permanent limitation, not a bug
to fix):** a homography maps points on a single plane — exact for court
keypoints (painted on the court surface) but only an *approximation* for the
ball whenever it's airborne (which is most of the time). Projecting an
airborne ball's pixel position through the ground-plane homography gives
"where the ball would be if it were touching the ground directly below its
image position," not its true 3D position. **`BallFramePosition.court` should
be read as reliable primarily at/near `contact_frame` and `bounce_frame`**
(where z≈0 genuinely holds), and as a rough approximation elsewhere. This
isn't something decision #4 "solves" — it's an inherent property of
recovering 3D position from a single camera view via ground-plane homography,
worth stating plainly rather than presenting all trajectory points as equally
trustworthy.

### 5. `SessionRecord` is a separate top-level entity

Model 4 operates on session-level aggregate performance vectors (one row per
player per session — `anomaly_detection/train.py:FEATURE_COLUMNS`), not
per-stroke. Bolting its output onto every `StrokeEvent` in a session would
either duplicate one session-level flag across many events (implying a
per-stroke signal that doesn't exist) or leave it null on most of them.
`SessionRecord` (`ir/schema.py`) is its own entity: `features` (validated
against `ANOMALY_FEATURE_COLUMNS`, mirroring `anomaly_detection/train.py`'s
schema exactly), `anomaly_flag`, `anomaly_score`, and `explanation` (the
human-readable top-k-deviated-features text from
`anomaly_detection/train.py:explain_anomaly()`).

### 6. Position fields are normalized to court-relative coordinates

`CourtPoint` (`x_m`, `y_m`, court-relative meters, via decision #4's
homography) is now the **canonical** position representation for things
being *located* — `BallFramePosition.court`. Raw pixel coordinates
(`PixelPoint`) are kept alongside as `BallFramePosition.pixel`, for
traceability/debugging, but are not the primary representation.

**One deliberate exception:** `CourtInfo.keypoints_px` stays in pixel space.
These 14 points are the *calibration input* used to compute the homography
in the first place — expressing them in the coordinate system the homography
itself produces would be circular (you can't calibrate a transform using
points already expressed in its own output space). Everything the homography
is used *to locate* (the ball) is court-relative; the thing used *to build*
the homography (court keypoints) stays in the space it was measured in.

---

## Not yet built (unchanged scope boundary from decisions #1/#2)

- Continuous match-video ingestion: stroke-boundary segmentation, player
  detection/tracking/re-identification, point/rally boundary detection.
- A trained hit/bounce classifier for `event_type` on new (unlabeled) footage
  — see decision #3's fast-follow recommendation.
- Anything for Model 3 (`momentum` stays `None` — see
  `momentum_tracking/DATA_ACQUISITION_CHECKLIST.md`, including the still-open
  momentum-label-definition question, unaffected by this revision).

## What's now built and verified (see `ir/populate.py`, `ir/example_output/`)

- `StrokeEvent` population from real Model 1A (TrackNet) + Model 1B
  (`CourtKeypointNet`, homography) + Model 2 (`StrokeGRU`) inference on real
  pre-segmented sample clips.
- `SessionRecord` population from Model 4 (`IsolationForest`) — necessarily
  using the same synthetic demo data as `anomaly_detection/train.py --demo`,
  since no real session dataset exists yet; output is clearly labeled as such.

## Composite examples vs. the true end-to-end example

**`ir/populate.py`'s three `stroke_event_{0,1,2}.json` files are composites**
(see that file's module docstring): real inference from three independently-
real but *unrelated* source clips (a Ball-Tracker rally, a Court-Keypoints
frame, a THETIS stroke sample) stitched into one record, because no dataset
in this project spans ball-tracking + court-keypoints + stroke-classification
for the same real rally. They exist to prove the schema validates against
genuine (not fabricated) model output and that the geometry pipeline runs —
**not** to demonstrate one coherent camera feed processed start to finish.

**`ir/populate_e2e.py`'s `stroke_event_end_to_end.json` is different: one
real clip, all three models, no compositing.** Ball Tracker+Bounce
`game1/Clip5`, frames 29–88 (60 frames, ~2s @ 30fps, Djokovic vs. Federer,
Shanghai 2014 — `game1` is in TrackNet's held-out *validation* split, so this
also exercises Model 1A's generalization, not a memorized clip):

- **Ball** (Model 1A): real TrackNet inference over the 60-frame window.
- **Court** (Model 1B): real `CourtKeypointNet` inference on frame 59 of
  *this same clip* (not a Court-Keypoints-dataset frame) — a genuine
  cross-domain generalization test, since `CourtKeypointNet` was never
  trained on Ball-Tracker footage. Homography reprojection error: **7.37cm**
  — inside the 3.6–10.5cm range seen on the composite examples, so this
  isn't an outlier; it's a fourth consistent data point for the same
  generalization gap (see "Error compounding" below).
- **Stroke** (Model 2): required extracting a real 99-dim MediaPipe Pose
  sequence from these frames ourselves, since StrokeGRU needs pose input and
  THETIS (its training data) ships only pre-extracted features, no raw
  video, in this repo. Two things had to be solved that don't generalize
  beyond this one clip, and are documented here rather than glossed over:
    1. **API mismatch**: `mediapipe==0.10.35` removed the legacy
       `mp.solutions.pose` API that whatever pipeline produced THETIS's
       `X_*.npy` almost certainly used. Extraction here uses the newer
       `PoseLandmarker` Tasks API instead — same underlying BlazePose
       33-landmark model family and output format, but not guaranteed
       bit-identical to the original extractor (unknown/undocumented).
    2. **Player localization**: THETIS clips are pre-cropped tightly to one
       player; this is a wide broadcast shot where the player is a small
       fraction of the frame — `PoseLandmarker` on the raw frame detects
       **0/60 frames**. Fix used: a simple HSV red-shirt color-blob tracker
       (this player wears red) crops a generous box around him per frame
       before pose detection (**60/60 frames detected** with this crop, vs.
       47/60 with a naive fixed crop that lost him mid-lunge near the net).
       This is a clip-specific hack, not a player-tracking solution —
       multi-player detection/tracking remains explicit future work (see
       decision #1). A pose skeleton overlay on frame 59
       (`ir/example_output/pose_spotcheck_frame59.png`) was checked visually
       and tracks the player's limbs correctly, so the extraction itself
       looks sound even though it isn't the original THETIS extractor.
  Result: **`predicted_class=backhand`, confidence=0.999.** There is no
  ground-truth stroke label for this clip (unlike the THETIS test examples,
  which have one) — Ball-Tracker+Bounce isn't a stroke-classification
  dataset — so this cannot be scored as correct/incorrect, only reported as
  the model's real output on real (if domain-shifted) input. Worth stating
  plainly: two independent domain shifts stack here (extractor API + camera
  framing vs. THETIS's amateur-court close-ups), so this prediction should
  be read as a pipeline-mechanics demonstration, not as evidence of Model 2's
  real-world accuracy — that claim still rests on the THETIS test-set F1.

  **What this example validates, and what it explicitly does not:**
  it validates that the *pipeline plumbing* runs coherently end-to-end on
  one real, unedited piece of footage — a real `StrokeEvent` can be
  constructed from Model 1A + 1B + 2 output on the same clip, satisfying the
  schema's "100% compliance" requirement on genuinely single-source data, not
  just on composited inputs. It does **not** validate stroke-classification
  accuracy on new footage, because two domain-shift factors stack on top of
  each other between this input and what Model 2 was trained/measured on:
  (a) the pose features come from a **different pose-extraction pipeline**
  than THETIS's training data — `mediapipe`'s `PoseLandmarker` Tasks API,
  not whatever produced `X_*.npy` (the original extractor is unknown and
  unrecoverable; see the API-mismatch point above); and (b) getting *any*
  detections at all on this footage's framing/scale required a **workaround
  color-based tracker** (the red-shirt HSV blob crop above) that has no
  counterpart in THETIS's pre-cropped clips or in any trained component of
  this project. Either factor alone would already put this input outside
  Model 2's validated distribution; both together mean this result carries
  no evidential weight on real-world accuracy, only on schema/pipeline
  mechanics.

### Error compounding through the IR pipeline (for the FYP error-analysis section)

Quantified, in one place, how uncertainty stacks as data flows through the
IR layer — useful evidence that "100% schema compliance" (structural
validity) is a different claim from "100% accurate," which the schema
doesn't and can't guarantee:

| Stage | Uncertainty source | Magnitude |
|---|---|---|
| Model 1B keypoint regression | `val_dist_px` (mean pixel error, val set) | ~6.1px @ 1280x720 |
| → Homography fit (ground-truth keypoints) | reprojection error | 2.2cm |
| → Homography fit (real model-predicted keypoints, 4 samples) | reprojection error | 3.6cm – 10.5cm (composites) / 7.37cm (true e2e) |
| → Ball position in court-space | homography z=0 assumption | exact at contact/bounce frames; approximate (unbounded by this table) while airborne — see decision #4 |
| Model 1A ball detection | `val_pck@4px` | 96.4% |
| Model 2 stroke classification | THETIS test macro-F1 | 82.8% (see `stroke_classification/README.md`) |

Each stage's error is measured against its *own* validation data; nothing
here composes these into a single end-to-end error bound (that would need
ground truth spanning the whole pipeline on one dataset, which doesn't
exist — the same gap this section documents). The practical takeaway for
the FYP writeup: court-relative position fields inherit both the keypoint
model's regression error *and* the homography fit's amplification of that
error (roughly px-error → cm-error, not 1:1 — a 6px average keypoint error
produces a several-cm reprojection error because the fit is over-determined
by 14 points but still sensitive to correlated noise), and this compounds
independently of, not in place of, each individual model's own reported
metric.

---

## Stroke-level biomechanical features — proposal, PENDING SIGN-OFF (2026-07-19)

This is the schema/feature-extraction groundwork that `rag/DESIGN.md`'s Step
2 formally deferred (stroke-level biomechanical coaching, blocked on "no
biomechanical fields on `StrokeEvent`, no derivation code"). Per the same
review discipline as the original six decisions above: **inventory and
design first, schema shown for sign-off, no extraction code written yet.**

### Step 1: What's actually extractable, and from what data

**No racket detection exists anywhere in this project** (confirmed:
grepped the whole repo for "racket" — the only hits are prose mentions in
`PROJECT_LOG.md`/`rag/DESIGN.md`/`rag/kb_placeholder.py`, and one explicit
disclaimer in `stroke_classification/README.md`: *"no racket/ball
trajectory signal in this feature set"*). Every field below is a **body
pose measurement**, and where it approximates something the proposal's
language implies is racket-related, it is **named for what it actually
measures**, not renamed to sound like a racket measurement it isn't:
"forearm/wrist orientation," never "racket angle."

**Two available pose data sources, with different capabilities — neither
is a superset of the other:**

| | `StrokeClassification` (THETIS, via `training/stroke_classification/dataset.py`) | `Pose_Estimation` (`data/.../Tennis Player Actions Dataset...`) |
|---|---|---|
| Format | `(60, 99)` per clip: 60-frame sequence × 33 MediaPipe BlazePose landmarks × (x,y,z), already extracted | Single static image + COCO-style 18-keypoint 2D annotation (`nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles, neck`), pixel coords + visibility flag |
| Size | 1980 clips: forehand=495, backhand=495, serve=495, volley=330, smash=165 | 500 images each for `backhand`/`forehand`/`serve`/`ready_position` (no volley/smash) |
| Temporal | Yes — 60-frame window, so features needing *change over time* (weight transfer, follow-through) are computable | No — one frame per sample, only single-instant features are computable |
| Coordinate space | x,y normalized to the per-frame **player crop** (confirmed in `ir/populate_e2e.py` docstring); z is MediaPipe's relative-depth estimate, not metrically validated | Raw pixel coordinates in the original (uncropped) image, no calibration, no depth |
| Contact-frame label | **None.** No ball-tracking data coexists with these clips at all — there is no way to know which of the 60 frames is the true contact frame from this dataset alone | Ambiguous — filename/label says "backhand" etc. generally, no guarantee the single frame *is* the contact instant vs. some other point in the swing |
| Handedness label | None | None |
| Landmark scheme | MediaPipe BlazePose, 33 points | COCO-style, 18 points — a **different convention**; would need an explicit index-mapping table if ever combined with the other source, not attempted here |

**One real exception with a true contact-frame label:** `ir/populate_e2e.py`'s
single end-to-end demo clip (game1/Clip5) has both a real MediaPipe Pose
sequence *and* a real TrackNet-labeled `event_type=="hit"` frame in the same
window — the only place in this project where "contact frame" is a
measured fact rather than a proxy. Everywhere else (all 1980
`StrokeClassification` clips), contact timing must be **approximated**.

**Recommendation: use `StrokeClassification` as the primary extraction
source** (it's the one with temporal information, and it's what
`StrokeInfo`/`StrokeGRU` already consume) — proxy contact frame as **the
frame of peak dominant-wrist speed** within the 60-frame window (a standard
biomechanics-literature proxy for ball contact, not invented here), and
**validate that proxy once against the one real ground-truth instance**
(game1/Clip5) before trusting it on the 1980 THETIS clips, since that's the
only place a check is even possible. `Pose_Estimation` is useful only as a
secondary, single-frame cross-check on the subset of features that don't
need a time window (elbow angle, knee flex, stance width, shoulder-hip
separation) — not proposed as a primary source, given the missing
volley/smash classes and the keypoint-scheme mismatch.

**Concrete feature list (all computed from 2D frame-plane (x,y) landmark
positions; z used nowhere below — MediaPipe's z has no metric validation
in this project, so out-of-plane rotation is explicitly not attempted):**

| Feature | What it measures | Landmarks used |
|---|---|---|
| `wrist_height_ratio` | Dominant wrist height relative to hip line, at contact, normalized by shoulder width (dimensionless — see coordinate-space caveat below) | dominant wrist, both hips, both shoulders |
| `shoulder_rotation_deg` | Shoulder-line angle vs. frame-horizontal, at contact | both shoulders |
| `hip_rotation_deg` | Hip-line angle vs. frame-horizontal, at contact | both hips |
| `shoulder_hip_separation_deg` | Angle between shoulder line and hip line ("X-factor" trunk-rotation differential) — a real, established biomechanics concept, fully computable from body pose alone | both shoulders, both hips |
| `knee_flex_left/right_deg` | Interior knee angle (hip–knee–ankle) | hip, knee, ankle (each side) |
| `stance_width_ratio` | Ankle-to-ankle horizontal distance / shoulder width | both ankles, both shoulders |
| `weight_transfer_ratio` | Hip-midpoint horizontal displacement from window start to contact, in shoulder-widths | both hips, both shoulders, across frames |
| `follow_through_extension_ratio` | Dominant wrist-to-shoulder distance at window end vs. at contact | dominant wrist, dominant shoulder |
| `forearm_orientation_deg` | Elbow→wrist vector angle vs. horizontal, at contact — **the honest rename of "racket angle"; a wrist/forearm-orientation proxy, explicitly not a racket measurement** | dominant elbow, dominant wrist |
| `elbow_angle_deg` | Interior elbow angle (shoulder–elbow–wrist), at contact | dominant shoulder, elbow, wrist |

**Coordinate-space caveat (important, same honesty standard as the IR
layer's homography section):** none of this data is camera-calibrated, so
there is no path to real-world centimeters the way `CourtPoint` has for
ball/court position. Every length/height feature above is expressed as a
**dimensionless ratio against a body-scale reference** (shoulder width),
not an absolute distance — this keeps values roughly comparable across
clips shot at different distances/crops, but "contact height" here means
"height relative to this player's own shoulder width in this frame," never
"contact height in centimeters." Angles are true degrees (scale-invariant
regardless of camera distance) and don't have this caveat.

**Two open methodological questions, flagged rather than decided
silently:**

1. **Dominant-wrist determination.** Neither dataset labels player
   handedness. Proposed heuristic: whichever wrist has greater total
   displacement during the window is "dominant" for that clip. This is
   reasonable for forehand/serve/smash/volley, but **two-handed backhand
   is a real edge case** — both hands are on the racket, so "one dominant
   wrist" is a simplification for that subclass specifically, not a clean
   fit. Proposing to apply the same heuristic uniformly and flag it as an
   approximation for two-handed backhand clips, rather than building
   separate handling — open to reconsidering if this turns out to distort
   results once real numbers are seen in Step 4.
2. **Contact-frame proxy validation.** Peak-wrist-speed-as-contact-proxy is
   standard in the literature but unverified *in this project* until it's
   checked against the one real labeled instance (game1/Clip5). Proposing
   to run that single validation as part of Step 4, report whether the
   proxy frame lands near the true hit frame, and treat the result as an
   honest confidence caveat on the other 1980 clips either way (this is
   n=1, not proof, but it's the only ground truth available).

### Step 2: Benchmark approach — population percentiles within this project's own data, not external norms

Consistent with the session-level precedent (`rag/DESIGN.md` Step 2,
RESOLVED): no licensed/vetted external biomechanics norms are sourced.
Reference ranges are proposed as **percentiles computed within our own
extracted feature population, grouped by collapsed stroke class**
(forehand/backhand/serve/volley/smash have genuinely different natural
ranges for all of the above — pooling across classes would blur real
differences, not measure anything meaningful). E.g. "this clip's
`wrist_height_ratio` is in the bottom 12th percentile *among forehands in
our dataset*," never a claimed-ideal/correct value.

**Flagging the small/skewed-sample limitation explicitly, not papering
over it:** `StrokeClassification`'s per-class counts (forehand=495,
backhand=495, serve=495, volley=330, smash=165) make percentiles
reasonably stable for the three larger classes, but **`smash` at n=165 is
thin for tail percentiles** (a "5th percentile" estimate over 165 samples
rests on roughly 8 data points) — low-percentile claims on smash-class
biomechanics should be reported with that caveat, not presented with the
same confidence as forehand/backhand/serve. `Pose_Estimation`'s 500-per-class
single-frame samples (backhand/forehand/serve only, no smash/volley) could
supplement percentile stability for those three classes' single-frame
features specifically, once the keypoint-scheme mapping is built — not
attempted in this phase, flagged as possible future work.

**Also flagging the population's composition, same caveat pattern as
Model 4's session-level percentiles:** both source datasets are general
tennis-player footage (THETIS is an action-recognition research dataset,
not a curated elite-player corpus; `Pose_Estimation`'s source/skill-level
is undocumented). A percentile here answers "how does this compare to our
sample of players," not "how technically correct is this" — the same
distinction already established and must be repeated in any coaching
output that uses these percentiles, not conflated.

### Step 3: Proposed schema addition — `BiomechanicsInfo`, attached to `StrokeEvent`

Proposed for `ir/schema.py`, following the same "validate at construction"
principle as every existing model, and the same optional-until-built
pattern already used for `MomentumInfo`:

```python
class BiomechanicsInfo(BaseModel):
    """Stroke-level biomechanical features derived purely from MediaPipe
    Pose body-landmark geometry (33 3D landmarks per frame over a
    StrokeEvent's pose sequence). NO RACKET DATA EXISTS ANYWHERE IN THIS
    PROJECT (confirmed by repo-wide search) -- every field here is a body
    pose measurement, or an explicitly-named body-pose PROXY for a
    racket-related concept (e.g. forearm_orientation_deg is NOT racket
    angle), never a renamed approximation pretending to be the racket
    measurement itself.

    Coordinate caveat: no camera calibration exists for the pose data this
    is derived from (unlike CourtPoint, which has real homography). Every
    length/height field is a DIMENSIONLESS RATIO against a body-scale
    reference (shoulder width), not an absolute distance -- comparable
    across clips, NOT convertible to real-world centimeters. Angles are
    true degrees computed from the frame-plane (x, y) projection only;
    MediaPipe's z is unvalidated in this project and is not used here, so
    out-of-plane (toward/away from camera) rotation is not captured.
    """

    contact_frame: int = Field(ge=0)
    contact_frame_source: Literal["ball_tracking_hit_frame", "peak_wrist_speed_proxy"]
    # "ball_tracking_hit_frame": a real event_type=="hit" frame exists in
    # this same StrokeEvent's ball.trajectory (true today only for the
    # game1/Clip5 end-to-end example). "peak_wrist_speed_proxy": no
    # ball-tracking data coexists with this pose sequence (true for every
    # StrokeClassification/THETIS-sourced clip) -- contact is approximated
    # as the frame of maximum dominant-wrist speed in the window. See
    # DESIGN.md open question #2.

    dominant_wrist: Literal["left", "right"]
    dominant_wrist_source: Literal["externally_supplied", "max_displacement_heuristic"]
    # Neither pose dataset labels player handedness -- see DESIGN.md open
    # question #1 re: two-handed backhand as a known edge case for this
    # heuristic.

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
    # landmarks this is computed from (checkpoint="pose_landmarker_full.task"),
    # consistent with how ModelProvenance already handles "which real
    # upstream output does this trace back to."


class StrokeEvent(BaseModel):
    ...
    momentum: MomentumInfo | None = None
    biomechanics: BiomechanicsInfo | None = None  # NEW -- None until Step 4 builds real extraction
    ...
```

**Deliberately NOT included in this schema proposal: percentile/benchmark
fields.** Unlike `SessionRecord` (which bakes in `anomaly_score`/
`explanation` because Model 4 computes those directly as its designed
output), a percentile rank here depends on a choice of reference
population that can be recomputed/extended independently of any single
`StrokeEvent` — proposing to keep `BiomechanicsInfo` limited to raw,
always-computable geometric measurements, and treat percentile lookup as a
separate downstream utility (analogous to `explain_anomaly()`, but for
biomechanics) to be built once real extracted values exist across the
dataset. This keeps today's schema change scoped to "what can this one
stroke's pose data prove about itself," not entangled with the population
data it would eventually be ranked against.

**Explicitly needs your sign-off before any extraction code is written:**
(a) the feature list above — anything missing, or anything that should be
cut; (b) the two open methodological questions (dominant-wrist heuristic,
contact-frame-proxy validation approach); (c) the schema shape itself,
including the decision to leave percentiles out of `BiomechanicsInfo` for
now.

### Sign-off received 2026-07-19 — both open questions resolved as proposed

Dominant-wrist heuristic (open question #1): apply the max-displacement
heuristic uniformly, including to two-handed backhand clips, flagged as an
approximation there rather than special-cased. Schema (Step 3): approved
exactly as proposed. `BiomechanicsInfo` added to `ir/schema.py`,
`StrokeEvent.biomechanics: BiomechanicsInfo | None = None` added following
the same optional-until-built pattern as `MomentumInfo`.

### Step 4: extraction built, run on real data, two real bugs found and fixed during sanity-checking

`ir/biomechanics.py` implements the 10 features exactly as designed above
(pure 2D frame-plane geometry — arctan2 for angles, Euclidean distance
ratios for lengths, no z-axis use). `ir/populate_biomechanics.py` runs it
on one real `StrokeClassification` (THETIS) example per collapsed class
(first match in the `test` split, not cherry-picked) plus the
contact-frame-proxy validation against `ir/populate_e2e.py`'s clip.
**Two real data/methodology issues were found while sanity-checking the
first pass of numbers — not assumed correct because the code ran without
error, exactly per the user's instruction — and both were fixed:**

**Bug 1 (found and fixed) — zero-padded frames poisoning the contact-frame
proxy.** The first run produced obviously broken output on 2 of 5 examples
(`weight_transfer_ratio = -478019`, every angle exactly `0` or `90`).
Root cause: some `StrokeClassification` clips have trailing all-zero
frames — the same "no pose detected, left as zeros" fallback already
documented in `ir/populate_e2e.py`'s docstring, present here too (confirmed
by inspection: `p53_foreopen_s2.avi` has 46 real frames then 14 all-zero
frames; `p27_smash_s2.avi` has 3). The naive peak-wrist-speed proxy picked
a zero-padded frame as "highest speed," because a jump from a real
landmark position to the origin looks like a huge (fake) displacement.
Fixed by computing a `_valid_frame_mask` (any all-zero frame is invalid)
and excluding any transition or candidate frame touching an invalid frame
from dominant-wrist determination, contact-frame selection, and the
window-start/end references used for weight-transfer/follow-through —
using the first/last *valid* frame instead of literally frame 0/-1.

**Bug 2 (found and fixed) — signed vector angle vs. undirected axis angle
for `shoulder_hip_separation_deg`.** Even after fixing bug 1, one real
example (`p4_backhand2h_s2.avi`, backhand) initially reported
`shoulder_hip_separation_deg = 167°` — anatomically impossible (a human
torso cannot twist the shoulders 167° relative to the hips). Root cause: a
line (shoulder-to-shoulder, hip-to-hip) has no inherent direction, but the
original calc treated `left→right shoulder` and `left→right hip` as
directed vectors and took the raw signed angle between them — so if the
shoulder line pointed "screen-left" and the hip line pointed "screen-right"
in that frame's projection (the same physical axis alignment, just labeled
in opposite directions), it read as a near-180° twist instead of a small
one. Fixed by reducing the raw angle-difference mod 180 and folding
anything past 90° back down, giving the true minimal angle between the two
undirected axes. Result: `12.89°` for that same example, and all 5
examples now fall in a `1.98°`–`25.03°` range — consistent with real
X-factor (trunk-rotation-differential) magnitudes reported in biomechanics
literature.

**Third issue found (fixed, same sanity pass) — single-frame shoulder-width
normalization was foreshortening-sensitive.** `wrist_height_ratio` on 2 of
5 examples was implausibly large (`+15.8`, `+13.6` shoulder-widths above
the hip). Diagnosed by printing raw landmark coordinates at the contact
frame: `left_shoulder`/`right_shoulder` were nearly coincident (e.g.
shoulder width of `0.0048` normalized units on one clip, vs. `~0.03–0.15`
in most other frames of the same clip) — the player was side-on to the
camera at that specific instant (common right at contact for serves and
one-handed strokes, where the body naturally rotates side-on into the
shot), so the 2D-projected shoulder line collapses toward a point, and
every ratio dividing by it blows up. Fixed by using the **window-averaged**
shoulder width across all valid frames as the scale reference, not the
single contact-frame value — smooths out one-instant foreshortening
without requiring 3D pose. Result: `wrist_height_ratio` dropped to a
`-0.37`–`+5.65` range, with **serve showing by far the highest value
(+5.65) vs. groundstrokes/volley (-0.37–+1.64)** — a real, physically
sensible pattern (serve contact is genuinely far overhead, well above
groundstroke/volley contact height), not a coincidence of the fix.
**Residual, undisturbed limitation:** this does not fix a clip where the
player stays side-on to the camera for the *entire* window (rare, but
possible, e.g. some one-handed-backhand framings) — that would need true
3D pose or camera calibration to resolve properly, out of scope here, and
left as a known limitation rather than chased further.

**Actual computed values, 2026-07-19, one real example per collapsed
class (`test` split, first match, not cherry-picked):**

| Feature | forehand | backhand (2H) | serve | volley | smash |
|---|---|---|---|---|---|
| dominant_wrist | left | right | right | left | left |
| contact_frame (of valid frames) | 29 (proxy) | 7 (proxy) | 32 (proxy) | 30 (proxy) | 33 (proxy) |
| wrist_height_ratio | +1.64 | +0.94 | **+5.65** | -0.37 | +0.65 |
| shoulder_hip_separation_deg | 10.71 | 12.89 | 1.98 | 6.31 | 25.03 |
| knee_flex_left/right_deg | 171.2/155.3 | 160.0/166.3 | 156.8/147.9 | 169.6/161.1 | 176.0/178.8 |
| stance_width_ratio | 0.98 | 0.54 | 0.53 | 1.39 | 1.48 |
| weight_transfer_ratio | -0.03 | -0.07 | -0.26 | +0.33 | +0.53 |
| follow_through_extension_ratio | 1.67 | 1.28 | 1.05 | 0.93 | 0.90 |
| forearm_orientation_deg | +153.8 | -86.5 | +99.8 | -110.7 | -125.5 |
| elbow_angle_deg | 89.9 | 178.0 | 155.9 | 153.8 | 149.8 |

Two data notes surfaced by the run itself (not hidden): `p53_foreopen_s2.avi`
(forehand) had 14/60 zero-padded frames, `p27_smash_s2.avi` (smash) had
3/60 — both excluded from selection per the bug-1 fix above, not silently
included.

**Contact-frame proxy validation against the one real ground-truth
clip** (`ir/populate_e2e.py`'s game1/Clip5): re-extracted that clip's real
pose sequence and compared the peak-wrist-speed proxy's picked frame
against the real TrackNet-labeled `hit` frames. **Honest complication found
here too:** this window is a multi-shot rally segment (5 real hit events:
local frames `10, 11, 30, 53, 54`), not one isolated stroke like every
`StrokeClassification` clip — `StrokeGRU` predicts one class for the whole
60-frame window with no per-frame localization, so there is no single
unambiguous "the" contact frame to validate against here. Using the hit
frame closest to the window's temporal center as the best available single
reference (frame 30, the shot the window is centered on): the proxy picked
frame **27**, a **3-frame** difference (~0.1s at 30fps) — a reasonable
result, but this is an n=1 sanity check with a real ambiguity in what it's
even comparing against, not proof the proxy generalizes; reported as such,
not oversold.

**What's built and real now:** `ir/biomechanics.py` (feature extraction,
2D geometry only), `ir/populate_biomechanics.py` (runs it on real THETIS
examples + the one contact-frame validation), `BiomechanicsInfo` in
`ir/schema.py`. **What's still not built (at the time of the 5-example
pass):** percentile/benchmark ranking, and a full population pass across
all 1980 clips — done next, see below.

---

### Full-dataset extraction pass (2026-07-19) — all ~1980 clips, not just the 5 sanity examples

The 5-example check above is not enough to trust class-level patterns
(one sample per class can't distinguish a real biomechanical difference
from individual variation) or to build a percentile reference population
(percentiles need the real distribution, not 5 points). `ir/populate_biomechanics_full.py`
runs the same extraction across all 1980 clips (all 3 splits combined:
train=1386, val=297, test=297) and buckets every clip by outcome, so
failures are counted and categorized, not silently dropped.

**Extraction outcome:**

| Bucket | Count | % | Meaning |
|---|---|---|---|
| `no_valid_frames` | 0 | 0.0% | Every frame zero (total pose-detection failure) — didn't occur |
| `insufficient_valid_frames` | 6 | 0.3% | Fewer than 30 real frames (half the 60-frame window) — threshold set from this project's own n_valid distribution (24-60 across all clips, only 6 below 30), all 6 are `backhand` clips |
| `degenerate_camera_angle` | 207 | 10.5% | Contact-frame shoulder width too small — see bug recurrence below |
| `degenerate_arm_extension` | 97 | 4.9% | Contact-frame wrist-to-shoulder distance too small — a 4th issue, found only at this scale, see below |
| **`clean`** | **1670** | **84.3%** | Used as the percentile reference population |

Clean per-class sample sizes: backhand=427, forehand=407, serve=398,
volley=301, **smash=137** (smallest, consistent with the small native
sample size already flagged in the Step 2 percentile discussion above).

**Watching for recurrence of the three already-fixed bugs, at scale — checked, not assumed generalized:**

1. **Zero-padded frames (bug 1): does not recur as a hard failure.**
   `insufficient_valid_frames` is only 6/1980 (0.3%), and there are zero
   clips with *no* valid frames at all — the fix generalizes.
2. **Signed-vector vs. undirected-axis angle (bug 2): cannot recur, by
   construction.** `shoulder_hip_separation_deg`'s formula
   (`min(raw_diff, 180 - raw_diff)`) mathematically bounds the result to
   `[0, 90]` regardless of input — confirmed by the full-dataset max
   across every class topping out at `89.877°`, never above. This is a
   guarantee, not an empirical observation.
3. **Single-frame shoulder-width collapse (bug 3): the RATIO fix
   generalizes, but the bug still affects fields the ratio fix doesn't
   cover.** The window-averaging fix protects every *ratio* feature — no
   near-zero cluster shows up in the window-averaged shoulder-width
   distribution at scale (0.032–0.181 across all 1980 clips, smooth, no
   collapse). But `shoulder_rotation_deg`/`hip_rotation_deg`/
   `shoulder_hip_separation_deg` are computed directly from the single
   **contact frame's** landmarks, not window-averaged, so they remain
   exposed. Checked, not assumed: the contact-frame shoulder-width
   distribution has a long left tail down to `0.0002` (near-total
   collapse), and clips in its bottom 10% (`sw <= ~0.0194`) show a mean
   `shoulder_hip_separation_deg` of **41.7° vs. 19.5° overall** — roughly
   double, a real, verified correlation between low contact-frame
   shoulder width and unstable angle output. **207 clips (10.5%)** fall
   below this data-justified threshold and are excluded from the clean
   population as `degenerate_camera_angle`.

**A 4th instance of the same root-cause pattern, found only at
full-dataset scale — not one of the three originally fixed bugs, but the
same underlying issue (dividing by a quantity that can collapse near zero
on a single frame), so treated the same way:** `follow_through_extension_ratio`
had a heavy right tail (max `52.7`, std 2.0–3.8 per class) on the
first full-dataset pass. Diagnosed the same way as bug 3: checked whether
extreme values correlate with a near-zero denominator before excluding
anything. They do — of the 116 clips with `follow_through_extension_ratio
> 5` across the whole dataset, **89 (77%) fall in the bottom 5% of
dominant wrist-to-shoulder distance at the contact frame** (small whenever
the arm is tucked close to the body at that instant — a compact arm
position, or a landmark-noise artifact; either way the ratio it feeds is
unstable). Excluded via a new `degenerate_arm_extension` bucket (97
clips, 4.9%), threshold set at that empirically-justified 5th-percentile
point. After exclusion, `follow_through_extension_ratio`'s max dropped
from `52.7` to `7.26` and its per-class std dropped to `0.66–1.28` — a
much better-behaved feature. **This is exactly the kind of thing the
full-dataset pass was for: a real instability that 5 examples didn't
happen to surface.**

**Do the 5-example class-level patterns hold at scale? Checked directly,
mixed result — reported honestly, not cherry-picked to confirm the
sanity-check:**

- **Holds: `wrist_height_ratio`.** Population means: serve **2.47**
  (highest by a wide margin), smash 2.04, backhand 1.49, forehand 1.30,
  volley **0.85** (lowest) — the same serve-highest, volley-lowest
  ordering seen on the 5 individual examples, now confirmed as a real
  class-level pattern, not a coincidence of which 5 clips got picked.
- **Holds: `weight_transfer_ratio`.** Serve's mean (`+0.31`) is clearly
  separated from every other class (`-0.12` to `+0.13`) — consistent with
  real serve biomechanics (a distinct forward weight drive). Caveat noted
  in `ir/biomechanics_percentiles.py`: this feature's sign depends on
  which screen-direction is "forward" for a given clip's camera framing,
  which isn't independently verified as consistent across all 1980 clips
  — the pattern is real in this data, but the sign-consistency assumption
  underneath it is not proven, only plausible.
- **Does NOT clearly hold: `elbow_angle_deg`.** The 5-example pass showed
  a striking forehand-vs-backhand contrast (89.9° vs. 178°). At full scale,
  every class's mean sits in a narrow `120°–134°` band with a **std of
  38–48° per class** — the individual example difference was real
  individual variation, not a systematic class difference. Reporting this
  plainly rather than only keeping the finding that happened to confirm
  the smaller check.
- **Does NOT hold as measured: `shoulder_hip_separation_deg`.** The
  5-example serve had the *lowest* separation (1.98°) of the 5; at scale,
  serve has one of the *highest* class means (20.1°, alongside smash's
  19.6°), with forehand/backhand/volley all lower (13–18°). The single
  serve example was not representative — a reminder that n=1-per-class
  checks are for catching broken code, not for reading off real class
  differences.
- **New observation, not from the 5-example pass: knee flexion doesn't
  differentiate stroke classes much.** All 5 classes cluster in a
  `151°–171°` mean range (i.e. legs close to straight at the contact
  frame across every stroke type) — plausibly because the peak-wrist-speed
  contact proxy tends to land on the moment of leg-drive extension
  regardless of stroke type, not a discriminating feature on its own.

**Percentile ranking built** (`ir/biomechanics_percentiles.py`),
**only now that the full-dataset pass has been run and checked** — per
the earlier instruction, percentile validity depends on this being a
real, verified distribution, not silently assumed. `BiomechanicsPercentileBenchmark`
loads the `clean`-bucket population from
`ir/example_output/biomechanics_full_dataset.csv` and answers "what
percentile is this value, for this class, for this feature," analogous to
`explain_anomaly()`'s z-score lookup but for biomechanics.

**Deliberately excludes 3 of the 10 fields from percentile ranking, and
raises rather than silently computing a number if asked for one:**
`shoulder_rotation_deg`, `hip_rotation_deg`, `forearm_orientation_deg` are
raw angles against a fixed frame-horizontal axis. Their full-dataset std
is **82–171° per class** — essentially spanning the whole `-180..180`
range — because which way a player happens to face the camera in a given
clip is a filming-setup artifact, not a real biomechanical quantity, so a
percentile against them would be meaningless, discovered by checking the
actual population spread rather than assumed reasonable because the
5-example pass didn't show it. `shoulder_hip_separation_deg` doesn't have
this problem (it's already an undirected axis-to-axis separation, bounded
`[0, 90]` by construction) and is ranked.

**Demo output (`ir/biomechanics_percentiles.py`'s `__main__`), ranking the
same 5 hand-inspected examples from the earlier sanity pass:**

| Class | Feature | Value | Percentile | n |
|---|---|---|---|---|
| forehand | wrist_height_ratio | 1.636 | 66.6th | 407 |
| backhand | wrist_height_ratio | 0.935 | 41.0th | 427 |
| serve | wrist_height_ratio | 5.651 | 93.0th | 398 |
| volley | wrist_height_ratio | -0.366 | 5.6th | 301 |
| smash | wrist_height_ratio | 0.652 | 16.8th | 137 |
| backhand | shoulder_hip_separation_deg | 12.89 | 59.5th | 427 |
| serve | elbow_angle_deg | 155.92 | 55.0th | 398 |

Attempting to rank an excluded feature (e.g. `shoulder_rotation_deg`)
raises `ValueError` rather than returning a number — verified directly.

**Still not built:** any coaching-generation integration (out of scope for
this phase, per the original instruction not to touch the RAG layer yet),
and no facing-direction normalization (would let `shoulder_rotation_deg`/
`hip_rotation_deg`/`forearm_orientation_deg` become usable, but requires
knowing/inferring which way the court/net is relative to the camera per
clip — real design work, not attempted here, see "Future work" below).

---

### Final summary table — what's usable for coaching grounding (for the FYP report)

Of the 10 `BiomechanicsInfo` fields, here is what's actually safe to cite
in generated coaching feedback, what isn't, and why — the single table
this whole section has been building toward:

| Field | Status | Why |
|---|---|---|
| `wrist_height_ratio` | **Validated at scale, percentile-ranked** | Full-population pattern confirmed: serve highest (2.47 mean), volley lowest (0.85) — matches the 5-example finding, not individual noise |
| `weight_transfer_ratio` | **Validated at scale, percentile-ranked** | Serve's mean (+0.31) clearly separated from all other classes at population level — caveat: sign-consistency across clips' camera framing is plausible but not independently verified |
| `knee_flex_left_deg` / `knee_flex_right_deg` | **Percentile-ranked, but low discriminative value** | Computed reliably at scale (interior angle, direction-independent, no camera-artifact issue) — but all 5 classes cluster in a narrow 151°–171° mean band, so a percentile is valid but not very informative about stroke-specific technique |
| `stance_width_ratio` | **Percentile-ranked** | Direction-independent distance ratio, no camera-artifact issue; overlapping but plausible per-class ranges (1.24–1.51 mean) |
| `follow_through_extension_ratio` | **Percentile-ranked, after excluding degenerate_arm_extension clips** | Direction-independent, but only reliable once the near-zero-denominator clips (4.9% of the dataset) are excluded — see bug 4 above |
| `elbow_angle_deg` | **Percentile-ranked, but the one striking pattern from the 5-example pass was FALSE at scale** | Direction-independent, no camera-artifact issue, computed reliably — but the forehand-vs-backhand contrast seen on 5 examples (89.9°/178°) is NOT a real class difference (population means: 120°–134° for every class, 38–48° std). Ranked, but should not be cited as "characteristic of class X" in coaching text without that caveat |
| `shoulder_hip_separation_deg` | **Percentile-ranked, but the one 5-example pattern was FALSE at scale** | Mathematically bounded to [0°, 90°] (bug 2 fix), immune to the camera-facing-direction problem below — but the 5-example serve had the *lowest* value (1.98°) while serve actually has one of the *highest* population means (20.1°). Ranked, but the specific "serve has low X-factor" claim from the sanity pass must not be repeated |
| `shoulder_rotation_deg` | **Excluded from percentile ranking** | Raw angle vs. fixed frame-horizontal axis — 82–171° std per class at full scale (spans nearly the whole -180..180 range) because which way the player faces the camera is a filming-setup artifact, not a comparable biomechanical quantity |
| `hip_rotation_deg` | **Excluded from percentile ranking** | Same camera-facing-direction problem as `shoulder_rotation_deg` |
| `forearm_orientation_deg` | **Excluded from percentile ranking** | Same camera-facing-direction problem — this was the field renamed from the proposal's "racket angle"; still unusable for benchmarking even under its honest name, for an unrelated reason (camera framing, not the racket-vs-body-proxy issue) |

**Contact-frame timing itself remains a proxy, not a measured fact, for
every field above except in the single `ir/populate_e2e.py` clip** — see
the contact-frame-proxy validation (3-frame/~0.1s difference from the one
real ground-truth hit frame, n=1, honestly caveated) earlier in this
section.

---

### Future work (scoped, not built): camera-angle normalization for the 3 excluded rotation fields

The 3 excluded fields (`shoulder_rotation_deg`, `hip_rotation_deg`,
`forearm_orientation_deg`) are unusable specifically because they are
**camera-relative, not viewpoint-corrected** — the same category of
problem `CourtPoint`/`CourtInfo.homography_matrix` already solves for
*ball and player position* (see decision #6 and `ir/geometry.py`): a raw
pixel position is camera-dependent and not comparable across different
camera setups, which is exactly why homography projects it into a
court-relative, camera-independent frame before it's used anywhere
downstream. **No equivalent correction exists yet for body-segment
rotation angles**, which is why they're excluded rather than ranked.

This is NOT a straight reuse of the existing homography — homography
solves for position on the (approximately planar) court surface using 14
known real-world court points, which is a different problem from
correcting a 2D-projected body-segment angle for an unknown camera
viewing angle relative to the player. But it is the same *family* of
problem (a camera-relative measurement that needs a viewpoint correction
before it's comparable across clips), and this project already has two
relevant pieces of scaffolding that a future attempt could build from,
rather than starting blind:

1. **`StrokeEvent.player_side` (`"near"`/`"far"`)** — already externally
   supplied per decision #1, a coarse signal for which way a player is
   facing relative to the camera, that this phase's biomechanics code
   does not currently consume at all.
2. **`CourtInfo.homography_matrix`**, when available for the same clip —
   gives the camera's relationship to the real court plane, which is a
   more precise version of the same "which way is the camera looking at
   this rally" signal `player_side` only approximates.

**Scoped next step, if picked up:** investigate whether combining
`player_side` (or, where available, the homography-derived camera
orientation) with the already-computed window-averaged shoulder width
(which acts as an implicit foreshortening signal — small = more side-on to
the camera, see bug 3) is enough to either (a) correct the raw
`*_rotation_deg`/`forearm_orientation_deg` values into a
camera-independent frame, or (b) at minimum bucket clips into coarse
facing-direction groups (e.g. "facing camera" vs. "side-on") within which
the raw angles *would* be comparable, without needing full 3D pose. This
is explicitly a future-work item, not started — no code has been written
toward it in this phase, since it requires either new StrokeEvent linkage
(pose sequences currently have no join key back to the `CourtInfo` of the
same rally in the `StrokeClassification` dataset — a real gap to resolve
first) or a from-scratch viewing-angle estimation method, either of which
is real design work beyond this phase's scope.
