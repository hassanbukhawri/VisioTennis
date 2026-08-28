# VisioTennis — Project Log

Running history of the whole project: what's been done, key results, bugs hit
and how they were fixed, and the reasoning behind non-obvious decisions.
Per-model implementation detail lives in each model's own `training/<model>/README.md`
(and `DATA_ACQUISITION_CHECKLIST.md` for Model 3) — this file is the single
narrative history, updated at the end of each completed task.

---

## 2026-07-16 — 2026-07-17: Model 1A — Ball Tracking (TrackNet)

**What was done:** Inspected `Ball Tracker+Bounce/game{N}/Clip{N}/Label.csv`
schema before writing any code. Built a TrackNet-style U-Net encoder-decoder
(2.84M params) taking a 3-frame RGB stack and regressing a Gaussian heatmap
per frame, with a weighted-MSE loss to counter the extreme background/ball
pixel imbalance. Split train/val by `game`, not by frame, to avoid clip
leakage. Smoke-tested (50-100 samples, shapes/loss direction/GPU memory
confirmed) before any full run, per the project's data-first/smoke-test-first
rule.

**Key results:** Final checkpoint stopped at epoch 25 of the resumed run
(cumulative ~53 effective epochs across 3 launches), **val_pck@4px = 0.964**.
Training was stopped intentionally rather than run to the original 60-epoch
target: val_pck plateaued around epoch 25 (best-so-far last improved there,
tied once at epoch 30, never beaten in the final 9 logged epochs) and val_loss
had been drifting up from its epoch-5-6 minimum — classic mild-overfitting
plateau. Continuing to epoch 60 was low-risk (checkpointing only keeps the
best) but low expected value, so training was stopped early by user decision
once the trend was shown. Exported to ONNX (`tracknet.onnx`), verified via
real onnxruntime inference (not just `onnx.checker`, see bug below).

**Bugs / environment issues:**
- **Missing `torch.no_grad()` during validation** (caught via a memory-usage
  anomaly, not a crash — a smaller batch size showed *higher* peak GPU memory
  than a larger one). Root cause: validation forward passes were building
  full autograd graphs. Fixed by wrapping the forward pass in
  `torch.set_grad_enabled(train)`; peak memory dropped from 8.39GB to 4.78GB.
  Applied proactively to every training script written afterward.
- **Harness-tracked background training got killed twice** (once at ~78 min,
  once at ~43-55 min, no Python error, clean GPU release both times — an
  external supervision issue, not a code bug). Fixed by switching all
  long-running training launches to genuinely OS-detached processes
  (PowerShell `Start-Process -WindowStyle Hidden -PassThru`), which are not
  subject to the harness's background-task supervision. Used for every full
  training run in this project from that point on.
- **`onnx.checker.check_model()` segfaults on this machine** — an `onnx==1.22.0`
  / `protobuf==5.29.5` native ABI mismatch, unrelated to any exported model
  (confirmed: a trivial single-conv-layer model's checker call crashes too).
  Fixed by dropping the checker call from every export script and relying on
  actual onnxruntime inference as the correctness check instead (arguably a
  more meaningful check anyway, since it exercises the exported graph).
  `onnxruntime-gpu`'s Windows wheel also failed to load its CUDA DLLs; pinned
  `onnxruntime==1.18.1` (CPU) everywhere instead — the export scripts only
  ever use `CPUExecutionProvider` for this sanity check.
- **`onnxruntime-gpu` and `onnxruntime` share the same import path** — later
  (during Model 1B), Ultralytics auto-installed `onnxruntime-gpu==1.27.0` as a
  missing-dependency side effect of an export call, which silently overwrote
  the working `onnxruntime==1.18.1` CPU install and reintroduced the DLL
  crash. Fixed by reinstalling the pinned CPU version and disabling
  Ultralytics' autoinstall (`YOLO_AUTOINSTALL=False`) for all export calls.

---

## 2026-07-17: Model 1B — Court Detection (YOLOv8-seg) + Court Keypoints

### Dataset schema findings (before writing any code)

- `Court_Detector/` (Roboflow export) turned out to be **YOLO-segmentation**
  polygon format, not bbox as the original spec assumed — confirmed by reading
  label files directly. No `data.yaml` or class-names file shipped; `"court"`
  was adopted as the (unverified) class name, flagged to the user.
  `prepare_data.py` hardlinks the raw export into Ultralytics' expected
  `images/`+`labels/` layout without mutating the source export.
- `Court_Detector_Keypoints/` — confirmed 14-keypoint schema via
  `data_train.json`/`data_val.json` inspection; not YOLO-pose format, so a
  custom direct-regression model (ResNet18 backbone + FC head, sigmoid-bounded
  normalized coordinates) was built instead of assuming YOLOv8-pose would work
  out of the box.

### Resume-safety verification (before trusting any multi-hour run)

Before launching Model 1B's long full runs, ran a live kill-mid-epoch test on
`court_keypoints/train.py`: launched real training, killed the process
mid-epoch-3, confirmed the last per-epoch checkpoint was intact (all tensors
finite, no partial write), and confirmed a fresh model+optimizer+scaler
reconstructed from that checkpoint reproduced the exact saved optimizer state
(Adam step count, `exp_avg`/`exp_avg_sq` sums, AMP scaler `growth_tracker`) —
not just "no exception thrown." Resuming training afterward continued smoothly
with no discontinuity in loss/metrics. This directly motivated fixing a real
gap found during the check: **checkpoints only ever saved `model_state`, never
optimizer/scaler state**, meaning every prior resume (across all three models)
had been silently restarting Adam's momentum from zero. Fixed in both
`ball_tracking/train.py` and `court_keypoints/train.py` (optimizer + AMP
scaler state now saved and restored on `--resume`), and added a "last"
checkpoint saved every epoch (not just on improvement) so a crash never loses
more than one epoch of progress.

### Training results

- **Court-detection (YOLOv8n-seg):** calibrated first (8 epochs, real
  batch=16/imgsz=640) — found the dataset is tiny (369 train images) and cheap
  (~5.5s/epoch), so the full 100-epoch run only takes ~9-10 min, not hours.
  Full run: **mAP50=0.995, mAP50-95=0.995** (mask). See split-leakage
  investigation below for why this number was checked twice.
- **Court-keypoints (ResNet18 regression):** calibrated (8 epochs, full
  dataset, batch=64) — no early val-loss drift. Full 40-epoch run: best
  checkpoint at epoch 30, **val_dist=6.14px, val_pck@10px=0.924**. Val
  performance plateaued after ~epoch 18-20 while train kept improving (mild
  overfitting onset), but best-checkpoint selection (saved every epoch)
  correctly avoided picking a worse late-epoch checkpoint — the run happened
  to *end* on a noisy bad epoch (epoch 40: val_dist=11.1px), which a "last
  epoch" checkpoint policy would have wrongly kept.

### Split-leakage investigation and fix (decision + reasoning)

The original court-detection run reported **mAP50 == mAP50-95 == 0.995**
exactly — suspicious, because both thresholds landing on an identical ceiling
value is a red flag for either a genuinely-too-easy eval set or a leaky split.
Investigated by:
1. Byte-hashing every image across train/valid/test — **zero exact duplicates
   across splits** (clean at that level).
2. Noticing 4 distinct source-video filename patterns (`{n}_...`, `frame_{n}_...`,
   `vid1_{n}_...`, `vid2_{n}_...`) and checking frame-ID proximity between
   splits within each group — median gaps as small as 1 frame.
3. Perceptual hashing (16x16 average hash) confirmed this as real
   near-duplicate leakage: **56% of the original 107 validation images were
   near-identical (Hamming distance ≤10/256) to a training frame**, rising to
   100% within the `vid1_` group alone. Root cause: the Roboflow split was
   frame-level random, not by source video.
4. **Control check:** re-validated against only the one source-video group
   that was NOT tightly leaked (median Hamming ~102, i.e. essentially
   uncorrelated) — got the identical 0.995 mask mAP. This showed the
   near-perfect score reflects a genuinely easy single-class task, not
   memorization, but the split was still worth fixing so the reported number
   doesn't need that side-channel argument to be trusted.
5. **Fix:** rewrote `prepare_data.py` to group all images by source-video tag
   and assign each group *wholly* to one split — no source video is ever split
   across train/valid/test. Landed close to but not exactly 70/20/10 (67.0% /
   19.6% / 13.4%) since there are only 4 groups to distribute; chose the
   assignment that kept train diverse (2 of 4 groups) and gave valid/test one
   held-out video each.
6. **Retrained on the fixed split** (identical 100-epoch config): mask
   mAP50-95 = **0.995**, matching the clean-subset control exactly. Box
   mAP50-95 came in slightly lower (0.969 vs. the original run's 0.995) — a
   small, expected difference, not a meaningful drop.

**Documented limitation for the FYP writeup** (`training/court_detection/README.md`):
all current footage traces to a small number of physical courts/camera setups
(at most 4 source videos). The split fix rules out frame-level memorization
*within this footage*, but the reported mAP should be read as strong
same-domain performance, not validated cross-venue generalization.

Both Model 1B checkpoints exported to ONNX and verified via onnxruntime
inference (court-detection re-exported from the corrected `court_seg_grouped`
checkpoint after the split fix, superseding the earlier leaky-split export).

---

## 2026-07-17: Model 2 — Stroke Classification

### Data inspection and proposal-alignment decision

`StrokeClassification/processed/X_{split}.npy` inspected before any code:
shape `(n, 60, 99)` float32, no NaNs — 60-frame MediaPipe Pose sequences
(33 keypoints × x/y/z = 99), already normalized. Cross-referencing
`class_map.csv` and `labels_*.csv` filenames (e.g. `p47_serflat_s3.avi`)
identified this as the public **THETIS** dataset: 55 players, 12 native stroke
classes, perfectly balanced at 165 clips/class (1980 total).

The FYP proposal specifies 5 categories (forehand, backhand, serve, volley,
smash) with an F1>82% target, not THETIS's native 12. Printed the full
`class_map.csv` and mapped all 12 native classes onto the 5 proposal
categories — a clean, unambiguous 1-to-many mapping (every native class maps
to exactly one proposal category), with **volley pulled out as its own
category** (`forehand_volley` + `backhand_volley`) rather than folded into
forehand/backhand, matching correct tennis technique grouping:

| proposal category | native classes | count |
|---|---|---|
| forehand | forehand_flat, forehand_openstands, forehand_slice | 495 |
| backhand | backhand, backhand2hands, backhand_slice | 495 |
| serve | flat_service, kick_service, slice_service | 495 |
| volley | forehand_volley, backhand_volley | 330 |
| smash | smash | 165 |

**Decision:** train directly on the collapsed 5-class labels (not native-12
with post-hoc grouping) — decided with the user specifically to match what's
defensible against the proposal document. This introduces a real 3:1 class
imbalance (smash vs. the three largest categories) not present in the native
labels, handled via inverse-frequency class-weighted loss rather than
resampling.

### Model and results

`StrokeGRU`: 2-layer bidirectional GRU (hidden=128, 0.54M params) over the
60-frame sequence, mean+max time-pooling, FC head to 5 classes. Dataset is
tiny enough (1386 train samples) that the full 40-epoch run completes in
seconds — no two-stream video fallback needed, since the pre-extracted pose
sequences were directly usable (confirmed by the shape/dtype inspection
above). Checkpointed on best val macro-F1 (epoch 39/40, 0.828).

**Held-out test set (n=297): accuracy=84.5%, macro-F1=0.834 — clears the
proposal's F1>82% target.** Per-class F1: forehand 0.828, backhand 0.879,
serve 0.933, volley 0.721, smash 0.809.

**Confusion analysis (why volley is weakest):** volley has 231 train / 50 test
support — less than forehand/backhand/serve (346-347 train) but *more* than
smash (115 train), and smash still scores a higher F1 (0.809) than volley
(0.721) despite half the training data. Since the class with less data scores
*better*, the gap is not primarily a data-volume effect. The confusion matrix
confirms a specific, directional cause: 14.0% of true volleys predicted
forehand (and 17.3% of true forehands predicted volley — the largest
off-diagonal cell in the whole matrix), while serve and smash stay clean
against everything except each other. Read as a **feature-set limitation**
(pose-only features can't distinguish a compact/open-stance forehand from a
forehand volley without racket/ball contact-point info), not something more
training data would obviously fix — noted in `training/stroke_classification/README.md`
for the FYP writeup. Full raw + normalized confusion matrices are in that
README; raw matrix also saved as
`training/stroke_classification/checkpoints/test_confusion_matrix.npy`.

ONNX export verified against the PyTorch model directly at both batch=1 and
batch=8 (max abs diff ~1.9e-6) — the exporter's dynamic-batch GRU warning
turned out to be a non-issue for this model.

---

## 2026-07-17: Model 3 — Momentum Tracking (scope-only, per spec)

No point-by-point match data (e.g. shot-by-shot charting data) exists
anywhere in the local repo — confirmed by search before writing anything.
Per spec, did **not** build a training script or fabricate placeholder data
to exercise one. Wrote `training/momentum_tracking/model.py`
(`TennisMomentumLSTM`: 2-layer LSTM, provisional 13-dim per-point feature
spec, Tanh-bounded `[-1, 1]` per-point momentum output — every design choice
in the docstring explicitly marked provisional/revisit-on-real-data) and
`DATA_ACQUISITION_CHECKLIST.md` (what data/schema is needed, and — deliberately
unresolved — how "momentum" itself should be operationalized as a training
target once real data exists, since that's a modeling decision that shouldn't
be defaulted silently).

**KNOWN OPEN ITEM (not resolved, do not treat as decided):** added a dedicated
"OPEN DESIGN QUESTION" section to `DATA_ACQUISITION_CHECKLIST.md` spelling out
that momentum has no ground truth in any point-by-point data source — it's a
construct that has to be derived from raw point outcomes via an explicit,
deliberately-chosen definition, not something the data will hand over
directly. Sketched 3 candidate label-construction approaches to evaluate
once real data exists (verify exact citations independently before use in
the FYP writeup — these are recalled from general tennis-analytics domain
knowledge, not re-derived from primary sources here):
1. Smoothed/EWMA recent win-rate (simplest, arbitrary window/decay choice).
2. Win-probability-derivative ("ΔWP" per point), requiring a separate
   point-level win-probability model built first.
3. Point-importance-weighted momentum (weighting points by their leverage on
   match outcome, e.g. break/set/match points count more than routine
   points) — most tennis-aware, most implementation work.
No approach has been chosen. This is explicitly flagged so a future session
doesn't silently default to whichever is easiest to code.

---

## 2026-07-17: Model 4 — Anomaly / Fatigue Detection (Isolation Forest)

No historical multi-session performance dataset exists locally. Per spec,
wrote `training/anomaly_detection/train.py` (CPU-only, sklearn
`IsolationForest`) against an explicit placeholder `FEATURE_COLUMNS` schema
(first-serve %, error/winner rates, court coverage speed, reaction time, etc.)
documented in the script itself. The script runs in two modes: `--data
<csv>` for real data (validates required columns exist) or `--demo` against
clearly-labeled synthetic data, purely to prove the fit → flag →
`decision_function` score → human-readable-explanation pipeline works
end-to-end before any real data exists. Explanation approach: per-feature
z-score deviation from the fitted training distribution, surfacing the
top-k most-deviated features per flagged session in plain text (e.g.
`unforced_error_rate=0.34 (3.4 std above average)`) — chosen over
IsolationForest's internal path-length structure for simplicity and stability.
Demo run recovers 100% of injected synthetic anomalies (expected, given a
well-separated synthetic population — not a real-world metric, and the script
prints an explicit disclaimer any time `--demo` is used so this can't be
mistaken for a real result later). **Sanity-checked explicitly** (not just
"script exits cleanly"): a crosstab of `anomaly_flag` against the synthetic
ground-truth label confirmed all 20 injected anomalous sessions got
`anomaly_flag == -1` and all 180 normal sessions got `+1`, with zero
misclassifications in either direction.

---

## 2026-07-17: Intermediate Representation (IR) layer — schema design (v1, superseded below)

Started objective 2 of the proposal: the schema-validated JSON layer bridging
CV model output with downstream ML/RAG components. **This is a design-phase
entry, not a completed one** — only the schema shape has been designed and
verified to construct/validate/reject correctly; no code populating an IR
record from real model output has been written, pending the user's review of
the schema shape.

**What was done:** `ir/DESIGN.md` (rationale + open questions) and
`ir/schema.py` (Pydantic v2 models) — top-level entity `StrokeEvent`, one
record per identified stroke, with field groups `ball` (Model 1A),
`court` (Model 1B keypoints), `stroke` (Model 2), and a `momentum` field that
stays `None` until Model 3 exists. "100% schema compliance" (the proposal's
stated requirement) is enforced structurally — every field is
type/range/enum-constrained via Pydantic so an invalid `StrokeEvent` cannot
be constructed in the first place, rather than being checked after the fact.
Verified: valid construction succeeds and round-trips to JSON;
`StrokeEvent.model_json_schema()` exports a standard JSON Schema for
non-Python consumers; four separate invalid-data cases (out-of-range
confidence, `class_probabilities` not matching the 5 known classes,
`predicted_class` not matching the probability argmax, wrong keypoint count,
`end_frame < start_frame`) were each confirmed to raise `ValidationError`.

**Bug fixed during design:** a field named `model_provenance` collided with
Pydantic v2's reserved `model_` namespace (harmless but noisy warning on
every construction) — renamed to `source_model` throughout.

**Decisions flagged for the user's sign-off, not made unilaterally**
(full detail in `ir/DESIGN.md` "Open questions"):
1. **Player identity/tracking is not solved by any of the four trained
   models** — `player_id`/`player_side` exist in the schema as optional
   fields with no current producer.
2. **Rally/point boundary detection is not solved** — none of the models
   detect point boundaries in continuous video, and Model 2 was trained on
   pre-segmented clips, so stroke-boundary segmentation in raw match footage
   is itself an unbuilt prerequisite component, not just a labeling gap.
3. **Ball contact/bounce-frame fields exist in the schema but can't currently
   be populated** — the raw `Ball Tracker+Bounce` labels do contain a
   per-frame `status` column (0=flight/1=hit/2=bounce, confirmed by
   re-inspecting `Label.csv`) that `ball_tracking/dataset.py` currently
   discards, and TrackNet's output head only regresses position, not event
   type. Real information exists in the source data that the trained model
   doesn't currently surface — flagged as concrete future work.
4. **Homography computation is not implemented** — `CourtInfo.homography_matrix`
   is schema-ready but nothing computes it from the 14 keypoints yet.
5. **Model 4 doesn't fit `StrokeEvent`'s per-stroke granularity** (it's
   session-level aggregate stats) — recommended a separate `SessionRecord`
   entity rather than bolting an anomaly flag onto every stroke in a session,
   but this is a recommendation for sign-off, not a decision made here.
6. **Coordinate-system strictness** — currently each model's native pixel
   space is kept as-is (not normalized to a shared `[0,1]` space), with
   `court_position` (real-world court-space via homography) as a separate
   derived field rather than a replacement. Flagged as an open tradeoff
   between raw precision and downstream geometry convenience.

**Not yet built** (explicitly, so scope stays clear going into the next
session): the serialization code that reads a real TrackNet /
`CourtKeypointNet` / `StrokeGRU` forward pass and constructs a validated
`StrokeEvent`; stroke-boundary segmentation for continuous video; homography
computation; player-identity/tracking; the `SessionRecord` entity for Model 4.

---

## 2026-07-17: IR layer — six decisions resolved, population code written, real examples verified

All six open items from the v1 design were resolved by the user and
implemented. `ir/DESIGN.md` and `ir/schema.py` (now schema_version `0.2.0`)
were rewritten to match; `ir/geometry.py` (homography) and `ir/populate.py`
(real-model-output population code) are new.

**Decisions and what was actually built:**

1. **`player_id` externally supplied.** No schema change needed (was already
   optional) — clarified in docstrings that populating code must pass it
   through from an external source, never guess it. Multi-player tracking
   stays explicit future work.
2. **IR population v1 scoped to pre-segmented clips**, matching Model 2's own
   training assumption. `ir/populate.py` takes one pre-segmented clip as
   input; continuous-video stroke segmentation remains unbuilt future work.
3. **Ball `event_type` (hit/bounce) surfaced from the raw label.**
   `training/ball_tracking/dataset.py`'s `ClipRecord` now parses and exposes
   the previously-discarded `status` column (see that file's updated
   docstring) — the training `Dataset` class (`BallTrackingDataset`) was left
   untouched to avoid touching the already-validated Model 1A training
   pipeline; only the lower-level `ClipRecord`/`discover_clips()` layer
   changed. **Reliability check** (30 clips / 4959 frames): every `NaN`
   status matched `visibility==0` exactly (120/120) and event rates were
   consistent across clips (flight 92.0% / hit 2.8% / bounce 2.7%) — clean
   ground truth for labeled clips. **But this only works because these clips
   are hand-labeled** — TrackNet itself has no output head for event type, so
   `event_type` can't be produced on new/unlabeled footage. Recommended a
   lightweight trained hit/bounce classifier (e.g. over trajectory
   velocity/acceleration) as a **fast-follow**, not built now.
4. **Homography implemented** (`ir/geometry.py`, `cv2.findHomography`,
   RANSAC over 14 points). **The 14 court keypoints' semantic order isn't
   documented anywhere in the source export** (no keypoint-name file ships
   with `Court_Detector_Keypoints`) — inferred it from raw pixel geometry
   (paired baseline/service-line rows; doubles-vs-singles corner x-spacing
   ratio matched the real ITF ratio 10.97m/8.23m=1.333 to within 0.3% across
   4 independently-checked sample frames). Verified by reprojecting the 14
   known real-world points back through the fitted homography on ground-truth
   keypoints: **2.2cm mean error**. On real *predicted* keypoints (via
   `populate.py`, not ground truth), reprojection error on 3 real sample
   frames came in at 3.6-10.5cm — consistent with Model 1B's ~6px val_dist.
   **Flagged a permanent limitation, not a bug:** homography assumes points
   on a single plane (the court surface, z=0) — exact for court keypoints,
   but only an approximation for the ball whenever it's airborne (most
   frames). `BallFramePosition.court` should be read as reliable mainly
   at/near `contact_frame`/`bounce_frame`, not uniformly across a trajectory.
5. **`SessionRecord` added as a separate top-level entity** for Model 4 —
   `features` (validated against `anomaly_detection/train.py:FEATURE_COLUMNS`
   exactly), `anomaly_flag`, `anomaly_score`, `explanation`. Not merged into
   `StrokeEvent`.
6. **Position fields normalized to court-relative meters** (`CourtPoint`,
   via decision #4's homography) as the canonical representation; raw pixel
   coordinates (`PixelPoint`) kept alongside for traceability. One deliberate
   exception: `CourtInfo.keypoints_px` stays in pixel space, since those
   points are the calibration *input* to the homography — expressing them in
   the homography's own output space would be circular.

**Small schema addition made during implementation (not one of the six, but
needed for #6's compositing to be traceable):** added
`ModelProvenance.source_file`, recording exactly which real input file backed
each field group.

**Population code and real examples (`ir/populate.py`, `ir/example_output/`):**
constructed 3 real `StrokeEvent`s and 3 real `SessionRecord`s from genuine
trained-model inference (TrackNet, `CourtKeypointNet`, `StrokeGRU`,
`IsolationForest`) on real sample data — not fabricated numbers. All 3 stroke
predictions matched their true labels (backhand/serve/forehand), with the
serve example's confusion mass landing on smash (0.163) rather than any other
class — consistent with the serve↔smash confusion already documented in
`stroke_classification/README.md`.

**Important honesty check, not glossed over:** no dataset in this project
spans ball-tracking + court-keypoints + stroke-classification for the *same*
real rally. `Ball Tracker+Bounce`, `Court_Detector_Keypoints`, and
`StrokeClassification` (THETIS) are three independent, unrelated video
sources. The 3 example `StrokeEvent`s are therefore **composites**: real
model inference on three independently-real but unrelated source clips,
stitched into one illustrative record per `ModelProvenance.source_file`
(fully traceable, not hidden) to demonstrate the schema validates against
genuine model output. This is explicitly *not* a demonstration of one
coherent end-to-end pipeline run on a single video — that needs continuous
match footage with all three signals attached to the same rally, which this
project doesn't have (the same gap as decision #2's "not yet built").
`SessionRecord` examples necessarily reuse the same synthetic demo data as
`anomaly_detection/train.py --demo`, since no real session dataset exists —
clearly labeled as such in `source_file`.

---

## 2026-07-17: IR layer follow-ups — keypoint order visually confirmed, true end-to-end example built

Two follow-up checks on the IR layer work above: a direct visual spot-check
of the inferred court-keypoint ordering (previously only checked indirectly,
via proportion-matching and reprojection error), and one genuinely
single-video `StrokeEvent` (previously all three examples were composites of
unrelated clips).

**1. Keypoint semantic ordering — direct visual spot-check.** Ran real
`CourtKeypointNet` inference on the strongest-prediction val frame found in a
60-entry scan (`S0Gn0tZQ-jg_900`, mean error 1.59px vs. ground truth), drew
all 14 predicted points on the source image labeled with their inferred
names, and manually checked each against the frame. **Result: confirms the
inferred ordering — every point (baseline-doubles/singles corners,
service-line-singles corners, service-line center "T" marks) lands exactly
where its label says, none swapped or mislabeled.** See
`ir/example_output/keypoint_spotcheck.png`; full writeup in
`ir/DESIGN.md` under decision #4. This was the one piece of the homography
work that had only been checked indirectly before; it's now confirmed by
three independent methods (proportion-matching, reprojection error, direct
visual inspection).

**2. True end-to-end `StrokeEvent`** (`ir/populate_e2e.py` →
`ir/example_output/stroke_event_end_to_end.json`), as distinct from the
three composite examples above: one real clip — Ball Tracker+Bounce
`game1/Clip5`, frames 29–88 (60 frames, Djokovic vs. Federer, Shanghai
2014; `game1` is in TrackNet's held-out validation split) — run through all
three models with no compositing.

- **Ball**: real TrackNet inference over the window.
- **Court**: real `CourtKeypointNet` inference on frame 59 of *this same
  clip* — genuine cross-domain generalization test, since that model was
  trained on the separate `Court_Detector_Keypoints` dataset, never on
  Ball-Tracker footage. Reprojection error **7.37cm**, inside the
  3.6-10.5cm range already seen on the composite examples — a fourth
  consistent data point, not an outlier.
- **Stroke**: this required extracting real MediaPipe Pose features
  ourselves, since StrokeGRU needs 99-dim pose input and THETIS (its
  training data) ships only pre-extracted features with no raw video in this
  repo. Two problems solved, both clip-specific rather than general
  solutions: (a) `mediapipe==0.10.35` removed the legacy `solutions.pose` API
  THETIS's extractor likely used, so extraction here uses the newer
  `PoseLandmarker` Tasks API instead — same BlazePose 33-point model family,
  not guaranteed identical output to the original; (b) `PoseLandmarker` on
  the raw broadcast frame detected 0/60 frames (player too small in a wide
  shot vs. THETIS's tight single-player crops) — fixed with a clip-specific
  HSV red-shirt color-blob tracker cropping around the player per frame
  (60/60 detected, vs. 47/60 with a naive fixed crop that lost him mid-lunge
  near the net). Visually verified the extracted skeleton tracks the player
  correctly on frame 59 (`ir/example_output/pose_spotcheck_frame59.png`).
  **Result: `predicted_class=backhand`, confidence=0.999.** No ground-truth
  stroke label exists for this clip (it's not a StrokeClassification
  sample), so this can't be scored right/wrong — reported as the model's
  real output on real but domain-shifted input (different pose-extractor API
  + broadcast framing vs. THETIS's close-up amateur courts), not as evidence
  of Model 2's real-world accuracy (that claim still rests on the THETIS
  test-set F1=0.834).

**Error compounding, quantified for the FYP error-analysis section**
(`ir/DESIGN.md` has the full table): Model 1B keypoint regression
(~6.1px val_dist) → homography reprojection error (2.2cm ground-truth
keypoints; 3.6-10.5cm real predicted keypoints across 4 samples now,
composites + this one) → ball court-position (exact only at contact/bounce
frames, per decision #4's z=0 caveat) is a distinct, independently-measured
error at each stage — nothing currently composes them into one end-to-end
bound, since that would need ground truth spanning the full pipeline on one
dataset, which is the same gap this whole follow-up was checking.

Full detail (including the "composite vs. true end-to-end" framing) is now
in `ir/DESIGN.md`.

**Closeout note on the two follow-ups above:** `ir/DESIGN.md`'s true-e2e
section now states explicitly what that example does and doesn't validate —
pipeline plumbing on real single-source footage, yes; stroke-classification
accuracy on new footage, no, given the two stacked domain-shift factors
(different pose-extraction pipeline than THETIS's training data, plus a
workaround color-based tracker needed just to get any pose detections at all
on this footage's framing/scale). That tracker (`ir/populate_e2e.py`,
`_demo_only_find_player_box_by_red_shirt`) is now explicitly labeled in code
— name, docstring, and a banner comment — as one-off demo scaffolding for
this specific clip, not a player-tracking component, so it doesn't get
mistaken for real infrastructure later.

**FYP writeup pointer:** the error-compounding table in `ir/DESIGN.md`
("Error compounding through the IR pipeline") — keypoint regression error →
homography reprojection error → ball court-position reliability — is written
to be pulled directly into the report's error-analysis section as-is; it's
the clearest single artifact this project has for "here is how uncertainty
propagates through the CV→IR pipeline," which is exactly the kind of honest
limitations discussion an FYP report should have.

**This is a good stopping point for the IR layer as a whole.** All six
original design decisions are implemented and verified; the keypoint-order
inference underlying the homography is now confirmed three independent ways;
both categories of example (schema-validation composites and one true
single-video end-to-end run) exist and are clearly distinguished from each
other in the docs; and every honesty caveat (compositing, z=0 ball
approximation, domain shift on the e2e stroke prediction, synthetic
`SessionRecord` inputs) is written down in the code, in `DESIGN.md`, and
here — not just carried in this conversation.

---

## 2026-07-17: RAG coaching layer (Objective 3) — design pass, two open questions logged, grounding harness built

Started objective 3 (RAG coaching feedback) with the same discipline as the
IR layer: inspect and design first, no pipeline code, and no fabricated
placeholder content, before the user signs off. Two open questions were
found and logged rather than resolved silently, matching the pattern already
used for Model 3's momentum-label problem. Full detail in `rag/DESIGN.md`.

**Step 1 — knowledge base inventory: confirmed empty, not assumed.**
Searched the whole repo for any coaching-reference material (ITF Coaching
Manual, USTA guidelines, anything else) — found nothing; the only `.pdf`
hits were unrelated `matplotlib` package files. `visiotennis_training_prompt.md`
(the closest thing to a local written spec) covers Models 1-4 only, nothing
about RAG content. **Logged as OPEN**: proposed three non-exclusive options
(licensed access; publicly-available openly-licensed content, e.g.
Wikipedia/Wikibooks CC-BY-SA or open-access biomechanics papers; a small
hand-authored placeholder set explicitly marked "authored, not sourced").
Recommended starting with the hand-authored placeholder to unblock pipeline
prototyping while pursuing licensed/open content in parallel as a
content-quality upgrade — but **nothing has been written yet**, pending the
user's decision on whether licensed ITF/USTA access is realistically
available.

**Step 2 — deviation-detection benchmarks: logged as OPEN, and it surfaced a
real schema gap.** `StrokeEvent` has no per-stroke biomechanical fields at
all (no contact height, racket angle, footwork metrics) — the exact kind of
measurement the proposal's own example ("normal contact point height for a
forehand") needs, and it doesn't exist in `ir/schema.py` today. Proposed
three candidate benchmark approaches (skill-tier-relative from coaching
literature; pro-player reference ranges from open-access biomechanics
papers; population percentiles computed from our own dataset) — all three
require either new external sourcing (options 1/2, same sourcing question as
Step 1) or new derived-feature extraction work from the pose landmarks
already proven working in `ir/populate_e2e.py` (option 3), so none are
buildable today without further scoping. **Found one thing that *is*
buildable now, no new work needed**: `SessionRecord.explanation` already
carries a real, grounded deviation (Model 4's `explain_anomaly()` z-scores
against population mean/std — see `ir/example_output/session_record_0.json`)
— session-level coaching feedback has a working "measured deviation" source
today; stroke-level biomechanical feedback does not yet. Recommended
prioritizing session-level feedback first for this reason, pending the
user's sign-off on sequencing and, eventually, which benchmark approach to
pursue for stroke-level work.

**Step 3 — pipeline architecture (`rag/DESIGN.md`).** Retrieval designed as
an abstraction (small local embedding or keyword index over short tagged
snippets) independent of Step 1's outcome, so the KB decision doesn't force
a re-architecture later. Prompt template design: rather than asking the LLM
to "cite sources" in free text (unverifiable except by fuzzy NLP), the LLM
must return structured JSON matching a new `rag/schema.py:CoachingResponse`
— every substantive claim is a `grounded_observation` item with an explicit
`Citation{record_type, record_id, ir_field_path, cited_value}`, enforced at
parse time by a Pydantic validator (same "100% compliance by construction"
principle as the IR schema); only non-factual filler may skip citation as a
`general_note`. LLM choice: prototype with the Claude API now (no
infrastructure to stand up, fast iteration); local quantized LLaMA 3.1 8B on
the 3060 noted explicitly as a future cost-reduction path, not started now
— no `anthropic` SDK installed, no API key configured, no LLM call made.

**Step 4 — grounding validation harness: built and demonstrated, not just
designed.** `rag/grounding.py`'s `check_grounding()` computes `grounding_rate
= (# grounded_observation items that pass) / (# grounded_observation items
total)`, where "pass" means three mechanical checks in order: the cited
`(record_type, record_id)` is among the real records actually provided (not
trusted from the LLM's own claim) → the cited `ir_field_path` resolves
against that record's real serialized data → the resolved value matches what
was cited (numeric tolerance / exact for strings). `general_note` items are
excluded from the rate but scanned for a stray digit as a soft warning (a
factual claim dodging citation by mislabeling itself). **Demonstrated**
(`rag/example_grounding_check.py`) against one real `SessionRecord` and a
hand-written, deliberately adversarial `CoachingResponse` (not
LLM-generated) exercising all 6 verdict paths — correct citation, wrong
cited value, nonexistent field, unprovided record, clean general note,
digit-smuggling general note. Result: `grounding_rate = 1/4 (25.0%)`,
exactly as designed — a deliberately bad score proving the harness *catches*
fabrication/mismatch/misattribution/mislabeling, which is what will make a
future ">95%" claim on real generated feedback mean something rather than
being asserted on the honor system.

**Not yet built, explicitly:** the knowledge base itself, any stroke-level
biomechanical benchmarks, the retrieval index (no content to index yet), any
LLM integration, and any actual generated coaching feedback (real or
synthetic). All of this is blocked on the user's sign-off on Steps 1-2.

---

## 2026-07-17: RAG coaching layer — Qwen3 8B set up locally via Ollama, evaluated against the grounding harness, adopted as the LLM (deviation from proposal's LLaMA 3.1 8B)

Before deciding Claude API vs. local for the RAG pipeline's LLM, set up and
evaluated a local model on the actual grounding harness rather than assuming
either option would work well. Originally asked to compare against Claude
API too, but no `ANTHROPIC_API_KEY` is configured in this environment — user
chose to proceed local-only for now rather than block on acquiring a key.
`rag/llm.py:ClaudeBackend` was still written to the same interface as the
Ollama backend, untested, so that comparison can happen later with zero
pipeline changes.

**Setup.** Installed Ollama via `winget` (not previously present), pulled
`qwen3:8b` (Ollama's default tag, confirmed via `ollama show` to be **Q4_K_M
quantization**, 8.2B params). Loads and generates within the RTX 3060's
12GB: ~1.1GB baseline VRAM, ~6.5-6.6GB with the model loaded and generating
— comfortable headroom.

**Reliability check — 6 trials, not 1.** Built `rag/kb_placeholder.py`
(small hand-authored, "authored not sourced" KB scoped to session-level
topics, per the Step 1/2 findings from the previous design pass) and
`rag/prompt.py` (shared prompt builder: keyword-retrieves matching snippets
from a `SessionRecord`'s real `explanation` field, embeds the full real
`SessionRecord` JSON, and forces structured JSON output matching
`rag/schema.py:CoachingResponse` — no free-text "please cite sources").
Ran this identical prompt against the same real `SessionRecord`
(`ir/example_output/session_record_0.json`) 6 times through Qwen3 8B and the
real `rag/grounding.py` checker:

| Trial | Result | Latency |
|---|---|---|
| 1 | 3/3 grounded (100%) | 64.2s |
| 2 | **0/6 grounded (0%)** — fabricated a fake numeric `record_id` and 2 feature names that don't exist in the schema (`net_approach_pct`, `backhand_slice_pct`) | 72.8s |
| 3 | Pydantic `ValidationError` — trailing summary item omitted the required `"type"` field (schema malformation, not a grounding failure — first 3 items were correctly grounded) | 29.9s |
| 4 | 3/3 grounded (100%) | 24.6s |
| 5 | 3/3 grounded (100%) | 30.5s |
| 6 | 3/3 grounded (100%) | 22.4s |

**Why 6 trials, not 1:** the first two ad-hoc runs already showed a single
run would mislead in either direction — a perfect 100% immediately followed
by a complete hallucination on an *identical* prompt at the same
temperature. **4/6 trials fully clean, 1/6 completely hallucinated, 1/6
schema-malformed on a non-critical trailing item.** Output quality on the
clean trials was genuinely good — coherent, on-topic, correctly paraphrased
the placeholder KB snippets for tone without copying them verbatim, and
correctly grounded a feature (`first_serve_points_won_pct`) that had *no*
matching KB snippet at all, sourcing it directly from the raw `SessionRecord`
JSON as instructed. Latency was bimodal (64-73s on the two ad-hoc runs vs.
22-31s, mean 26.8s, on the 4-trial batch) — plausibly variable-length hidden
"thinking" tokens (Qwen3 is a reasoning model), not confirmed with only 6
samples.

**Decision: adopt Qwen3 8B (Q4_K_M, local via Ollama) as the pipeline's
default LLM, replacing the proposal's originally-scoped LLaMA 3.1 8B.**
Same underlying goal (free, local, on-device coaching generation) — this is
a better-current-fit substitution, not a scope change, found by checking
what's actually available now rather than assuming the original pick was
still best. **Required follow-up, not yet implemented:** the observed 1/6
hallucination rate means `rag/grounding.py`'s checker must run as a runtime
retry-gate (reject/regenerate any response that fails to parse or scores
below 100% grounding) before any response reaches a user, not only as an
offline eval tool — this is cheap to add (the harness and schema validator
already exist) and turns raw 4/6 reliability into much higher delivered
reliability at the cost of occasional extra latency, acceptable given this
isn't a real-time interaction.

Full detail, including the schema-gap discussion this connects to, is in
`rag/DESIGN.md` ("LLM choice — RESOLVED"). Raw trial outputs and per-trial
grounding reports saved under `rag/example_output/qwen3_trials/`.

---

## 2026-07-18: RAG coaching layer — runtime retry-gate built, before/after reliability measured

Implemented the retry-gate flagged as a required follow-up in the previous
entry: `rag/gated_pipeline.py:generate_grounded_response()` wraps a single
LLM call in a check-and-retry loop — generate, parse against
`CoachingResponse`, run `check_grounding()`, accept only at exactly 100%
grounding on every `grounded_observation` item, else regenerate from scratch
(same prompt, same temperature — retries rely on Ollama's run-to-run
sampling variance, not a corrected prompt) up to a max of 3 attempts. If no
attempt qualifies within budget, returns an explicit failure state
(`success=False`, no response) — a hard requirement, so nothing
ungrounded or malformed can reach a user even if all retries fail.

**Re-ran the identical 6-trial reliability test through the gate**
(`rag/run_gated_multitrial.py 6 3`, same prompt/KB context/SessionRecord as
the ungated run above):

| Trial | Attempts | Outcome | Total latency |
|---|---|---|---|
| 1 | 3 | Accepted on attempt 3 — attempts 1-2 both failed Pydantic validation (missing `ir_field_path`/`cited_value`/`type` fields) | 192.6s |
| 2 | 1 | Accepted, 3/3 grounded | 41.0s |
| 3 | 1 | Accepted, 3/3 grounded | 31.3s |
| 4 | 1 | Accepted, 3/3 grounded | 22.1s |
| 5 | 1 | Accepted, 3/3 grounded | 23.6s |
| 6 | 1 | Accepted, 3/3 grounded | 33.7s |

**Before/after:**
- **Ungated: 4/6 (66.7%) trials clean** — the 1/6 hallucination and 1/6
  schema-malformed trial would both have reached the user unmodified.
- **Gated: 6/6 (100%) trials delivered a fully grounded, well-formed
  response** — zero ungrounded or malformed output crossed the user-facing
  boundary.
- 1/6 trials needed a retry (2 extra generations, both parse failures this
  time — a different failure mode from the earlier hallucination case,
  confirming the gate has to catch both, not just one).
- Cost: mean attempts per trial 1.33, mean latency per trial including
  retries 57.4s (vs. ~26.8s single-shot on the earlier clean batch); one
  retried trial (192.6s) accounts for most of that, the other 5 trials cost
  no more than an ungated call would have.

**This is the concrete before/after result for the FYP report's evaluation
section**: raw model reliability 4/6 (66.7%) → gated delivered reliability
6/6 (100%), at a measured, bounded latency cost rather than an assumed one.
Only 6 trials — not enough to claim a stable retry-rate percentage, but
enough to demonstrate the gate does what it was built for.

Full design writeup, including why the gate uses a hard 100% threshold
rather than a softer one, is in `rag/DESIGN.md` ("Retry-gate design and
evaluation"). Per-trial results saved under
`rag/example_output/qwen3_gated_trials/`.

---

## 2026-07-18: RAG coaching layer — KB expanded, real semantic retrieval added, Step 2 resolved, validated across 6 diverse anomaly types

Closed out the two open design questions from the original RAG design pass
(Steps 1 and 2) with concrete decisions and evidence, and fixed a gap in
the reliability evaluation so far: every prior grounding/retry-gate result
had been measured on **the same repeated SessionRecord**, which cannot show
whether retrieval generalizes since the same KB snippets were retrieved
every time.

**Step 1 — KB expanded from 6 to 24 entries.** `rag/kb_placeholder.py` now
covers all 12 `ANOMALY_FEATURE_COLUMNS` (2 differently-framed entries per
feature), each still explicitly labeled "authored, not sourced." This is
deliberately more than one obviously-matching snippet per test case, so
retrieval has to actually rank, not do a 1:1 lookup.

**Step 1 — retrieval is now real embedding-based semantic search, not
keyword matching.** Pulled `all-minilm` (45MB) locally via Ollama (same
serving infra as the Qwen3 8B coaching LLM — no new API dependency).
`rag/embeddings.py:OllamaEmbedder` + cosine similarity replace the previous
exact keyword/topic-string lookup in `rag/prompt.py:retrieve_snippets()`:
each SessionRecord deviation phrase is embedded and matched against
embeddings of all 24 KB entries, picking the closest match per phrase.

**Step 2 — RESOLVED, not left open.** Session-level coaching feedback's
grounding mechanism is formally adopted: Model 4's `explain_anomaly()`
z-scores (session vs. that player's own history — a statistical deviation,
not a comparison to an external "ideal" value), so no external benchmark
sourcing is needed and none is being pursued. Stroke-level biomechanical
benchmarking (contact height, racket angle, etc.) is formally logged as
**deferred**, not merely open — blocked on the schema gap (no
biomechanical fields on `StrokeEvent`, no derivation code from the already-
proven MediaPipe Pose landmarks) and a benchmark source, neither being
pursued in this phase. Full reasoning in `rag/DESIGN.md` Step 2 "Decision —
RESOLVED."

**Step 3 — validation across 6 genuinely different anomaly types, not the
same session repeated.** The first 3 real `SessionRecord`s
(`session_record_0/1/2.json`) all happen to share the same dominant
deviated features, because the synthetic anomalous population shifts all
12 features together. `ir/populate_diverse_sessions.py` crafts 6 new
sessions, each targeting a distinct theme (serve breakdown, second-serve/
ace decline, movement/fatigue, rally-style shift, clutch/break-point
breakdown, general unforced-error spike) — run through the same real
fitted `IsolationForest` + `explain_anomaly()`, confirmed genuinely flagged
anomalous (`anomaly_flag=-1`) before use. (Honest finding worth keeping: a
pure single-theme perturbation, however extreme, often wasn't enough to
trip this particular fitted forest — 4/6 profiles needed a mild blend
toward the real anomalous population's mean layered under the targeted
override to get flagged, without losing the theme features' dominance in
the top-3 explanation.)

**Retrieval recall across the 6 diverse sessions: 15/18 (83.3%)** —
reported as measured, not rounded up:

| Session | Recall |
|---|---|
| clutch_breakdown | 3/3 (100%) |
| movement_fatigue | 3/3 (100%) |
| serve_breakdown | 3/3 (100%) |
| unforced_error_spike | 3/3 (100%) |
| rally_style_shift | 2/3 (67%) — lost a slot to a dedup collision, not a ranking error |
| second_serve_ace_decline | 1/3 (33%) — a genuine semantic miss (`ace_rate` query matched a `first_serve_points_won_pct` snippet instead) |

**Gated pipeline result on these 6 diverse sessions, each using its own
freshly-retrieved KB context: 6/6 (100%) gated success, 0/6 needed a
retry**, mean 1.00 attempts/trial, 364.3s total latency across all 6.

**Key finding for the FYP writeup: grounding rate and retrieval recall are
two independent reliability axes.** Even on the two sessions where
retrieval missed a KB entry, the model still produced a fully grounded
response on the first attempt — a retrieval miss degrades the coaching
advice's relevance/framing, not the grounding the harness checks (grounding
is about correctly citing real `SESSION_RECORD` fields, independent of
which KB snippets were used for framing). On this data: grounding 6/6
(100%), retrieval recall 15/18 (83.3%) — good evidence these should be
reported and improved separately, not conflated into one number.

Full writeup in `rag/DESIGN.md` ("Retrieval accuracy across diverse anomaly
types"). Per-session results, retrieval decisions, and grounding reports
saved under `rag/example_output/qwen3_gated_diverse_trials/`.

---

## 2026-07-18: RAG coaching layer — two retrieval bugs found and fixed, semantic-miss output inspected, phase closed out

Two closeout follow-ups on the diverse-session retrieval result above
(15/18, 83.3% recall), then a stopping point for this phase.

**Inspected the actual generated output for the genuine semantic-miss
session** (`second_serve_ace_decline` — the one true ranking error, not the
dedup issue). Result (100% grounded): three bare, correctly-cited factual
restatements with essentially no coaching interpretation, e.g. *"The
player's second-serve points won percentage is 9.07..., which is 6.0
standard deviations below average."* The model did not misapply the
wrong-topic KB snippet's advice to the right numbers — it simply fell back
to thin, generic output instead of inventing something wrong. **Named
limitation now documented in `rag/DESIGN.md`: the grounding harness
validates citation accuracy, not retrieval relevance — a response can be
100% grounded while being retrieval-degraded, and the harness has no way to
detect or penalize that. Grounding rate and retrieval recall are two
independent axes and must be reported separately, not conflated.**

**Investigated the dedup collision — found two real bugs, both quick and
isolated, both fixed:**
1. `retrieve_snippets()` picked each phrase's single best match and
   dropped the phrase entirely if that index was already claimed, instead
   of falling through to its next-best unclaimed candidate. Fixed to walk
   the full ranked candidate list per phrase.
2. Fixing (1) alone didn't fix `rally_style_shift`, because two KB entries'
   authored text happened to literally contain another topic's exact
   snake_case field name as a cross-reference (e.g. "...compare against
   winners_rate in the same session..."), which a small 45MB embedding
   model weighted heavily enough to outrank that topic's own snippets.
   Audited all 24 entries mechanically for the same pattern, found exactly
   2, reworded both in natural language instead of the raw identifier.

**Recall after both fixes: 17/18 (94.4%)**, re-verified directly against
all 6 diverse sessions (retrieval-only check, no LLM calls needed since
retrieval and generation are independent). One miss remains and is left
alone deliberately: `second_serve_points_won_pct` vs
`first_serve_points_won_pct` confusion, a genuine embedding-model
limitation (two short passages differing mainly in "first"/"second") that
would need a better embedding model, a rerank step, or KB restructuring to
fix properly — real design work, not a quick edit, so it's documented in
`rag/DESIGN.md` as a known, understood limitation rather than pursued now.

**One more honest data point:** re-confirming the `rally_style_shift` fix
with a targeted single-session re-run hit the retry-gate's failure path for
the first time in this project — all 3 attempts failed to parse (the same
recurring "trailing item missing `type`" pattern seen in the earlier Qwen3
reliability evaluation), so the pipeline correctly returned
`GatedResult(success=False, ...)` rather than any output. Not a regression
from the retrieval fixes (verified independently, no LLM call involved) —
it's the gate's designed failure state actually firing on real generation
variance, confirming that path is reachable in practice, not just a
defensive branch.

### RAG / session-coaching layer — scoped completion point

**Built and verified:**
- Structural citation-grounding schema (`rag/schema.py`) and a mechanical,
  non-NLP grounding checker (`rag/grounding.py`), demonstrated against a
  hand-written adversarial example before any LLM was involved.
- A 24-entry placeholder KB (`rag/kb_placeholder.py`) covering all 12
  session-level anomaly features, and real embedding-based semantic
  retrieval (`rag/embeddings.py`, local `all-minilm` via Ollama) — recall
  94.4% (17/18) across 6 genuinely diverse anomaly types after fixing two
  retrieval bugs found during evaluation.
- Qwen3 8B (Q4_K_M, local via Ollama) adopted as the pipeline's LLM,
  replacing the proposal's LLaMA 3.1 8B, with a documented 6-trial
  reliability profile (4/6 clean, 1/6 hallucinated, 1/6 schema-malformed).
- A runtime retry-gate (`rag/gated_pipeline.py`) that only ever surfaces a
  100%-grounded, well-formed response or an explicit failure state —
  measured at 6/6 (100%) delivered reliability on the repeated-session test
  (vs. 4/6 ungated) and 6/6 on the diverse-session test, with the failure
  path itself confirmed to fire in practice, not just in theory.
- Step 2 (deviation-benchmark question) formally RESOLVED for session-level
  coaching: `explain_anomaly()`'s z-scores are the adopted grounding
  mechanism, self-referential (player vs. own history), no external
  benchmark sourcing required.

**Deferred, not pursued in this phase, and not silently dropped:**
- **Claude API comparison** — no `ANTHROPIC_API_KEY` configured;
  `rag/llm.py:ClaudeBackend` is interface-complete and untested, ready to
  run the identical prompt through the identical grounding harness the
  moment a key exists.
- **KB content-authority upgrade** — the 24-entry KB remains explicitly
  "authored, not sourced." Licensed ITF/USTA access or vetted
  openly-licensed content (Step 1 options 1/2) would raise coaching
  authority but require real acquisition/licensing work outside this
  codebase's control.
- **Stroke-level biomechanical coaching** — formally deferred (not merely
  open), blocked on two unaddressed things: no biomechanical fields on
  `StrokeEvent`, and no derivation code from the already-proven MediaPipe
  Pose landmarks, plus an eventual benchmark-source decision among the
  three candidate approaches already recorded in `rag/DESIGN.md`.

**Key architectural finding for the report: grounding and retrieval
relevance are decoupled.** Across every trial run in this phase, grounding
rate and retrieval recall moved independently — a response can be
mechanically 100% grounded (every citation traces to a real field, exactly
as claimed) while being retrieval-degraded (generic, low-value coaching
content because the wrong reference snippet was surfaced). The grounding
harness catches fabrication and misattribution; it was never designed to,
and does not, catch retrieval quality — that has to be measured and
reported as a separate number, which is why this phase built and reported
both (94.4% retrieval recall, 100% grounding, on the same 6-session test).

**This is a good stopping point for the RAG coaching layer, same discipline
as the IR layer's closeout:** every component that's buildable without new
external sourcing or new schema work is built, evaluated with real
(non-cherry-picked) data, and its remaining limitations are named in
`rag/DESIGN.md` and here rather than left implicit.

---

## 2026-07-19: Stroke-level biomechanical coaching — feature-extraction inventory and schema proposal (design phase, pending sign-off)

Picking up the item `rag/DESIGN.md` formally deferred: stroke-level
biomechanical coaching, blocked on "no biomechanical fields on
`StrokeEvent`, no derivation code." This entry logs the design work only —
same discipline as the IR layer's original six decisions: inventory real
data, propose a schema, get sign-off, *then* write extraction code. No
extraction code has been written yet.

**Confirmed no racket detection exists anywhere in this project** (repo-wide
search — the only "racket" hits are prose mentions, including an explicit
disclaimer in `stroke_classification/README.md`: "no racket/ball trajectory
signal in this feature set"). Every proposed feature is a body-pose
measurement; anything approximating a racket concept is named for what it
actually measures (`forearm_orientation_deg`, not "racket angle").

**Two available pose data sources, inventoried in full in `ir/DESIGN.md`:**
`StrokeClassification` (THETIS, `(60, 99)` MediaPipe-landmark sequences,
1980 clips across forehand/backhand/serve/volley/smash, no contact-frame
label, no handedness label) and `Pose_Estimation` (500-per-class single
static frames, COCO-18-keypoint scheme — different convention than
MediaPipe's 33, no volley/smash classes, no temporal information).
Recommended `StrokeClassification` as the primary extraction source since
it's the only one with a time window (needed for weight-transfer/
follow-through features). One real exception exists where contact frame is
a measured fact rather than a proxy: `ir/populate_e2e.py`'s single
end-to-end demo clip, which has both real pose data and a real
TrackNet-labeled hit frame — proposed as a one-instance validation check
for the peak-wrist-speed contact-frame proxy needed everywhere else.

**Proposed feature list (10 fields):** `wrist_height_ratio`,
`shoulder_rotation_deg`, `hip_rotation_deg`, `shoulder_hip_separation_deg`
("X-factor" trunk rotation differential), `knee_flex_left/right_deg`,
`stance_width_ratio`, `weight_transfer_ratio`,
`follow_through_extension_ratio`, `forearm_orientation_deg`,
`elbow_angle_deg` — all body-pose geometry, all expressed as dimensionless
ratios/degrees (no camera calibration exists for this pose data, so no
path to real-world centimeters the way `CourtPoint` has).

**Benchmark approach proposed: population percentiles within our own
extracted feature data, grouped by stroke class** — same non-external-
sourcing precedent as the session-level Step 2 decision. Flagged
explicitly: `smash` at n=165 is thin for tail percentiles; both source
datasets are general/mixed-skill footage, so a percentile answers "unusual
for our sample," not "technically correct," same distinction already
established for session-level coaching and must not be conflated.

**Two open methodological questions flagged, not decided silently:**
dominant-wrist determination (max-displacement heuristic, with two-handed
backhand noted as a real edge case where "one dominant wrist" is a
simplification), and contact-frame-proxy validation (peak-wrist-speed,
to be checked against the one real ground-truth instance before trusting
it on the other 1980 clips).

**Proposed schema addition:** a new `BiomechanicsInfo` model in
`ir/schema.py`, attached as `StrokeEvent.biomechanics: BiomechanicsInfo |
None = None` (same optional-until-built pattern as `MomentumInfo`).
Deliberately excludes percentile/benchmark fields from the schema itself —
proposed as a separate downstream utility once real extracted values
exist across the dataset, keeping today's schema change scoped to
per-stroke raw geometry only.

Full inventory, feature table, and schema draft in `ir/DESIGN.md`
("Stroke-level biomechanical features — proposal, PENDING SIGN-OFF").
**Awaiting sign-off before Step 4 (extraction code + real computed
values).**

---

## 2026-07-19: Stroke-level biomechanics — sign-off received, schema added, extraction built and run on real data, two bugs found and fixed

Sign-off received on both open questions from the proposal above:
dominant-wrist heuristic applied uniformly (including to two-handed
backhand, flagged as an approximation there), and the schema approved
exactly as drafted. `BiomechanicsInfo` added to `ir/schema.py`
(`StrokeEvent.biomechanics`, optional, same pattern as `MomentumInfo`).

**Built `ir/biomechanics.py`** (the 10 designed features, pure 2D
frame-plane geometry) **and `ir/populate_biomechanics.py`**, run on one
real `StrokeClassification` (THETIS) example per collapsed class
(first match in `test`, not cherry-picked) plus a contact-frame-proxy
validation against the one clip with real ground truth
(`ir/populate_e2e.py`'s game1/Clip5).

**Sanity-checking the first pass of numbers (not assumed correct because
the code ran) surfaced three real issues, all fixed:**

1. **Zero-padded frames poisoning the contact-frame proxy.** Some THETIS
   clips have trailing all-zero frames (undetected pose during original
   preprocessing — the same fallback already documented in
   `ir/populate_e2e.py`). A naive peak-wrist-speed search mistook a jump
   from a real position to the origin for a huge fake speed spike, on 2 of
   5 examples (visible as `weight_transfer_ratio = -478019` and every
   angle exactly `0`/`90` before the fix). Fixed by excluding any
   zero/missing-pose frame from contact-frame selection, dominant-wrist
   determination, and the weight-transfer/follow-through window endpoints.
2. **Signed-vector vs. undirected-axis angle bug in
   `shoulder_hip_separation_deg`.** One real example reported a 167°
   shoulder-hip separation — anatomically impossible. A line has no
   inherent direction, but the original calc treated the shoulder line and
   hip line as directed vectors, so opposite-labeled-but-same-axis
   alignment read as a near-180° twist. Fixed by reducing to the true
   minimal angle between the two undirected axes; that example became
   12.89°, and all 5 examples now sit in a 1.98°–25.03° range consistent
   with real X-factor magnitudes reported in the biomechanics literature.
3. **Single-frame shoulder-width normalization was foreshortening-sensitive.**
   `wrist_height_ratio` was implausibly large (+15.8, +13.6 shoulder-widths)
   on 2 examples; traced to the shoulder line nearly collapsing to a point
   at that exact frame because the player was side-on to the camera at
   contact (common for serves/one-handed strokes). Fixed by using the
   window-averaged shoulder width as the scale reference instead of the
   single contact-frame value. Result: `wrist_height_ratio` now ranges
   -0.37 to +5.65, with **serve showing by far the highest value** — a
   real, physically sensible pattern (serve contact is genuinely far
   overhead), not a coincidence of the fix. Residual limitation
   (undisturbed): a clip where the player stays side-on for the *entire*
   window would still be affected — would need true 3D pose/calibration
   to fix properly, out of scope, documented not chased.

**Actual computed values are in `ir/DESIGN.md`** (full table, one example
per class) — headline sanity check: serve's `wrist_height_ratio` (+5.65)
is far higher than groundstrokes/volley (-0.37 to +1.64), and the
two-handed backhand's `elbow_angle_deg` (178°, nearly straight) contrasts
with the forehand example's (89.9°, bent) — both match real expected
stroke-technique differences, not arbitrary noise.

**Contact-frame proxy validated against the one real ground-truth
instance:** picked frame 27 vs. the nearest real TrackNet-labeled hit
frame at 30 (3-frame / ~0.1s difference) — a reasonable result, but
honestly caveated: that clip is a multi-shot rally window (5 real hit
events), not one isolated stroke, so there's no single unambiguous "true"
contact frame to compare against; this is an n=1 sanity check, not proof
the proxy generalizes across the other 1980 clips.

**Not yet built:** percentile/benchmark ranking (deliberately excluded
from `BiomechanicsInfo` itself, proposed as a separate downstream utility)
and a full population pass across all 1980 `StrokeClassification` clips
(only 5 individual examples computed and inspected so far).

---

## 2026-07-19: Stroke-level biomechanics — full-dataset pass (all ~1980 clips), a 4th bug found at scale, percentile ranking built

The 5-example pass above was a code-sanity check, not proof that
class-level patterns or a percentile reference population are real — this
entry is the actual scale validation, and **explicitly separates what's
now confirmed at scale from what was only true of those 5 individual
clips.**

**Ran `ir/populate_biomechanics_full.py` across all 1980 real
`StrokeClassification` clips** (all 3 splits combined). Outcome, bucketed
by why a clip is or isn't in the percentile-reference population:

| Bucket | Count | % |
|---|---|---|
| `no_valid_frames` | 0 | 0.0% |
| `insufficient_valid_frames` (<30/60 real frames) | 6 | 0.3% |
| `degenerate_camera_angle` (contact-frame shoulder width collapsed) | 207 | 10.5% |
| `degenerate_arm_extension` (contact-frame arm distance collapsed — new, see below) | 97 | 4.9% |
| **`clean` (used for percentiles)** | **1670** | **84.3%** |

Clean per-class sizes: backhand=427, forehand=407, serve=398, volley=301,
**smash=137** (thinnest, as already flagged for the session-level layer's
percentile discussion).

**Watched specifically for recurrence of the three bugs just fixed, at
scale, rather than assuming the fixes generalized:**
1. Zero-padded-frame bug — does not recur as a hard failure (0.3% below
   the 30-frame threshold, 0 clips with zero valid frames).
2. Signed-vs-undirected-angle bug — **cannot** recur; the fix
   mathematically bounds `shoulder_hip_separation_deg` to `[0, 90]`
   regardless of input, confirmed by the full-dataset max never exceeding
   `89.877°`.
3. Single-frame shoulder-width collapse — the ratio-feature fix
   generalizes (no near-zero cluster in the window-averaged distribution
   at scale), but `shoulder_rotation_deg`/`hip_rotation_deg`/
   `shoulder_hip_separation_deg` are still computed from the single
   contact frame directly, so the underlying issue persists there:
   clips in the bottom 10% of contact-frame shoulder width show **41.7°
   mean separation vs. 19.5° overall** (verified correlation, not
   assumed) — excluded as `degenerate_camera_angle` (207 clips).

**A 4th instance of the same root-cause pattern, found only at this
scale:** `follow_through_extension_ratio` had a heavy right tail (max
52.7) on the first full-dataset pass. Checked before excluding anything:
of the 116 clips with a ratio above 5, **89 (77%) had a near-zero
dominant wrist-to-shoulder distance at the contact frame** — the same
"dividing by something that can collapse on one frame" issue as bug 3,
just on a different feature. Excluded via a new `degenerate_arm_extension`
bucket (97 clips); after exclusion the max dropped to 7.26 and per-class
std dropped from 2.0–3.8 to 0.66–1.28.

**Did the 5-example class patterns hold at scale? Mixed, reported
honestly rather than only keeping the confirming half:**
- **Held:** `wrist_height_ratio` — serve highest (2.47), volley lowest
  (0.85), same ordering as the 5-example check, now a confirmed
  population-level pattern.
- **Held:** `weight_transfer_ratio` — serve's mean (+0.31) clearly
  separated from all other classes (-0.12 to +0.13), consistent with real
  serve biomechanics (caveat: sign depends on an unverified
  camera-facing-direction consistency assumption).
- **Did NOT hold:** `elbow_angle_deg` — the striking 5-example
  forehand/backhand contrast (89.9°/178°) was individual variation, not a
  class difference; population means all sit in a narrow 120°–134° band
  with 38–48° std per class.
- **Did NOT hold:** `shoulder_hip_separation_deg` — the 5-example serve
  had the *lowest* value (1.98°) of the 5; at scale serve has one of the
  *highest* class means (20.1°). That single example was not
  representative.
- **New finding, not visible in 5 examples:** knee flexion doesn't
  differentiate stroke classes much — all 5 classes cluster in a
  151°–171° mean range.

**Percentile ranking built (`ir/biomechanics_percentiles.py`), only now
that the full-dataset pass is run and checked** — percentile validity
depends on this being a real, verified distribution, not assumed.
**Deliberately excludes `shoulder_rotation_deg`/`hip_rotation_deg`/
`forearm_orientation_deg`** from ranking (raises `ValueError` if asked):
their population std is 82–171° per class, essentially spanning the full
range, because which way a player faces the camera is a filming-setup
artifact, not a comparable biomechanical quantity — discovered by
checking the real population spread, not assumed fine because the
5-example pass didn't surface it. Demoed against the same 5 hand-inspected
examples (e.g. serve's `wrist_height_ratio` of 5.651 → 93.0th percentile
of 398 clean serve clips).

**Bottom line for the FYP writeup:** the full-dataset pass did what it was
for — it held up two of the four sanity-checked class patterns, falsified
two others that looked convincing on 5 examples, and surfaced a 4th real
instability (`degenerate_arm_extension`) that the 5-example pass simply
didn't happen to hit. Percentile ranking is now built against a checked,
84.3%-clean population of 1670 real clips, not an assumed-good one.

---

## 2026-07-19: Stroke-level biomechanics — closeout table, future-work item logged, feature-extraction stage complete

Two closeout items, then a stopping point for this phase.

**Final usability table written into `ir/DESIGN.md`** — the single table
the FYP report should use, covering all 10 `BiomechanicsInfo` fields:
`wrist_height_ratio` and `weight_transfer_ratio` are validated at scale
and percentile-ranked with their 5-example pattern confirmed;
`knee_flex_left/right_deg` and `stance_width_ratio` are percentile-ranked
and reliable but not strongly discriminative between stroke classes;
`follow_through_extension_ratio` is percentile-ranked after excluding the
`degenerate_arm_extension` clips; `elbow_angle_deg` and
`shoulder_hip_separation_deg` are percentile-ranked but **their specific
5-example patterns were contradicted by the full population** (forehand
vs. backhand elbow-angle contrast was individual noise; serve does not
have low shoulder-hip separation, it has one of the highest) — flagged
explicitly in the table so those two false leads don't quietly resurface
in the report. `shoulder_rotation_deg`, `hip_rotation_deg`, and
`forearm_orientation_deg` remain excluded from percentile ranking
(camera-facing-direction artifact, not a real biomechanical signal).

**Logged a scoped future-work item, not built:** the 3 excluded rotation
fields are camera-relative, unlike ball/player position, which already
gets a viewpoint correction via `CourtInfo.homography_matrix`
(`ir/geometry.py`). No equivalent correction exists for body-segment
rotation angles. Documented in `ir/DESIGN.md` as an actionable next step —
not a vague "revisit this later" note — pointing at two concrete pieces
of existing scaffolding a future attempt could use
(`StrokeEvent.player_side` as a coarse facing-direction signal, and
`CourtInfo.homography_matrix` where available for a more precise one),
and naming the real blocker: `StrokeClassification` pose sequences
currently have no join key back to a `CourtInfo` for the same rally, so
that linkage would need to be built first. No code written toward this —
explicitly deferred, not attempted.

### Stroke-level biomechanics (feature-extraction stage) — scoped completion point

**Built and verified:**
- `BiomechanicsInfo` schema (`ir/schema.py`), attached as
  `StrokeEvent.biomechanics`, following the same optional-until-built
  pattern as `MomentumInfo`, reviewed and signed off before implementation
  (same discipline as the IR layer's original six decisions).
- `ir/biomechanics.py` — 10 pure-2D-geometry features, computable from
  MediaPipe Pose landmarks alone, with every racket-adjacent field
  explicitly renamed to what it actually measures (no fabricated racket
  data).
- `ir/populate_biomechanics.py` (5-example sanity pass) and
  `ir/populate_biomechanics_full.py` (full 1980-clip pass) — real
  extraction run on real `StrokeClassification` (THETIS) data, both
  results kept and reported, not just the final one.
- Four real bugs found and fixed by sanity-checking actual output rather
  than trusting code that ran without error: zero-padded frames poisoning
  a contact-frame proxy; a signed-vector-vs-undirected-axis angle bug
  producing an anatomically impossible 167° separation; single-frame
  shoulder-width collapse inflating ratio features; and a 4th, same-root-cause
  instability in `follow_through_extension_ratio` found only at full-dataset
  scale.
- `ir/biomechanics_percentiles.py` — population-percentile benchmarking
  (Model-4-style, no external sourcing), built only after the full-dataset
  pass was run and checked, against 1670 clean clips (84.3% of the
  dataset).
- A full usability table (`ir/DESIGN.md`) distinguishing validated/ranked
  fields from excluded ones, and explicitly flagging which 5-example
  patterns turned out to be false at scale.

**Validated at scale (1980 clips) vs. only sanity-checked (5 examples) —
kept distinct throughout, not conflated:** bug-recurrence checking,
per-class distribution stats, and percentile ranking are all built from
the full 1980-clip pass. The 5-example pass remains valuable only as the
original code-correctness sanity check and the source of two now-falsified
individual-clip impressions (elbow angle, shoulder-hip separation) that
the full pass corrected.

**Excluded, with a stated reason, not silently dropped:** 3 of 10 fields
(camera-facing-direction artifacts, see above); ~15.7% of clips
(insufficient valid frames, degenerate camera angle, degenerate arm
extension — all three thresholds set from this project's own measured
distributions, not guessed).

**Explicitly deferred, not started:**
- **RAG/coaching-generation integration** — these features and percentiles
  are not yet wired into any coaching-text generation. Per the original
  instruction for this phase ("do not touch the RAG/coaching-generation
  layer yet"), that integration is future work, structurally similar to
  how `rag/DESIGN.md`'s Step 2 already anticipated this (session-level
  coaching via `explain_anomaly()` is live; stroke-level coaching via this
  new percentile mechanism is the natural next consumer, not yet built).
- **Camera-angle normalization** for the 3 excluded rotation fields — see
  the scoped future-work item above.
- Everything the RAG layer itself already deferred (Claude API comparison,
  KB content-authority upgrade) remains deferred, unaffected by this phase.

**Key finding for the report, same shape as the RAG layer's
grounding/retrieval-relevance finding:** a code fix that works on a small
sanity sample does not automatically generalize — of the four bug
patterns checked at scale, one was mathematically guaranteed fixed, one
generalized empirically, and one only partially generalized (protecting
ratio features but not the raw angle fields computed from the same
collapsing input), while full-dataset scale surfaced an entirely new
instability the small sample never triggered. The same discipline applied
to biomechanical *content*: two of four class-level patterns that looked
real on 5 examples were contradicted once the full population was
checked. Small-sample validation and full-population validation are
different questions, and this phase deliberately answered both rather than
stopping at the first.

**This is a good stopping point for the stroke-level biomechanics
feature-extraction stage, same discipline as the IR and RAG layers'
closeouts:** everything buildable without new external sourcing or new
cross-dataset linkage is built, evaluated with real (non-cherry-picked)
data at both small and full scale, and every remaining limitation is named
in `ir/DESIGN.md` and here rather than left implicit.

---

## 2026-07-19: RAG coaching layer extended to stroke-level biomechanics — Steps 1-3 solid, Step 4 surfaced a structural prompt-size finding, stopped to flag rather than push further

Extended the RAG coaching layer to consume `ir/biomechanics_percentiles.py`,
same design-before-code discipline as every phase so far.

**Step 1 (scope check):** confirmed partial reuse, not full. The citation/
grounding schema, the field-path checks, and the retrieval ranking
algorithm all reuse cleanly (`Citation.record_type` already supported
`"StrokeEvent"`; refactored `retrieve_snippets()` into a KB-agnostic
`retrieve_snippets_for_phrases()`, verified byte-identical session-level
prompt text and identical retrieval output post-refactor). Two genuinely
new pieces were needed, not assumed away: (1) percentile citations require
RECOMPUTING the percentile at verification time
(`Citation.percentile_of`, a new `check_grounding()` branch), not a static
field lookup; (2) a new `select_notable_biomechanics_features()` — a
StrokeEvent has no upstream "which feature is worth mentioning" signal the
way `SessionRecord.explanation` provides.

**Step 2 (KB):** 14 entries in `rag/kb_biomechanics.py`, scoped to exactly
the 7 validated fields from `ir/DESIGN.md`'s usability table. No entries
for the 3 excluded rotation fields. `elbow_angle_deg`/
`shoulder_hip_separation_deg` entries written generically per instruction
— no stroke-class claims, since the specific patterns seen on 5 examples
were falsified at full-dataset scale.

**Step 3 (grounding extension), tested adversarially before trusting it:**
5/5 expected verdicts confirmed exactly
(`rag/example_percentile_grounding_check.py`) — correct percentile passes,
fabricated percentile caught as `PERCENTILE_MISMATCH`, a percentile
claimed for an excluded field caught as `PERCENTILE_FIELD_NOT_RANKED`, a
wrong-record citation caught, and omitting `percentile_of` correctly falls
back to raw-value checking (also caught as a mismatch, since the cited
number was a percentile, not the raw value).

**Step 4 (real-data validation): 0/4 gated success — two real bugs found
and fixed en route, then a third, more fundamental issue surfaced that
was NOT pushed past.**

Built 4 real `StrokeEvent`s with real biomechanics (one per stroke class,
`ir/populate_biomechanics_stroke_events.py`), same composite pattern as
`ir/populate.py`. Found and fixed 2 real bugs on the first run (100%
parse failure, all 4 trials): the prompt told the model to name the
citation's own JSON key `"event_id"` for StrokeEvent instead of the fixed
`"record_id"` `Citation` actually requires (a genuine bug in this
session's own prompt design, not model unreliability), plus a missing
comma in the percentile-field JSON example. Fixed both, verified the
session-level prompt stayed byte-identical.

**Re-ran: still 0/4, but the failure mode changed completely.** Every
attempt on every stroke class now abandons the required JSON schema
entirely and invents its own ad-hoc structure (e.g. `{"stroke_class":
..., "biomechanical_metrics": {...}, "coaching_insights": [...]}` on one
attempt, `{"feedback": [...]}` on another) — a different failure mode
than anything the session-level evaluation showed (which mostly followed
the schema correctly, with occasional hallucinated content inside it).

**Important, separate finding: this is a schema-compliance failure, not a
content-quality failure.** The free-form text the model produces when it
abandons the schema is specific and substantively useful (e.g. "Your
right knee flexion is exceptionally high (98.8th percentile), indicating
a highly compact leg drive... may increase strain on the knee joint if
not balanced with proper core engagement") — not the thin/generic pattern
from the session-level semantic-miss case. The model can produce good
stroke coaching content; it is failing to package it in the schema this
prompt requires.

**Working hypothesis, checked against real numbers:** the stroke-level
user prompt is ~3.8x larger than the session-level one (10,364 vs. 2,755
chars) because the full `StrokeEvent` JSON includes a complete ball
trajectory and 14 court keypoints that biomechanics coaching can't even
cite (citations are restricted to the `biomechanics` object) — unlike
`SessionRecord`'s compact features dict. Plausible, not proven on 4
trials with no ablation run.

**Stopped here to flag this rather than continue iterating on the prompt
unilaterally, per the explicit instruction for this phase.** A next step
(trimming the STROKE_EVENT JSON shown in the prompt to just the citable
fields, while keeping the full record available to the grounding harness)
is real and scoped, but changes what's shown vs. what's citable in a way
that touches this project's original "the prompt is the only source of
truth" design principle — flagged for a decision, not silently
implemented.

---

## 2026-07-19: Stroke-level RAG integration — trimmed the prompt, found and fixed a second real bug, reached 4/4 (100%) gated success

Chose to trim the STROKE_EVENT JSON shown in the prompt down to only the
citable fields (`event_id`, `player_side`, `stroke.predicted_class`/
`confidence`, `biomechanics.*`) — the grounding harness still receives the
FULL record for verification, so nothing about what's actually checkable
changed, only what the model is shown. User prompt size dropped from
10,364 to 3,405 chars.

**This fixed the schema-abandonment problem almost completely, confirming
the size hypothesis** — but immediately surfaced a second, different real
bug: every parsed response now cited the RAW value while still correctly
setting the `percentile_of` flag (e.g. citing `0.147` while `percentile_of:
"wrist_height_ratio"` was set and the text correctly said "14.5th
percentile"). Diagnosed via live raw-output inspection: the design asked
the model to override what `cited_value` means based on a *separate* flag
field, breaking the one invariant ("`ir_field_path` resolves to
`cited_value`") this schema relies on everywhere else — and the model
reliably defaulted back to that invariant rather than the exception.
**Fixed by folding the percentile signal into `ir_field_path` itself**
(`"biomechanics_percentile.<feature>"` vs. `"biomechanics.<feature>"`),
removing the separate `percentile_of` field entirely. Re-ran the
adversarial test — all 5 expected verdicts still confirmed under the new
design, plus a new case (percentile number cited against a raw-value
path, correctly caught).

**Final result: 4/4 (100%) gated success, 0/4 needed a retry**, mean 1.00
attempts/trial, 132.7s total latency (21.4s–43.6s per trial), all 12/12
grounded_observation items grounded.

**Spot-checked the actual generated text, not just the grounding rate, per
the explicit instruction:** all 12 items are substantively specific and
interpretive (not the thin/bare restatement seen in the session-level
semantic-miss case), and correctly reflect the deliberately generic KB
framing for `elbow_angle_deg`/`shoulder_hip_separation_deg` — no response
claimed a stroke-class pattern now known to be false. **One real
overclaiming case found anyway, which the grounding pass rate alone would
not have caught:** volley's `weight_transfer_ratio` citation is correctly
grounded (88.4th percentile, verified), but the generated text adds
"suggesting a highly proactive weight shift **toward the ball direction**"
— a directional claim the measurement's own documented sign-ambiguity
caveat (`rag/kb_biomechanics.py`) does not support. **This is the
stroke-level echo of the earlier grounding/retrieval-relevance finding, in
a new form: grounding verifies the cited NUMBER, not an interpretive claim
layered on top of a correctly-cited number** — a response can be 100%
grounded while still overclaiming in the surrounding prose.

**Two real, structurally distinct bugs found and fixed in this
extension** (citation-key/id-field-name mismatch; percentile-value
override via a separate flag field), both confirmed via live LLM output,
neither assumed fixed just because the code read correctly — consistent
with this project's running discipline of diagnosing against real
generations rather than reasoning from the schema alone.

---

## 2026-07-19: MVP slice — design/scoping pass (Steps 1-4), no code written yet

New phase, explicit shift in focus: the ML core (5 models, IR layer, RAG
session + stroke coaching, gated pipeline) is done; this phase scopes the
smallest real end-to-end product path — one phone screen, one video, one
piece of grounded coaching feedback. Design/scoping only, per instruction
— no backend or mobile code written.

**Step 1 (inventory):** wrote `ir/verify_onnx_exports.py` and
independently re-verified all 5 existing ONNX exports via `onnxruntime`
right now (not trusting each export script's own export-time check) —
introspected real input/output shapes from each graph, ran real forward
passes, and for 3 of the 5 also ran REAL data through (not just random
noise): ball tracking (real clip, sensible varying heatmap peaks 0.93–0.95
confidence), stroke classification (18/20=90% real accuracy vs. reported
0.834 macro-F1), court keypoints (real image, plausible normalized
output range). All 5 load and run cleanly, no drift found.

**Real finding: `court_detection` (YOLOv8-seg court segmentation) is not
wired into the actual IR pipeline at all** — confirmed by a repo-wide
grep; only `court_keypoints` (direct `CourtKeypointNet` regression) is
ever used for court position in every real `StrokeEvent` built so far.
A trained, exported, working model that appears to be unused exploratory
work, not dead weight to silently drop or silently keep.

**Not exported and why (each has a real reason, not just "TODO"):** Model
3 (momentum, no weights, pre-existing known gap), Model 4 (anomaly
detection — session-level aggregate stats, structurally out of scope for
a single-stroke video, not just unexported), MediaPipe Pose (a separate
bundled runtime, not ONNX, already used successfully elsewhere in this
project), the coaching LLM (Ollama service, not an export).

**Step 2 (backend):** proposed a single synchronous FastAPI endpoint
(`POST /api/v1/analyze-stroke`) covering pose extraction → StrokeGRU →
biomechanics → percentile lookup → gated coaching generation, returning
`StrokeEvent` + coaching JSON in one response. Proposed keeping this fully
synchronous (no job queue/polling) given the gated pipeline's own
previously-measured latency (21–83s/attempt, up to 3 attempts) is already
bounded, and a job-queue system would be exactly the kind of extra
architecture this phase is explicitly not supposed to add.

**Step 3 (mobile client):** proposed Expo (React Native) over a bare-native
prototype, justified by speed-to-real-device (Expo Go + QR scan, no
native build step) with no corresponding benefit yet from going native.
Single screen: record/pick a video, upload, display coaching text +
predicted stroke class + a couple of headline biomechanics percentiles.

**Step 4 (real blockers) — the substantive part of this pass:** confirmed
the user's own guess (no stroke-boundary-finding step exists; every model
assumes pre-segmented input) and found several more, each with a proposed
smallest fix documented in `mvp/DESIGN.md`: no code anywhere in this repo
reads a raw video file at all (checked, not assumed — every existing
script reads pre-extracted frames or numpy arrays); MediaPipe Pose's
raw-frame reliability on a real self-shot clip is untested (broadcast
footage needed a throwaway crop hack; THETIS was pre-cropped; neither
answers what a phone video needs); TrackNet's generalization to
phone-shot footage is an unverified out-of-distribution risk.

**Real scope-narrowing finding, surfaced rather than silently acted on:**
re-reading `rag/stroke_prompt.py`'s actual prompt construction shows the
stroke-level coaching path never references `ball`/`court` at all — only
`biomechanics.*` and `stroke.predicted_class` are shown to the LLM or
citable. The literal task description named ball tracking as part of the
target slice, but the coaching output doesn't actually need it — flagged
as an open scope question (recommend cutting it, since it also removes
two of the real generalization risks) rather than decided unilaterally.

Full inventory, endpoint design, client proposal, and the complete
numbered blocker list are in `mvp/DESIGN.md`.

**Decisions received same day:** ball tracking / court detection are OUT
of the MVP slice (confirmed — the coaching path never used them, and
cutting them removes the untested TrackNet-on-phone-footage risk plus
half the video-preprocessing surface); stroke pre-trimming is
instruction-only for v0 (record "one swing, ≤3s," server just uniformly
resamples to StrokeGRU's hard-coded 60-frame input, no motion-energy/ML
auto-trim). With ball/court cut, the MVP pipeline is now: video →
decode/normalize → MediaPipe Pose → `StrokeGRU` → `ir/biomechanics.py` →
percentile lookup → gated coaching generation. The single biggest
remaining untested risk is now MediaPipe Pose's raw-frame reliability on
an actual self-shot phone clip (Step 4 gap #3) — everything else upstream
of it has been cut. Still design/scoping only — no backend or mobile code
written yet; implementation is the next step.

---

## 2026-07-19: MediaPipe Pose stress-test on real footage — result: viable, proceeding to backend/mobile

Tested the single biggest remaining untested risk (gap #3) before any
backend/mobile code, per explicit instruction.

**No genuinely self-shot phone footage was available.** Checked first:
no video files anywhere in this repo or in common user directories on
this machine. One video existed elsewhere on the machine, but inspection
showed it was professional ATP broadcast footage from an unrelated
separate project — the same wide-broadcast condition already known to
need a workaround, so using it would only re-confirm something already
documented. Per user direction, fetched 3 free, no-attribution Pexels
clips as real-footage stand-ins instead — genuine amateur/practice
footage, not curated dataset clips, each clearly flagged as a stand-in:
one solo/close/bright-sun serve-practice clip, one solo/medium/overcast
groundstroke-practice clip, one multi-person/distant/hazy group-lesson
clip.

**Result: the red-shirt-tracker problem does NOT recur.** `mvp/stress_test_pose.py`
ran MediaPipe Pose on 90 real consecutive frames per clip, RAW — no crop,
no workaround. All 3 clips: **100% detection rate (90/90 frames)**,
including the hazy/overcast one. A visual spot-check (landmarks drawn on
a real frame) confirms the skeleton accurately tracks real joints, not
just "detects something." The broadcast-specific hack
`ir/populate_e2e.py` needed was specific to a genuinely wide TV shot; none
of these closer, more deliberate framings needed it.

**One real, separate finding, checked rather than assumed away:**
`num_poses=1` has no subject re-identification — a multi-person scene can
silently switch which person it tracks mid-window while still reporting
a detection every frame. Verified via frame-to-frame hip-center
displacement: the two solo clips showed a clean, tiny, continuous
trajectory (max displacement 0.011–0.017, zero large jumps); the
multi-person clip showed 6 large jumps (up to 0.48) out of 89 transitions
— consistent with the tracked identity changing partway through. This
would corrupt features relying on temporal continuity even with perfect
per-frame detection. Scoped to multi-person framing, not the primary
single-user MVP case — carried forward as a product-guidance note
("keep the frame clear of other people"), not a new engineering
requirement.

**Conclusion: the re-scoped MVP pipeline (video → decode → MediaPipe Pose
→ `StrokeGRU` → `ir/biomechanics.py` → percentiles → coaching) is viable
as designed — no new crop/detection step needed for the single-user case.
Proceeding to backend/mobile scaffolding next**, per the original
instruction's explicit go/no-go condition.

---

## 2026-07-19: MVP backend + mobile scaffolding built, tested locally end-to-end, real-device test handed off

**Structurally surprising finding, fixed before building the endpoint:**
`StrokeEvent.ball`/`.court` were REQUIRED fields (no default) in
`ir/schema.py` — every StrokeEvent built in this project so far always
populated them. The MVP decision to cut ball tracking/court detection
(made in the previous phase) has a direct schema consequence that wasn't
called out at the time: a StrokeEvent literally could not be constructed
without them. **Fixed by making both Optional** (`ball: BallInfo | None =
None`, `court: CourtInfo | None = None`) — a backward-compatible
relaxation (every existing example StrokeEvent still populates both, nothing
existing relied on the requirement), re-verified by re-loading all
existing `stroke_event_*.json`/`session_record_*.json` example files
successfully.

**A second, unrelated structural surprise, also fixed:** installing
`fastapi`/`uvicorn`/`python-multipart` pulled in `pydantic` 2.5.3 →
2.13.4 as a transitive dependency upgrade. Checked, not assumed fine:
re-validating an existing stored `StrokeEvent` JSON (with
`ModelProvenance.metric_value` originally set via `float("nan")` for the
biomechanics component, which has no accuracy metric) now failed —
`model_dump_json()` had always serialized that `nan` as JSON `null`, and
the older pydantic silently accepted `null` back into a plain
(non-Optional) `float` field on reload, while the newer one correctly
rejects it. **Fixed properly, not version-pinned around:** made
`ModelProvenance.metric_value` `float | None = None` and switched the two
call sites (`ir/populate_biomechanics.py`,
`ir/populate_biomechanics_stroke_events.py`) from the `float("nan")`
sentinel to an explicit `None` — a more honest representation of "no
metric applies here" that doesn't depend on any particular pydantic
version's leniency. Re-verified: all existing example StrokeEvents/
SessionRecords still load, the percentile-grounding adversarial test
still passes 5/5.

**Backend built** (`backend/app.py`): single FastAPI service, orchestration
only — every actual inference/logic step reuses existing, already-validated
code (StrokeGRU via the ONNX export verified in `ir/verify_onnx_exports.py`,
`ir/biomechanics.py`, `ir/biomechanics_percentiles.py`,
`rag/stroke_prompt.py`, `rag/gated_pipeline.py`, `rag/llm.py`). New code
written: video decode + uniform resample to exactly 60 frames (StrokeGRU's
ONNX graph hard-codes `seq_len=60`, confirmed via introspection), and raw
(no-crop) MediaPipe Pose extraction, matching what
`mvp/stress_test_pose.py` just validated. `POST /api/v1/analyze-stroke`
and `GET /api/v1/health`.

**Tested locally before touching mobile, per instruction:** ran the real
server (`uvicorn backend.app:app --host 0.0.0.0 --port 8000`) against
`clip1_serve_practice.mp4` (one of the 3 Pexels stand-ins from the pose
stress-test). Result: `200 OK` in ~39s, 100% pose detection, a real
grounded coaching response (3/3 items, 1 attempt). Honest caveat
surfaced by this same test: this Pexels clip is an ~11.5s unstructured
practice montage (288 real frames), not a single trimmed swing — the
uniform-resample-to-60 step picked up whatever motion landed in the
sample and predicted "volley," not "serve." This is the expected,
already-documented consequence of the instruction-only pre-trimming
decision (Step 4 gap #1), not a pipeline bug — the pipeline itself ran
correctly end to end.

**Mobile scaffolded** (`mobile/`, Expo SDK 57): single screen
(`App.js`) — Record Video / Pick from Library (`expo-image-picker`),
upload via `fetch`+`FormData` to the backend, display predicted stroke
class, coaching text with its percentile citations, and detection-rate/
latency diagnostics. **One real bug caught before the device test, not
after:** the first draft manually set a `Content-Type: multipart/form-data`
header on the fetch call — this overrides the boundary parameter
`FormData` auto-generates, which silently breaks server-side multipart
parsing. Removed the manual header. Also checked Expo's SDK 57 API
surface before writing code (`ImagePicker.MediaTypeOptions` is deprecated
in this version in favor of a `mediaTypes: ['videos']` array) rather than
assuming an older, remembered API still applied.

**Networking, checked per explicit instruction before assuming a
connection failure means broken code:** backend LAN IP confirmed
(`192.168.1.9`), Metro/Expo dev server confirmed reachable on the same
LAN IP (`curl` to `192.168.1.9:8081/status` → `200`). Windows Firewall
inbound rules could NOT be queried or created from this session (`Get-NetFirewallPortFilter`
denied — needs admin rights this session doesn't have) — flagged
explicitly for the user to check/open if the phone can't reach either
port, rather than silently assumed fine.

**Both services are running now**, left up for the real-device test:
backend on port 8000, Expo dev server (Metro) on port 8081.

**Handed off: the true end-to-end test needs a real phone**, which this
session doesn't have. Waiting on the user to: install Expo Go, connect to
the same Wi-Fi as this machine, open `exp://192.168.1.9:8081` in Expo Go
(manual URL entry, since no interactive terminal is available here to
scan a QR code), record or pick a real stroke clip, and report what
happens — the mobile equivalent of the IR layer's true end-to-end
example, reported the same way: what it validates vs. doesn't, honestly,
whichever way it goes.

**Update, same day — user reported the phone connected (after a manual
SDK downgrade to 54), then hit a bundler-side codegen error:**
`TransformError`/`SyntaxError` in `VirtualViewExperimentalNativeComponent.js`
("Unable to determine event arguments for onModeChange"). Root cause
confirmed, not guessed: `package.json` had `"expo": "54"` but
`expo-image-picker` (`~57.0.5`), `expo-status-bar` (`~57.0.1`), `react`
(`19.2.3`), and `react-native` (`0.86.0`) were all still pinned to the
prior SDK 57-era versions — a real leftover mismatch from a manual
partial downgrade, not a code bug.

**Fixed with `npx expo install --fix` alone — a clean `node_modules`
reinstall was not needed.** Verified the fix directly rather than
declaring it resolved on faith: confirmed via `require(...)` that
`react-native` (`0.81.5`), `react` (`19.1.0`), and `expo` (`54.0.36`) are
now the only versions present anywhere in `node_modules` (no duplicate/
conflicting nested installs); restarted Metro with `-c` (cache clear);
then fetched the actual compiled bundle directly
(`curl http://localhost:8081/index.bundle?platform=android&dev=true`) —
`200 OK`, `Bundled 6422ms index.js (701 modules)`, no errors. Checked the
specific previously-broken component's compiled output by hand: the
`VirtualView` native component now generates a well-formed
`directEventTypes`/`validAttributes` config (exactly what the original
error said was missing). The `TransformError`/`onModeChange` strings
still findable in the bundle are React Native's own LogBox error-parsing
regex patterns, present in every dev bundle regardless of whether an
error occurs — checked their surrounding context directly rather than
treating a keyword match as evidence of a live error.

**What this confirms vs. doesn't, stated honestly per the explicit
instruction not to declare this resolved on the bundler starting alone:**
confirmed — the version mismatch is fixed, the bundle compiles cleanly,
the specific reported error's code path now produces correct output.
**Not yet confirmed — the app actually loading and running on the real
phone.** That still requires the user to reload it on-device; both
services remain up and reachable on the LAN (`192.168.1.9`, both `200` on
health/status checks) for that reload.

---

## 2026-07-20 — Failure-mode hardening pass

The true end-to-end path now works on real hardware. Per instruction,
before adding anything new: systematically tested and hardened the
pipeline against 5 realistic failure modes, each tested for real
against the running backend (and, as far as testable without a
physical device, the mobile code path) — not just reasoned about. Full
detail, including the exact test data and a summary table, is in
`mvp/DESIGN.md` ("Failure-mode hardening pass"); this entry covers the
narrative.

**1. Low/degraded pose detection.** The existing floor (reject below
50% detection) had never been validated against a real degraded clip —
it was a guessed number. Built an actual detection-rate curve by
applying graduated darkening+blur to the real `clip1_serve_practice.mp4`
(100% clean baseline) rather than guessing where MediaPipe would start
failing. Found the real problem: a moderately degraded version (63.3%
detection — over a third of frames with no real pose) **passed the old
50% floor and produced a fully normal-looking 200 response — 92.5%
stroke-classification confidence, full coaching text, no visible
distinction from a clean run.** That's exactly the "sparse data
presented as normal" failure this task asked about. Fixed by raising
`MIN_POSE_DETECTION_RATE` to 80% in `backend/app.py` — clearly above
the demonstrated-bad case, at the low end of where real clean footage
in this pipeline actually sits (98-100%). Verified: the same degraded
clip now returns 422 with an explicit message including the actual
rate achieved; clean clips are unaffected.

**2. No person / wrong content.** A synthetic no-person video was
already correctly rejected (422, clear message) — confirmed, not
assumed. The real multi-person `clip3_group_distant.mp4` doesn't crash
or hang (100% detection, produces a plausible result), but the earlier
stress-test finding stands unfixed: MediaPipe's `num_poses=1` has no
subject re-identification, so a multi-person scene can silently switch
who it's tracking. Left as a known, flagged limitation (product-
guidance item: ask the user to keep other people out of frame) rather
than new engineering work, matching the original stress-test
conclusion. Also confirmed a separate, inherent limitation while
testing: the classifier has no "this isn't really a stroke" rejection
class — a person present but not mid-swing still gets a confident
classification (seen twice now, including in this pass's own
Ollama-down test). Not fixed this pass; would require real
out-of-distribution detection, which is new ML scope, not a hardening
bug fix.

**3. Bad upload / malformed input.** Before this pass there were no
file-size or duration/frame-count limits at all — a long, high-res
upload would be decoded entirely into memory before any check could
reject it. Tested concretely: a non-video file (random bytes and,
separately, plain text with a `.mp4` extension) was already correctly
rejected. A real video file truncated to 30% of its size was tolerated
by OpenCV and processed normally (not a bug — flagged as "handled
either way"). A 2-frame synthetic clip and a real 4-frame clip reached
pose extraction before failing with a slightly misleading message. A
5400-frame (3-minute) synthetic clip and the untrimmed real `clip3`
had no limit at all — measured that 5400 low-res frames alone decode
to ~1GB of raw arrays in memory, and a real 1080p phone video of the
same length would be over an order of magnitude larger, a genuine
resource-exhaustion risk. A 110MB dummy upload would have been fully
buffered into memory with no limit. Fixed all of these in
`backend/app.py`: added `MAX_UPLOAD_BYTES` (100MB, enforced while
streaming the upload in chunks, not after buffering it), and
`MIN_RAW_FRAMES`/`MAX_RAW_FRAMES` (5-900 frames), with the max enforced
*during* the decode read loop so an oversized file is rejected within
about a second regardless of its real length, not after being fully
read. Verified: the short/long/oversized cases now all return clear
422/413 responses quickly; clean real footage is unaffected.

**4. Backend/network failure paths on the mobile side.** Before this
pass, `mobile/App.js`'s fetch call had one generic catch block that
displayed `err.message` verbatim — for a real connection failure this
would have shown React Native's raw `"Network request failed"` message
with no guidance. Investigated concretely (via Node's fetch against an
unreachable port, matching how React Native's fetch behaves for this
case) what a connection failure, a timeout, and an HTTP error each
actually look like from the catch block's perspective. Fixed: the
catch block now distinguishes `AbortError` (timeout) from `TypeError`
(network-unreachable) from a handled HTTP error (now carrying the
backend's own, much-improved message per items 1-3 and 5), and shows a
specific, actionable message for each, plus a "Try Again" button.
Verified as far as possible without a physical device: rebuilt the
Metro bundle and fetched it directly — 701 modules, no errors, all
three new message strings present in the actual served bundle. **Not
verified: how this renders on the physical phone** — requires the
user.

While testing this, found and fixed something unrelated to the 5
failure modes but that would have silently blocked all on-device
testing regardless: the backend machine's LAN IP had changed since the
last session (DHCP re-lease, `192.168.1.9` → `192.168.1.8`) —
`BACKEND_URL` in `mobile/App.js` was stale. Updated it and reconfirmed
both the backend and Metro are reachable at the new IP.

**5. The Ollama/gated-pipeline failure path.** Two distinct failure
modes exist here. The retry-gate's own exhaustion path (LLM responds
but never produces a fully grounded answer within 3 attempts) was
already handled correctly end-to-end — confirmed to fire in practice
during the earlier RAG validation work, already surfaced by the
backend as `coaching.success: false` + `failure_reason`, and mobile's
UI already had a correct branch for it. Reviewed, no change needed.
But **Ollama being unreachable entirely had never been tested, and was
a real, confirmed bug**: added a temporary `OLLAMA_HOST` env var to
`backend/app.py` specifically so this could be tested by pointing at
an invalid host, without touching the actually-running Ollama service.
With Ollama unreachable, a real request ran stroke classification and
biomechanics successfully, then threw an uncaught
`urllib.error.URLError` from deep inside the gated-pipeline call,
which FastAPI turned into a bare `500 Internal Server Error` with a
plain-text, non-JSON body — **silently discarding the already-correct
classification and biomechanics** and leaving the mobile client with
nothing usable. Fixed by wrapping the coaching-generation call in a
`try/except Exception`, converting any failure into the same
`coaching.success: false` / `failure_reason` shape the retry-gate's
own exhaustion path already produces, so the mobile client only ever
has to handle one failure shape, and a coaching failure never discards
an already-successful classification. Verified: re-ran the same
Ollama-unreachable test post-fix → `200 OK`, full stroke_event intact,
`coaching.success: false` with a clear reason.

**Honest status of what's confirmed vs. not, same discipline as every
other phase:** all five backend-side fixes were tested for real against
the running server and confirmed working. The mobile-side error-message
fix (item 4) was verified as far as possible without a physical device
(bundle compiles cleanly, correct strings present) but **not verified
rendering on-device** — needs the user. Both services are up and
reachable at the corrected LAN IP (`192.168.1.8`) for that test, along
with a fresh true end-to-end run now that the stale-IP issue is also
fixed.

---

## 2026-07-28 — Model 3 (momentum tracking): feasibility check against existing Tennalysis MCP data

Before scoping new data acquisition, checked whether the Match Charting
Project (MCP) data already ingested for the sibling Tennalysis-web project
(`D:\Tennalysis-web` — a separate repo from this one, hence the 2026-07-17
entry below saying "no point-by-point data anywhere in the local repo" was
right for *this* repo but incomplete for the user's wider project set)
already covered what Model 3 needs, rather than assuming a fresh acquisition
effort was required. Verified directly: the 10,022 rows referenced were
match-level aggregate stats (one row per charted match/player in Postgres'
`charted_match_stats` table, built from MCP's `Overview.csv`/`NetPoints.csv`
"Total" rows) — not point sequences. Inspected the 5 raw MCP files actually
downloaded and confirmed every one is a match/set/situational-bucket
aggregate (e.g. "8 of 11 break points won for the whole match"), never a
per-point row with server/winner/score-state. None of the 3 candidate
momentum-label approaches were computable from this data as scoped.
**Conclusion: acquisition still required, but narrowly — the actual MCP
point-by-point files (`charting-m-points*.csv`) exist in the same source
repo the already-ingested stats files came from and were simply never
downloaded.** Full detail in `training/momentum_tracking/DATA_ACQUISITION_CHECKLIST.md`.

## 2026-07-28 — Model 3: real point-by-point files pulled and schema-confirmed

Pulled 3 real files directly from `raw.githubusercontent.com/JeffSackmann/
tennis_MatchChartingProject` (`charting-m-points-2020s/2010s/to-2009.csv`,
1.28M point rows / 7,532 matches) and inspected the actual schema before
writing any ingestion code. Confirmed real: `match_id, Pt, Set1, Set2, Gm1,
Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd, Notes, PtWinner` — a genuine per-point
log with server identity, point winner, and running score, all with zero
nulls in the core columns across all three files. Found and flagged a real
data-quality quirk: ~18% of matches have a non-`+1` jump in `Pt` (investigated,
not just noted). Cross-checked match IDs directly against the live
`charted_match_stats` table: **4,997 of the 5,011 already-ingested matches
(99.7%) also have point-level data available**, meaning ingestion could join
straight onto Tennalysis-web's existing reconciliation mapping with no new
matching pass. Confirmed all 3 candidate momentum-label approaches (EWMA,
ΔWP, point-importance-weighting) are now data-supported in principle — the
modeling work for #2/#3 is separate, unblocked work, not a data gap.

## 2026-07-28 — Model 3: first real pipeline — ingestion + EWMA label + sanity check

Scoped deliberately narrow, per instruction: ingestion, the EWMA
recent-win-rate momentum label (candidate approach #1 only), and a sanity
check on real matches — explicitly **no** ΔWP, no point-importance-weighting,
no `Dataset`/training loop, held off until EWMA is shown to be sensible.
Wrote `training/momentum_tracking/ingest_points.py` and `sanity_check_ewma.py`.

**Dedupe/sort fix, made the first thing the pipeline does.** Root-caused the
~18% non-sequential-`Pt` figure rather than patching around it: 1,521 of
7,532 matches (20.2%) turned out to be pure row-ordering artifacts in the raw
files — each match's points were already complete and correct, just not
written in `Pt` order; sorting alone fixes these. A separate, smaller set of
11 matches had genuine duplicate rows for the same `Pt`; deduplicating on the
score-relevant columns (ignoring shot-notation-only differences) resolved 6
of these cleanly. **5 matches had real, conflicting duplicate data and were
dropped rather than guessed at** (named individually in the checklist).
Verified explicitly, not assumed: every remaining match's `Pt` column is
strictly `+1` increasing from 1, checked with a hard assertion. Final count:
7,532 raw matches → 7,527 clean matches (99.93% retention) → **4,996 already
linked to real ATP match/player identity** — within 1 of the 4,997 estimated
during schema confirmation, confirming the dedupe fix doesn't shrink the
usable corpus.

**A real bug the sanity check itself caught.** While picking example matches
to plot, naively reading each match's last row's `Set1`/`Set2` to determine
the winner was wrong in 31.2% of matches (checked against Tennalysis's real
`matches.winner_id`) — the raw score columns record state *entering* each
point, not after it, so the last row reflects the score before the final
point was played. Fixed by using the real `winner_id` instead. Flagged in the
checklist for whoever builds approach #3 later (break/set/match-point
detection has the same trap).

**EWMA sanity check: looks like real signal, not noise.** Plotted the
momentum curve (halflife=10 points, provisional) for 3 real matches, each
verified against `matches.score`: the 2004 Roland Garros final (Gaudio's
famous comeback from 2 sets to love down, score `0-6 3-6 6-4 6-1 8-6`) shows
sustained negative momentum for the first ~100 points that crosses positive
around the match's midpoint and mostly holds — the expected shape for that
exact comeback. The 2015 Rome Masters SF (Federer d. Wawrinka 6-4 6-2 — a
tidy scoreline) shows real point-level texture underneath: Wawrinka built an
early lead the curve captures before Federer's momentum turns it around
mid-match. (An earlier draft caption for this match, written from memory,
incorrectly called it "Wawrinka d. Federer, a real upset" — the plot's own
computed winner label caught that mistake before it went further, worth
noting as the discipline working as intended.) A dominant control match
(Djokovic d. Cilic 6-0 6-1 6-2) stayed mostly positive throughout with no
deep dip, as expected for a real blowout. **Conclusion: EWMA is validated
well enough to move forward with** — approaches #2 and #3 remain deliberately
deferred until this label is actually used to train something.

Processed output: `training/momentum_tracking/data/processed/
points_ewma_momentum.parquet` (1,281,126 rows, 7,527 matches). Raw CSVs
(130MB) now live in `training/momentum_tracking/data/raw/`, not yet
committed — needs a deliberate git-tracking decision, not made here. Full
detail in `training/momentum_tracking/DATA_ACQUISITION_CHECKLIST.md`.

## 2026-07-28 — Trial readiness: reachability options report + passcode gate (tunnel setup blocked, in progress)

**Reachability options.** Before committing to anything, reported real
tradeoffs (cost/setup time/reliability) for getting the backend reachable
beyond the LAN for a bounded trial with known testers: Cloudflare Tunnel,
ngrok, Tailscale Funnel, vs. actual cloud hosting. Checked the backend's real
constraints first, since they matter: coaching runs Qwen3 8B locally via
Ollama on this machine's RTX 3060 (6.5-6.6GB VRAM, already-validated latency
profile from `rag/DESIGN.md`), so "just deploy it to the cloud" really means
either renting GPU compute (~$400+/mo on-demand for a comparable instance) or
swapping to a hosted LLM API (`ClaudeBackend` exists in `rag/llm.py` but is
untested, no key configured) and re-validating latency/reliability from
scratch. Recommended Cloudflare Tunnel: reuses the exact already-working
local setup at zero/near-zero cost, at the honest cost of uptime being tied
to this PC and home internet staying up — an acceptable tradeoff for a short
trial with known, informed testers. Also confirmed via grep that the mobile
app doesn't surface ball tracking, momentum, or anomaly detection anywhere
(only `App.js` exists, single screen, three result sections: stroke,
coaching, diagnostics) — stroke classification + coaching are genuinely the
only two trial-facing features.

**Passcode gate — done and verified.** Added a hardcoded shared-passcode
check (`X-Trial-Passcode` header, `backend/app.py`) as global middleware
(covers every request/endpoint, not just the ones that exist today) —
explicitly not real auth, just enough that the URL isn't wide open to anyone
who stumbles on it. Mobile app (`mobile/App.js`) sends the matching header on
every request. Verified for real against the running backend: no header →
401, wrong header → 401, correct header → 200, and a full real-clip round
trip through `/api/v1/analyze-stroke` with the header present succeeded
end-to-end (`clip1_serve_practice.mp4` → volley, 72.3% confidence, coaching
success, 100% pose detection) — confirms the gate doesn't break the actual
multipart upload path.

**Cloudflare Tunnel setup — blocked, not yet done.** Installed `cloudflared`
(2026.7.3, via winget). A *stable named hostname* (the user's explicit
requirement, not the rotating `trycloudflare.com` quick-tunnel URL) requires
a domain added as a zone to a Cloudflare account, which doesn't exist yet —
asked the user rather than guessing or substituting the rotating URL. Also
fixed an unrelated but blocking issue found in passing: the LAN IP had
DHCP-re-leased again since the last session (`.8` → `.9`, the same recurring
issue as 2026-07-20) — updated `mobile/App.js`'s `BACKEND_URL` so local
testing isn't broken while the tunnel is pending. **Next, once the user has
an active domain in Cloudflare:** `cloudflared tunnel login` (interactive,
needs the user), `tunnel create`, DNS route, ingress config, run the tunnel,
update `BACKEND_URL` to the stable hostname, and a real phone round-trip test
through the tunnel (not just LAN) — same standard as the original MVP
device test. Consent/data-handling flow remains explicitly deferred to after
this.

**Note for future sessions:** the recurring DHCP LAN-IP-change issue (hit
and patched three times now — 2026-07-20, and twice in this entry alone)
is a symptom of `BACKEND_URL` pointing at a LAN IP at all, not a one-off
bug each time. Once the Cloudflare Tunnel is live, `BACKEND_URL` becomes a
stable hostname instead of a LAN IP, and this class of fix goes away
structurally — no further LAN IP patches should be needed after that
point. If a "stale IP" issue shows up again after the tunnel is live, that's
a sign `BACKEND_URL` reverted to a LAN address (e.g. during local dev),
not the same recurring problem.

## Phase B scoping (2026-07-28): multi-screen navigation + session history

Reachability work is paused pending the user registering a domain (their
action, not something to work around). In the meantime, scoped Phase B —
moving `mobile/App.js` from its current single-screen state machine to
real screens (Home/Capture, Processing, Results, History, Settings) —
per instruction, proposal only, no screens written yet. Full detail and
justification in `mvp/DESIGN.md` ("Phase B scoping"); summary here.

**Navigation: Expo Router recommended, not React Navigation.** Checked
current docs before deciding (`mobile/AGENTS.md` explicitly warns Expo has
changed since the installed SDK 54 and to check versioned docs first —
correctly, since it surfaced two real changes that would have been wrong
to assume from older knowledge: Expo Router is now built directly on
`react-native-screens` rather than on top of React Navigation, and the
file-based root has moved to `src/app/`). Recommended Expo Router mainly
because this is a genuinely green-field navigation decision — today's app
has zero existing navigation code either way — combined with it being the
docs' explicit recommendation for new apps, file-based routing needing
less boilerplate, and typed routes catching bad navigation targets at
compile time. Real tradeoff acknowledged: file-based routing conventions
have already changed across recent SDK versions and could again.

**Proposed structure:** tab bar for Home/History/Settings; Processing and
Results as non-tab stack screens (Home → push Processing → replace with
Results on success, so the back stack from Results skips straight to
Home). Large result data (the full analysis JSON) proposed to pass via a
small React Context, not URL params — current docs are explicit that
params are for small serializable values, not full response payloads.

**Session history: on-device only recommended, not backend-persisted.**
Real tradeoff table in `mvp/DESIGN.md`; the deciding factor is that
backend persistence directly depends on the consent/data-handling
decision that's explicitly still deferred (retention, access, deletion) —
building server-side storage now would mean building it twice or making
those decisions by accident. On-device keeps Phase B unblocked and
independent of that pending conversation; proposed writing History against
a small local storage-repository module so a later move to
backend-persisted history, if ever justified, is a contained change.

## 2026-07-28 — Phase B implementation: Expo Router structure + on-device history, wired screen by screen

Scoping approved; built the proposed structure from `mvp/DESIGN.md`'s Phase B
section. No visual polish (Phase E is separate) -- structure and navigation
only, plain styling throughout.

**Setup had two real install snags, both resolved:** `expo install` initially
picked a `babel-preset-expo` version (`~54.0.10`) one patch behind what
`expo@54.0.36` itself requires (`~54.0.12`) -- fixed by pinning the correct
version. Separately, a plain `npm install` failed on an unrelated peer
conflict: `expo-router` pulls in `vaul`/`@radix-ui/*` (web-only dev-tooling
dependencies, unrelated to native functionality) which wanted a newer
`react-dom` than this project's pinned `react` version satisfies -- resolved
with `--legacy-peer-deps`, standard practice for this class of RN/Expo
transitive-peer mismatch, not a workaround specific to this project.

**A real, non-obvious bundling gotcha found and fixed:** after setting
`package.json`'s `"main": "expo-router/entry"`, the dev server's
`/index.bundle` URL kept serving the OLD `App.js` content — confirmed by
literally finding old UI text ("VisioTennis MVP") in the fetched bundle
after the router files were already written. Root cause: Metro's
`/<name>.bundle` dev-server endpoint resolves `<name>` against a literal
source file path, not `package.json`'s `main` field — the leftover root
`index.js` (which still imported old `App.js`) was exactly that file. Fixed
by deleting `index.js` entirely (correct per current docs: not needed when
using `expo-router/entry` directly) and switching verification to the
correct entry path, `/node_modules/expo-router/entry.bundle`. Worth
remembering for any future mobile bundle-compile check in this project — the
old `/index.bundle` pattern from the pre-router single-screen app no longer
applies.

**Built, in order, verifying each via Metro bundle-compile checks before
moving on** (no physical device or emulator available this session --
confirmed with the user before proceeding, so this is compile-level
verification only, not on-device rendering, same limitation as the mobile
error-UX work in the 2026-07-20 hardening pass):
1. Routing skeleton: `app/_layout.js` (root `Stack`, wraps app in
   `SessionContext`), `app/(tabs)/_layout.js` (`Tabs`: Home/History/Settings),
   `processing.js`/`results.js` as sibling stack screens outside the tab bar.
2. `lib/historyRepository.js` -- on-device session storage (`expo-sqlite`),
   built as the swappable seam the design doc calls for
   (`addSession`/`listSessions`/`getSession`, screens never touch
   `expo-sqlite` directly), wired into the root layout via `initHistoryDb()`.
3. Home (`app/(tabs)/index.js`) -- record/pick logic moved from the old
   `App.js`, now navigates to `/processing` with the video URI/filename as
   router params (small serializable values, per the design doc's guidance
   on what params are for).
4. Processing (`app/processing.js`) -- the actual upload/passcode-header/
   timeout/error-handling logic moved here unchanged in substance. On
   success: saves to the history repository, sets `SessionContext`, then
   `router.replace('/results')` (replace, not push, so Results' back stack
   goes straight to Home). On failure: same inline error message + Try Again
   as before, now returning to Home via `router.replace('/(tabs)')`.
5. `components/ResultView.js` -- the result-rendering JSX extracted out of
   the old single-screen `App.js` so both `results.js` (live result, from
   `SessionContext`) and the History detail screen (a past, on-device-stored
   result) render identically without duplicating markup.
6. History list (`app/(tabs)/history/index.js`) -- reloads via
   `useFocusEffect` every time the tab gains focus, so a session just saved
   by Processing shows up without an app restart. Detail screen
   (`history/[sessionId].js`) loads one full session by id and renders it
   via `ResultView`.
7. Settings (`app/(tabs)/settings.js`) -- placeholder only, nothing
   configurable yet (`BACKEND_URL`/passcode are still hardcoded in the new
   `lib/config.js`).

**Final verification:** after removing the now-fully-superseded `App.js`,
a clean Metro restart + full bundle fetch confirmed all 7 screens' real
content present exactly once each (not placeholder text — actual UI strings:
"Record Video", "Processing... this can take", "Predicted stroke", "No
sessions yet", etc.), zero remaining trace of the old single-screen app, and
both Android and iOS platform bundles compiling with no errors or warnings.
1123 modules, clean build.

**Explicitly not verified this pass:** on-screen rendering, tap-through
navigation, and `expo-sqlite`'s actual runtime behavior (only its JS
import/syntax was checked at bundle-compile time) — all require a physical
device or emulator, neither available this session. Flagged to the user
before proceeding; same honest-limitation pattern as prior device-dependent
work in this project.

## 2026-08-01 — Phase B: real device/emulator verification pass (all 5 scenarios pass)

The prior entry's bundle-compile-only verification was explicitly flagged as
insufficient; this session replaced it with a real execution pass, since a
clean compile cannot catch runtime routing/storage bugs. No physical device
available this session, so stood up an Android emulator from scratch on a
machine with no prior Android tooling: Android Studio (for the JDK) +
standalone cmdline-tools installed headlessly, `platform-tools`/`emulator`/
`platforms;android-34`/a `google_apis` x86_64 system image via `sdkmanager`,
AVD `vt_test` (Pixel 6, API 34) via `avdmanager`. The emulator's windowed GUI
crashed on first boot (Qt native crash — this sandboxed environment can't
render it); booted successfully headless instead
(`-no-window -no-audio -no-boot-anim -gpu swiftshader_indirect`).

**A real Expo Router bug was found only by actually running the app** —
direct vindication of doing this pass at all: `app/(tabs)/history/` lacked
its own `_layout.js`, so the folder wasn't recognized as a valid nested tab
route (`WARN [Layout children]: No route named "history" exists...`). A
clean bundle-compile check never would have surfaced this, since it's a
runtime routing-resolution warning, not a syntax/import error. Fixed by
adding `app/(tabs)/history/_layout.js` (a `Stack` wrapping `index` and
`[sessionId]`); warning gone on rebuild.

Deliberately avoided the emulator's synthetic camera feed for the happy-path
test (a checkerboard fake feed with no real human pose would predictably
fail pose detection for reasons unrelated to what's being tested) — instead
pushed the known-good real clip (`clip1_serve_practice.mp4`) onto the
emulator's storage and selected it through the real Android system photo
picker, so the test exercises genuine pose detection/classification/coaching.

**All 5 scenarios verified by actually tapping through, screenshotting real
device state at each step (not assumed):**
1. **Happy path (Home → pick → Processing → Results): PASS.** Real upload to
   the real backend produced volley/72% confidence, coaching feedback, 100%
   pose detection, 44.0s processing time.
2. **Failure case + Try Again: PASS.** Killed the backend process, retried
   the same flow — the inline error rendered correctly ("Couldn't reach the
   server...") with a working Try Again button; Try Again returned to Home;
   restarted the backend and confirmed a fresh attempt from Home succeeded
   end-to-end (57.6s, same clip).
3. **History live update via `useFocusEffect`: PASS.** Both completed
   sessions appeared in the History tab immediately on switching tabs, no
   app restart needed.
4. **History detail rendering: PASS.** Tapping a History row rendered the
   same `ResultView` content as the live Results screen, with the session
   timestamp.
5. **SQLite persistence across a real restart: PASS.** Force-stopped the app
   entirely (`am force-stop`, not just backgrounding) and relaunched from
   cold — both sessions were still present in History, confirming
   `expo-sqlite` data survives an actual process restart, not just an
   in-memory session.

No further Phase B work pending. Phase C (mapping the 5 hardened backend
failure modes to specific screen states) is next.

## 2026-08-01 — Reachability: Tailscale Funnel evaluated as a free alternative to Cloudflare Tunnel (domain purchase avoided)

User decided against buying a domain (which the pending Cloudflare Tunnel
setup from 2026-07-28 was blocked on) and asked for a genuinely free
alternative instead. Evaluated Tailscale Funnel against the same 4 questions
before committing to setup, rather than assuming it fits: (1) hostname
stability — confirmed deterministic (`<device-name>.<tailnet>.ts.net`), not
a rotating URL like `trycloudflare.com`'s quick-tunnel mode; (2) public
reachability — confirmed Funnel is specifically designed to expose to the
general internet, not just devices on the same tailnet; (3) free-tier
availability — found a real discrepancy between sources (official docs say
"available for all plans"; the pricing page's comparison table read as
Standard-tier-only) and resolved it empirically rather than trusting either
source blindly, by installing the real client and checking; (4) bandwidth
limits — undisclosed by Tailscale but not documented as blocking file
uploads at this scale.

Installed via winget (`Tailscale.Tailscale`, 1.98.10). Started
`tailscale up --hostname=visiotennis-trial`; this requires one interactive
browser login (Tailscale account auth) that only the user can complete —
paused here, waiting on that step before finishing Funnel setup (enabling
Funnel on port 8000, confirming the resulting hostname, updating mobile's
`BACKEND_URL`, and a real round-trip test through the public URL).

**Update, same day: setup completed and verified.** The first browser-auth
attempt didn't actually take (the backgrounded `tailscale up` process got
orphaned before completing the handshake — `tailscale status` still showed
"Logged out" locally even though the admin console showed a device, which
turned out to be a stale registration from the abandoned attempt). Caught
this by checking the local CLI state directly rather than trusting the
admin console alone, re-ran `tailscale up`, got a fresh login URL, user
completed that one — confirmed via `tailscale status` showing
`visiotennis-trial` connected (100.105.175.29).

**Free-tier question resolved empirically: yes, Funnel works on the free
Personal plan.** `tailscale funnel 8000` initially returned "Funnel is not
enabled on your tailnet" with a one-time admin-console enablement link —
this is Tailscale's standard opt-in step for any tailnet's first use of
Funnel, not a plan restriction. User completed it; `tailscale funnel --bg
8000` then started cleanly with no plan-gate error of any kind. Public
hostname confirmed stable and deterministic:
`https://visiotennis-trial.tail458008.ts.net`.

Updated `mobile/lib/config.js`'s `BACKEND_URL` to the new hostname (from the
LAN IP that had needed patching three separate times for DHCP re-leases —
that class of fix goes away structurally now). Verified twice, at
increasing levels of realism: (1) a direct `curl` multipart upload through
the public hostname reproduced the full pipeline (volley 72.3%, coaching
success, 100% pose detection); (2) the actual mobile app, freshly relaunched
on the Android emulator to pick up the new `BACKEND_URL`, completed a real
Home → pick → Processing → Results run entirely through the public URL —
confirmed genuinely public-routed, not a LAN shortcut, because the
backend's own request log showed the request arriving from `139.135.33.77`
(a Tailscale relay address), not a LAN or loopback address.

**Known limitation, stated plainly:** both verifications above ran from
this same machine/network — no genuinely separate external device or
network was available this session to test from. The network path itself
is confirmed real-internet-routed (public DNS, real TLS cert, non-LAN
source IP at the backend), which is the strongest test available without
a second physical location; a true outside-network test with an actual
tester's phone is the next natural confirmation once testers are lined up.

## 2026-08-01 — Roadmap adopted: `visiotennis_roadmap.md` as the consolidated plan

User introduced `visiotennis_roadmap.md` (repo root) as the consolidated
plan from current state through full product vision, and asked to work
Phase 0 and Phase 1 in order, checking in before each numbered item and
stopping on any conflict with what's already built or any unspecified
decision. Mapped roadmap items against work already done this session
before starting anything new:

**Phase 0 — already satisfied, no new work needed:**
- Reachability (free public URL, no domain/paid tier, verified round trip
  from off-LAN) — done via Tailscale Funnel, see the two entries above.
- App foundation's device-verification sub-item (real tap-through
  verification, SQLite-across-restart, History-refresh-on-focus) — done,
  see the Phase B verification entry above.

**Phase 0 — remaining, to work in order:** Phase C (map the 5 backend
failure modes to screen states, the other half of "App foundation"),
Consent and data handling, Coach review document.

**Phase 1 — not yet started:** Automatic session instrumentation,
structured feedback (reliability/usefulness tracks), training-data capture
with consent, confidence signalling in the UI.

Checking in with the user before starting the first remaining item (Phase C)
per instruction.

## 2026-08-01 — Phase C: mapped the 5 hardened backend failure modes to distinct screen states

User approved proceeding. Read `backend/app.py` in full first to establish
exactly how each of the 5 modes actually surfaces today, rather than
assuming from memory: 3 of the 5 (bad/oversized upload, wrong content in
frame, low pose detection) come back as HTTP 413/422 with a `detail`
string; network failure is entirely client-side (no HTTP response at all);
coaching-service outage does NOT fail the request -- it's a 200 OK with
`coaching.success: false`, by design, so the already-successful stroke
classification and biomechanics are never discarded.

**Backend change:** the 3 upload/content error paths previously returned
`detail` as a bare prose string, which the mobile client would have had to
pattern-match to tell apart -- fragile, and would silently break if the
wording ever changed. Restructured `detail` into `{"error_code", "message"}`
for all 4 relevant `HTTPException` raises (`_read_upload_capped`'s 413, and
the 3 in `analyze_stroke`: bad-video-file/no-player-detected/
low-pose-detection), with 3 possible `error_code` values
(`bad_video_file` covers corrupt/too-long/too-short/zero-frame/oversized --
all "something's wrong with this file" from a tester's perspective; the two
others are `no_player_detected` and `low_pose_detection`). Checked first
that nothing else parses `detail` as a plain string (only
`mobile/app/processing.js` does) before changing its shape. Also reworded
the backend's own message text away from dev-facing phrasing ("this MVP
expects...") toward plain, actionable, tester-facing language while
preserving the specific guidance already written in the 2026-07-20
hardening pass.

**Mobile changes:** `processing.js` now renders a bold category title above
the backend's specific message (`lib/errorMessages.js` maps `error_code` ->
title), so each of the 3 upload/content modes plus the 2 client-side network
modes (timeout vs. connection-unreachable, kept distinct as before) reads as
5 visually distinct states, not one undifferentiated red box. Network
failure's message now says plainly "this isn't a problem with your video"
per the roadmap's instruction to say so when the fix isn't on the user's
side. `ResultView.js`'s coaching-failure branch no longer shows the raw
`failure_reason` (a Python exception repr, e.g. from an unreachable Ollama
backend -- developer-facing, not tester-facing); it now shows a fixed,
reassuring message that explicitly says the stroke analysis above is still
valid and this isn't something wrong with their video, while logging the
real reason to the console for debugging.

**Verified all 5, for real, not assumed:**
- `bad_video_file` (corrupt file), `no_player_detected` (blank/no-person
  video), and `low_pose_detection` (a constructed 50%-detection real clip:
  every other frame from `clip1_serve_practice.mp4` replaced with black) --
  all 3 confirmed via direct backend requests to return the correct
  `error_code`/`message` shape.
- `low_pose_detection` additionally verified on-device on the emulator:
  screenshot shows the new bold title ("Couldn't get a clear read on the
  player's movement") above the specific 30/60-frame guidance text.
- Coaching-service outage verified on-device: temporarily pointed
  `OLLAMA_HOST` at an invalid address (same technique as the original
  2026-07-20 discovery of this failure mode), ran a real clip through the
  actual app -- Results screen correctly showed volley/72%/100% pose
  detection with the new clean "isn't available right now... still valid"
  message in place of the old raw exception text, confirming the
  already-successful classification is never discarded. Restored real
  Ollama afterward.
- Network failure (timeout/unreachable) and the happy path were already
  verified in the Phase B device pass above and unchanged by this work.

Not separately re-verified: `bad_video_file`'s too-large/too-long/too-short
sub-cases and `no_player_detected` on-device (code-identical rendering path
to the on-device-verified `low_pose_detection` case, differing only in
which title/message string is looked up -- confirmed correct via the direct
backend tests above).

## 2026-08-01 — Coach review document exported from the coaching knowledge base

Per the roadmap's Phase 0 "Coach review" item (explicitly the item to get
out of the user's hands earliest since it's the only one blocked on someone
else's schedule, not ours). Exported all 38 entries across both KB files
(`rag/kb_placeholder.py`, 24 session-level entries across 12 match-stat
metrics; `rag/kb_biomechanics.py`, 14 stroke-level entries across 7
biomechanics metrics) verbatim into `rag/KB_COACH_REVIEW.md` -- count
cross-checked against both source files (`grep -c "KBSnippet("` = 24 and 14,
matching the doc's 38 `**E<n>**` markers exactly) so nothing was dropped or
altered in transcription.

Per entry: metric name, a plain-English gloss of what it measures (pulled
from the exact code-comment definitions in
`training/anomaly_detection/train.py`'s `FEATURE_COLUMNS` for Section A,
and the KB snippets' own descriptive first sentences for Section B -- not
invented, to avoid presenting the coach with characterizations they aren't
actually the ones reviewing), the coaching text itself, and 3 blanks (1:
technically correct? 2: would you say this to a player? 3: what's wrong or
missing?). Whole-set questions at the end, as specified: what common faults
are missing entirely, and whether anything is actively bad advice, plus a
general-comments blank.

Explicitly flagged in the intro, per instruction: (1) all content is
authored by the project team, not sourced from any coaching manual or
published guideline -- stated plainly, not hedged; (2) the biomechanical
percentile framing (Section B) is the area most needing the coach's
judgement, specifically the question of whether expressing a single-camera,
geometry-derived measurement as a specific-sounding percentile implies more
precision/authority than it actually has; (3) two entries
(`elbow_angle_deg`, `shoulder_hip_separation_deg`) are deliberately
non-committal because an early 5-example pattern was contradicted by the
full 1,980-clip population (documented already in `ir/DESIGN.md` and the KB
files' own comments) -- explained so a reviewer doesn't mistake
deliberately-generic wording for an oversight. Also noted honestly that the
exact sentence a player sees may be an LLM paraphrase grounded in this
content, not always verbatim -- what's being reviewed is the substance the
AI is allowed to draw from, not guaranteed final UI copy.

Suggested (not required) that two coaches review independently rather than
jointly, since coaching philosophy varies and disagreement between
reviewers is itself informative.

Next: consent and data handling (last Phase 0 item) -- checking in with the
user first, since the roadmap specifies *that* this is needed but not the
specific retention/deletion policy, which is a real product decision, not
an implementation detail.

## 2026-08-01 — Consent and data handling: stopped, deliberately not started

Per the standing instruction to stop on any decision the roadmap specifies
is needed but doesn't itself resolve. Two real open decisions here, both
now resolved as far as they can be resolved right now:

1. **Default retention for testers who consent to processing only (not
   training-data retention): confirmed as "not retained at all."** This
   matches the backend's actual current behavior already -- every upload
   is written to a temp file and `os.unlink`'d immediately after the
   request completes, success or failure (`backend/app.py`). No new
   engineering needed for this default case; the consent copy just needs
   to describe real existing behavior accurately.

2. **Ethics/IRB approval: confirmed required or already in progress**,
   given this trial involves real university tennis players and their
   video as part of an FYP. This changes the shape of the remaining work:
   consent wording and the exact data-handling policy (especially for the
   separate training-data-retention opt-in, where video WOULD be kept) has
   to match whatever the ethics process actually requires -- drafting that
   language unilaterally risks conflicting with it. Asked the user how to
   proceed (hold off entirely vs. build placeholder-copy scaffolding now);
   **user chose to hold off entirely.**

**Consent and data handling is therefore deliberately not started**, not
forgotten -- waiting on the approved (or draft-approved) consent
language/policy from the ethics process before implementing anything here.
This also blocks Phase 1's "training-data capture, with explicit consent"
item, which depends on the same consent mechanism.

**Phase 0 status:** 3 of 4 items done (reachability, app foundation, coach
review sent out for review); consent and data handling blocked on the
user's ethics process, not on us.

## 2026-08-01 — Phase 1: confidence signalling in the UI (first of 3 items not blocked on consent)

Per instruction, continuing only the 3 Phase 1 items that don't depend on
the (currently blocked) consent mechanism: session instrumentation,
structured feedback, confidence signalling. Started with confidence
signalling since it changes what testers see directly. Proposed thresholds
and exact message wording to the user before writing any code, per their
explicit request -- user approved with 3 changes, all implemented as
specified:

**1. Two independent triggers, not one.** Absolute confidence < 60% is one
trigger; a top1-top2 probability gap < 20 points is a SEPARATE trigger that
also names the runner-up class, regardless of absolute confidence. This
matters concretely: the stroke classifier has a documented volley/forehand
confusability (`training/stroke_classification/README.md`'s confusion
matrix), which can produce a confidently-wrong top-1 prediction that a
single confidence threshold would structurally miss (e.g. top1=68%,
top2=53% -- "confident" by a flat threshold, but a near-coin-flip against
the runner-up). Verified both triggers independently and jointly via 3
Node-level test vectors covering exactly this case, plus confirmed against
real backend output. Pose detection gets a separate, single threshold
(<90%) for the 80-90% band that already passes Phase C's hard 80% floor
but is below the 98-100% clean-footage baseline.

**2. Flattened wording, no editorializing about trust.** Dropped
"treat this loosely" / "treat with caution" phrasing per instruction --
messages state the fact and the filming tip only, e.g. "Lower confidence on
this one (52%) -- it could also be a backhand. For a clearer read next
time: film closer to the player with their full swing visible in frame."
and "Pose tracking was inconsistent on this clip (87% of frames). For a
cleaner result next time: get closer to the player, use brighter or more
even lighting, and make sure only one person is in frame."

**3. Logs every session, not just flagged ones.** Added
`log_session_metrics()` in `backend/app.py`, called unconditionally right
after `stroke_event_dict` is built (regardless of whether a caveat displays
client-side, and regardless of whether coaching generation later succeeds
or fails) -- appends one JSON line to `backend/logs/sessions.jsonl`
(gitignored) with: `event_id`, `created_at`, `predicted_class`,
`confidence`, `runner_up_class`, `top1_top2_gap`, `pose_detection_rate`.
Logging only flagged sessions would have biased the calibration data away
from exactly the cases most worth finding -- confident-and-wrong
predictions -- so this was made unconditional per explicit instruction, not
tied to whether a caveat fired. **Deliberately excludes player_id and
source_video** (a tester could name a file after themselves) and any
IP/device info, per the cross-cutting non-identifying-by-default
instruction -- confirmed by inspecting every field written, not just
by omission.

Implementation lives entirely client-side for display (`components/
ResultView.js`) since the backend response already carries everything
needed (`class_probabilities`, `confidence`, `pose_detection_rate`) --
no backend response-shape change, only the new logging side effect.

**Verified for real:**
- Confidence-caveat logic: 3 Node test vectors (high-conf/big-gap -> null;
  low-conf/big-gap -> caveat without runner-up; high-conf/close-gap ->
  caveat WITH runner-up) all matched expected output.
- On-device, real backend, real clip: constructed a clip at exactly 86.7%
  pose detection (blanking every 7th frame of `clip1_serve_practice.mp4`)
  that happened to also land at 52.4% classification confidence -- both
  caveats rendered simultaneously on the actual emulator, in the right
  positions, with the exact flattened wording, confirming both the UI
  mechanism and the "no runner-up named when gap >= 20pts" branch (gap was
  24.4% here, and no runner-up appeared, matching the logged value exactly).
- Logging: confirmed writes happen even when the HTTP client disconnects
  before the full response completes (a curl call with too-short
  `--max-time` cut off mid-coaching-generation; the log entry was still
  written, since logging happens before the coaching step, not after the
  full response is sent) -- this was a genuine, not contrived, confirmation
  that logging doesn't depend on a completed response round-trip.
- Inspected all 4 logged entries from testing directly: no player_id,
  source_video, or any other identifying field present in any of them.

Test video and the test-run log entries were scratch artifacts, not real
trial data -- deleted after verification (`backend/logs/sessions.jsonl`
reset to empty).

**Explicitly documented as provisional, not final:** both threshold values
(60% confidence, 20-point gap) are placeholders pending real trial data --
Phase 1's own structured feedback (next item) is what will let these
actually be calibrated against "was this correct?" answers, same logic the
roadmap already applies to pose detection vs. usefulness ratings.

Next: session instrumentation (the log just added covers confidence
signalling's inputs specifically; the roadmap's fuller instrumentation ask
also wants retry count, processing time, and failure mode logged) and
structured feedback (reliability/usefulness ratings) -- both still to do.

## 2026-08-01 — Phase 1: session instrumentation (second of 3 items)

Extended the logging added for confidence signalling into the roadmap's
full "automatic session instrumentation" ask: pose detection rate,
predicted class + confidence (already done above), whether the coaching
gate retried and how many times, processing time, and any failure mode
triggered. Two gaps closed:

**Rejected sessions were previously invisible to instrumentation
entirely.** The confidence-signalling log call only fires after a session
reaches classification -- Phase C's 3 upload/content rejection paths
(bad_video_file, no_player_detected, low_pose_detection) raise an
HTTPException before that point and were never logged at all, silently
undercounting how often testers actually hit them. Added `log_rejection()`,
called at all 5 raise sites for those 3 error codes (413 upload-too-large,
the 3 `bad_video_file` sub-cases, no_player_detected, low_pose_detection),
logging `event`, `failure_mode`, `processing_time_s`, and a fresh
log-local `event_id` (no real StrokeEvent exists yet at rejection time --
this id exists only to give the log row a unique key, not to identify
anyone).

**Coaching outcome (retry count, success, processing time) wasn't logged
at all.** Added `log_session_outcome()`, called once after the coaching
try/except resolves (success or failure), logging `coaching_success`,
`coaching_attempts`, `coaching_failure_reason`, and total
`processing_time_s` -- correlated with the earlier classification log entry
via the same `event_id`, written as a SEPARATE call rather than merged into
one, specifically preserving the property (already verified for confidence
signalling) that classification data survives even if this later step
never completes.

Log schema is now 3 event types in one file (`backend/logs/sessions.jsonl`,
gitignored), distinguished by an `"event"` field: `classification`,
`outcome`, `rejection`. Same non-identifying-by-construction rule applied
throughout: no player_id, no filename, no IP/device info in any of the 3
record types -- checked field-by-field, not by omission.

**Verified for real, all 3 event types:**
- `rejection`: sent a corrupt file, confirmed `{"event": "rejection",
  "failure_mode": "bad_video_file", "processing_time_s": 0.36, ...}`.
- `classification` + `outcome` (success): real clip through the full
  pipeline, confirmed both entries share the same `event_id`, and
  `outcome` shows `coaching_success: true, coaching_attempts: 1`.
- `classification` + `outcome` (coaching failure): repeated the
  OLLAMA_HOST-invalid technique from the 2026-07-20/Phase C work,
  confirmed `outcome` shows `coaching_success: false, coaching_attempts:
  0`, and `coaching_failure_reason` carries the real exception text (this
  field is server-side-only, never shown to a tester -- Phase C already
  handles that on the display side) -- restored real Ollama afterward.

All test-run log entries were scratch data, deleted after verification.

Next and last of the 3 unblocked Phase 1 items: structured feedback
(reliability/usefulness ratings, kept as two separate tracks per the
roadmap).

## 2026-08-01 — Phase 1: structured feedback (third and last of the 3 unblocked items)

Proposed the design before building (where it appears, the 3 questions,
backend endpoint/log shape) since it's the other item that changes what
testers see. User approved with 2 changes:

**1. Reliability question made concrete.** "We said {predicted_class}. Was
that right?" instead of a generic phrasing -- ties the question directly to
the specific prediction the tester just saw.

**2. Wrong/partly-wrong answers get a one-tap follow-up: what was it
actually?** (the 5 stroke classes). Logged as `actual_class`. Flagged by
the user as the highest-value field in the whole schema -- real ground
truth on real phone footage, paired with the model's own wrong prediction,
which the project has had none of until now (all existing training/test
data is the curated THETIS dataset, not real trial footage).

**Design fork surfaced and resolved: the coach-only question ("would you
say this to a student?") needs to know who's answering, which is in
tension with "no persistent per-tester identifier."** Proposed a bare
ROLE CATEGORY ("player"/"coach") -- not a name, not an ID, stored on-device
only (new `settings` table in the existing history SQLite db,
`mobile/lib/historyRepository.js`'s `getRole()`/`setRole()`), sent as a
plain string alongside a feedback submission, never tied to anything else.
User confirmed this reading of the constraint is correct: a role isn't an
identifier.

**Built:**
- `mobile/app/(tabs)/settings.js` -- role selector ("I am reviewing as:
  Player / Coach"), the settings screen's first real content (previously a
  placeholder).
- `mobile/components/FeedbackForm.js` -- the 3-question form (reliability
  with predicted-class-specific wording; actual-class follow-up shown only
  when reliability is no/partly; usefulness 1-5 + optional free text;
  coach-only yes/no question shown only when role=="coach"), submits to
  the new backend endpoint, shows "Thanks -- feedback recorded." in place
  of the form on success, a plain retry-able error on failure.
- `mobile/components/ResultView.js` -- took a `children` prop so
  `FeedbackForm` renders inside the same ScrollView, after Diagnostics.
- `mobile/app/results.js` -- passes `<FeedbackForm>` as ResultView's child
  -- **deliberately only here, not from History's detail screen**
  (`app/(tabs)/history/[sessionId].js` renders `ResultView` without
  children), so the form shows exactly once, right after a live session,
  never re-prompted when revisiting a past session.
- `backend/app.py` -- new `POST /api/v1/feedback`, a pydantic
  `FeedbackSubmission` model with cross-field validation (usefulness_rating
  1-5; `actual_class` required when reliability is no/partly and must be
  one of the 5 real stroke classes), logged to `backend/logs/
  feedback.jsonl` (gitignored, same directory as the session logs) --
  joinable to those via the shared `event_id`, which is what actually makes
  the roadmap's "does low pose detection predict low usefulness" analysis
  possible.

**Verified for real:**
- Backend: valid submission logged correctly; both cross-field validation
  errors (reliability=no without actual_class; rating out of 1-5 range)
  correctly rejected with 422 and never written to the log.
- Full on-device round trip: set role to Coach via the real Settings
  screen (confirmed the SQLite write persisted by reading it back), ran a
  real clip through Home -> Processing -> Results, filled out every field
  on the actual rendered form (Partly -> actual-class picker correctly
  appeared -> forehand -> usefulness 4 -> coach question Yes), submitted
  against the real backend, confirmed the exact submitted values in
  `feedback.jsonl` matched what was tapped, and confirmed "Thanks --
  feedback recorded." replaced the form.
- Confirmed the History detail screen does NOT show the feedback form
  (scrolled to the bottom of a past session, form absent) -- the
  children-prop mechanism works as designed.

All test-run log entries were scratch data, deleted after verification.

**Phase 1 status: all 3 unblocked items done** (confidence signalling,
session instrumentation, structured feedback). Training-data capture
remains blocked on the same consent/ethics process as Phase 0's consent
item -- not started, per standing instruction.

## Status summary (as of 2026-07-19)

| Model | Status | Key metric |
|---|---|---|
| 1A — Ball tracking | Done, exported | val_pck@4px = 0.964 |
| 1B — Court detection | Done, exported, split-leakage fixed | mAP50-95(mask) = 0.995 |
| 1B — Court keypoints | Done, exported | val_dist = 6.14px |
| 2 — Stroke classification | Done, exported | test macro-F1 = 0.834 |
| 3 — Momentum tracking | Real ingestion pipeline + EWMA label (approach #1) built and sanity-checked on real matches; ΔWP/point-importance-weighting deferred until EWMA is used in training; no `Dataset`/training loop yet | 7,527 clean matches (4,996 linked to real ATP identity) after dedupe/sort fix |
| 4 — Anomaly detection | Pipeline done, demo-verified (all synthetic anomalies flagged -1) | N/A (no real data yet) |
| IR layer (Objective 2) | Schema + homography + population code done; 3 real composite `StrokeEvent`s + 1 true single-video end-to-end `StrokeEvent` + 3 `SessionRecord`s verified; keypoint order visually confirmed. **Stroke-level biomechanics: scoped completion point reached.** `BiomechanicsInfo` schema + extraction + full-dataset pass (1980 clips) + percentile ranking done; 4 real bugs found/fixed (3 on 5-example pass, 1 at full scale); 7/10 fields validated/percentile-ranked, 3/10 excluded (camera-facing-direction artifact, future-work item logged); RAG integration deferred | homography reprojection error 3.6-10.5cm; biomechanics: 84.3% (1670/1980) clean clips used for percentiles, contact-frame proxy within 3 frames (~0.1s) of the one real ground-truth hit frame |
| RAG coaching layer (Objective 3) | **Scoped completion point reached.** 24-entry KB + embedding retrieval + grounding harness + retry-gate done; Step 1 resolved (placeholder KB, content-authority upgrade deferred); Step 2 RESOLVED (session-level via `explain_anomaly()`, stroke-level formally deferred); LLM: Qwen3 8B (Q4_K_M) local via Ollama. Deferred: Claude API comparison, KB authority upgrade, stroke-level coaching | Retrieval recall 17/18 (94.4%) after 2 bug fixes; gated reliability 6/6 (100%) on repeated + diverse sessions (gate's failure path also confirmed to fire in practice); grounding/retrieval-relevance decoupling flagged as key finding |

## 2026-08-02 — KB rewritten for player-facing tone; grounding re-verified at 100%

Real player-facing testing on real output surfaced a tone problem the
grounding harness can't catch: coaching text was correct but read as
measurement reporting, not coaching (e.g. "your follow-through extension
ratio (0.216) is at the 0.0th percentile for backhand strokes, suggesting
a highly abbreviated follow-through motion" -- a real, fully-grounded
citation, but not something a club player knows what to do with).

**Rewrote all 38 KB entries** (`rag/kb_placeholder.py` 24 entries,
`rag/kb_biomechanics.py` 14 entries) to: lead with plain-language meaning
(no "ratio"/"percentile"/"extension"/"separation angle" jargon in the main
sentence), end every entry with something the player can try next time,
state findings directly instead of hedging on a metric already measured as
unusual, and keep entries to ~2 sentences so several stacked results stay
skimmable on one screen. Trimmed bracket notes that didn't carry a
specific fact (kept the one that did: `wrist_height_ratio`'s "(contact
height: unusual for this shot)"). The two deliberately non-committal
entries (`elbow_angle_deg`, `shoulder_hip_separation_deg` -- see the
2026-07 biomechanics entry below on why) were kept honest: what was
measured, told directly, no manufactured "try X" fix, since the specific
patterns behind those two were already falsified against the full
1980-clip population.

**Did not touch citation/grounding architecture** -- confirmed by code
read before starting (`rag/grounding.py` never inspects KB text;
`rag/embeddings.py`/`rag/prompt.py` embed and retrieve on the KB's full
text same as before, whatever it says) and then confirmed for real by
re-running both grounding harnesses against the rewritten KB:

- `rag/run_gated_diverse_trials.py` (6 session-level trials, distinct
  anomaly profiles): **6/6 (100%) gated success.** Mean retrieval recall
  83.3% (2 of 6 trials retrieved a KB topic that didn't match one of the
  session's actually-deviated features -- a retrieval-relevance question,
  not a grounding failure; every attempt that completed was still 100%
  grounded regardless of which topics were retrieved).
- `rag/run_gated_stroke_trials.py` (4 stroke-level trials, one per stroke
  class): **4/4 (100%) gated success.**

**Infrastructure finding, unrelated to the KB content:** the local Ollama
server intermittently stalled on the raw HTTP request itself (client-side
`TimeoutError`, no response at all within 300s) after a handful of
sequential `/api/chat` calls in one process -- reproduced 3 times across
otherwise-identical clean runs of the 6-trial script, each time at a
*different* trial index (3rd, 4th, and 4th call respectively), which
rules out any specific prompt or KB entry as the cause. A standalone
single-call reproduction of the exact prompt that hung came back in 65s
with a fully grounded response. Added a validation-script-only retry
wrapper (`generate_with_network_retry` in both `run_gated_*_trials.py`
scripts, 3 attempts with a 5s pause) that retries the whole trial on a
network-level exception -- deliberately NOT inside
`rag/gated_pipeline.py`'s retry-gate, which is about rejecting ungrounded
content, not working around a dead connection; those are different
failure modes and the fix belongs at the layer that owns each one. The
4th full run, with the wrapper in place, passed both harnesses cleanly on
the first pass through all trials.

**Regenerated `rag/KB_COACH_REVIEW.md`** from the 38 rewritten entries
(same E1-E38 numbering as the version already in front of the coach, so
old notes stay addressable). 10 entries introduce a concrete new "try
this" that wasn't in the prior wording and are marked **⚠ NEW ADVICE**
inline, with a one-line note on what changed, so the coach's review can
tell reworded-only content apart from genuinely new claims: E10, E12,
E13, E14, E22 (session-level); E27, E30, E31, E33, E34 (biomechanics).
Also flagged for the coach separately: E27's prior wording carried a real
data-quality caveat (the weight-transfer sign depends on per-clip camera
framing and hasn't been independently verified) that is no longer
repeated in the player-facing text -- worth their attention regardless of
the advice-tone question.

## 2026-08-02 — Stroke segmentation: multi-stroke detection in a single clip (detection-only, first real-footage pass)

Scoped as segmentation + per-stroke classification only, no coaching --
coaching over multiple strokes (one session-level generation covering
shot counts/breakdown/patterns, not one LLM call per stroke, to keep
processing time sane) is deliberately deferred to a later pass.

**Built, no new model / no new training data:**
- `ir/stroke_segmentation.py` -- `detect_candidate_strokes()` finds
  candidate contact points as local maxima in dominant-wrist speed across
  a full (unwindowed) pose sequence, reusing `ir/biomechanics.py`'s
  existing zero-padded-frame exclusion (`_valid_frame_mask`) and dominant-
  wrist heuristic (`determine_dominant_wrist`) rather than reimplementing
  them. Windows each candidate to StrokeGRU's fixed 60-frame input
  (`training/stroke_classification/model.py`), shifted (not zero-padded)
  to stay inside the sequence when a candidate sits near either end.
  Three tunable suppression knobs, picked analytically (not fit to the
  validation clip below): `min_separation_frames` (no two candidates
  closer than this), `height_percentile` (a peak must exceed this
  percentile of the clip's OWN speed distribution, since wrist speed here
  is uncalibrated and doesn't transfer as an absolute threshold between
  clips), `min_window_valid_fraction` (a per-window analog of
  `backend/app.py`'s existing whole-clip `MIN_POSE_DETECTION_RATE` gate).
- `ir/run_stroke_segmentation_trial.py` -- validation harness: decodes
  every frame of a real (unresampled) clip, runs MediaPipe pose
  extraction + candidate detection + StrokeGRU classification per
  candidate, and writes a wrist-speed plot with detected peaks marked,
  frame-index-labeled contact sheets, and full-res crops at each
  candidate's contact frame -- specifically so detection accuracy could
  be hand-verified against the real footage rather than trusted on the
  numbers alone.

**First real-footage validation** (`mvp/real_footage_test/
clip1_serve_practice.mp4` -- 288 frames, 25fps, 11.5s, 100% pose detection
rate; one of only 3 real non-THETIS, non-synthetic clips in the repo, no
prior ground-truth annotation existed for any of them):
- **4 candidates detected.** Hand-verified against full-res frame crops
  at each candidate (not just the low-res contact sheet, which was
  ambiguous on its own for 2 of the 4): **2 real strokes in the clip**
  (both serves), **2 true positives** (frame 235 landed almost exactly at
  a jump-serve's contact point; frame 155 landed a little early, during
  the ball toss rather than contact, but attributable to the same real
  serve), **2 false positives** (frames 48 and 108 -- both confirmed on
  the full-res crop to be the player walking with the racket swinging at
  their side, not a stroke; frame 108 was the single highest wrist-speed
  peak in the entire clip, i.e. walking arm-swing here scored higher than
  either real serve), **0 false negatives**. Precision 50% (2/4), recall
  100% (2/2) on this one clip -- not tuned to fit it, this is the first
  pass with the three knobs above left at their analytically-picked
  defaults.
- This is exactly the over-detection failure mode anticipated before
  building this: walking-with-racket motion produces wrist-speed peaks on
  the same order as, and in this case exceeding, genuine stroke swings.
  `min_separation_frames`/`height_percentile` alone were not enough to
  suppress it on this clip; next tuning pass should look at whether a
  velocity-direction or wrist-relative-to-shoulder feature (a swing
  rotates through/across the body; walking arm-swing doesn't) separates
  the two better than speed magnitude alone -- not attempted yet.
- **Separate finding, classification not detection:** StrokeGRU
  classified all 4 windows (including both true-positive serves) as
  "volley," never "serve." Both real serves were detected at the right
  place; the classifier's label was wrong. Likely a real-footage domain
  gap from THETIS training data (different camera distance/framing,
  motion blur, a genuinely fast jump serve) rather than a segmentation
  problem -- flagged, not investigated further here, since classification
  accuracy on real footage is a distinct question from finding candidate
  windows in the first place.
- Only one clip validated so far (single player, clay court, bright sun,
  static camera) -- explicitly not enough to trust these three defaults
  generally. Next: validate against `clip2_teen_practice.mp4` (longer,
  more player movement between strokes, a harder over-detection test) and
  `clip3_group_distant.mp4` (multi-person, distant -- likely a pose-
  detection-rate problem before a segmentation one) before tuning
  anything.

## 2026-08-02 — Stroke segmentation: kinetic-chain rotation filter + validation across all 3 real clips

Follow-up to the pass above. Wrist-speed magnitude alone under-suppressed
on clip1 (walking-with-racket outscored both real serves). Added a second,
independent signal and validated detection across all 3 real clips, not
just clip1 -- per instruction, thresholds were NOT compromise-tuned to
make all three clips come out clean; where a threshold that helps one
clip doesn't transfer to another, that's reported as a limitation.

**Added `compute_trunk_rotation_series` / rotation-gated filtering to
`ir/stroke_segmentation.py`**, no new model: reuses
`ir.biomechanics._angle_vs_horizontal_deg` (the same formula already used
once per clip for `shoulder_rotation_deg`), evaluated every frame and
turned into a rate of change. Hypothesis: a real stroke turns the trunk
into the shot; a walking arm-swing is a pendulum with the torso facing
forward.

**First version (shoulder + hip combined) was itself broken by pose-
tracking noise, found and fixed before it was ever trusted:** on
clip1's frame-108 false positive, `shoulder_width` collapsed to 0.0045 vs.
0.0387 at the true-positive serve (an 8.6x difference, the same
foreshortening problem `ir.biomechanics.extract_biomechanics_features`
already works around by averaging shoulder width over a whole window) --
the resulting near-degenerate line's angle swung 77->113->115 degrees
across 3 consecutive frames from pure landmark jitter, producing the
single HIGHEST rotation reading in the entire clip despite not being a
real rotation. Added `_foreshortening_stable_mask` (a per-frame,
clip-relative line-length stability gate, same reasoning as the existing
shoulder-width-averaging code, applied per-frame instead of once per
window) to fix this class of noise generally.

**Second-order finding: hip line dropped entirely, not just gated.**
Even after the stability gate, the hip line's naturally shorter baseline
(hip landmarks sit closer together than shoulder landmarks in 2D
projection) meant the same relative threshold left it more exposed to
tracking jitter -- a hip-line reading at 42% of its own clip median still
cleared the 35% stability cutoff and produced a 166.94 deg/frame spurious
value. Switched to shoulder-line rotation only. This is a geometric
argument (shorter baseline -> more noise-sensitive angle under the same
tracking jitter), confirmed on one clip so far -- flagged in the code as
a decision that could turn out wrong on footage where the shoulder line
is the noisier one, in which case that would be a real limitation to
report, not a case for re-tuning the threshold to force a fix.

**Even after both fixes, the rotation filter did NOT cleanly separate
clip1's specific false positive from its real stroke** -- reported
honestly rather than chased further: false positive frame 108 scored
10.70 deg/frame, true positive frame 155 (the toss-initiation moment of a
real serve, not exact contact -- itself a sign the wrist-speed contact-
frame proxy mistimed that swing) scored 9.20 deg/frame. No single
threshold can separate an inverted pair; raising the bar to exclude the
false positive would exclude the true positive first. Left as-is (default
`rotation_percentile=75.0`) rather than hand-picking a value that happens
to work on this one clip's specific numbers.

**Validation across all 3 real clips** (`ir/run_stroke_segmentation_trial.py`,
now runs detection with AND without the rotation filter and reports both,
plus a second plot panel for the rotation signal):

- **clip1_serve_practice.mp4** (11.5s, single player, sparse well-separated
  serves): 4 candidates both with and without the filter (0 removed).
  Unchanged from the first pass: 2 true positives, 2 false positives
  (walking with the racket), 0 false negatives. The rotation filter
  provided no net improvement here, per the finding above.
- **clip2_teen_practice.mp4** (25.2s, single player, dense continuous
  ball-feeding drill -- a much harder, higher-cadence case than clip1):
  10 candidates without the filter, 9 with it -- the filter correctly
  removed the one candidate at frame 619, where the player had actually
  left the frame entirely (confirmed on the frame crop: empty of the
  player, only background pedestrians), which is a real category of
  false positive the filter is suited for. All 9 survivors hand-verified
  as genuine mid-swing frames on their full-resolution crops -- 9/9
  precision. Recall was NOT perfect, unlike clip1: a complete, clearly
  visible stroke (backswing -> contact -> follow-through, ball departing
  the racket, confirmed on individually-extracted frames 76-104) produced
  zero candidates -- the `height_percentile=75` threshold, calibrated to
  this clip's loudest peaks, silently dropped a real but less explosive
  swing. Given the swing density in this drill, this is reported as "at
  least one confirmed miss," not an exhaustive count -- hand-verifying
  every swing in continuous dense play from static thumbnails is
  meaningfully harder than clip1's sparse, well-separated serves, and
  that difficulty is itself a finding about the validation method, not
  swept under the rug.
- **clip3_group_distant.mp4** (23.1s, distant, multi-person coach-led
  children's group lesson): 20 candidates, unchanged by the filter.
  On inspection this clip does NOT validate the segmentation algorithm at
  all: MediaPipe's `num_poses=1` pose tracking does not consistently
  track one person across a multi-person scene. Two spot-checked candidate
  frames (22, 191) show the tracked motion does not correspond to any one
  person mid-stroke -- at frame 22 the nearest adult is standing still
  holding a racket at their side; at frame 191 nobody in frame is
  swinging. The large "wrist speed" readings here most likely come from
  pose-tracking identity jumping between different real people in the
  group as MediaPipe's single-pose selection shifts frame to frame, not
  from any one person's real fast movement. This is a pose-identity-
  stability limitation (would need multi-person tracking/re-identification
  to fix), not a peak-detection or rotation-filter problem, and squarely
  out of scope for this pass -- flagged as a hard prerequisite gap for
  ever running this pipeline on group/coach footage, not something these
  suppression knobs can address.

**Net assessment:** the rotation filter is a real, principled addition
(the foreshortening-stability fix in particular generalizes beyond this
one use) but is NOT a solved over-detection problem -- it helped on
clip2 (correctly rejected a no-player-in-frame candidate), did nothing on
clip1 (the one case it was built to fix), and is moot on clip3 (wrong
problem entirely). Precision was good wherever the input pose stream
was valid (9/9 on clip2, and clip1's two real serves both survived);
recall was NOT reliable even on a single-player clip once play density
increased (clip2's confirmed miss). No thresholds were tuned to make
these three numbers look more consistent than they are.

**Still not done, explicitly out of scope for this pass:** any coaching
generation over multiple strokes (per original scope -- one session-level
generation, not one call per stroke); a directional/velocity feature
beyond rotation magnitude that might separate clip1's remaining FP/TP
pair; multi-person tracking for clip3-style footage; systematic (not
spot-check) false-negative counting on clip2's dense drill.

## 2026-08-02 -- StrokeGRU classifier diagnostic on real footage (diagnosis only, no fix)

Ran the unmodified StrokeGRU ONNX classifier (`ir/run_classifier_diagnostic.py`,
new script) against every real-stroke candidate window on 5 real clips, and
hand-verified predicted vs. actual by eye against saved contact-frame crops.
No model or threshold changes -- this pass is diagnosis only, per explicit
instruction.

**Clips used:** the same 3 practice clips from the stroke-segmentation work
(`mvp/real_footage_test/`) plus 2 new ones pulled from
`data/VisioTennis/Dataset/Ball Tracker+Bounce/` (broadcast frame-sequences,
originally collected for ball tracking -- the only other real, non-THETIS
tennis video in the repo), chosen from a visual scan of 81 available clips
specifically for having a closer/larger player than the broadcast norm.
Two of the 5 clips turned out not to produce usable predictions (see below);
this is reported as-is rather than swapped out for a better-looking clip.

**Results (only confidently hand-verifiable instances graded):**

| clip | contact frame | predicted | actual (hand-verified) | correct? |
|---|---|---|---|---|
| clip1_serve_practice | 235 | volley (0.91) | serve (jump serve, full extension -- unambiguous) | **NO** |
| clip1_serve_practice | 48/108/155 | volley (0.61-0.70) | not strokes (walking/idle with racket) | n/a -- model still outputs a confident class for non-strokes |
| clip2_teen_practice | 34 | backhand (1.00) | backhand (2-handed) | yes |
| clip2_teen_practice | 142 | backhand (0.99) | backhand (2-handed) | yes |
| clip2_teen_practice | 213 | volley (0.99) | backhand (2-handed) | **NO** |
| clip2_teen_practice | 259 | backhand (1.00) | backhand (2-handed) | yes |
| clip2_teen_practice | 340 | backhand (0.83) | backhand (2-handed) | yes |
| clip2_teen_practice | 385 | backhand (0.98) | backhand (2-handed) | yes |
| clip2_teen_practice | 439 | backhand (0.98) | backhand (2-handed) | yes |
| clip2_teen_practice | 493 | forehand (0.99) | backhand (2-handed) | **NO** |
| clip2_teen_practice | 574 | backhand (0.99) | backhand (2-handed) | yes |
| game4_clip1_broadcast_close | 110 | backhand (1.00) | serve (classic trophy/back-scratch position, unambiguous) | **NO** |
| game4_clip1_broadcast_close | 64/166/302/457 | backhand (0.96-1.00) | ambiguous -- single frame + camera angle insufficient to confirm forehand vs. backhand vs. scramble | not graded |

Clip2's "actual" column is not a per-frame guess: all 9 candidates were
verified against the two contact sheets covering the full clip
(`ir/example_output/stroke_segmentation_trial/clip2_teen_practice/
contact_sheet_*.png`) -- every visible swing in the clip uses the same
two-handed grip and the same swing path (ball fed from her right, sweeps
across body), i.e. this is one repetitive two-handed-backhand feeding
drill, not a mix of stroke types.

**clip3_group_distant:** excluded from grading (already invalidated for
pose-tracking-identity reasons in the segmentation work); included in the
raw run for completeness but not counted.

**game6_clip1_broadcast_medium:** pose detected on only 33/596 frames
(5.5%) -- the same distant/broadcast camera-scale problem that already
invalidated clip3, reproduced on a second, independent clip. 0 candidates,
nothing to grade. Not swapped for an easier clip; reported as the honest
result of trying it.

**Confidently graded aggregate: 7/11 correct (63.6%)**, vs. 84.5% on the
model's own held-out THETIS test set
(`training/stroke_classification/README.md`).

**The error pattern is not diffuse -- it is concentrated entirely on serve.**
Both real, unambiguous serve instances in this diagnostic (clip1 frame 235,
game4 frame 110) were misclassified: **0/2 real-footage serves correct**,
to two *different* wrong classes (volley, then backhand) -- not one
consistent confusion pair. This is the opposite of what the model's own
in-sample numbers predict: on the THETIS test set, serve is the *most*
reliable class (95.9% recall, raw confusion only with smash, zero
confusion with volley or backhand -- see the confusion matrix in
`training/stroke_classification/README.md`). Backhand recall on real
footage (7/9 = 78%) is close to its THETIS test recall (83.8%) --
consistent with the known in-sample volley/forehand/backhand
confusability already documented in the README (`backend/app.py:220-222`
calls this out explicitly) -- but serve going from the *best*-performing
class in-sample to 0% on real footage is a distinct, new failure, not an
extension of a known weakness.

Hypothesis for *why* serve specifically breaks (not verified, flagged as
such): THETIS clips are pre-trimmed tightly around one stroke, so a
60-frame serve clip is close to prep-through-follow-through with little
else. Our windows are instead a fixed ±30-frame crop centered on a
peak-wrist-speed proxy pulled from a continuous, un-trimmed real video --
the framing/camera angle and the phase of the swing centered in the window
both likely differ from how THETIS was cropped and shot. Serve has the
most distinctive, singular motion shape of the 5 classes in THETIS, which
may be exactly why a shift in camera angle or window-centering hurts it
disproportionately versus classes with more visually generic swing shapes.
This is a hypothesis for a future investigation, not something this pass
verified.

**Cheap fallback rule checked: "wrist substantially above the head" as a
serve indicator -- does NOT hold up.** Using the wrist and nose landmarks
already computed (no change to `ir/biomechanics.py`), checked both at the
single contact frame and scanned across the full 60-frame window for every
candidate in all 5 clips. Result: wrist was above the nose at NO point, in
ANY window, for ANY candidate in the entire diagnostic -- including both
confirmed real serves. On clip1's serve (frame 235), the closest the wrist
ever gets to the nose across the whole 60-frame window is still 2.7% of
normalized frame height below it. The one exception (game4 frame 457,
wrist above nose) is one of the ambiguous/ungraded candidates, not a
confirmed serve. Root cause, from direct landmark inspection: the
peak-wrist-speed contact-frame proxy tends to land *after* the swing's
true highest point, during the fast downward part of the motion (the
racket is already snapping through contact by the time speed peaks) --
and even setting that aside, the WRIST joint itself doesn't reach much
above head height at a real serve's contact point; it's the racket head
(unmeasured -- no racket detection exists) that extends well above the
head, not the wrist. **Conclusion: this is not a viable fallback rule as
literally stated** -- not a promising direction to chase further without
a racket-tip estimate, which the pipeline doesn't have.

**No fix attempted.** Per explicit instruction, this pass is diagnosis
only. `ir/run_classifier_diagnostic.py` and its output
(`ir/example_output/classifier_diagnostic/`) are the artifacts; no model,
threshold, or biomechanics-feature code changed.

---

## 2026-08-03 — Phase E: visual design of the trial mobile app

**What was done:** Designed and applied a shared design system across all 5
trial-app screens (Home, Processing, Results, History, Settings) — no
backend/logic changes. Palette (deep navy primary, teal reserved for a
narrow positive signal, warm neutral background, amber caveats, red for
genuine errors only), a serif/sans type pairing built entirely from
platform-default fonts (`Platform.select({ios:"Georgia", android:"serif"})`
for headings, no override for body — zero new font dependencies), and a
4px spacing/radius scale, all centralized in `mobile/lib/theme.js` as the
single source screens import from (no per-screen inline hex codes).

Design-system proposal went through one real revision cycle before
implementation: the first artifact draft had the "Applied" phone mockups
inadvertently following the *document's* light/dark viewer theme (shared
CSS custom properties between document chrome and app mockups), which
made the proposed app look dark-themed when it was never meant to be —
fixed by pinning the mockups to fixed, non-theme-reactive tokens. Serif
rendering was verified for real on the `vt_test` Android emulator (not
just documented) before building any screen; cross-OEM Android font
variation and untested iOS rendering were flagged as known, unverified
risks in the design doc rather than assumed away.

Icon set for the tab bar (Ionicons, outline/filled by focus state) was
picked and shown before wiring in, then revised once on user feedback:
History's icon was `time-outline` (reads as duration/timer) in the first
pass, swapped to `list-outline` (reads as "a log of past results") after
the user flagged the mismatch; confirmed the active-tab tint is navy, not
teal, keeping teal's meaning undiluted.

Each screen was verified on-device against the real backend, not just
visually inspected:
- **Home** — confirmed "Pick from Library" genuinely opens the native
  picker with real device clips.
- **Processing** — all 5 hardened error states (`bad_video_file`,
  `no_player_detected`, `low_pose_detection`, network-unreachable,
  timeout) were triggered for real against a locally-run instance of the
  actual backend (temporarily repointed via `mobile/lib/config.js`,
  reverted byte-for-byte after), not just inferred from shared styling.
  `low_pose_detection` needed a purpose-built clip (real person-frames
  from `mvp/real_footage_test/clip1_serve_practice.mp4` interleaved with
  blank frames to land at 30% detection, between the 0% and 80%
  thresholds) since it's not easy to trigger with existing footage.
- **Results** — full pipeline run end-to-end (real clip → pose → StrokeGRU
  → coaching → feedback submission), confirmed in the backend's own logs.
  One deliberate design call: the confidence pill switches from teal to
  amber whenever the existing low-confidence/close-call caveat fires, so
  it never shows a positive-color signal next to a caveat telling the
  reader to distrust the number.
- **History** — list + detail verified against real on-device session
  data spanning both high- and low-confidence sessions (the low-confidence
  one incidentally confirmed the confidence-caveat and pose-detection-
  caveat boxes render correctly together, not just individually).
- **Settings** — role toggle re-verified functionally after restyling;
  added the placeholder consent/data-handling section as a visually
  distinct dashed-border card with an amber "PLACEHOLDER -- NOT FINAL
  COPY" tag and bracketed insert points, so it reads as structurally
  ready rather than as real, final legal copy.

**Bug found and fixed along the way (not the original ask, but blocking
"looks intentional"):** History was rendering two stacked "History"
headers — the tab navigator's own header plus the nested Stack's header
for the same route, both present before this pass but only visually
obvious once themed. Fixed with `headerShown: false` on the outer tab for
that route, since the nested Stack already supplies its own themed header
for both the list and detail screens.

Also repointed `TRIAL_PASSCODE`-gated `BACKEND_URL` at
`http://10.0.2.2:8000` (or other unreachable/black-hole addresses)
multiple times during this pass purely to trigger specific error/success
paths against a locally-run backend instance — each time reverted and
confirmed byte-identical to the original Tailscale Funnel URL afterward,
including one real request run through the actual public URL (not a
local override) to confirm the revert was functionally correct, not just
textually matching.

---

## 2026-08-04 -- Metro/Expo Go off-LAN reachability: Tailscale Funnel diagnosed as fundamentally incompatible, Cloudflare quick tunnel adopted (verification interrupted, not completed)

Backend reachability off-LAN was already solved via Tailscale Funnel
(2026-08-01 entry above). This entry is the separate problem of getting
**Metro's dev server** (not the backend) reachable from a phone on a
different Wi-Fi network than the PC, so a fresh Expo Go connect works for
an in-person supervisor demo.

**Root cause found, not assumed.** `npx expo start --tunnel` (the
official path) failed outright: Expo's bundled `@expo/ngrok` requires an
authenticated ngrok account (`ERR_NGROK_4018`), confirmed by running
`ngrok http 8081` directly. Did not create an account on the user's
behalf; user chose Tailscale Funnel as the alternative (second Funnel
port, 8443, alongside the existing backend Funnel on port 443, using
`EXPO_PACKAGER_PROXY_URL` to fix Metro's manifest bundle-URL port --
verified via direct manifest/bundle `curl` fetches, both 200 OK).

Despite the manifest and bundle both being fetchable via `curl`, a real
Expo Go connect (via `adb shell am start -d "exp://...:8443"`, and later
via Expo Go's own "Enter URL manually" UI, both after a `pm clear` for a
genuinely fresh app state) consistently failed with
`java.io.IOException: Failed to download remote update` /
`Http1ExchangeCodec.readResponseHeaders`. Isolated the real cause by
testing the identical flow against Metro over plain LAN
(`exp://10.0.2.2:8081`, no funnel involved) and reading the resulting
error text directly: **"Packager is not running at `http://<host>:8443`"**
-- Expo Go's own connect flow always precedes the real bundle fetch with a
**plain-HTTP** reachability check against whatever host:port is given,
regardless of URL scheme (`exp://`, `exps://`, or a manually-typed
`https://` URL -- none of these change Expo Go's own precheck; `https://`
typed directly isn't even routed to Expo Go at all, since Expo Go only
registers the `exp`/`exps` intent schemes, not `https`). **Tailscale
Funnel serves HTTPS/TLS only on port 8443 -- it has no plain-HTTP
listener -- so this precheck always fails**, before Expo Go ever attempts
the (real, working) HTTPS fetch underneath. Confirmed via Metro's own log
staying at zero new requests across every failed connect attempt: the
request never left the phone. This is a hard incompatibility between Expo
Go's client logic and any HTTPS-only tunnel, not a config mistake on our
side, and isn't fixable from the Metro/backend side.

**Decision (user-selected, not defaulted):** switch to `cloudflared`'s
free "quick tunnel" (`cloudflared tunnel --url http://127.0.0.1:8081`, no
account required, already installed locally) as the Metro tunnel
mechanism instead of Tailscale Funnel. Got a live quick-tunnel URL
(`https://<random-words>.trycloudflare.com`) and confirmed it resolves
and serves 200 OK externally (via Google DNS `8.8.8.8` directly, since
this machine's default resolver couldn't resolve `trycloudflare.com` --
a local DNS-config issue, not a tunnel problem). **Not yet re-verified
whether Cloudflare's quick tunnel avoids the same plain-HTTP-precheck
problem** (plausible it does, since `trycloudflare.com` is a hostname
Expo's client-side tooling may special-case, unlike an arbitrary
Tailscale hostname -- not confirmed either way yet). Metro was mid-restart
against the new tunnel URL (`EXPO_PACKAGER_PROXY_URL` pointed at the
Cloudflare URL) when this session's focus shifted to Phase 1 (cloud
migration, see below) at the user's explicit direction. **Left in an
honest incomplete state, not claimed as working:** the Cloudflare tunnel
process is up and externally reachable, but a real fresh Expo Go connect
through it has not been tested. Picking this back up should start with
that test, exactly as this whole investigation insisted on testing
Tailscale Funnel for real rather than assuming it would work.

---

## 2026-08-07 -- Phase 1 (cloud migration) kicked off: hosted-provider options researched, Fireworks AI selected, backend wired (untested pending API key)

Started Phase 1 of the cloud-migration plan (roadmap Phase 4's "cloud
migration" item, see `visiotennis_roadmap.md` -- now marked IN PROGRESS).
Confirmed by reading the code, not assumed: the FastAPI backend's core
pipeline (onnxruntime `CPUExecutionProvider`, MediaPipe pose) has **no GPU
dependency at all** -- only Qwen3 8B via Ollama needs the GPU, and
`OLLAMA_HOST` was already read from an env var, meaning the LLM host was
already decoupled from the FastAPI process in code before this phase
started.

**Options presented to the user (not provisioned -- explicit constraint:
no signups or payment info handled on the user's behalf):** cloud GPU
hosting (self-managed, always-on ~$150-400/month, or serverless/autoscale
~$5-30/month with added cold-start latency) vs. swapping to a hosted
inference API (Claude via the existing untested `ClaudeBackend` stub, or a
hosted Qwen3 8B via an inference provider). User picked hosted Qwen3 8B via
an inference provider -- lowest infra/cost, smallest step away from the
already-validated reliability numbers.

**Provider selection -- checked live, not assumed.** Of the 4 candidates
named (Groq, Together.ai, Fireworks.ai, DeepInfra), **only Fireworks AI
actually hosts Qwen3 8B** as of this writing -- Groq only has 32B
(deprecated) and 27B; Together and DeepInfra have both moved their
catalogs to newer Qwen3.5/3.6/Max variants. Full comparison, sourcing, and
the honest quantization caveat (no hosted provider offers literal Q4_K_M;
Fireworks serves FP8, which is *higher* precision than local Q4_K_M, not
an exact match) written up in `rag/DESIGN.md` ("Hosted-provider choice for
cloud migration -- RESOLVED: Fireworks AI"), including real sourced
numbers: $0.20/M input + $0.20/M output tokens, 0.64s TTFT / 144.8 tok/s
(Artificial Analysis' independent cross-provider benchmark, not
Fireworks' own marketing).

**Code wired, explicitly not yet trusted.** `rag/llm.py` gained
`OpenAICompatibleBackend` (provider-agnostic -- base_url + model only,
since Fireworks/Together/DeepInfra/Groq all speak the same OpenAI
chat-completions wire format, so this one class covers any of them if
ever needed again). `backend/app.py` startup now selects the LLM backend
via `LLM_BACKEND` env var (`ollama` default, `fireworks` opt-in via
`FIREWORKS_API_KEY`) -- restarted the backend and confirmed the **default
path is unchanged**: `llm_backend: "ollama:qwen3:8b"`,
`llm_reachable: true`. `/api/v1/health` now reports which backend is
actually configured (`llm_backend` field) and checks reachability
correctly for either backend type (previously hard-coded to Ollama's
`/api/tags`, which would have thrown `AttributeError` against a
non-Ollama backend -- fixed before it could ever fire).

**Confirmatory re-run script written and verified to fail cleanly without
a key** (`rag/run_fireworks_confirmatory.py`): a smaller re-run than the
original 6-trial + 6-diverse-session evaluation -- 3 trials (configurable)
against the same repeated-session prompt `run_gated_multitrial.py` used,
plus one diverse-session prompt (`session_record_diverse_serve_breakdown.json`)
as an independent data point, through the identical gated pipeline. Not
yet run for real: blocked on the user creating a Fireworks account and
handing over an API key (their explicit ask -- account creation and
payment info stay on their side, never handled here).

**Unrelated port-8000 conflict found and handled correctly, not
overridden.** While restarting the local dev backend for testing, found
port 8000 occupied by a different FastAPI app ("Tennalysis API" -- an ATP
match-prediction project, unrelated code, unrelated conda env). Did not
assume it was stale or kill it; asked the user first. Confirmed it's the
user's own separate, still-needed project -- moved the VisioTennis dev
backend to port 8001 instead, left the other process untouched. Purely a
local-dev-port choice; does not affect the eventual cloud-hosted
`BACKEND_URL`, which supersedes any local port once Phase 1 step 4
(deployment) lands.

**Not yet done, explicitly:** the confirmatory re-run itself (blocked on
the API key), choosing/configuring the actual deploy host for the FastAPI
app (Fly.io/Render-class, ~$5-20/month, deliberately not locked in before
Fireworks is confirmed good), and updating `BACKEND_URL` in the mobile app
and web frontend (blocked on both of the above).

---

## 2026-08-07 -- Phase 1 (cloud migration) PAUSED, not abandoned: no budget for Fireworks, reverted to local Ollama + Tailscale Funnel

**Why paused.** Fireworks AI's `/models` endpoint returned "Account
[user-account] is suspended, possibly due to reaching the monthly spending
limit or failure to pay past invoices" when the user's real API key was
tested (real request, not assumed -- see previous entry for the
model-not-found red herring this produced on the chat-completions
endpoint first, before the real cause surfaced on `/models`). Checked live
whether this was resolvable for free: Fireworks' new-account trial credit
is $1 (confirmed via live search, not memory), which would have covered
this project's tiny confirmatory-eval token volume, but the suspension
itself gates on a verified payment method being on file, not the trial
credit balance. **User decided: no budget for this right now** -- paused,
not a technical dead end.

**Revert, verified for real, not assumed:**
1. `LLM_BACKEND` confirmed defaulting to `ollama` in the running process
   (health endpoint: `"llm_backend": "ollama:qwen3:8b"`,
   `"llm_reachable": true`) -- no Fireworks key anywhere in the active
   path. The `OpenAICompatibleBackend` code and `LLM_BACKEND=fireworks`
   opt-in path stay in the repo, unused, exactly as designed for this
   scenario (config-only switch, not a refactor to undo).
2. Cloud FastAPI hosting step skipped entirely, as instructed -- never
   actually deployed anywhere during Phase 1 (deliberately held off
   pending the Fireworks confirmatory result, see previous entries), so
   there was nothing to revert here.
3. **Found and fixed a real, active problem, not just confirmed the
   status quo:** Tailscale Funnel's main URL
   (`https://visiotennis-trial.tail458008.ts.net`) was still forwarding to
   `127.0.0.1:8000` -- but port 8000 now belongs to the user's unrelated
   "Tennalysis API" project (confirmed with the user in the prior
   port-conflict exchange, left untouched), meaning the public URL was
   silently misrouted to the wrong app. Fixed via
   `tailscale funnel --bg 8001` (repoints the *internal* target only --
   the external hostname is byte-identical, confirmed via
   `tailscale funnel status` before/after). `mobile/lib/config.js`'s
   `BACKEND_URL` was never touched during Phase 1 and still points to the
   same hostname, confirmed by reading the file directly. **No separate
   "web frontend" exists anywhere in this repo** -- searched the full
   tree; only `mobile/` (the Expo app) references `BACKEND_URL` or the
   Funnel hostname. Flagged to the user rather than silently skipped or
   fabricated.
4. **Real end-to-end pipeline test through the (now-correct) public Funnel
   URL**, using real footage (`mvp/real_footage_test/clip1_serve_practice.mp4`,
   288 real frames, pose_detection_rate=1.0): stroke classification
   (volley, 72.3% confidence) and biomechanics extraction succeeded
   cleanly (HTTP 200, ~204s). **First attempt's coaching-generation step
   hit the gated pipeline's own documented failure path** (3/3 attempts
   failed Pydantic validation on the citation's `record_type`/`record_id`
   fields -- the same class of schema-malformation failure mode already
   recorded in `rag/DESIGN.md`'s original 6-trial Qwen3 evaluation and
   confirmed firing for real once before, see the `rally_style_shift`
   entry above) -- reported honestly as a partial result, not glossed
   over, and not treated as a regression from the Fireworks work (no
   coaching-path code changed by that work; this is the documented,
   expected occasional gate behavior on real generation variance). A
   second real request was run rather than stopping on the first result
   -- **this one succeeded cleanly: `success=true`, 1 attempt, 34.4s,
   3/3 grounded_observation items, all correctly citing real
   `StrokeEvent` biomechanics-percentile fields** (e.g. wrist height
   ratio at the 98.7th percentile for volley clips). Feedback submission
   (`POST /api/v1/feedback`, real `event_id` from that response) also
   confirmed working (`{"status": "recorded"}`, HTTP 200). Read together,
   the two runs are exactly what the retry-gate is supposed to look like
   in the wild: one real, on-the-record instance of its documented
   failure path firing, immediately followed by a clean success on retry
   -- consistent with, not contradicting, the original evaluation's
   reliability profile, and confirms the full pipeline (stroke
   classification -> biomechanics -> gated coaching -> feedback) works
   end-to-end through the reverted local + Funnel setup.
5. This entry.

**Explicitly not lost:** `rag/llm.py:OpenAICompatibleBackend`, the
Fireworks-specific config plumbing in `backend/app.py`, the live
4-provider comparison in `rag/DESIGN.md` ("Hosted-provider choice for
cloud migration"), and `rag/run_fireworks_confirmatory.py` all stay in the
repo as-is. Resuming Phase 1 once there's budget should need only a valid
Fireworks API key and `LLM_BACKEND=fireworks` -- no re-research, no
re-coding.

## 2026-08-07 -- KB tone regression fixed (real prompt-template bug, not stale cache); latency profiled and explained (retry-gate attempt count, not a system regression)

**Trigger**: a real analyze-stroke result showed old-style technical phrasing
("your wrist height is at the 98.7th percentile...") instead of the
player-facing tone approved in the 2026-08-01/08-02 KB rewrite, plus a
109.4s total latency far above the stated 26-50s baseline. Both flagged as
urgent, both investigated for real against the live backend, not assumed.

**Issue 1 -- KB tone regression, root cause confirmed NOT a stale build/cache:**
- Verified the running backend process (pid 24244, started 2026-08-07
  14:27:24) started well after both `rag/kb_placeholder.py` (mtime 08-02
  13:54) and `rag/kb_biomechanics.py` (mtime 08-02 13:54) were last
  modified, and `rag/__pycache__/kb_*.cpython-312.pyc` were compiled at
  08-02 13:55 -- in sync with source, not stale. No duplicate copies of
  either KB file exist anywhere else in the repo (checked). Ruled out.
- Reproduced the regression live: `POST /api/v1/analyze-stroke` against
  the real local backend returned `"Your wrist height is at the 98.7th
  percentile for volleys, indicating a very high contact point..."` --
  exactly the old-style phrasing, with real citation data
  (`biomechanics_percentile.wrist_height_ratio`, `cited_value: 98.7`).
- Real root cause: `rag/prompt.py`'s `build_schema_instructions(...,
  include_percentile_rule=True)` (used only by the stroke/biomechanics
  path, `rag/stroke_prompt.py`'s `STROKE_SCHEMA_INSTRUCTIONS`) generated a
  schema rule (rule 8) whose own worked EXAMPLE told the model to write
  `"your contact height is at the 80th percentile for this stroke type"`
  -- i.e. the prompt template itself re-taught the exact percentile-first
  technical phrasing the KB rewrite was trying to eliminate, and did so
  with more authority (a hard schema rule) than the KB snippets get (rule
  6 only said the model "may use" them "for tone"). The KB rewrite was
  real and correct; the schema instructions around it were never updated
  to match and silently overrode it on every stroke-level generation.
- Confirmed via `rag/grounding.py` that the grounding harness never
  inspects `item.text` for `grounded_observation` items (only scans
  `general_note` for a stray digit) -- so the fix is safe and cannot
  affect retry-gate pass/fail mechanics.
- Fix (`rag/prompt.py`): rewrote rule 8's example and added an explicit
  instruction that the citation JSON encodes the percentile number but
  the prose `"text"` must never spell out "percentile"/"ratio" or a raw
  percentile number -- describe direction in plain language ("unusually
  high", "on the higher end") like the KB snippets. Strengthened rule 6
  from "may use the snippets for tone" to "match the plain, direct,
  player-facing tone of the snippets in every text sentence."
- Backend restarted (still on port 8001, Tailscale Funnel repoint to 8001
  confirmed unchanged after restart via `tailscale funnel status`) and
  re-verified against a REAL result (not just the grounding harness):
  `"Your contact point is unusually high for this shot, which could
  indicate a flatter swing path. If it feels natural, there's no issue
  here."` / `"Your right knee bend at contact is on the higher end for
  this shot type..."` / `"Your weight shift into the shot is on the
  higher end for this stroke..."` -- no raw percentile/ratio jargon in any
  item's text, citation JSON still carries the exact percentile
  (`cited_value: 98.7`, `97.0`, `97.0`) unchanged, `n_attempts: 1`,
  `success: true`.

**Issue 2 -- latency profiled per-stage; 109.4s explained, not a regression:**
- Added real per-stage timing to `backend/app.py`'s `/api/v1/analyze-stroke`
  (`diagnostics.stage_timings_s`: `video_decode`, `pose_extraction`,
  `stroke_classification`, `biomechanics_extraction`,
  `prompt_build_and_retrieval`, `llm_generation`) and exposed each retry
  attempt's individual latency (`coaching.attempt_latencies_s`, from
  `rag/gated_pipeline.py`'s already-tracked `AttemptRecord.latency_s`,
  previously computed but never surfaced past the aggregate
  `total_latency_s`).
- Ran 3 real end-to-end requests post-fix against the local backend:
  32.1s, 40.4s, 26.8s total LLM-generation time, all `n_attempts: 1`.
  Non-LLM stages are consistently small: video decode 1.7-3.7s, pose
  extraction ~1.4s, stroke classification and biomechanics extraction both
  ~0ms, prompt build + KB retrieval (embedding calls) 2-8s. LLM generation
  alone is 85-95% of total latency in every run, and its single-attempt
  cost (27-40s) matches the previously-stated 26-50s baseline exactly --
  no evidence of a slowdown in the model or the GPU (`nvidia-smi`: RTX
  3060, 6977/12288 MiB used by the two resident Ollama models, 15% util,
  no contention found).
- The 109.4s flagged result is explained by the retry-gate
  (`rag/gated_pipeline.py`, `max_attempts=3`) needing more than one
  generation attempt, not a per-generation slowdown: a 2026-08-07 test run
  earlier this session logged `total_latency_s: 127.67` for `n_attempts:
  2`, and a separate real Funnel test logged 203.97s total (consistent
  with 3 attempts at this per-attempt cost). 109.4s sits squarely between
  the 1-attempt (~27-40s) and 2-attempt (~128s) real data points measured
  today. This is the retry-gate's documented, intentional trade
  (`rag/gated_pipeline.py`'s own docstring: "trades latency (extra
  generations on failure) for a hard guarantee") -- working as designed,
  not a bug -- but it was previously invisible to the client, since only
  the aggregate total was ever returned. `attempt_latencies_s` now makes
  this directly verifiable per-request instead of inferred.
- Not fixed (out of scope of this investigation, flagged for awareness):
  the retry-gate's variance IS the latency variance the user is seeing
  session to session. Reducing worst-case latency would mean either
  raising the KB/prompt/schema reliability so fewer attempts are needed
  (attempt failures are Pydantic/grounding validation misses, see
  `rag/gated_pipeline.py`'s module docstring for known failure rates), or
  lowering `max_attempts` at the cost of a higher outright failure rate --
  a product tradeoff, not something to decide unilaterally here.

## 2026-08-07 -- Retry-gate latency: decided to keep max_attempts, add honest progress messaging instead

**Decision**: given the latency-profiling data gathered earlier today (1
attempt ~27-40s, 2 attempts ~110-130s, 3 attempts ~200s, all consistent
with `rag/gated_pipeline.py`'s `max_attempts=3` retry-gate, not a system
regression), `max_attempts` stays at 3. Reliability (never showing an
ungrounded/malformed response) is kept as-is rather than traded for
speed. Instead, `mobile/app/processing.js` now sets honest expectations
during the wait via progress messaging, so the Processing screen reads as
the system doing something deliberate rather than stalling.

**Why**: lowering `max_attempts` would raise the outright-failure rate
(more sessions ending in "coaching unavailable" instead of a slow-but-
correct result) purely to shave worst-case latency, and the retry-gate's
whole purpose (see its own module docstring) is trading latency for a
hard grounding guarantee -- undoing that for a UI complaint would be
solving the wrong layer. A generic spinner with no context was the actual
problem: 109s of silence reads as broken, 109s with an explanation reads
as thorough.

**Implementation** (`mobile/app/processing.js`):
- On submit, shows `"Analyzing your stroke -- usually takes 30-40
  seconds"` (`INITIAL_STATUS_MESSAGE`), matching the real measured
  single-attempt range.
- A `setTimeout` at `RETRY_STATUS_THRESHOLD_MS = 47000` (comfortably past
  the ~27-40s single-attempt range, well before a 2nd attempt would
  finish at ~110-130s) switches the message to `"Double-checking the
  coaching feedback for accuracy -- this can take a bit longer"`
  (`RETRY_STATUS_MESSAGE`) if the request is still in flight at that
  point -- i.e. a retry is very likely already underway. Cleared
  alongside the existing request-timeout `setTimeout` on both the
  success and error paths so it never fires after the screen has already
  moved on.
- Deliberately does NOT expose `n_attempts`, retry counts, or any other
  internal retry-gate mechanics -- both messages stay in plain,
  player-facing language, consistent with the same 2026-08-01/08-02 KB
  tone rewrite and today's earlier prompt-template fix (see the KB tone
  regression entry above).

**Verified for real, not assumed** -- on the `vt_test` Android emulator
(booted fresh via `emulator -avd vt_test`, Expo Go app data cleared with
`pm clear host.exp.exponent` to drop a stale cached LAN-IP manifest from
a 2026-08-02 session that was otherwise blocking reconnection, then
relaunched against Metro via `adb reverse tcp:8081 tcp:8081` +
`exp://127.0.0.1:8081`): pushed `clip1_serve_practice.mp4` to
`/sdcard/Movies`, ran it through the real Home -> Pick from Library ->
Processing -> Results flow against the live backend over the Tailscale
Funnel. Screenshots confirm the exact sequence: `"Analyzing your stroke
-- usually takes 30-40 seconds"` on submit, then (real request, not
forced) `"Double-checking the coaching feedback for accuracy -- this can
take a bit longer"` once the retry-gate's first attempt had clearly
finished and a 2nd attempt was underway, then a clean Results screen
(volley, 52% confidence, 3 grounded coaching items in player-facing
language, e.g. "Your weight transfer into the volley is on the higher
end for this shot type...", 87% pose detection, 43.4s diagnostic
latency for that stage) -- the message transition happened on a real
2-attempt run, not a simulated delay.

**Not changed, flagged for revisit**: this is a product tradeoff, not a
final answer. If first-attempt reliability improves later (e.g. from
prompt/schema tuning, or a stronger model), worst-case latency drops on
its own and this messaging becomes less necessary -- revisit whether
`max_attempts` and/or these thresholds still make sense at that point,
rather than assuming today's numbers hold indefinitely.

## 2026-08-07 -- Accounts + backend-persisted history: two decisions recorded before building

Two explicit decisions requested on record before starting the accounts
build (Phase 2 of the roadmap), per the standing rule that a deliberate
trial-stage gap must be a stated decision, not something a tester
discovers by getting locked out.

**Decision 1 -- password reset is deliberately OUT OF SCOPE for this trial.**
Fallback: admin-assisted reset -- a tester who's locked out contacts the
team directly, and the password_hash is updated via direct DB access.
Deliberately NOT "just re-signup with a new email" -- that would orphan
the tester's existing `stroke_history` under an account they can no
longer log into, which defeats the entire point of tying history to an
account. Reasoning: the tester group is small and known (the user's own
framing), there is no email-sending infrastructure anywhere in this
project (no SMTP/mail service configured), and building a full
forgot-password email flow is disproportionate infrastructure for a
handful of testers the developer can already reach directly. Revisit if
the tester pool grows past "small and known," or once real hosting +
email infra exist.

**Decision 2 -- the anonymous instrumentation logs (`backend/logs/*.jsonl`)
stay PERMANENTLY and deliberately separate from the new `users`/
`stroke_history` tables, not just "temporarily separate while both are
built."**
`log_session_metrics()`'s own existing docstring already states its
anonymous-by-construction design was "per explicit instruction" -- a
prior deliberate privacy call, not an oversight to eventually fix.
Cross-referencing `user_id` into those calibration logs would mean using
real account data for a purpose (aggregate model-calibration analysis)
that was never disclosed to a tester at signup, and the trial's consent
copy is itself still an unapproved placeholder
(`mobile/app/(tabs)/settings.js`: "PLACEHOLDER -- NOT FINAL COPY...
[Insert approved language here on: who can access trial data...]").
Joining identifying account data into those logs is a real product/
consent decision this project has not made -- not a technical wiring
change -- so the correct default is permanent separation unless a
future, explicit, informed-consent decision says otherwise.

## 2026-08-07 -- Phase 2 (accounts + backend-persisted history) built and verified end-to-end

Full build per the approved plan (see the prior "Plan: Accounts +
Backend-Persisted History" report and the two decisions logged above).
Built against the local backend only, per the paused Phase 1 cloud
migration -- see that note's own entry for what changes once cloud
hosting resumes (still accurate: DB file needs a persistent volume/managed
DB, CORS needs tightening, no signing-key secret to migrate since the
opaque-token design was chosen specifically to avoid that).

**Backend** (backend/db.py, backend/auth.py, wired into backend/app.py):
- SQLAlchemy + SQLite (backend/app.db) -- both already installed in the
  Python env (checked via direct import before writing any code), so this
  needed zero new backend dependencies. users / auth_tokens /
  stroke_history tables, deliberately separate from the existing
  backend/logs/*.jsonl anonymous instrumentation logs (per the decision
  logged above).
- Email + password (bcrypt), opaque secrets.token_urlsafe(32) session
  tokens stored server-side (30-day TTL, lazily deleted on first expired
  use) -- not JWT, no signing-key material to manage. Email format checked
  with a plain regex rather than pydantic's EmailStr, to avoid adding the
  email_validator dependency for a trial-scale form.
- New endpoints: POST /api/v1/auth/{signup,login,logout},
  GET /api/v1/auth/me, GET /api/v1/history,
  GET /api/v1/history/{id} -- all still behind the existing
  TRIAL_PASSCODE middleware (unchanged, layered under the new auth, not
  replaced by it).
- POST /api/v1/analyze-stroke now requires auth (Depends(get_current_token))
  and uses the authenticated user's id as StrokeEvent.player_id (previously
  a hardcoded "mobile_test_user" string from the mobile client -- the
  player_id field existed in the schema for exactly this purpose and was
  unused until now). A StrokeHistory row is now written automatically
  inside this same request, atomically with the analysis, keyed by the
  StrokeEvent's own event_id (no second id minted) -- deliberately not a
  separate client-initiated "save to history" call, so a client that
  crashes/backgrounds right after receiving the response can't silently
  lose the session.
- Real bug found and fixed during smoke-testing (not assumed away):
  SQLite's DateTime column strips tzinfo on read, so a value written as
  tz-aware datetime came back naive on the next request, and comparing it
  against a fresh datetime.now(timezone.utc) raised
  "TypeError: can't compare offset-naive and offset-aware datetimes" on
  every /me and /history call. Fixed with a single utcnow() helper
  (backend/db.py) used everywhere a DB timestamp is written or compared,
  keeping everything naive-but-UTC consistently. Relatedly: JSON timestamps
  sent to the client now get an explicit "Z" suffix appended
  (row.created_at.isoformat() + "Z") -- without it, JS's new Date(...)
  on the client would have silently parsed the naive-UTC string as LOCAL
  time, shifting every displayed timestamp by the device's UTC offset.
- Smoke-tested directly via curl before any mobile work started: signup,
  duplicate-email 409, login, wrong-password 401, /me with/without token,
  logout + revoked-token 401, analyze-stroke 401 without auth, and a real
  authenticated analyze-stroke call confirmed player_id tied to the
  account and a matching stroke_history row created -- all before
  touching the UI.

**Mobile** (mobile/lib/api.js, AuthContext.js, rewritten
historyRepository.js, new app/login.js + app/signup.js, updated
app/_layout.js, app/processing.js, app/(tabs)/settings.js):
- One new dependency: expo-secure-store (~15.0.8, via expo install
  for SDK-54 compatibility) -- the session token lives there
  (Keystore/Keychain-backed), deliberately not the existing on-device
  SQLite settings table used for the anonymous role value.
- historyRepository.js: confirmed the "swappable seam" claim in its own
  prior docstring for real (grepped all call sites) before touching
  anything -- exactly 4 screens called only this module's exported
  functions, never expo-sqlite directly. listSessions/getSession
  bodies now call the backend instead of expo-sqlite, keeping identical
  return shapes; the one real deviation from "zero other changes needed"
  is that both gained a token parameter (backend-persisted history is
  per-account), so their 2 call sites now pass useAuth().token in.
  addSession() is gone entirely (server persists automatically now);
  getRole/setRole are untouched, still on-device, still deliberately
  anonymous.
- app/_layout.js: AuthProvider wraps the tree; Stack.Protected
  (expo-router 6 -- confirmed present in the installed
  node_modules/expo-router build before using it, per this project's
  mobile/AGENTS.md rule to check the actual installed API rather than
  assume) gates (tabs)/processing/results behind !!token and
  login/signup behind !token, with a loading screen while a stored
  token is validated against /me on startup (not just trusted).
- login.js/signup.js: built from theme.js tokens only, no new
  colors/type -- the app's first screens with real TextInput fields.
  Signup deliberately stays to email + password + optional name
  (playing_hand has no UI yet, since nothing reads it today -- adding
  that is a separate later change, not smuggled in here).
- processing.js: analyze-stroke request now sends
  Authorization: Bearer <token> alongside the unchanged trial-passcode
  header; the client-side addSession() call is gone (server persists
  automatically).
- settings.js: added an Account section (email + Log Out), matching the
  existing card/pill styling.

**Verified for real, on the vt_test emulator, not assumed:**
Signup through the actual UI (Emulator_Tester /
emulatortester@example.com) -> correctly landed on Home behind
Stack.Protected's auth gate -> Settings correctly showed the account
email -> Home -> Pick from Library -> a real analyze-stroke request went
through Processing (both status messages -- initial and the
2026-08-07-added retry message -- correctly observed live again) -> History
tab showed the real result (Volley, 52%, 8/7/2026, 9:23:34 PM,
matching the backend's created_at exactly once the device's UTC+5 offset
is accounted for) -> tapped into it, full coaching detail rendered
correctly from GET /api/v1/history/{id} -> force-stopped and relaunched
the app (am force-stop + relaunch, not just a JS reload) -> landed
straight on Home, no re-login required -> History still showed the same
entry, confirming both session persistence (expo-secure-store survives
an app kill, as it should) and backend persistence (re-fetched fresh from
backend/app.db, not a client-side cache) after a real restart.

**New finding, NOT fixed, flagged for follow-up:** 3 consecutive real
analyze-stroke requests through the Tailscale Funnel during this
verification session hit the client's REQUEST_TIMEOUT_MS (5 min) and
showed the "This is taking a while" error -- but the backend/app.db
history row for one of these exact requests shows success: true,
n_attempts: 1, total_latency_s: 36.25 (a fast, single-attempt, fully
successful generation). This means the server completed the request
normally while the client believed it had timed out -- a real client/
network-side discrepancy (most likely upload-transfer time over the
Funnel tunnel, which is NOT included in the server's own t0/timing
instrumentation, so it's invisible to every latency number this project
has measured so far) that is DIFFERENT from the already-documented
retry-gate latency variance (today's earlier "keep max_attempts, add
progress messaging" entry). Not investigated further today -- this
session's Ollama process was also observed fully unloaded
(ollama ps empty) shortly after, which was a live hypothesis at the
time but the timeline doesn't actually support it as the cause (the
successful request predates that observation). Worth a real investigation
before relying on REQUEST_TIMEOUT_MS = 5 minutes as a safe client-side
budget for the Funnel path specifically.

## 2026-08-08 -- Funnel timeout investigation: REQUEST_TIMEOUT_MS raised 5 min -> 7 min, root cause only partially resolved

Follow-up on the finding logged above, requested explicitly before moving
on to further work. Investigated with real, direct tests -- not assumed.

**1. Confirmed the actual value and compared against real data.**
`mobile/lib/config.js`: `REQUEST_TIMEOUT_MS` was `5 * 60 * 1000` = 300s.
Compared against this project's own measured retry-gate worst case: 3
attempts ~204s of LLM generation alone (logged 2026-08-07), plus ~3-10s
for pose/classification/biomechanics/prompt-retrieval -- a realistic
server-side worst case of ~215-225s, which is UNDER 300s by ~75-85s. By
the server's own numbers, the old timeout was not provably too short for
the documented retry-gate ceiling -- the original hypothesis ("if the
client timeout is shorter than the 3-attempt worst case, that's the bug")
does not hold up against the real measured data.

**2. Tested Funnel vs. local directly, with a controlled probe, not just
inference from analyze-stroke's own variance.**
- Local vs. Funnel comparison on a real `analyze-stroke` call: locally,
  curl's `time_total` matched the server's own
  `diagnostics.total_processing_time_s` almost exactly (98.65s vs
  98.58s -- no hidden overhead). Over the Funnel, curl's `time_total` was
  28.29s against a server-reported 23.05s -- a ~5.2s gap that matches the
  TLS handshake time (`time_appconnect` = 5.14s) almost exactly. Normal
  Funnel overhead is small and fully explained by TLS setup, not by
  transfer speed or added latency.
- Added a TEMPORARY `GET /api/v1/_debug_sleep?seconds=N` endpoint
  (removed immediately after this investigation, not left in the
  codebase) that just holds the connection open for exactly N seconds
  before responding -- a controlled way to test the Funnel's own
  connection-duration tolerance without depending on retry-gate
  randomness. Ran it over the live public Funnel URL at 30s, 60s, 100s,
  130s, 205s (the exact 3-attempt worst case), and 260s (87% of the OLD
  300s budget) -- every single one returned cleanly with `curl_total`
  matching the requested sleep duration almost exactly. **This directly
  disproves the hypothesis that the Tailscale Funnel imposes its own
  idle/total connection timeout shorter than what a legitimate
  worst-case request needs** -- the tunnel itself is not the bottleneck.

**3. What the user actually sees on a timeout, confirmed unchanged and
still accurate:** `processing.js`'s `AbortError` branch shows an honest,
non-misleading error ("This is taking a while... Check your connection
and try again") -- not silent data loss. But per the prior entry's
finding (still true, re-confirmed conceptually, not re-proven today): if
the server had already completed and persisted a `stroke_history` row for
that exact request before the client's abort fired, that successful
result sits in History without the user ever being shown it or told it's
there -- an invisible orphaned success, not data loss, but easy to miss.
Not fixed today (would need correlating an aborted request to a
just-created history row, a real design task of its own, not a
quick addition) -- flagged for a deliberate follow-up decision, same as
the items above.

**4. Fix applied: `REQUEST_TIMEOUT_MS` raised from 5 minutes to 7 minutes**
(`mobile/lib/config.js`), with the reasoning (real measured worst case,
the Funnel-duration test results, and the explicit non-claim that this
fixes a genuinely dead connection) written directly into the code comment
so the number stays justified in place, not just in this log. No change
needed to the retry-gate progress-message text (`processing.js`) -- both
messages are already duration-agnostic ("usually takes 30-40 seconds" /
"this can take a bit longer"), so they stay accurate regardless of the
exact timeout value.

**Re-verified with a real request through the Funnel, on the `vt_test`
emulator, logged back in as `test1@example.com`** (the emulator and Metro
had both stopped running since the last session and were restarted fresh
for this) -- one real `analyze-stroke` call via Pick from Library reached
Results successfully. Its backend `stroke_history` row shows
`n_attempts: 1`, `total_latency_s: 43.47` (normal single-attempt LLM
time), `total_processing_time_s: 51.34` -- a perfectly ordinary,
fast, successful server-side run. **But the wall-clock time observed on
the emulator itself (Processing screen shown at device-local 4:38, Results
shown at 4:43) was close to 5 minutes** -- meaning roughly 190s of real,
observed client-side time is NOT accounted for by the server's own
`total_processing_time_s`, and is NOT explained by anything ruled out in
steps 1-2 above (not the LLM/retry-gate, not Funnel idle-timeout, not
generic Funnel transfer overhead, which the earlier controlled test
showed is only ~5s). This run would very plausibly have hit the OLD
300s timeout and shown a false failure; it succeeded specifically because
of today's fix.

**Honest status: the fix is confirmed effective in a real run, but the
underlying root cause of the client-side gap is only partially
understood, not fully closed out.** Curl-based testing (which is how
steps 1-2 were verified) does not exercise the actual mobile app's
video-picker-to-upload code path at all, so it could not have caught
this -- the likely remaining candidates are video-file read/preparation
time from the picker's `content://` URI and/or the emulator's own
(known-to-be-slower-than-a-real-device) I/O and virtual network stack,
neither of which was isolated today. Do not assume the 7-minute budget is
proven sufficient on a real physical device over a real network until
this gap is actually explained -- it might be smaller on real hardware
(emulator-specific), or it might not be. Worth a real investigation
(e.g. instrumenting the exact moment `uploadVideo()` starts building
`FormData` vs. the moment `fetch()` is actually called) before trusting
this number long-term, but out of scope to chase further right now.

## 2026-08-08 -- History Insights charts built and verified on genuinely mixed real data

Built per the approved proposal (stroke-type breakdown, confidence trend,
one filtered biomechanics trend), with one addition from the approval
itself: the biomechanics chart's stroke-class filter is captioned, never
silent.

**Dependency**: `react-native-svg@~15.12.1` (SDK-54 pinned version, matched
against `expo/bundledNativeModules.json` rather than guessed) -- the one
new package, same footprint category as `expo-secure-store` last phase.
Installing it hit a real, pre-existing upstream conflict: `expo-router`
optionally bundles `vaul` (a web-only drawer component) which pulls in
`@radix-ui/*` React-DOM packages requiring `react-dom@19.2.8`, conflicting
with this pure-React-Native project's `react@19.1.0` (no `react-dom`
anywhere in this app). Unrelated to `react-native-svg` itself -- installed
with `--legacy-peer-deps` (npm's standard escape hatch for exactly this
class of transitive, never-actually-exercised peer conflict), then fixed
the resulting `^15.12.1` back to `~15.12.1` to match this project's
existing Expo-managed version-pinning convention.

**No charting library** -- hand-rolled two components on top of raw
`react-native-svg`/`View` primitives, styled directly from `theme.js`
tokens, per the approved proposal:
- `components/charts/StrokeBreakdownChart.js` -- plain `View`s, no SVG
  needed for bars.
- `components/charts/TrendChart.js` -- `Polyline` + `Circle`, with a
  min/max range caption underneath instead of a full axis (deliberately
  minimal for the 3-15 points this app's trial-stage session counts
  realistically produce).

**Data**: `lib/historyInsights.js` -- pure functions, no JSX, kept separate
so the grouping/filtering logic is easy to check on its own:
- `computeStrokeBreakdown` -- counts per class, sorted desc.
- `computeConfidenceTrend` -- chronological confidence values.
- `findMostCommonClass` / `computeWeightTransferTrend` -- filters to the
  user's single most common stroke class before charting
  `weight_transfer_ratio`, per the correctness issue found during the
  proposal stage (biomechanics values are percentile-ranked PER STROKE
  CLASS in `ir/biomechanics_percentiles.py`, so mixing classes on one raw
  trend line would silently splice together different baselines). Ties
  broken by whichever class's most recent session is more recent. Returns
  `strokeClass` alongside the values specifically so the screen can
  caption which class is being shown -- never filters silently, per the
  explicit approval condition.
- `historyRepository.js`'s `listSessions()` now also extracts
  `weightTransferRatio` per session -- free, since `/api/v1/history`
  already sends each row's full `result_json` to the client (confirmed
  before adding anything: no backend change was needed for this).

**Placement**: a new `InsightsSection` at the top of the existing History
screen (`app/(tabs)/history/index.js`), via `FlatList`'s
`ListHeaderComponent` -- not a new tab, not a nested scroll view. The
existing chronological session list below is unchanged.

**Empty/sparse-data gate**: single unified threshold,
`MIN_SESSIONS_FOR_INSIGHTS = 3` -- below it, one placeholder line instead
of the whole section; at or above it, all three charts render together
(never a half-populated section). A second, narrower threshold,
`MIN_SAME_CLASS_FOR_TREND = 3`, applies only to the biomechanics chart:
even with >=3 total sessions, if the dominant stroke class itself has
fewer than 3, that one chart alone shows a fallback line ("Not enough
sessions of one stroke type yet...") while the other two charts still
render normally off the full mixed-class session set.

**Real mixed-class test data**: found by testing, not guessed --
`mvp/real_footage_test/clip2_teen_practice.mp4` classifies as `backhand`
at 99.3% confidence (vs. `clip1_serve_practice.mp4`'s consistent
`volley`). Ran it twice more through the real local backend (authenticated
as `test1@example.com`) to bring that account to 7 real sessions: 4 volley
+ 3 backhand -- deliberately structured so the dominant-class filter
(volley, 4 vs 3) and the caption logic would both be genuinely exercised,
not just pass on a trivially single-class account. (Separately noted: a
real `[user-email]` account already existed with 2 real
backhand sessions from the user's own physical-device timeout-fix
verification in the prior session -- not used here since its password
isn't known to this session, but confirms real multi-class usage is
already happening outside of manufactured test data too.)

**Verified end-to-end on the `vt_test` emulator, logged in as
`test1@example.com`, against that real 7-session/2-class account** --
History correctly shows: Insights section present (7 >= 3), stroke
breakdown bar chart (Volley 4, Backhand 3), confidence trend (7 points,
"52% - 99%" caption matching the real min/max), and the weight-transfer
trend correctly filtered to the dominant class with the caption "Showing
your Volley sessions" (4 points, "0.73 - 2.78" caption matching the real
data) -- not the fallback text, since volley's own count (4) clears
`MIN_SAME_CLASS_FOR_TREND`. The unchanged "All sessions" list below still
shows all 7 real entries correctly. Nothing broken or half-populated.

**One real bug found and fixed during this verification, not shipped
unnoticed**: the biomechanics caption's `textTransform: "capitalize"`
style was applied to the whole sentence, rendering "Showing Your Volley
Sessions" (every word capitalized) instead of "Showing your Volley
sessions." Moved the capitalization to just the stroke-class name in the
string itself; re-verified live after the fix.

**Two environment-only gotchas hit during verification, not code bugs,
noted here since they'll recur**: (1) after the emulator process died
and was restarted (same intermittent non-persistence across turns seen
earlier in this project), the app reconnected before the port-forward was
re-established, silently fell back to a bundle cached from hours earlier,
and showed no Insights section at all -- not because the code was wrong,
but because the new code had never actually loaded. Force-stopping and
relaunching after confirming `adb reverse` was set fixed it. (2) Metro
itself then failed to resolve the brand-new `components/charts/` files
with "Unable to resolve module" even though they existed on disk exactly
where imported from -- Metro's own file-watcher had been running since
before those files were created and never picked up the new directory.
Fixed with `npx expo start --clear`. Both are worth checking first
(stale bundle vs. real code error) the next time something "isn't
showing up" after a background process gets restarted mid-session.

## 2026-08-08 -- Chatbot grounding architecture: design approved, turn-router + mechanical backstops built and verified standalone

**Design** (approved before any code, same discipline as the auth/accounts
design): a multi-turn stroke chatbot, scoped to ONE stroke's real data (not
cross-session history -- no retrieval mechanism exists to pick relevant past
sessions, and building one undesigned mid-build was explicitly ruled out).
Every turn is classified into one of 4 routes before any generation happens,
so the expensive retry-gate (`rag/gated_pipeline.py`, 27-200s+) only runs
when a turn genuinely needs it:

- `existing_citation` -- replay an already-grounded item from this session's
  real `CoachingResponse` verbatim. **Never generates new text** -- a wrong
  topic-match can only surface a real, already-verified (if slightly
  off-topic) answer, never a fabricated one. That bounded failure mode is
  why an imperfect matcher is acceptable to ship.
- `new_grounded_claim` -- needs the full retry-gate. Also the fail-safe
  default whenever classification is uncertain, unparseable, or an
  `existing_citation` claim doesn't check out mechanically.
- `general_note` -- pure generic advice, no personalization implied.
- `unavailable_comparison` -- sounds personalized/comparative but isn't
  groundable here (cross-session trends, "what's normal for my level").
  Gets a **fixed, code-authored disclaimer**, never one the LLM is trusted
  to remember to write itself -- same "don't trust the model's self-report"
  philosophy as `rag/grounding.py`.

**Built and verified standalone, before any UI or prompt-path code
exists** (`rag/schema.py`'s new `ChatRouteDecision`, `rag/chat_router.py`,
`rag/run_chat_router_trials.py`):

1. *Real-model routing trials* -- 8 test cases (the 6 documented in the
   design proposal + 2 more) against real `qwen3:8b` via Ollama, scoped to
   one real StrokeEvent's real citations
   (`qwen3_gated_stroke_trials/backhand_result.json`, event_id
   `6bc75eb4-...`). First pass: 5/8 correct. Added 3 worked examples to the
   router prompt targeting the exact 3 misses (generic-advice-after-a-topic,
   ambiguous-referent-vs-benchmark-request, cross-session-comparison-vs-new-claim)
   -- re-ran: 7/8 correct. The one remaining miss (`"was that normal?"`
   after two knee-flex topics were just discussed) routed to
   `unavailable_comparison` instead of the intended `new_grounded_claim`;
   defensible either way (`"normal"` is itself a benchmark-flavored word),
   and -- critically -- still resolves to a SAFE outcome via the backstop
   below, not a fabricated answer. Not chased further to avoid overfitting
   the prompt to a hand-picked test set; qwen3:8b's imperfect reliability on
   an identical prompt is already a documented property of this model (see
   `rag/gated_pipeline.py`'s own module docstring).

2. *Mechanical backstop #1* (existing-citation field-path verification) --
   proven with a scripted fake backend that deliberately claims a match to
   `"biomechanics.elbow_angle_deg"`, a field that was never actually cited
   for this stroke. `classify_turn()` correctly downgraded it to
   `new_grounded_claim` rather than trusting the claim -- same pattern as
   `grounding.py`'s `WRONG_RECORD` check.

3. *Mechanical backstop #2* (tier-3 misclassification safety net,
   `check_tier3_response_for_smuggled_claims`) -- tested against the same
   standard as the original retry-gate's fabricated-citation test
   (`rag/example_grounding_check.py`'s hand-written adversarial data), not
   just trusted to exist: (a) a genuinely clean `general_note` -> correctly
   allowed through (no false-positive escalation); (b) a `general_note`
   deliberately smuggling the player's real `0.147` `wrist_height_ratio` as
   an uncited prose claim -> correctly escalated; (c) a tier-3 response
   containing a full fabricated `grounded_observation` citation -> correctly
   escalated (automatic `WRONG_RECORD`, since tier-3 is checked against an
   EMPTY `source_records` -- nothing is legitimately citable on that tier).
   Reuses `check_grounding()` completely unchanged; no new verification
   logic was needed. Also ran one real-model adversarial attempt (a question
   baking the real `0.147` value into "in general, is X..." phrasing,
   designed to tempt a `general_note` misroute) -- the real router actually
   resisted it and correctly routed to `existing_citation` both times, so
   nothing was there for the backstop to catch on that specific attempt;
   the deterministic case (b) above is what proves the catch mechanism
   itself, independent of whether any single real attempt happens to trip it.

**Not yet built** (at that checkpoint): the three prompt paths and the chat
UI. Per explicit instruction, reported back after the classifier + backstops
were verified, before building on top of them.

## 2026-08-10 -- Chatbot: three prompt paths, chat endpoint, and UI built and verified end-to-end on the emulator

**Prompt paths** (`rag/chat_pipeline.py`, new; `rag/stroke_prompt.py` and
`rag/prompt.py` extended):
- Tier 1 (`existing_citation`): pure template-stitch, `_find_existing_item()`
  looks up the matched citation's stored `CoachingFeedbackItem` and replays
  it verbatim -- confirmed no new generation call happens on this path.
- Tier 2 (`new_grounded_claim`): `build_stroke_chat_prompt()`, the multi-turn
  analog of `build_stroke_prompt()` -- same citable view/notable-percentile
  selection/KB retrieval, plus conversation history, through the *unchanged*
  `generate_grounded_response()` retry-gate. Needed `build_schema_instructions()`
  to grow an `item_count_rule` parameter (default value byte-identical to the
  original hardcoded string, so every existing caller's prompt is unaffected)
  since a one-question chat reply needs 1-3 items, not the original 3-6.
- Tier 3 (`general_note`/`unavailable_comparison`): a bare single-shot prompt
  with no citation mechanism offered at all. Given its own small retry budget
  (`TIER3_MAX_PARSE_ATTEMPTS = 2`) after discovering, via
  `rag/run_chat_pipeline_trials.py`, that qwen3:8b hit the *exact* "field
  omitted on a trailing item" parse failure already documented in
  `gated_pipeline.py`'s own docstring on both real tier-3 test messages --
  without this, every such slip would escalate a plain formatting error to
  the expensive tier-2 gate, defeating tier 3's entire cost purpose.
  `unavailable_comparison` prepends the fixed `UNAVAILABLE_COMPARISON_DISCLAIMER`
  string (never LLM-generated).
- `handle_chat_turn()` orchestrates: classify -> dispatch -> (tier 3 only)
  run `check_tier3_response_for_smuggled_claims()` and escalate to tier 2 on
  failure. Verified end-to-end against a real StrokeEvent + real Ollama
  backend for all 4 outcomes before any UI existed (`run_chat_pipeline_trials.py`):
  3/4 live cases matched the expected tier+route on the first pass; the lone
  live mismatch was the same documented classifier-consistency noise from the
  prior checkpoint (not a pipeline bug). A deterministic forced-route test
  (a fake backend that scripts only the router's answer, letting every other
  call hit the real model) separately proved the `unavailable_comparison`
  disclaimer-prepending dispatch path itself, independent of which way the
  classifier happens to lean on a given run.

**Backend**: `POST /api/v1/history/{history_id}/chat` (`backend/app.py`) --
deliberately stateless server-side per the approved design (chat memory lives
only in client React state); the client resends `conversation_history` each
turn. Returns 409 `no_coaching_available` if the session's original coaching
generation failed (nothing for tier 1 to replay, no real citations to build
router context from).

**Mobile UI**: `mobile/components/StrokeChat.js` (new) + `mobile/lib/chatRepository.js`
(new), wired into both `app/results.js` and `app/(tabs)/history/[sessionId].js`
via `ResultView`'s existing `children` extension point (same pattern
`FeedbackForm` already used) -- gated on `coaching.success`, matching the
backend's 409 case. Three bubble styles keyed off the response `route`
(grounded teal+citation / plain general / amber caveat with the disclaimer),
plus a "Thinking..." -> "Checking your data -- this can take a bit longer..."
loading-message swap after 12s, mirroring `app/processing.js`'s existing
retry-gate messaging pattern rather than inventing new UI language.

**Real bug found and fixed during verification** (not shipped unnoticed):
the input+Send row sits at the bottom of `ResultView`'s `ScrollView`, and on
Android nothing was pushing it above the keyboard when focused -- confirmed
via `uiautomator dump` that the `EditText`/`Send` bounds didn't move at all
after focus. `login.js`/`signup.js`'s existing `KeyboardAvoidingView` pattern
(`behavior={Platform.OS === "ios" ? "padding" : undefined}`) doesn't cover
this case -- their `undefined` on Android relies on native `adjustResize`
alone, which works for their short, near-top forms but not for an input at
the bottom of a long scrollable page. Fixed in two parts: (1) wrapped both
screens' outer container in `KeyboardAvoidingView` with an explicit
`behavior="height"` on Android too; (2) added a keyboard-height-sized spacer
in `StrokeChat` itself (`Keyboard.addListener` tracking real height, not a
guessed constant) since the shrunk container alone still left no scrollable
slack below the last element to bring it fully clear. Verified via
`uiautomator dump` bounds before/after: `Send` moved from behind the keyboard
to fully above it.

**End-to-end verification on the emulator** (`vt_test`, real
`test1@example.com` backhand session, real Ollama qwen3:8b, real Tailscale
Funnel-reachable backend) -- a genuine multi-turn conversation, not a
scripted single-happy-path test, hitting all 3 bubble styles and both
loading states for real:
1. Tier 1: "why was my weight transfer flagged" -> teal bubble, exact stored
   text, correct `weight_transfer_ratio: 100th percentile` caption.
2. Tier 2 (failure case): "what about my elbow angle" -> full retry-gate ran
   (visible "Checking your data..." swap), exhausted 3 attempts, correctly
   showed the honest failure message rather than a fabricated answer -- real
   LLM variance, not a bug (elbow angle is a real percentile-ranked field;
   this specific generation just didn't reach 100% grounded in 3 tries).
3. Tier 3 general_note: "what's a good drill to improve my backhand
   follow-through" -> plain bubble, non-personalized advice, no citation.
4. Tier 2 (success case): "what's a good weight transfer ratio for a player
   at my level" -> classifier judged this answerable from real data rather
   than an unavailable comparison; fresh grounded generation, teal bubble,
   real citation ("top 1% of backhand strokes in our dataset").
5. `unavailable_comparison`: "how does my weight transfer compare to other
   players" -> amber caveat bubble, fixed disclaimer verbatim ("I don't have
   benchmark data to compare you against -- here's general guidance:")
   followed by real generated general advice.

Confirmed incidentally along the way: chat memory really is ephemeral as
designed -- navigating away from the session screen and back reset the
conversation to empty, exactly as intended.

**One environment/tooling gotcha worth recording**: `adb shell input text`
sending keystrokes while the target `EditText` was NOT actually focused (a
stale tap coordinate after the keyboard changed the layout) got intercepted
as Expo Go's hardware-keyboard dev shortcuts ("r" = reload, "m" = open dev
menu) instead of typed text -- silently discarded the in-progress message
and reloaded the JS bundle back to Home. Not an app bug. Fixed by making
"confirm `focused="true"` via a fresh `uiautomator dump` immediately before
every `input text` call" a hard rule for this kind of test, not an
occasional sanity check -- every UI coordinate (input field, Send button)
was re-dumped fresh after each layout change (new message bubble, keyboard
open/close) rather than reused from a prior screenshot.

## 2026-08-10 -- Crop-alignment hypothesis for serve misclassification: tested, NOT confirmed

Follow-up to the 2026-08-02 classifier diagnostic entry above, which found
0/2 real-footage serves correctly classified and floated (explicitly
unverified) a hypothesis: THETIS clips are tightly pre-trimmed around one
stroke, while this pipeline's fixed +/-30-frame window centered on a
peak-wrist-speed proxy likely lands the model on a differently-framed/
differently-centered 60 frames than what it was trained on. Per explicit
instruction, this pass tests ONLY that hypothesis, on ONE of the two real
misclassified serves (`clip1_serve_practice.mp4`, contact_frame=235,
original window 205-265 -> volley 0.907) -- no production windowing code
touched, diagnostic script only (`ir/run_crop_alignment_check.py`).

**Method:** decoded every frame of the raw clip, hand-identified the serve's
real prep/backswing start, contact, and follow-through end by eye from
individual frames (not the segmentation algorithm's own peak-speed logic) --
idle stance ends and backswing visibly begins ~frame 198-208, contact is
~236-238 (close to but not identical to the peak-speed proxy's 235), and
follow-through completes by ~248, with pure walking/recovery after that.
Re-ran the SAME unmodified StrokeGRU ONNX classifier against the same
pose sequence, re-cropped to 5 different windows spanning that hand-identified
motion with varying amounts of margin, each resampled to the model's required
60 frames via `np.linspace` (the same technique `backend.app.decode_and_resample_video`
already uses for any real clip -- the closest available proxy in this repo for
"how THETIS clips were framed," since the original THETIS preprocessing script
itself isn't in this repo, only its already-processed output). First
reproduced the documented original-window result exactly (volley, 0.9071)
as a pipeline-fidelity sanity check before trusting any new crop's result.

| crop | raw frames | predicted | confidence | serve probability |
|---|---|---|---|---|
| original +/-30 (205-265) | 60 | volley | 0.9071 | 0.0000 |
| manual, backswing-to-follow-through (196-250) | 54 | volley | 0.5174 | 0.0004 |
| tighter, load-to-follow-through (208-248) | 40 | volley | 0.5044 | 0.0003 |
| very tight, uncoil-to-follow-through (218-244) | 26 | **backhand** | 0.9205 | 0.0019 |
| wider margin (190-252) | 62 | volley | 0.6575 | 0.0001 |

**Result: hypothesis NOT confirmed.** Every one of the 5 crops -- including
ones with much less dead time than the original and centered on different
parts of the swing -- still misclassifies. Confidence in the wrong class did
drop substantially with tighter crops (0.907 -> 0.50-0.66), and the tightest
crop flipped to a different wrong class entirely (backhand, 0.92) rather
than converging toward serve. Most telling: `serve`'s predicted probability
never rose above 0.0019 in any of the 5 crops -- the model isn't
moderately-but-wrongly leaning toward serve under better framing, it isn't
considering serve as a real possibility AT ALL regardless of crop. If
crop/window misalignment were the real driver, a well-centered tight crop
should have pushed at least some probability mass toward serve even without
flipping the top prediction; that didn't happen here.

**What this means:** the problem is deeper than window alignment. Crop
timing is ruled out as the primary cause on this instance -- something else
(most plausibly a genuine distribution gap between this real single-camera
footage and THETIS's specific camera setup/framing/scale for serve
specifically, given serve is the *only* class that fails this badly on real
footage while being the *best*-performing class in-sample, per the
2026-08-02 entry) needs investigation before any fix is attempted. Not
investigated further here, per explicit scope (single diagnostic test only).

**No production change.** `ir/stroke_segmentation.py`'s windowing logic is
unchanged; `ir/run_crop_alignment_check.py` is a standalone diagnostic
script, not wired into the real pipeline.

**Status: closed for now, not pursued further this session.** Explicit
decision after this result: crop/window alignment is ruled out as the
primary cause, but the remaining open hypothesis (real-footage-vs-THETIS
camera-setup/framing distribution gap, specific to serve) is untested and
not yet investigated -- recorded here as a characterized, unresolved
limitation for later/for the paper, not something mid-fix. Do not resume
this investigation without a fresh scoping decision.

---

## 2026-08-12: Account role/skill-level profile -- schema, setup UI, tone-of-voice wiring into every prompt path, verified end-to-end with real accounts

New feature set, built and verified per explicit spec: a `role`
(`"player"`/`"coach"`) plus role-specific skill fields on the account, wired
into every place coaching text is generated so the *language* -- not just the
underlying analysis -- actually changes based on who's using the app.

**Schema.** `backend/db.py`'s `User` gained 4 nullable columns: `role`,
`skill_level` (player only: beginner/intermediate/advanced), `years_coaching`
and `certification_level` (coach only: club_coach/national/itf_certified/
other, chosen as a small fixed set rather than free text). Coaches
deliberately get no scaling mechanism for tone -- always technical, regardless
of `years_coaching` -- per explicit instruction not to build one. Since
`backend/app.db` already existed with real accumulated trial accounts, added a
manual migration (`_migrate_add_missing_columns()`: `PRAGMA`-based column
check + `ALTER TABLE ADD COLUMN` for anything missing) rather than relying on
`Base.metadata.create_all()`, which only creates missing tables and silently
no-ops on an existing one. Verified live against the real `app.db`, no data
loss.

**Two "role" concepts now coexist, deliberately not merged.** The pre-existing
on-device anonymous `role` in `historyRepository.js` (feedback-question
selection only, explicitly unlinked from any account per its own comment in
`settings.js`) is a different thing from the new account-level `role` that
actually drives coaching tone. Flagged as a naming collision in code comments
on both sides, not consolidated -- out of scope for this feature.

**UI.** New `mobile/app/profile-setup.js`: role first ("I am a..."), then a
role-appropriate step 2 (3 skill buttons for players; years-coaching input +
4 certification pills for coaches). Pushed straight after signup
(`signup.js` now routes to `/profile-setup` instead of `/(tabs)`) and
reachable again from a new "Coaching profile" section in `settings.js` to
edit later. Styled entirely from `mobile/lib/theme.js`, no new design tokens.

**Tone wiring -- the part the spec called out as critical.** `rag/prompt.py`
gained `build_tone_instruction(tone_profile)`: returns `None` for no
role/profile (byte-identical prompts to before this feature existed, verified
via direct `==` comparison), a fixed technical-language paragraph for
`role="coach"`, or one of three distinct paragraphs for `role="player"` +
beginner/intermediate/advanced. Threaded as an optional `tone_profile: dict |
None` parameter through every real prompt-construction path: single-session
analysis (`rag/stroke_prompt.py:build_stroke_prompt`), and both chatbot tiers
(`rag/chat_pipeline.py:build_stroke_chat_prompt` for tier-2,
`build_tier3_prompt` for tier-3, including the tier-3-escalates-to-tier-2
fallback). `backend/app.py`'s live `/analyze-stroke` and chat-turn endpoints
now pass `user_tone_profile(token_row.user)` through on every real request --
not stored-but-unused.

**End-to-end verification on the emulator, with real accounts, not just unit
checks.** Created two real trial accounts and ran the identical real clip
(`clip2_teen_practice.mp4`) through both, so the underlying biomechanics
(stroke prediction, confidence, percentile metrics, pose-detection rate) are
held identical and only the generated language differs:

| | beginner-player | coach (8yr, National) |
|---|---|---|
| Predicted stroke | Backhand, 99% confidence | Backhand, 99% confidence |
| Pose detection | 98% | 98% |
| weight_transfer_ratio (100th pct) | "Your weight shift is among the strongest in our dataset, which helps generate powerful backhand shots. Keep driving off your back foot to maintain this forward momentum." | "Your weight transfer into the shot is on the higher end for a backhand, which could indicate a strong forward shift from your back foot. If this feels effective, it may be optimal for your style." |
| knee_flex_right_deg (97.9th pct) | "Your right knee bend at contact is on the higher end for backhands, but this alone doesn't indicate a specific issue. Focus on keeping your legs stable during the swing." | "Your right knee flex at contact is on the higher end for a backhand, though this measurement alone doesn't indicate a specific adjustment is needed." |
| wrist_height_ratio (93.4th pct) | "Your contact point is higher than most players, which could mean you're taking the ball later or using a flatter swing style. If it feels natural, no changes are needed." | "Your wrist height ratio is on the higher end for a backhand, which may reflect a flatter swing path. If this doesn't feel problematic, no adjustment is needed." |

Confirms genuine, consistent tone differentiation: the player version uses
lay terms ("weight shift", "knee bend", "contact point") and directive,
simplified cues; the coach version names the actual metric being discussed
("weight transfer", "knee flex", "wrist height ratio" -- mirroring the raw
field names) and stays clinical/hedged throughout. Not just the field
saving correctly -- the generated text is verifiably different.

**Environment note (recurring, not new):** backend/emulator/Metro had all
died again between turns. Recovery hit one new wrinkle this time: the backend
was first restarted on port 8000 (the old convention from early in the
project), but Tailscale Funnel has pointed at **port 8001** since the
2026-08-07 entry -- caused a real "Something went wrong" save failure on the
coach profile until caught via `tailscale serve status` showing the
`127.0.0.1:8001` target and a 502 from the funnel URL. Fixed by restarting
uvicorn on 8001; documenting here since the port-8000-by-habit mistake is
easy to repeat.

---

## 2026-08-12: Chatbot -- third mechanical backstop built (skill-tier/percentile mismatch), confirmed against the real 2026-08-10 failure

Closed the gap identified in the prior status-check entry (skill-tier
comparison backstop was designed in the router prompt but never mechanically
enforced). Built `rag/chat_router.py:check_for_skill_tier_mismatch()` --
backstop #3, alongside the two from the 2026-08-08 chatbot entry, same "never
trust the model's self-report, verify the output" philosophy.

**What it catches:** a "new_grounded_claim" generation that IS fully,
mechanically grounded (passes `check_grounding()`, real citation, real
number) but answers a skill-tier-framed question ("at my level", "for my
skill level", "someone at my stage", etc. -- checked via a regex pattern
list, deliberately excluding bare "beginner"/"advanced" as standalone
triggers to avoid false-positiving on legitimate technical text) with a
citation into `ir/biomechanics_percentiles.py`'s population -- confirmed to
have no skill/tier stratification anywhere in that module. When both
conditions hold (skill-tier language in the player's message or the
response text, AND a `biomechanics_percentile.*` citation present), the
result is discarded and replaced with a fresh `unavailable_comparison`
tier-3 response instead, regardless of what `classify_turn()` originally
decided. Implementation: `PERCENTILE_PREFIX` hoisted from
`rag/grounding.py`'s `check_grounding()` to module level so both files share
one constant; `rag/chat_pipeline.py`'s tier-3 generation logic factored into
`_generate_tier3_response()`/`_finish_tier3_result()` so the normal tier-3
dispatch and the new backstop's fallback path share one implementation, and
`_run_tier2_checked()` wraps every path that reaches `_run_tier2()` (direct
`new_grounded_claim` routing AND tier-3-escalation) with the new check.

**Deterministic unit test (no LLM, mirrors backstop #2's original
verification standard): 8/8 correct** -- hand-scripted phrase list against
`_contains_skill_tier_language()` (5 true positives incl. the exact failing
phrase, 2 true negatives incl. a deliberately-adjacent "how do advanced
players usually generate more power" to confirm bare skill-level words don't
false-positive alone) plus 3 hand-constructed `CoachingFeedbackItem` cases
through the full `check_for_skill_tier_mismatch()` (skill-tier msg +
percentile citation -> correctly unsafe; non-skill-tier msg + same citation
-> correctly safe; skill-tier msg + general_note only, no citation ->
correctly safe, confirming the backstop is about the *citation* being
mislabeled, not the phrase alone).

**Live re-test** (`rag/run_skill_tier_backstop_trials.py`, real qwen3:8b via
Ollama, same real StrokeEvent used throughout the chatbot work,
event_id=`6bc75eb4-...`): the exact 2026-08-10 phrase plus 3 rephrasings, run
both naturally (Part A) and with the router forced to `new_grounded_claim`
(Part B, guaranteeing tier 2 actually attempts generation). **Zero percentile
citations leaked to the user across all 8 live attempts** -- the true safety
metric, re-derived by hand after the script's first version used too strict
a grading criterion (it required every message to literally end on
`route=="unavailable_comparison"`, which also flagged an honest
tier-2-ungrounded failure message or a router that legitimately picked
`general_note`/`unavailable_comparison` on its own as if they were backstop
failures; fixed in the script to check "was a percentile citation ever
shown," the actual thing that matters).

**Honest finding, not glossed over:** only 1 of the 4 forced-`new_grounded_claim`
attempts (Part B) actually produced a real, LLM-generated percentile
citation for the backstop to catch -- the other 3 had tier 2 itself fail to
ground within its 3-attempt budget (real qwen3:8b variance on this specific
message class, the same documented reliability characteristic noted in
every prior chatbot/gated-pipeline entry, not something this backstop is
responsible for). The one case that DID produce a real grounded percentile
citation (`"what would be normal for someone at my stage of playing?"`,
forced route) was correctly caught and downgraded to the disclaimer --
genuine live confirmation the mechanism fires against a real LLM output, not
just the scripted unit test. **The exact original 2026-08-10 phrase did not
reproduce its original misrouting today** (qwen3:8b's router classified it
as `general_note` this time, producing clean non-personalized advice with no
citation at all) -- consistent with the router's already-documented run-to-run
variance (2026-08-08 entry, 7/8 -> 8/8-ish reliability range on repeated
trials), not evidence the original bug was a one-off that can't recur; the
backstop exists precisely so it doesn't matter which way the router leans on
any given run.

**Before/after:** before this fix, a live run existed (2026-08-10) where a
skill-tier-framed question received a real, numerically-correct-but-mislabeled
percentile citation with no mechanical check catching it. After: the
identical failure shape is now checked on every tier-2 result, confirmed via
both a deterministic unit test and a live run against a real (if rare, this
session) instance of the exact failure condition.

---

## 2026-08-12: Video constraints + zero-shot tennis-content classifier -- built, client caps + rejection UX complete, classifier threshold PROVISIONAL and explicitly not locked

Third item of the role/skill-level-adjacent feature set. Two parts, one
fully done, one deliberately left open pending a real blocker.

**1. Client-side video constraints (DONE).** `mobile/lib/config.js` gained
`MAX_PICKED_VIDEO_DURATION_S = 30` / `MAX_PICKED_VIDEO_BYTES = 100MB`,
explicitly cross-referenced in comments against `backend/app.py`'s
`MAX_RAW_FRAMES`/`MAX_UPLOAD_BYTES` (no shared source of truth between the
JS and Python codebases -- flagged, not hidden). `mobile/app/(tabs)/index.js`'s
`pickVideo()` (the path with NO existing picker-level cap, unlike
`recordVideo()`'s `videoMaxDuration: 10`) now checks the picked asset's
`duration`/`fileSize` -- both confirmed optional/nullable against the real
locally-installed `expo-image-picker` (~17.0.11, Expo SDK 54) type
definitions before writing the check, not assumed from possibly-stale docs.
Violation shows an `Alert.alert(...)` before navigating away, matching the
existing camera/library-permission alerts already in that file. `recordVideo()`
also gets the size check defensively (duration is already bounded by the
picker itself).

**2. Zero-shot "is this tennis" classifier (`ir/tennis_classifier.py`,
NEW).** CLIP zero-shot (`open_clip_torch`, ViT-B-32/laion2b_s34b_b79k, new
dependency -- installed cleanly; flagged 2 pre-existing, unrelated pip
conflicts on this machine, `gtts`/`streamlit`, neither used by this backend).
Runs server-side in `backend/app.py`'s `analyze-stroke`, right after
`decode_and_resample_video()` and before the expensive pose/classification/
coaching stages -- reject cheap, before paying for the rest of the pipeline.
Samples 4 evenly-spaced frames from the 60 resampled frames; standard CLIP
zero-shot (softmax over ALL prompts per frame, sum the positive group's
mass, average across frames). Positive prompt group (3, ensembled) plus a
negative group of 7 deliberately chosen to include the closest visual
neighbors (badminton, table tennis, squash, basketball, soccer), not just
generic scenes. New `not_tennis` error_code, wired identically to the 3
existing Phase C failure modes (`log_rejection`, `HTTPException(422,
detail={"error_code", "message"})`, `mobile/lib/errorMessages.js`'s
`UPLOAD_ERROR_TITLES`).

**Threshold status: PROVISIONAL, explicitly NOT locked -- flagged in the
module docstring itself, not just here.** Per explicit user instruction, the
two real cases this feature exists to catch (a Spider-Man premiere clip and
a "ball going over a net" clip, referenced from "the very first 5-video test
session") are the deciding validation cases. **Neither could be located**:
searched the entire repo (filename glob across everything including the
596K-file `data/` Google-Drive mirror, plus `PROJECT_LOG.md` content search)
for "Spider-Man," "premiere," "ball-over-net," and any record of a 5-video
test session -- nothing matches. The only real video files anywhere in this
repo are the 3 Pexels stand-ins already documented throughout this log
(`clip1_serve_practice`, `clip2_teen_practice`, `clip3_group_distant`).
Flagged to the user directly; not resolved as of this entry.

**Partial validation run** (`ir/run_tennis_classifier_validation.py`,
DEFAULT_THRESHOLD=0.5), what's available now, explicitly NOT a substitute
for the two blocked clips:
- **Positive side**: `clip1_serve_practice` (tennis_score=0.9914) and
  `clip2_teen_practice` (tennis_score=0.9148) both correctly classified.
  `clip3_group_distant` couldn't be tested via the real pipeline path at
  all -- it's rejected by the PRE-EXISTING `MAX_RAW_FRAMES` check before
  ever reaching the classifier (its real fps apparently pushes it over 900
  frames despite a 23.1s runtime) -- not a bug in this new code, a
  pre-existing constraint this validation run surfaced incidentally.
- **Negative side, gathered per "any other negatives you gather"**: 5 real
  clips from Penn Action (`data/VisioTennis/Dataset/Penn_Action`, a
  multi-sport action dataset already present in this repo, previously
  undocumented in this log), chosen to include the closest real analog to a
  tennis stroke (a swing motion): `baseball_swing` (0.0012),
  `golf_swing` (0.2102 -- the closest real negative to the threshold found
  so far, worth remembering for any future threshold change), `bowl`
  (0.0730), `strum_guitar` (0.3677), `jumping_jacks` (0.0630). **7/7
  labeled clips classified correctly at threshold=0.5**, with a real margin
  picture: the closest real negative (golf_swing, 0.21) sits ~0.29 below
  threshold; the weakest real positive (clip2, 0.91) sits ~0.41 above it.

**This is real evidence the mechanism works and the 0.5 starting point isn't
obviously wrong** -- but it is NOT the validation the user specified. The
Spider-Man and ball-over-net clips could plausibly be harder cases than
anything tested so far (a premiere event likely has people/motion/crowds
that could read as "sport-adjacent" to CLIP; a ball going over a net outside
a tennis context is precisely the kind of near-miss these prompts were
designed to catch, untested). **Do not treat 0.5 as locked. Re-run
`ir/run_tennis_classifier_validation.py <path-to-clip>` once the two clips
are located, per that script's own support for extra command-line paths,
before treating this feature as complete.**

---

## 2026-08-12: Tennis-content classifier threshold LOCKED at 0.5 -- resolution on the two blocked validation clips

Follow-up to the entry above. User resolution on the two clips:

1. **The ball-over-net clip is out of scope for this classifier, not a
   blocked test case.** Identified source: mixkit.co/free-stock-video/
   tennis-ball-flying-over-the-net-23639 -- genuine tennis content, but with
   no visible player. That's a DIFFERENT failure mode than what
   `ir/tennis_classifier.py` exists to catch -- it's already handled by
   `backend/app.py`'s pre-existing `no_player_detected` check (a real player
   with a real pose stream, or none at all; separate from "is the content
   tennis"). Correctly excluded from this classifier's validation set, not a
   gap in it.
2. **The Spider-Man premiere clip's source is not recoverable right now.**
   Not pursued further -- explicit instruction not to keep chasing it.

**Decision: threshold LOCKED at 0.5**, based on the validation already
completed (prior entry): 7/7 real clips correct (2 real tennis positives via
the actual pipeline decode path, 5 real Penn Action negatives including a
golf swing as the hardest plausible confusable case at 0.21), with real
margins on both sides (~0.29 below threshold on the closest negative, ~0.41
above on the weakest positive). Judged sufficient evidence per explicit
instruction, without the two originally-referenced clips.

**Documentation updated to reflect a validated-with-a-known-gap status, not
a bare "provisional" flag removed silently.** `ir/tennis_classifier.py`'s
module docstring now states explicitly, in one place, both what WAS
validated (the 7/7 breakdown above, real footage through the real decode
path) and what was NOT (no adversarial/edited/produced video content --
e.g. a movie premiere's crowds/flashes/produced lighting is a materially
different visual distribution from anything tested; no threshold sweep, 0.5
is the one value tested, not an ROC-searched optimum). `backend/app.py`'s
call site and `ir/run_tennis_classifier_validation.py`'s docstring/summary
message updated to match -- no lingering "PROVISIONAL" language left
pointing at a since-resolved status.

**Feature now complete**: client-side duration/size caps, `not_tennis`
rejection UX, and the classifier itself (with its threshold locked and its
validation scope honestly documented) are all done.

---

## 2026-08-12: Ball-tracking + court-homography validated on real footage for the first time -- CONFIRMED generalization failure, visually verified, no fix attempted

Next item in the queue. Neither TrackNet (Model 1A) nor CourtKeypointNet +
homography (Model 1B) had ever been run against the 3 real out-of-dataset
clips (`mvp/real_footage_test/`) that every other pipeline component
(MediaPipe Pose, StrokeGRU, stroke segmentation, the new tennis-content
classifier) has been extensively validated against -- confirmed by searching
this log before writing any code. Both models had previously only run
against (a) their own training dataset's held-out split, and (b) ONE
cross-domain spot-check (CourtKeypointNet on a single frame from the
UNRELATED ball-tracking dataset, `ir/populate.py`'s "true end-to-end"
example, 7.37cm reprojection error). This is the first time either model
has seen genuinely different-source real footage.

**New script**: `ir/run_ball_court_real_footage_validation.py` -- reuses
`ir/populate.py`'s exact inference code paths (same checkpoints, same
preprocessing) but adapted to run at EACH clip's own real native resolution
(1920x1080 for all 3) rather than assuming the training datasets' fixed
1280x720, since an arbitrary real clip has no reason to share that. Per this
project's established discipline (the IR layer's original keypoint-order
spot-check), also writes annotated debug frames (predicted keypoints/ball
position drawn on the real frame) so results are visually confirmed, not
just trusted as a number -- and that discipline mattered a lot here (see
below).

**Ball tracking (TrackNet), 20 sampled seq_len=3 windows per clip:**

| clip | visibility rate (>=0.5 conf) | confidence mean |
|---|---|---|
| clip1_serve_practice | 45/60 (75%) | 0.715 |
| clip2_teen_practice | 39/60 (65%) | 0.609 |
| clip3_group_distant | 15/60 (25%) | 0.265 |

Degrades with clip difficulty in the same order as every other component in
this project (clip1 easiest, clip3 hardest -- multi-person, distant,
low-quality). Not independently hand-verified frame-by-frame here (the
debug overlay plots multiple frames' detections onto one static frame,
which is informative for a density/spread check but not a per-detection
correctness check) -- flagged as a real limitation of this pass, not
glossed over. clip1's scene also has a real confound worth noting: dozens
of stray tennis balls sitting on the ground from the ball-basket drill,
which is a genuinely ambiguous case for "the" ball even for a human, not
just this model.

**Court homography -- CONFIRMED generalization failure, visually verified:**

| clip | reprojection_error_m | compare: prior best-known range |
|---|---|---|
| clip1_serve_practice | **1.22m** | 2.2cm (ground-truth kps) -- 10.5cm (real predicted kps, composite examples) |
| clip2_teen_practice | **5.65m** | same |
| clip3_group_distant | **6.27m** | same |

10x to 50x+ worse than anything previously seen. **Visually confirmed, not
just inferred from the number** (`ir/example_output/ball_court_real_footage_validation/*_court_debug.png`):
on all 3 clips, the predicted 14 "court keypoints" cluster around the
PEOPLE in frame (bodies, rackets, arms) rather than any real court line
intersection -- clip1's predicted quadrilateral is a garbled shape spanning
the player and the ball basket, nowhere near the actual baseline/sideline
geometry visible in the same frame; clip2 and clip3 show the same pattern.
This is not a borderline or ambiguous case -- the reprojection error number,
bad as it already looked, actually UNDERSTATES how broken this is; a
homography fit through garbage keypoints can still produce a superficially
parseable meters value.

**Most likely cause, not confirmed further (out of scope for this
validation-only pass):** `Court_Detector_Keypoints`'s training images most
likely follow a specific framing convention (full court boundary visible,
roughly centered, moderate distance) that none of these 3 real amateur
clips match -- clip1 is a much closer/tighter angle than typical broadcast
framing, clip2 is a tight cropped shot where the actual court boundary
lines may not even be in frame, clip3 is a wide multi-person shot the model
was never trained to parse. This is the SAME class of camera-framing/
distribution-shift concern already flagged (untested, at the time) for
TrackNet specifically in the 2026-07-20 hardening-pass entry -- now
CONFIRMED, and shown to affect CourtKeypointNet at least as badly.

**No fix attempted, by design -- this was a validation pass, not a repair
task**, matching this project's established diagnostic-only scoping (e.g.
the 2026-08-02 StrokeGRU real-footage diagnostic, the 2026-08-10
crop-alignment test). **Status for the paper/future work:** ball-tracking
and court-homography are confirmed NOT to generalize to real amateur
footage in their current trained state -- consistent with why
`backend/app.py`'s MVP scope already excludes both from the live pipeline
(`StrokeEvent.ball`/`.court` stay `None` in production; see that file's own
top docstring, a decision made independently of this validation but now
directly supported by it). Any future work re-enabling these components for
real footage would need either a re-trained/fine-tuned model on
framing-diverse data, or a documented, tested set of camera-framing
constraints for users to follow -- neither pursued here.

---

## 2026-08-12: "Path B" (video-based fine-tuning, as an alternative to pose-sequence StrokeGRU) -- feasibility report, no training code written

Requested as a 4-part honest feasibility check before any training code, in
response to the serve-misclassification domain-shift problem characterized
across the 2026-08-02 and 2026-08-10 diagnostic entries above. Verdict up
front: **hardware and model licensing are NOT the blocker -- data is.**

**1. Raw video data availability -- HARD BLOCKER, confirmed.** Exhaustive
search of `data/VisioTennis/Dataset/StrokeClassification/` (and repo-wide
for any `.avi`/`.mp4`/`.mov`/`.mkv`): only `processed/{train,val,test}/
X_{split}.npy` (pre-extracted 60-frame MediaPipe pose-landmark sequences)
and label CSVs exist. **No raw THETIS video files anywhere in this repo.**
The label CSVs reference original THETIS filenames (e.g.
`p47_serflat_s3.avi`) but the actual video was apparently never downloaded
-- only the pre-extracted features. This independently confirms something
already noted in passing in the IR-layer true-e2e entry ("THETIS...ships
only pre-extracted features with no raw video in this repo"), now checked
directly rather than recalled. **Not independently verified this session
whether THETIS's original public distribution still offers raw video for
re-download** -- flagged as the one open question that could change this
blocker's status, not chased further here (would need a real acquisition
effort, out of scope for a feasibility check).

**2. Candidate pretrained backbones -- 3 identified, all license-clear:**

| model | params | license | source | notes |
|---|---|---|---|---|
| `torchvision.models.video.r2plus1d_18` | 31.5M | BSD-3-Clause (torchvision) | already installed, zero new dependency | officially maintained by the PyTorch team; Kinetics-400 pretrained, 67.5% top-1 |
| X3D-XS (PyTorchVideo) | 3.80M | Apache 2.0 (confirmed via repo LICENSE) | `pip install pytorchvideo`, Meta-maintained | smallest of the three by a wide margin |
| MoViNet-A0 | 3.11M | MIT (confirmed via repo) | `pip install` from `Atze00/MoViNet-pytorch` | **unofficial third-party PyTorch port** of Google's official (Apache 2.0, TF-native) MoViNet -- MIT license itself is fine for product use, but this is an independent reimplementation, not Google's maintained code; a real (if secondary to licensing) maintenance/correctness-provenance risk worth naming |

All three are permissively licensed and usable in an academic and product
context. r2plus1d_18 is the safest choice by a different axis than size:
zero new dependency, officially maintained, already installed on this
machine. Full VideoMAE/TimeSformer-large-class models were not evaluated
further, per explicit instruction to rule them out outright.

**3. Local hardware feasibility -- checked empirically, not assumed
(`ir/run_video_model_vram_probe.py`, new -- a memory/timing PROBE, single
forward+backward pass on dummy tensors, explicitly NOT a training script).**
Real RTX 3060, real qwen3:8b loaded via a live Ollama `generate` call
(confirmed via `nvidia-smi` before/after: 7108-7120MiB used throughout, not
assumed from an earlier session's documented figure), batch_size=4:

| model | peak VRAM (fwd+bwd) | measured throughput |
|---|---|---|
| r2plus1d_18 | 2.10-2.34 GB | 15.9 clips/s |
| X3D-XS | 0.58-0.61 GB | 91.7 clips/s |
| MoViNet-A0 | 0.74-0.76 GB | 94.1 clips/s |

**All 3 candidates ran successfully, no OOM, while qwen3:8b was genuinely
loaded and consuming ~7.1GB** (confirmed real, live, concurrent -- not two
separate measurements stitched together) -- even the largest (r2plus1d_18,
2.3GB) fits comfortably inside the ~4.9GB `nvidia-smi`-confirmed free
headroom with room to grow batch size. **Hardware is feasible, including
concurrently with Ollama serving coaching generation.**

**One unresolved measurement anomaly, reported rather than hidden:**
`torch.cuda.mem_get_info()` inside the probe script reported ~10.98GB free
even while `nvidia-smi` (ground truth, OS-level) showed only ~5.0GB free
with Ollama loaded -- a real discrepancy, most likely a Windows WDDM
driver-mode CUDA memory-virtualization quirk (not confirmed further). Not
treated as authoritative; `nvidia-smi`'s system-wide used/free numbers and
the actual successful-with-no-OOM runs are what this feasibility verdict
rests on, not the confused self-reported "free" figure.

**Caveat on the throughput numbers:** measured on synthetic in-memory
tensors already resident on GPU -- real training also involves video
decode/augmentation/dataloader overhead, which for video pipelines is
often the actual bottleneck, not the GPU forward/backward pass. These
clips/s numbers are a GPU-compute ceiling, not a promised real epoch time.

**4. Real-footage validation data sufficiency -- NOT sufficient, and this
compounds directly with finding #1.** Exactly 3 real hand-labeled clips
exist (`mvp/real_footage_test/`, the same 3 used throughout this project's
validation work). This was already judged insufficient for far simpler
decisions than validating a fine-tuned deep video model -- e.g. this
project's own stroke-segmentation work explicitly flagged 3 clips as "not
enough to trust these three defaults generally" for tuning a single scalar
height threshold. Validating (or meaningfully training) a 5-class video
classifier needs enough real examples PER CLASS, across enough distinct
players/courts/cameras/lighting conditions to say anything about
generalization -- 3 source videos, regardless of how many individual
stroke instances can be segmented out of them, cannot provide that
diversity by construction (they're 3 camera setups, not 3-to-infinitely-many).

**Combined verdict: Path B is blocked on data, not on hardware or
licensing.** Finding #1 (no raw video for the one large existing labeled
dataset) means there is currently no substantial real or public dataset to
fine-tune a video model ON. Finding #4 (3 real clips) means there is no
substantial real dataset to validate one AGAINST either, even if a trained
checkpoint existed. Findings #2-3 (backbones, hardware) are not the
constraint -- multiple well-licensed, small, empirically-confirmed-feasible
options exist and would run comfortably on this machine, including
alongside the coaching LLM. **Next-best options, not started:** (a)
investigate whether THETIS's original public distribution still offers raw
video for re-download -- the single highest-leverage next step, since it
would resolve the larger of the two data gaps; (b) independently of (a),
real labeled footage collection would need to expand well beyond 3 clips
before a fine-tuned model's real-world performance claim would mean
anything. No training code was written this session, per explicit
instruction.

---

## 2026-08-12: Raw THETIS RGB video located and downloaded; domain-shift hypothesis confirmed with real specificity FROM THE PRIMARY SOURCE, with one correction to the initially-stated recollection

Follow-up to the Path B feasibility report's #1 finding (no raw video in
this repo). A source was identified: `github.com/THETIS-dataset/dataset`, a
real, maintained GitHub org hosting the dataset directly (verified via the
GitHub REST API, not just a rendered page).

**1. License -- checked, reported plainly, real ambiguity for product use.**
GitHub's own license detector reports `license: null` for this repo (the
`/license` API endpoint returns 404) -- **no LICENSE file exists.** The
README states no explicit terms of use, redistribution rights, or
commercial-use restrictions. The only stated requirement is a citation
ask: "If you use the THETIS dataset in your work, please cite one of the
following papers" (three named academic references, 2013/2017/2018).
**This is normal for an academic dataset released under implicit
research-and-cite norms, but it is not a formal open license grant** --
under default copyright law, "no license" does not mean "free to
redistribute or use commercially." Fine for the same academic/FYP research
purpose this project has already been using THETIS's pose-extracted
features for all along (no new risk category vs. existing practice) --
**but genuinely unresolved for "eventual product context"** use, exactly
the same caveat already flagged for MoViNet-A0's unofficial port in the
Path B report. Would need explicit permission from the dataset authors
before any commercial/product use, not assumed from citation-norm silence.

**2. RGB format/size -- confirmed via the GitHub Contents API (real file
sizes, not Git LFS pointers) before downloading anything.** `VIDEO_RGB/`
contains exactly 12 class subfolders matching this project's already-known
native taxonomy, 165 `.avi` files each = 1980 total, matching the
externally-reported count exactly. Sampled 3 classes directly:
`backhand` (319.2MB/165 files), `flat_service` (341.7MB/165 files), `smash`
(306.2MB/165 files) -- average ~1.9-2.1MB/file, consistent across classes.
**Extrapolated total for VIDEO_RGB alone: ~3.8GB** -- confirmed real video
byte counts (not tiny LFS-pointer-sized entries), so this is genuinely
downloadable, not a stub.

**3. Downloaded RGB only, not the full corpus.** `ir/download_thetis_rgb.py`
(new) pulls only `VIDEO_RGB/` via the Contents API (12 calls, well under the
unauthenticated 60/hour rate limit) + direct `raw.githubusercontent.com`
file downloads (uncounted against that limit) -- explicitly skips
`VIDEO_Depth`/`VIDEO_Mask`/`VIDEO_Skelet2D`/`VIDEO_Skelet3D`, which together
would be roughly 4-5x the data for modalities this project doesn't use.
Output: `data/VisioTennis/Dataset/StrokeClassification/raw_rgb/<class>/*.avi`.

**Domain-shift hypothesis -- confirmed with real specificity from the
PRIMARY SOURCE PAPER, with one correction against the initially-stated
recollection.** Downloaded and text-extracted the original CVPR 2013
Gourgari et al. paper (`papers/Gourgari_THETIS...pdf` in the same repo) --
checked directly rather than taken on recollection, per this project's
established discipline. **What the paper actually says (Section 2.1,
"Camera set up and capture conditions"):**
- Recording device: Microsoft Kinect, **indoors only** -- explicitly because
  Kinect's infrared depth sensing fails under direct sunlight, not a
  staging choice.
- Camera: **static, fixed at 1.6m height**; the "action performing point"
  (the subject's position) was set at **~1.5m** from the capture point --
  a genuinely close range, consistent with (though not identically worded
  as) a close-up/camera-facing framing.
- **Correction: backgrounds are explicitly documented as NOT staged/static**
  -- the paper states background "does not remain static since most of the
  times varies from one shot to the other, containing multiple persons
  acting in different ways behind the action scene (passing, playing
  basketball etc.)." This is the opposite framing from "staged environments
  -- changing room or basketball court backgrounds": the paper describes
  varying, uncontrolled real-world background activity, not deliberately
  chosen staged backdrops. **The literal phrase "changing room" does not
  appear anywhere in the paper text** (checked directly, not just recalled).
  "Basketball" appears exactly once, as one example of incidental
  background activity (someone playing basketball behind the subject), not
  as a description of a basketball-court recording venue.
- Ball usage: **not addressed either way** in the paper -- no mention of
  "ball" (checked directly) confirming or ruling out whether players struck
  a real ball during capture.

**Net effect on the domain-shift explanation: still real, still citable,
still more specific than a vague "different distribution" guess -- just
different specifics than initially stated.** The concrete, paper-confirmed
distribution-shift factors are: (a) fixed close-range (~1.5m) indoor Kinect
capture vs. this project's real footage (further/variable-distance,
natural outdoor daylight, phone/consumer camera), and (b) THETIS's
deliberately-varying, uncontrolled backgrounds (not the controlled clean
court backgrounds this project's own real clip1/2 happen to have) -- if
anything, an argument that THETIS's background variation should have made
the model MORE robust to background clutter, which sharpens rather than
explains away the observed serve-specific failure; the framing/distance
difference remains the more likely driver.

**Not found, consistent with the earlier unresolved flag from this
session:** searched `visiotennis_roadmap.md` and `visiotennis_training_prompt.md`
(the only two root-level `.md` files besides this log) for any
domain-shift/crop-alignment/research-compilation content -- neither
contains any. **No "research materials" document beyond this log has been
located anywhere in this repo** across this entire session's searches; this
entry is written to `PROJECT_LOG.md` only, since that remains the one
confirmed real location for this kind of finding.

**Download completed and verified.** `ir/download_thetis_rgb.py` finished
in 3614s (~60 min): **1980/1980 files downloaded, exactly 165 per class
across all 12 classes, 3783.8MB (matches the ~3.8GB pre-download estimate
closely).** Verified independently after completion (not just trusting the
script's own summary line): re-counted files per class directory on disk,
confirmed 165/165 for every one of the 12 classes, total dir size 3.7GB.
Output: `data/VisioTennis/Dataset/StrokeClassification/raw_rgb/<class>/*.avi`.

**Path B feasibility report finding #1 (raw video availability) is now
RESOLVED -- no longer a blocker.** The other three findings from that
report are unaffected by this: model/license choice (#2) and hardware
feasibility (#3) were already confirmed workable; real-footage validation
data sufficiency (#4, only 3 real hand-labeled clips) is a separate gap
this download does not address. Path B's overall verdict updates from
"blocked on data (both #1 and #4)" to "blocked on #4 only" -- raw video to
fine-tune ON now exists; real diverse footage to validate a fine-tuned
model AGAINST still does not.

---

## 2026-08-12: Real-footage validation set expanded to 6 clips -- THIRD independently-confirmed real-footage serve misclassification, the cleanest case yet

Directly addresses Path B's finding #4 (real-footage validation data
sufficiency) -- partial progress, not a fix: 3 new real clips added, real
labels confirmed by direct frame-by-frame visual inspection (not assumed
from filename, per explicit instruction), run through the exact real
pipeline path (`decode_and_resample_video` -> `run_pose_extraction` ->
`run_stroke_classification` -- the same path a real single-clip upload
takes in production, not the segmentation/candidate-window diagnostic
path).

**Source and provenance:** 3 Mixkit stock clips (free, real amateur/staged
tennis footage, same general provenance category as the original 3
Pexels-sourced clips), moved into `mvp/real_footage_test/` as
`clip4_game_rally.mp4`, `clip5_serve_setup.mp4`, `clip6_solo_practice.mp4`
(descriptive names matching the existing `clipN_<description>` convention
-- names describe visible scene content, not an asserted stroke class).

**Results** (`ir/run_new_clips_pipeline_check.py`, new):

| clip | pose detection | true label (confirmed by eye) | predicted | confidence | correct? |
|---|---|---|---|---|---|
| clip4_game_rally | 54/60 (90.0%) | backhand (two-handed grip, ball approaching) | backhand | 55.2% | **yes** |
| clip5_serve_setup | 60/60 (100.0%) | **serve** (confirmed via ball-toss frame + full trophy-position swing frame -- unambiguous) | backhand | **99.3%** | **NO** |
| clip6_solo_practice | 58/60 (96.7%) | forehand (one-handed swing prep) | forehand | 97.1% | **yes** |

All 3 clear the 80% pose-detection production trust threshold comfortably
-- none flagged as degraded validation data.

**clip5_serve_setup is the THIRD independently-confirmed real-footage
serve misclassification** (following the two documented in the 2026-08-02
classifier diagnostic entry, `clip1_serve_practice` frame 235 and
`game4_clip1_broadcast_close` frame 110) -- and, on the specifics, **the
cleanest case seen yet**: close camera, well-lit, single unobstructed
player, an unambiguous serve motion (ball toss + full trophy position
confirmed frame-by-frame), called backhand at 99.3% confidence -- not a
borderline or low-confidence miss, a confident, decisive wrong answer.
Serve is no longer plausibly a marginal/edge-case failure across a small
sample: **0/3 real-footage serves correctly classified**, now across 3
independently-sourced clips (2 Pexels, 1 Mixkit) and 2 different original
provenances (this project's own footage + the Ball Tracker+Bounce
broadcast set), with zero counterexamples of a correct real-footage serve
classification anywhere in this project's history.

**Updated real-footage validation set: 6 clips total** (was 3).
True-labeled instances now on record: clip1 (2 confirmed real serves, per
the 2026-08-02 diagnostic, plus 2 non-stroke false-positive candidates),
clip2 (multiple confirmed backhand instances, per the 2026-08-02 stroke-
segmentation validation), clip4 (backhand), clip5 (**serve**), clip6
(forehand). clip3 remains excluded from classification-accuracy
conclusions (known pose-tracking failure on that clip, per the
2026-08-02 diagnostic).

**Strengthens the domain-shift finding directly, not just generically.**
Combined with the same day's primary-source-verified finding (THETIS's
fixed ~1.5m indoor Kinect capture vs. this project's real footage's
more distant/variable, natural-light framing) and the ruled-out
crop-alignment hypothesis (2026-08-10: serve probability never exceeded
0.0019 across 5 re-crops of one of these exact serve misses, regardless of
window timing), a third clean, independently-sourced serve miss under
otherwise-favorable conditions (good lighting, close camera, unobstructed
view) makes a distribution-shift explanation specific to the serve class
harder to dismiss as coincidence, and correspondingly raises the practical
stakes for Path B (or an equivalent fix) actually addressing serve
specifically, not stroke classification in general -- every other class
checked across this validation set (backhand, forehand) has classified
correctly on real footage; serve is the one specific, repeatedly-confirmed
failure.

---

## 2026-08-12: Two feature-set items built and verified end-to-end -- quality warning (item 5) and practice/match session-type selection (item 3)

**1. Item 5 -- quality warning.** `mobile/components/ResultView.js`: the
existing `pose_detection_rate` signal (already computed server-side, already
shown, but only as small plain text inside the bottom "Diagnostics" section)
now also gets a prominent pill ("Tracking quality: 96%") directly under the
confidence pill near the top of the screen -- not a new signal, purely more
visible framing of what already existed, exactly as scoped. Reuses the
existing `poseCaveat()` threshold/color logic (90% cutoff) so the pill turns
amber under the same condition the existing bottom-of-screen caveat text
already uses, for visual consistency rather than a second, independent
threshold. The original bottom Diagnostics text is untouched (still shows
pose detection % + processing time together).

**2. Item 3 -- practice/match session-type selection.** New multi-step flow
built directly into `mobile/app/(tabs)/index.js` (replacing the old
single-screen Home), mirroring `profile-setup.js`'s established step-state +
"< Back" pattern:
- **Practice** -> **By Stroke** (grid of the 5 `ir/schema.py:STROKE_CLASSES`
  stroke types, each with a plain themed-color placeholder swatch instead of
  a real illustration -- animated/real stroke art is explicitly deferred,
  item 1's mascot/asset work) -> select one -> capture.
- **Practice** -> **Beginner -> Intermediate -> Advanced** (the single fixed
  roadmap, no further sub-choice) -> capture.
- **Friendly Match** -> capture directly, singles only, no doubles option
  shown anywhere in the flow, per explicit scope.
- The final "capture" step keeps the existing Record Video / Pick from
  Library buttons unchanged, plus a summary row ("Practice · Serve",
  "Friendly Match (singles)", etc.) with a "Change" link back to step 1.

**Wiring, not a pipeline change, exactly as scoped.** The selected context
(`{mode, practiceType, strokeType}`) is JSON-stringified and passed through
router params into `processing.js`, which appends it as a new
`session_context` form field on the existing upload -- `backend/app.py`'s
`analyze_stroke()` accepts it as an optional, best-effort `Form(None)` field
(malformed/missing JSON never fails the request), and stores the parsed
dict verbatim in the response body -- which was already being persisted
whole into `StrokeHistory.result_json`, so this needed no new DB column or
migration. Classification, biomechanics, and coaching generation are
completely untouched; `session_context` is inert metadata riding along for
future use (e.g. item 7's cross-session digest).

**Verified end-to-end on the emulator, real sessions, not just UI
click-through:**
- **Practice -> By Stroke -> Serve**: UI correctly showed "Practice ·
  Serve" through to a real analyze-stroke upload; queried
  `backend/app.db` directly afterward and confirmed the exact stored
  `session_context = {"mode": "practice", "practiceType": "stroke",
  "strokeType": "serve"}`. Results screen showed the new "Tracking
  quality: 98%" pill correctly (teal, since 98% clears the threshold).
- **Friendly Match**: UI correctly showed "Friendly Match (singles)"
  through to a real upload; confirmed stored `session_context =
  {"mode": "match", "practiceType": null, "strokeType": null}`. Results
  screen showed "Tracking quality: 100%".
- **Practice -> Roadmap**: confirmed via UI click-through ("Practice ·
  Beginner -> Intermediate -> Advanced" correctly displayed) rather than a
  third full session -- same capture/upload code path as the other two,
  already proven correct, so a third ~90s+ session would have re-tested
  plumbing already confirmed working, not new code.

**One real environment gotcha hit and fixed during verification:** the
newly-pushed real clip (`clip5_serve_setup.mp4`, pushed to the emulator's
`Movies` folder) didn't appear in the system photo picker immediately after
a `MEDIA_SCANNER_SCAN_FILE` broadcast -- the picker's index lagged behind
the broadcast's completion. Worked around by using an already-indexed clip
for the first session (wiring verification doesn't depend on which real
clip is used) and confirming the picker had caught up by the second
session, where clip5 did appear and was used for real.

Both items now fully built and verified, per explicit instruction. Next up:
item 7, the cross-session digest -- the biggest remaining piece in this
feature set.

---

## 2026-08-12: Item 7 -- cross-session progress digest ("Option C"), digest mechanism + grounding + generation verified; UI not yet built, per explicit two-phase instruction

The biggest, most architecturally significant item on the list. This entry
covers parts 1-4 (digest computation, grounding extension, generation
wiring, adversarial testing) -- all verified working. Part 5 (UI) is a
proposal only in this entry, not yet built, per explicit instruction to
report back before UI polish.

**1. Digest computation (`ir/session_digest.py`, new; `SessionDigest` added
to `ir/schema.py`).** Fresh-computed from real stored `StrokeHistory` rows
every call -- no persisted digest table, no incremental update, no
"chain from a prior digest" code path anywhere. `compute_session_digest()`
takes a player's sessions (newest-first), filters to those with a real
biomechanics block, and computes, over a trailing window of `lookback_k`:
- `trailing_averages`: average PERCENTILE RANK (not raw value) per
  `ir/biomechanics_percentiles.py:RANKED_FEATURES` feature.
- `trend_deltas`: percentile-point change vs. the prior `lookback_k` window
  (empty dict, not omitted, if fewer than `2*lookback_k` usable sessions
  exist -- an explicit "no trend yet" signal, not a silently missing key).
- `stroke_type_mode`, `session_count`, `total_sessions_considered`,
  `trend_prior_window_size`, `source_session_ids` (the audit trail).

**Why percentile rank, not raw value -- a real methodological decision, not
incidental.** A raw feature's typical value differs by stroke class (e.g.
`wrist_height_ratio`'s demo values span ~-0.4 to ~5.7 across classes per
`ir/biomechanics_percentiles.py`'s own docstring) -- averaging raw values
across a mixed-stroke-type session history would silently conflate classes
into a meaningless number. Percentile rank is already class-normalized
(computed per-session against that session's own `predicted_class`), so
it's the only value that stays meaningful when averaged across a real
player's naturally mixed practice history. Reuses
`BiomechanicsPercentileBenchmark.percentile()` unchanged -- no new
percentile logic, this is the same one call sites already trust.

**Same 3-feature exclusion as percentile ranking, with MORE force here, not
less.** `shoulder_rotation_deg`/`hip_rotation_deg`/`forearm_orientation_deg`
are camera-facing-direction-dependent and already excluded from
`RANKED_FEATURES` for not being comparable across clips -- across SESSIONS
(potentially different days, different camera setups) that problem is
strictly worse, so the digest inherits the exclusion automatically (it only
ever touches `RANKED_FEATURES`).

**`DEFAULT_LOOKBACK_K = 5`, a judgment call, reasoned and stated as such,
not empirically tuned (no real usage data exists yet).** Three factors,
documented in the module: (a) noise floor -- this project's own real-footage
work already documented single-session pose-detection swinging 65-100%
clip-to-clip, so k=2-3 would let one unusual session dominate a "trend"
claim; (b) trial reachability -- the Phase 2 trial (`visiotennis_roadmap.md`)
runs over a bounded window with real players; a larger k (e.g. 10, needing
20 total sessions before ANY trend exists) would likely mean most
participants never see this feature fire, while k=5 needs only 10 sessions
for a real trend and gives a trailing average after just 1; (c) k=5/k=10
trailing windows are an unremarkable, common choice elsewhere (fitness/
analytics UIs), picked to be unsurprising, not novel. **Flagged explicitly
for revisiting once Phase 1's session-instrumentation logging has
accumulated real per-player session-frequency data during the actual
trial.**

**Real-data honesty check found and fixed a real gap before calling this
done:** running `compute_session_digest()` against `test1@example.com`'s 7
real stored sessions (from earlier work this session) produced a real,
non-empty `trend_deltas` -- but compared against only 2 prior sessions, not
a full `lookback_k=5`. Nothing in the original design distinguished a
thin-prior-window trend from a full one, which could present a
low-confidence comparison with the same apparent authority as a solid one --
the same "low_confidence tail" discipline `ir/biomechanics_percentiles.py`
already applies to thin percentile tails, just missing here. Added
`trend_prior_window_size` to the schema and wired an explicit caveat
instruction into the generation prompt whenever it's below `lookback_k` --
confirmed working in the live test below (the model spontaneously wrote
"though this trend is based on a small comparison window" unprompted by the
test, driven by the new instruction).

**2. Grounding extension (`rag/schema.py`) -- genuinely zero new
verification logic, as required.** `Citation.record_type` extended to
`Literal["StrokeEvent", "SessionRecord", "SessionDigest"]`. A `SessionDigest`
citation's `ir_field_path` (e.g. `"trailing_averages.wrist_height_ratio"`,
`"trend_deltas.knee_flex_right_deg"`) is a PLAIN dotted path -- it does NOT
start with `"biomechanics_percentile."`, so it never touches the special
percentile-recompute branch in `rag/grounding.py:check_grounding()` at all;
it falls straight through to the exact same `resolve_field_path()` +
`_values_match()` mechanism every other plain citation already uses. No new
`if` branch, no new function, exactly the "no new logic class" requirement.

**3. Wired into generation via a 5th chat route, `"progress_digest"`**
(`rag/schema.py:ChatRouteDecision`, `rag/chat_router.py`'s router prompt +
2 new worked examples distinguishing it from `"unavailable_comparison"`,
`rag/chat_pipeline.py`). A player asking "how have I been progressing?" (or
comparative phrasing like "is my backhand better than before?") gets routed
here instead of the old `unavailable_comparison` catch-all -- because a real
digest now exists for exactly this question. `build_progress_digest_prompt()`
(no KB retrieval -- no progress-framing knowledge base exists, out of scope
here) generates through the SAME unchanged `generate_grounded_response()`
retry-gate as tier 2, sourced from the digest instead of a `StrokeEvent`.
Uses the existing tone-conditioning (`build_tone_instruction`) unchanged.
`backend/app.py`'s chat endpoint gained `_compute_digest_for_user()` --
fetches ALL of a user's stored history (not a hard-capped query: a row limit
could silently undercount if recent sessions failed pose detection, since
`compute_session_digest` filters to usable ones AFTER fetching) and computes
a fresh digest before every single chat turn, never cached across turns.

**Skill-tier backstop (2026-08-12 earlier entry) extended to cover digest
citations too -- a real gap found by reasoning through the new feature, not
by accident.** `trailing_averages`/`trend_deltas` are themselves computed by
averaging the SAME flat, non-skill-stratified percentile population a direct
`biomechanics_percentile.` citation uses -- so a message combining trend
language with skill-tier framing ("has my progress been good for my
level?") carries the identical mislabeling risk. `check_for_skill_tier_mismatch()`'s
`_has_percentile_citation()` now also matches any `record_type=="SessionDigest"`
citation, not just the `biomechanics_percentile.` prefix. No digest exists
yet -> a fixed, code-authored `NO_DIGEST_AVAILABLE_MESSAGE`, never an LLM
attempt to describe a trend from nothing (same philosophy as
`UNAVAILABLE_COMPARISON_DISCLAIMER`).

**4. Adversarial testing -- both layers, same standard as the original
grounding harness and the skill-tier backstop, not just "the design was
sound on paper."**

*Deterministic (`rag/run_session_digest_grounding_check.py`, no LLM, a real
digest hand-copied from this project's own dev DB): 7/7 cases behaved
exactly as designed* -- wrong `record_id` -> `WRONG_RECORD`; fabricated
feature name -> `FIELD_NOT_FOUND`; a real field with an INFLATED, wrong-sign
fake trend value (+45.0 vs. the real -17.5, the actual "false trend claim"
case the instruction asked for) -> `VALUE_MISMATCH`; citing a trend field
against a digest whose `trend_deltas` is genuinely empty (a 1-session
player) -> `FIELD_NOT_FOUND` automatically, no special-casing needed, the
key simply isn't in the real dict; a legitimate correct citation ->
`GROUNDED` (proving the checker isn't just rejecting everything); the
skill-tier+digest combination -> mechanically `GROUNDED` but correctly
caught unsafe by the extended backstop; and a control case (same citation,
no skill-tier framing) -> correctly left alone, proving the backstop isn't
over-triggering on every digest citation regardless of message content.

*Live (`rag/run_progress_digest_pipeline_trials.py`, real qwen3:8b, the
real 7-session digest above): 4/4 cases behaved safely* -- "how have I been
progressing lately?" and "is my backhand better than it was before?" both
correctly routed to `progress_digest`, generated real grounded citations,
and the model spontaneously self-caveated the thin prior window per the new
instruction; "has my progress been good for a player at my level?" was
routed by the classifier itself straight to `unavailable_comparison`
(rule 4) without even needing the backstop to intervene -- one of two valid
safe outcomes, consistent with this session's earlier skill-tier backstop
verification (the deterministic test is what proves the backstop's own
mechanism, independent of whether any single live attempt happens to
exercise it); `digest=None` correctly showed the fixed message with zero
generation attempted.

**Digest mechanism and its grounding: verified working, both layers,
before any UI was touched, per explicit instruction.**

**5. UI -- proposed, not yet built.** See the chat reply for the proposal
itself (a suggested-prompt chip in the existing chat UI, e.g. "How am I
trending?", plus a small trend indicator on Results when a real digest with
a non-empty trend exists) -- reported back for confirmation before building,
per explicit instruction not to build UI without proposing first.

---

## 2026-08-18: Item 7 part 5 (UI) built and verified -- closes out item 7 and this session's full feature list

**Built exactly the proposed chip, nothing more.** `mobile/components/StrokeChat.js`:
a `PROGRESS_PROMPT_TEXT = "How am I trending?"` pressable pill rendered
unconditionally under "Ask about this stroke" (not gated on digest
availability -- the pipeline already handles "no digest" gracefully, so the
UI doesn't need to know in advance), sending that exact string through the
existing `handleSend` path. `handleSend` refactored to accept an optional
`textOverride` so the chip and the free-text input share one code path.

**Two real bugs found and fixed before any testing, by reasoning through
the diff, not by trial and error:**
- `Pressable onPress={handleSend}` and `TextInput onSubmitEditing={handleSend}`
  both implicitly pass a synthetic event object as the first positional
  arg -- since `handleSend` now takes an optional string override, both call
  sites would have silently passed an event object where a string was
  expected. Fixed to `() => handleSend()` wrappers at both sites.
- `citationCaption()` only stripped `biomechanics_percentile.`/`biomechanics.`
  prefixes, so a digest citation like `trend_deltas.wrist_height_ratio` would
  have rendered as the raw dotted path with no sign or percentile framing.
  Added two dedicated branches (`trailing_averages.*` -> "X: Nth percentile
  avg", `trend_deltas.*` -> "X trend: +/-N percentile pts").
- `CoachBubble`'s `grounded` (teal-styling) check didn't include
  `"progress_digest"` at all -- a real grounded digest response would have
  rendered as plain, uncited text. Fixed, requiring `message.citation != null`
  specifically for that route so the `digest=None` fixed-message case (same
  route, no real citation) correctly does NOT get teal styling.

**Verified end-to-end on the emulator, two real accounts, per the explicit
two-part instruction:**

1. **Real multi-session history** (`test1@example.com`, 7 real stored
   sessions, password reset via direct DB manipulation since the original
   wasn't recorded -- reasoned acceptable, this project's own dev DB, not
   production data): tapped the chip, got a real teal-styled grounded
   response citing `shoulder_hip_separation_deg` (-51.7 pt trend),
   `stance_width_ratio` (+10.8), `elbow_angle_deg` (+24.8), each with a
   spontaneous "based on a small comparison window" caveat (the
   `trend_prior_window_size < lookback_k` case) -- numbers matched exactly
   what had been independently computed and verified earlier in this
   session against this same account's real data. Backend log confirmed a
   clean 200 OK chain (login -> history -> chat).

2. **Fresh/thin history.** **A reachability nuance found and reported
   honestly rather than silently substituted:** the true `digest=None` /
   `NO_DIGEST_AVAILABLE_MESSAGE` code path requires zero stored sessions --
   but zero sessions also means no session exists to open the chat screen
   from at all, making that exact state structurally unreachable via the
   live UI (it was already verified directly via
   `rag/run_progress_digest_pipeline_trials.py`'s automated test instead).
   The realistic UI-reachable equivalent is a fresh account with exactly
   ONE real completed session (`digest` exists, `trend_deltas={}`, a
   genuine "thin history" state) -- built one: created
   `thinhist@example.com` fresh, logged in, completed one real
   Practice-roadmap session on the emulator (backhand, 100% pose detection,
   confirming item 5's "Tracking quality: 100%" pill), then tapped the
   chip. Got a real, correctly-grounded, teal-styled response: *"With only
   1 session recorded so far, we don't yet have enough data to identify any
   trends in your progress. Let's track across more sessions to see
   patterns."* -- cited `session_count: 1`, no fabricated trend, no error.
   Backend log confirmed a clean 200 OK chain (signup -> login ->
   analyze-stroke -> chat).

**Item 7 (digest computation + grounding extension + generation wiring +
adversarial testing + UI) is now fully built and verified, both mechanically
(deterministic adversarial harness) and live (real qwen3:8b, two real
accounts spanning full and thin history).**

---

### Full feature-list status, this session's plan, as requested

**Built and verified this session:**
- **Item 5 -- quality warning.** "Tracking quality: X%" pill on Results,
  reusing the existing pose-detection signal and its existing teal/amber
  threshold, made prominent instead of buried at the bottom.
- **Item 3 -- recording flow.** Practice (By Stroke, 5 types, placeholder
  swatches) or Beginner->Intermediate->Advanced roadmap, or Friendly Match
  (singles only) -- wired as inert `session_context` metadata on the
  existing upload, zero pipeline changes.
- **Item 7 -- cross-session digest.** Fresh-computed trailing
  average/trend/mode/count over `lookback_k=5`, `SessionDigest` grounding
  citation type, 5th chat route (`progress_digest`), skill-tier backstop
  extended, adversarially tested (7/7 deterministic + 4/4 live), UI chip
  built and verified against both full and thin real history.

**Explicitly deferred, not started this session:**
- **Item 1 -- mascot/logo/illustration assets.** Stroke-type picker in item
  3 uses plain themed-color placeholder swatches instead of real
  animated/illustrated art, by explicit scope decision when item 3 was
  built.
- **Item 6c -- post-point trim.** Not started.
- **Path B (video-based fine-tuning as a StrokeGRU alternative), actual
  training.** Feasibility report only (2026-08-12): verdict "blocked on
  data, not hardware" -- no training code written. A follow-up real-footage
  validation dataset was later obtained (addressing feasibility finding #1),
  but no training run has been executed against it.

No other item on this session's plan remains open.

---

## 2026-08-18: Path B training started -- data prep + backbone done, smoke test passed with a real bug found and fixed; full run NOT started, per explicit two-phase instruction

**1. Data prep.** `training/stroke_classification/prepare_video_split.py`
(new) splits the 1980 raw THETIS clips (downloaded 2026-08-12,
`data/VisioTennis/Dataset/StrokeClassification/raw_rgb/<class>/*.avi`) into
train/val/test.

**Went one level stricter than "video-level, not frame-level" and found a
real gap in EXISTING work while checking.** Each `.avi` is already one
atomic clip (no per-frame leakage risk by construction), so the real
question was whether repeated temporal-window sampling from a clip could
leak -- it can't, since window sampling always reads from a file already
assigned to one split. But checking prompted a check of subject identity
too: querying the EXISTING pose-based split
(`processed/{train,val,test}/labels_*.csv`, used by this same directory's
StrokeGRU model) found **all 55 players present in all 3 of its splits --
100% player overlap, not just partial.** That split is safe against frame
leakage but not against a model learning a player's body/clothing instead
of the stroke. Not fixed retroactively (out of scope, StrokeGRU already
trained and evaluated on it), but this script does NOT repeat it: split at
the PLAYER level instead -- 55 players -> 39 train / 8 val / 8 test
(fixed seed 42), each player's full 36-clip set (12 classes x 3 shots)
assigned wholly to one split. Confirmed by direct inspection before
splitting: every one of the 55 players contributes exactly 3 shots to
every one of the 12 native classes, zero exceptions -- so a player-level
split is automatically class-balanced with no stratification logic needed.
Verified post-split, not assumed: zero player overlap across all 3 splits,
zero duplicate filepaths, exact per-class counts printed (train
351/351/351/234/117, val and test 72/72/72/48/24 each, for
forehand/backhand/serve/volley/smash -- the same 3:1 smash-minority
imbalance already known from the pose-based model).

**Label mapping: reused, not re-decided.** Same native-12 -> collapsed-5
collapse already chosen for StrokeGRU (`dataset.py:NATIVE_TO_COLLAPSED`),
re-keyed by folder name instead of `label_idx` so both models target an
identical taxonomy -- an `assert` at prep-time confirms the raw_rgb folder
set matches this mapping's keys exactly.

**2. Backbone.** `r2plus1d_18` (torchvision, Kinetics-400 pretrained), per
the feasibility report's own accuracy/compute verdict. Real head
replacement (`model.fc = nn.Linear(512, 5)`) in `train_video.py`, not the
VRAM probe script's adapter-on-top-of-400-way-head shortcut (that was
explicitly flagged in the probe's own docstring as a measurement
simplification, not the real architecture).

**3. Smoke test -- ran, and caught a real bug via the results, not
trusted blind.** `training/stroke_classification/dataset_video.py` (new,
OpenCV decode straight from the raw `.avi`s, preprocessing matched exactly
to `R2Plus1D_18_Weights.DEFAULT.transforms()` checked directly --
resize 128x171/crop 112x112 -- THETIS's native 640x480 is the same 4:3
aspect, so no letterboxing needed; Kinetics mean/std; random
window+crop+flip for train, deterministic center for val/test, consistent
across all 16 frames of a clip). First smoke run's `max_samples` slice
(`df.iloc[:32]`) silently pulled 32 rows that were ALL "backhand" (the
manifest's alphabetical native-class ordering puts backhand/backhand2hands/
backhand_slice first) -- class weights came out `[inf, 1.0, inf, inf, inf]`
and macro-F1 was trivially pinned at 0.2 test, a degenerate single-class
run, not a real check. **Fixed**: `VideoClipDataset` now round-robins
across the 5 collapsed classes when subsetting, so even a tiny smoke subset
stays genuinely multi-class.

**Re-run, real signal.** 30 train clips (6/class, balanced) + 5 val clips
(1/class), 3 epochs, batch_size=4, **with a real qwen3:8b actually loaded
in Ollama** (confirmed via a live `/api/generate` call + `nvidia-smi`
before starting, 7155MiB used, matching the feasibility report's earlier
~7.1GB figure):

| epoch | train_loss | train_acc | train_macroF1 | val_macroF1 |
|---|---|---|---|---|
| 1 | 1.568 | 0.300 | 0.244 | 0.267 |
| 2 | 1.140 | 0.600 | 0.583 | 0.267 |
| 3 | 0.721 | 0.833 | 0.832 | 0.233 |

Train loss decreases monotonically and substantially, train accuracy/F1
climb sharply -- the loop is genuinely learning (overfitting a 30-sample
subset, exactly as expected for a smoke test, not a generalization claim).
Val (n=5) is too small to mean anything and isn't being read as a real
result -- confirms the eval path runs without crashing, nothing more.

**VRAM, real, with Ollama loaded, `nvidia-smi`-verified (not
`torch.cuda.mem_get_info()`, per the feasibility report's own documented
distrust of that call on this machine):** peak_alloc=1.63GB,
peak_reserved=1.83GB for the training process itself; `nvidia-smi` total
8.83GB (Ollama's ~7.15GB + this run's ~1.7GB) against 12GB total, comfortably
inside budget, zero OOM. **Same ballpark as, and slightly under, the
feasibility probe's synthetic-tensor estimate** (2.10-2.34GB peak at the
same batch_size=4) -- the real head-replacement model is a bit leaner than
the probe's adapter-wrapped one, consistent with expectations, not a
discrepancy worth chasing further.

**Epoch time -- measured, but flagged honestly as not yet a trustworthy
full-run projection.** train_time 5.6-6.2s / val_time 4.7-4.9s for this
tiny 30/5-sample, 8/2-batch smoke run. A large fraction of that is almost
certainly one-time `DataLoader` worker-process startup (`num_workers=4`,
spun up fresh per split, amortized over only 2-8 batches here) plus cuDNN
warmup on epoch 1 (6.2s vs. 5.6-5.7s on epochs 2-3) -- NOT a reliable
per-batch steady-state number to linearly extrapolate to the full
1404-clip train / 288-clip val split (351 / 72 batches at batch_size=4).
Real per-epoch time on the full run will be measured directly and reported
after it runs, not projected from this smoke sample -- consistent with the
feasibility report's own caveat that real dataloader/decode overhead, not
raw GPU compute, is expected to be the actual bottleneck for video
training.

**Smoke test passed. Full training run NOT started -- reporting back
first, per explicit instruction.**

## 2026-08-20: KNOWN OPEN ITEM (high-priority, not urgent) -- existing StrokeGRU split is leaky, headline numbers unvalidated

Flagging explicitly so this isn't lost. Building Path B's player-level split
(`prepare_video_split.py`, above) required checking the EXISTING pose-based
StrokeGRU split for the same leakage class, since the court-detection project
already burned us once on this. The check: direct Python comparison of the
player-ID sets across `data/VisioTennis/Dataset/StrokeClassification/processed/
{train,val,test}/labels_*.csv`.

**Result: 100% player overlap across all three splits** -- all 55 players
appear in train, val, AND test simultaneously (`train&val overlap: 55`,
`train&test overlap: 55`, `val&test overlap: 55`). This is the same class of
leakage as the original court-detection bug (`training/court_detection/
prepare_data.py`'s `GROUP_SPLIT` fix), just never caught until this session's
Path B data-prep work surfaced it by comparison.

**Consequence: the currently-reported StrokeGRU headline numbers -- 84.5%
accuracy / 0.834 macro-F1, already written into the research materials --
were measured under leaky (subject-identity-overlapping) conditions.** The
model may have partly learned to recognize *players* (body shape, clothing,
court/camera setup per recording session) rather than purely the *stroke*
being performed. These numbers have NOT been re-validated on a clean,
player-disjoint split the way court-detection's numbers were after its fix.

**Open item, not urgent right now but must not be lost:** once Path B's video
model training run is done, the original StrokeGRU should get the same
clean-split retrain-and-compare treatment court-detection already received --
retrain (or at minimum re-evaluate) on a player-disjoint split and compare
against the existing 84.5%/0.834 numbers, so the paper's headline result is
either confirmed solid on clean data or corrected. Left resting on an
unverified split until then.

## 2026-08-20: Path B full (non-smoke) training run started

Smoke test approved by explicit user go-ahead. Starting the full run per step
4 of the original instruction: `python training/stroke_classification/
train_video.py` with real (non-smoke) defaults -- `--batch-size 4 --epochs 30
--lr 1e-4`, full 1404-clip train / 288-clip val manifests (no `max-samples`
subsetting), checkpointed every epoch to `training/stroke_classification/
checkpoints/video/{r2plus1d_last.pt every epoch, r2plus1d_best.pt on val
macro-F1 improvement}` -- same standard as every other model in this project,
protects against losing progress to any interruption. Run launched with
qwen3:8b concurrently loaded in Ollama, matching the smoke test's validated
concurrent-usage condition (the realistic deployment scenario being tested).
Real per-epoch time and final per-class results (including test-set
evaluation against the real 6-clip footage set) to be reported once the run
completes, not projected in advance.

## 2026-08-20: Path B full training run complete + real-footage validation -- serve failure NOT fixed

**Training complete, 30/30 epochs, checkpointed every epoch as specified**
(`checkpoints/video/{r2plus1d_last.pt, r2plus1d_best.pt}`). Real per-epoch
timing held rock-steady across the whole run: ~57.6-59.4s train + ~16.4-18.2s
val, ~75s/epoch average, matching the epoch-1 projection closely -- no
dataloader-worker-startup skew at full scale. **Best checkpoint: epoch 20,
val_macroF1=0.735** (val_acc=0.764, per-class F1: forehand=0.780
backhand=0.794 serve=0.835 volley=0.705 smash=0.560).

Honest note on the full trend, not glossed over: train_macroF1 climbed
steadily to 0.836 by epoch 30 while val_macroF1 was noisy and did NOT
improve monotonically -- it peaked at epoch 20 and epoch 30's val_macroF1
was only 0.681, a widening train/val gap consistent with overfitting on this
dataset size in later epochs. This is exactly why "best" (not "last") is the
checkpoint carried into evaluation. Also note: qwen3:8b's Ollama residency
was only confirmed for the first ~3-4 epochs of this run (`nvidia_smi_used`
dropped from 7.71GB to 2.39GB starting epoch 4, consistent with Ollama's idle
keep-alive timeout unloading it) -- most of the run did not actually have
the concurrent-Ollama condition holding, reported plainly rather than
silently claimed.

**Step 5: real-footage validation (`evaluate_real_footage.py`, new).**
Evaluates the best checkpoint against every hand-verified real (non-THETIS)
stroke instance already on record in this log from prior diagnostic work:
clip1's 2 confirmed serves (frames 155/235), clip2's 9 confirmed backhand
instances (frames 34-574), and clip4/5/6's whole-clip labels (backhand/
serve/forehand respectively). clip3 excluded (known pose-tracking failure,
never valid for classification-accuracy conclusions). 14 labeled instances
total, 16-frame clips sampled centered on each hand-verified contact frame
(or clip center for the whole-clip cases), same preprocessing as val/test.

| class | correct | total |
|---|---|---|
| serve | **0** | 3 |
| backhand | 5 | 10 |
| forehand | 0 | 1 |
| volley | -- | 0 (no real instance exists anywhere in this project) |
| smash | -- | 0 (no real instance exists anywhere in this project) |
| **overall** | **5** | **14** (35.7%) |

**The specific problem Path B exists to fix is NOT fixed on this set: 0/3
real-footage serves correctly classified, identical to the pose-based
model's pre-existing 0/3 record** (clip1 frame235, clip1 frame155, and
clip5 all still misclassified -- clip1's two windows both called "backhand,"
clip5 called "volley"). This is the entire point of this exercise per the
original instruction, and the honest result is that fine-tuning a
video-based model on THETIS's raw RGB footage, at least at this data scale
and epoch budget, did not close the real-footage serve gap. Overall
real-footage accuracy (35.7%) is also far below both the training accuracy
(85.5%) and the THETIS-native val macroF1 (0.735) reached during training --
a real domain gap between THETIS-trained performance and actual real-world
footage, the same class of gap already documented for the pose-based model,
not resolved by switching model architecture alone.

**Caveats, stated in the same breath as the result, not separately:**
this is a 14-instance set covering only 3 of 5 classes -- no real volley or
smash instance exists anywhere in this project, and forehand has exactly
ONE instance (0/1, statistically meaningless on its own). clip2's 9 backhand
instances all come from a single video/player/session, so they are not 9
independent samples of "real-world backhand" -- more like 9 correlated draws
from one recording condition. This set should not be read as a rigorous
test-set accuracy number, and it isn't being reported as one -- it's a
small, honestly-graded spot check, and on that spot check, the serve problem
persists. Whether a larger real-footage validation set, more training data,
different frame-sampling within each stroke window, or a fundamentally
different fix is needed remains open -- not concluded here.

## 2026-08-20: THETIS serve visual-distinctness check -- dataset is NOT the ceiling on serve

Cheap, fast diagnostic before investing further in a real-footage-data fix
for serve specifically: are THETIS's own serve clips visually distinctive
from its other classes, even to a human? `ir/run_thetis_serve_visual_check.py`
(new) pulled 3 frame grabs (30%/55%/80% through) each from 6 serve clips (2
players x flat/kick/slice) and 6 backhand/forehand clips, tiled into one
contact sheet (`ir/example_output/thetis_serve_visual_check.png`), inspected
directly by eye -- not automated, same manual-verification spirit as the
real-footage frame grabs done earlier this session.

**Result: THETIS's serve clips are clearly, consistently visually distinct
from backhand/forehand -- not an ambiguous, hard-to-tell-apart class in the
training data itself.** All 6 serve clips show the same unmistakable
overhead pose across all 3 sampled frames: arm/racket raised straight up
during a ball toss, body upright facing the net, contact at full overhead
extension, follow-through back down past the head. Backhand/forehand clips
never look like that -- racket stays at waist/shoulder height, body rotated
sideways, swinging across or forward. A human could sort these into
serve/not-serve from the overhead-arm silhouette alone, in a single frame,
with no ambiguity.

One real but different finding: **the 3 serve subtypes (flat/kick/slice)
look nearly identical to each other frame-by-frame** -- differentiated by
racket-face angle and ball spin/trajectory (not visible in a still frame),
not overall body pose. Not a problem for the collapsed 5-class taxonomy;
if anything it means the 3 native subclasses reinforce a single coherent
"serve" visual signature rather than diluting it.

**Conclusion: THETIS's own serve data is not the bottleneck.** The dataset
gives a clean, human-visually-distinctive serve class. Path B's repeated
real-footage serve failures (0/3, this session's earlier entry) and the
pose-based model's pre-existing 0/3 record point to a genuine domain-gap
problem (camera framing/distance, real-world variability -- consistent
with the already-documented THETIS-fixed-camera-vs-real-footage finding
from 2026-08-12), not mislabeled or visually-ambiguous training data.
Pursuing a different/larger real-world serve data source looks like the
right next lever, not something to abandon because THETIS itself is
supposedly bad at representing "serve" -- it isn't.

## 2026-08-20: StrokeGRU clean-split retrain -- headline 84.5%/0.834 does NOT hold, corrected to 68.8%/0.647

Closes the high-priority open item flagged earlier this session (100%
player overlap across the original StrokeGRU train/val/test split). Same
treatment court-detection's leaky split already received: clean re-split,
retrain, honest before/after comparison, research materials corrected.

**`prepare_clean_pose_split.py` (new)** merges the existing (leaky-split)
`processed/{train,val,test}/X_*.npy` + `labels_*.csv` back into one
1980-clip pool -- the extracted pose data itself was never the problem,
only the split assignment -- then re-splits by player identity using
**the exact same `split_players()` function already built and verified for
Path B** (imported directly, not reimplemented, `SPLIT_SEED=42`, 39/8/8
players), guaranteeing an identical, independently-verified-zero-leakage
split to the one Path B's video model trained on
(`data/.../processed_clean/`, verified zero player overlap, same 1404/288/288
clip counts and balanced per-class counts as Path B's split).

**Retrained StrokeGRU on the clean split**, same architecture/
hyperparameters/40-epoch budget as the original run (`train.py`, now with
a `--data-root` override added so both splits can be trained from the same
script without duplication). Best checkpoint: epoch 36/40, val_macro_f1=0.741
(vs. the original leaky run's val_macro_f1=0.828 at epoch 39/40 -- already a
visible drop before even touching the test set).

**Test-set result (`evaluate_clean_split.py`, new; n=288, disjoint players
from train/val): accuracy 68.8%, macro-F1 0.647 -- both well below the
original 84.5%/0.834, and below the FYP proposal's F1>82% target.**

| class | original F1 (leaky) | clean F1 | delta |
|---|---|---|---|
| forehand | 0.828 | 0.694 | -0.134 |
| backhand | 0.879 | 0.764 | -0.115 |
| serve | 0.933 | 0.731 | -0.202 |
| volley | 0.721 | 0.685 | -0.036 |
| smash | 0.809 | 0.364 | **-0.445** |

**The drop is real and substantial, not noise -- the honest answer is that
the original number does NOT hold.** Smash collapses hardest (0.809 ->
0.364): 14/24 true smashes are now misclassified as serve, a confusion that
was minor and in the opposite direction in the leaky matrix (6/25). Forehand
loses most of its drop to volley (17/72), the same direction as the original
run's own genuine forehand/volley confusability finding -- now worse without
leaked player-identity cues propping up recall. Serve and backhand hold up
comparatively best but still drop meaningfully (-0.202, -0.115).
**Interpretation: a real, non-trivial share of the original 84.5%/0.834 was
the model partially learning individual players' body proportions/movement
signatures that happened to co-occur across the leaky train/val/test split,
not purely stroke technique** -- precisely the failure mode player-disjoint
splitting exists to catch, and precisely why this check mattered before
letting the number stand in the paper.

**Research materials updated**: `training/stroke_classification/README.md`
now leads with the clean-split numbers as the current, citable result, with
the original leaky-split section kept below as historical record (its
volley/forehand confusability analysis is still a valid, independent
finding) but explicitly marked superseded. **The FYP proposal's F1>82%
target is not currently met on a clean, player-disjoint split** -- this is
the number future work (including any further architecture or data changes)
should be measured against, not 84.5%/0.834.

---

## 2026-08-20: System-adaptive light/dark theme -- mockup revised, task 2 (live emulator toggle) verified for real, one real root-cause bug found and fixed

Picks up the UI/UX phase kicked off earlier this session: a system-adaptive
(light/dark, OS-driven) theme replacing Phase E's single hardcoded palette,
"inspired by SwingVision's energy, not a copy of their specific palette or
mascot." Mockup went through one real revision cycle before any code was
touched, same review process as every design phase in this project:

**Mockup round 2 -- two user-requested fixes, both applied and re-verified:**
1. **A third, neutral color for dark-mode buttons, not accent.** Round 1's
   mockup reused the new jade/emerald accent as the dark-mode primary-button
   fill (reasoning: navy-on-near-black nearly disappears) -- correctly
   flagged as wrong, since it made accent common instead of rare, diluting
   the one signal it's meant to carry. Replaced with a new neutral slate
   (light `#B7C0D4`, navy `#16233F` reused as its text -- a deliberate
   callback to the brand anchor). Audited all 8 mocked screens for real
   accent usage afterward and found two more mapping bugs in the mockup
   itself (the stroke-picker placeholder swatches and the "Change" link were
   both wrongly wired to accent, not matching the real app's actual
   `colors.tealTint`/`colors.navy` usage) -- fixed those too. Net result:
   accent appears exactly once across all 8 screens, identically in both
   themes (the grounded chat bubble).
2. **Hairline card/input border strengthened to real 3:1, not exempted.**
   Checked first, as asked, whether the border is genuinely the sole
   delimiter: `surface` vs `bg` sit at 1.11-1.15:1 on their own in both
   themes -- visually indistinguishable, no shadow/elevation cue doing
   backup work. So it's structural, not decorative. Strengthened both
   themes' `border`/`borderStrong` to clear 3:1 against both adjacent
   surfaces (light `#E1D9C8`->`#8A7857`, dark `#2A3448`->`#5F6F99`).

**Task 2: wired real code, not just proposed it.** `mobile/lib/theme.js`
now exports a `useTheme()` hook (`useColorScheme()`-driven) alongside the
existing static `colors` (kept as the light-only fallback for
not-yet-converted screens). Converted `app/_layout.js` (root Stack +
loading screen), `app/(tabs)/_layout.js` (tab bar), and `app/(tabs)/index.js`
(Home, including moving its `StyleSheet.create` into a per-render
`makeStyles(colors)` since a static stylesheet can't react to a live OS
change) to the reactive hook. Found and fixed a real token collision while
doing this: `colors.surfaceAlt` was being used for two different things in
the existing code -- an actual alternate-surface background
(`history/index.js`'s `insightsPlaceholder`) vs. "text sitting on a
`colors.navy` fill" everywhere else (Home, Settings, Login, Processing,
ProfileSetup, Results, Signup, FeedbackForm, StrokeChat all do the latter).
Introduced a new `onPrimary` token for the second meaning instead of
overloading `surfaceAlt` further, and renamed Home's 3 call sites --
whichever screen is converted next needs the same rename, noted directly in
theme.js's own comments so it isn't missed. Also swapped
`strokeImagePlaceholder`'s `tealTint` for a new neutral `placeholderTint`
in the real component, matching the mockup audit's finding, not just in
the mockup.

**Real bugs hit and fixed along the way, not glossed over:**
- Dropped the `Platform` import while rewriting theme.js's top (still
  needed by `fonts.serif`'s `Platform.select`) -- caused an immediate
  uncaught error on load (`Property 'Platform' doesn't exist`), caught by
  actually loading the app on the emulator rather than assuming the edit
  was safe. Fixed by importing both `Platform` and `useColorScheme`.
- **The real root-cause bug**: after the fix above, the app loaded fine but
  toggling the emulator's OS dark mode (`adb shell cmd uimode night yes`,
  confirmed for real via `dumpsys uimode` -- `mComputedNightMode=true`, not
  just assumed from the command's own claim) did nothing, even after a full
  app restart. Isolated by comparing against Expo Go's own chrome, which
  DID go dark correctly on the same toggle -- ruling out "the OS toggle
  didn't really work" and pointing at something project-specific. Found by
  reading `mobile/app.json` directly: `"userInterfaceStyle": "light"`,
  left over from Phase E's deliberate single-fixed-theme era, hard-locking
  `Appearance`/`useColorScheme()` to always report light regardless of the
  real OS setting. Changed to `"automatic"`, restarted Metro with a cleared
  cache (an app.json change isn't picked up by Fast Refresh), force-stopped
  and relaunched Expo Go fresh to get the new manifest.

**Verified live on the vt_test emulator, the actual point of task 2:**
after the `userInterfaceStyle` fix, toggled `adb shell cmd uimode night
yes`/`no` twice each direction *while the app sat open in the foreground*
(no restart, no background/foreground cycle) and the real Home screen
switched palettes instantly both times -- dark navy bg, light heading ink,
the new neutral-slate primary button with navy text, slate outline
button/tab-bar icons; back to warm-paper bg, navy primary button, navy
outline on the reverse toggle. Matches the approved mockup exactly, screens
saved (`theme_dark6.png`, `theme_live1.png`, `theme_live2.png`).

**Honest scope note -- not fully done.** Only Home + the two shared layout
files are converted so far. Everything else (History, Settings, Login,
Signup, Processing, Results, ProfileSetup, FeedbackForm, StrokeChat) still
renders the static light-only palette regardless of OS setting -- confirmed
by the code, not assumed. The full re-skin (same `useTheme()` +
`makeStyles()` pattern, plus the `surfaceAlt`->`onPrimary` rename
everywhere it applies) is the remaining work, not yet started.

**Cleanup, reverted exactly.** `mobile/lib/config.js`'s `BACKEND_URL` was
temporarily repointed at a locally-run backend (`http://10.0.2.2:8000`,
same pattern as Phase E's own verification passes) since the Tailscale
Funnel wasn't up this session -- reverted back to the real Funnel URL
afterward, and the local backend process stopped. Emulator's OS appearance
setting restored to light (its state before this test began).

---

## 2026-08-21: System-adaptive theme -- remaining 9 screens converted and verified live, one more real bug caught before it shipped

Finished the re-skin left open at the end of the previous entry: History,
Settings, Login, Signup, Processing, Results, ProfileSetup, FeedbackForm,
StrokeChat, plus every shared component they depend on
(`components/ResultView.js`, `components/charts/TrendChart.js`,
`components/charts/StrokeBreakdownChart.js`,
`app/(tabs)/history/_layout.js`, `app/(tabs)/history/index.js`,
`app/(tabs)/history/[sessionId].js`) all converted to the same
`useTheme()` + per-render `makeStyles(colors)` pattern established for
Home. Verified each one live on the vt_test emulator in both light and
dark before moving to the next, per the explicit instruction this phase
was scoped to.

**The `surfaceAlt` token-collision bug, run to ground everywhere.** Grepped
every remaining `colors.surfaceAlt` call site across the converted screens
and renamed each "text sitting on a `colors.navy` fill" usage to the
`onPrimary` token introduced last entry (Settings, Login, Signup,
ProfileSetup, Processing, Results, FeedbackForm, StrokeChat -- 8 files, ~10
call sites). Left `history/index.js`'s `insightsPlaceholder` alone, the one
genuine alternate-surface-background usage, confirmed by re-reading the
code rather than assuming. Two components needed an extra step because
they'd been closing over a module-scope `styles` constant that no longer
exists now that styles are computed per-render: `OptionRow` (profile-setup),
`ChoiceRow` (FeedbackForm), `CoachBubble` (StrokeChat), and
`InsightsSection` (history/index) now all take `styles` (and `colors` where
needed) as an explicit prop instead.

**A second real bug, caught by reading before it was ever exercised live:**
`FeedbackForm.js`'s submit spinner was `<ActivityIndicator color="#fff" />`,
hardcoded white from before the theme work existed. Against the new
dark-mode slate button fill (`#B7C0D4`) a white spinner would have been
low-contrast but not invisible -- exactly the kind of regression that's
easy to miss in a quick look and only shows up mid-submit. Fixed to
`colors.onPrimary` before it ever rendered for real.

**Session got interrupted mid-verification (context compaction) with
Processing/Results/FeedbackForm still unverified** -- the emulator, Metro,
and local backend had all been torn down by the time work resumed, and the
previous session's blocker (Android's system video picker not registering
taps) was still unresolved. Recovered by relaunching everything from
scratch: emulator (`vt_test` AVD failed to load its `default_boot`
snapshot on the first relaunch attempt and silently exited -- fixed by
relaunching with `-no-snapshot-load` to force a clean cold boot instead),
local backend (`uvicorn`, already found running from before the
interruption -- port 8000 was already bound, confirmed via a real request
rather than assumed), and Metro/Expo Go. App resumed already logged in
(the auth token persists in AsyncStorage across a Metro/emulator restart).

**The photo-picker blocker from last session, actually solved.** The
previous session's failure was mis-diagnosed as "the picker isn't
registering taps" -- rereading this session's own established discipline
(always pull real element bounds via `uiautomator dump`, never estimate
from a screenshot's displayed pixels) and applying it properly to the
picker itself immediately found the fix: the system picker's grid items
*are* real accessibility nodes (`content-desc="Video taken on Aug 1, 2026,
5:02:08 PM with duration 00:03"`, `clickable="true"`), and tapping the
dumped bounds' exact center selected the pushed test clip on the first
try. The earlier session's repeated failures were the same screenshot-
scale mismatch bug already diagnosed and fixed for in-app taps, just not
yet applied to picker automation specifically.

**Verified live with a real end-to-end analysis run, not a static
inspection.** Selecting the test clip drove the actual pipeline through to
completion (~90s: pose detection, StrokeGRU classification, biomechanics
percentiles, Ollama-generated coaching text) and landed on a real Results
screen with genuine low-confidence-caveat content (52% confidence, 87%
tracking quality -- both caveat-styled) -- checked in both themes,
including scrolling to and interacting with FeedbackForm's chip
selection/rating/Submit states (disabled -> `colors.borderStrong` fill,
enabled -> `colors.navy` fill with `colors.onPrimary` text, both legible in
both themes) rather than only screenshotting the static unselected form.
Signup was re-screenshotted live in both themes too, since its `makeStyles`
conversion from earlier hadn't been re-confirmed live in this specific pass.

**Result: all 9 named screens plus every shared dependency are now
`useTheme()`-driven and confirmed live in both light and dark on the real
emulator.** No screen in the app still renders the static light-only
`colors` export. Cleanup repeated exactly as before:
`mobile/lib/config.js`'s `BACKEND_URL` reverted to the Tailscale Funnel
URL, local backend and Metro processes stopped, emulator OS appearance
left on light.

---

## 2026-08-21: Confidence calibration (temperature scaling) + two coaching-text prompt refinements, both verified live with real before/after examples

Two independent refinement tasks, scoped to existing components only.

### Task 1: confidence calibration

**Confirmed which model is actually live before touching anything, per
explicit instruction -- and the answer wasn't what was assumed.**
`backend/app.py` loads `checkpoints/stroke_gru.onnx`, exported from
`stroke_gru_best.pt` -- the ORIGINAL **leaky-split** checkpoint (README.md
calls it the deliverable checkpoint in its leaky-split section: 84.5%
accuracy / 0.834 macro-F1, 100% player overlap train/test). The 2026-08-20
clean-split retrain (`checkpoints/clean_split/stroke_gru_clean_best.pt`,
68.8% / 0.647 honest player-disjoint numbers) was never exported to ONNX
and was never wired into the deployed pipeline. Flagged this to the user
before proceeding rather than silently calibrating whichever one was
assumed; user chose to calibrate the model actually live today (leaky-split),
not swap in the weaker clean-split checkpoint as an undiscussed side effect.

**Calibration script** (`training/stroke_classification/calibrate_temperature.py`):
loads the deployed checkpoint directly (not the ONNX export, for
gradient-based fitting), fits a single scalar T via LBFGS on the
leaky-split's own held-out val split (this checkpoint's real val set, 297
samples) minimizing NLL (Guo et al. 2017 method), reports ECE via a
10-bucket reliability breakdown before/after on the leaky-split test set
(297 samples, never touched during fitting).

**Result: test-set ECE 7.01% -> 3.83%** (T=1.4474), accuracy unchanged at
84.5% (temperature scaling is argmax-preserving by construction). Reliability
table before showed the model over-confident in every bucket above 60%
(e.g. 90-100% bucket: 91.7% actual accuracy vs 98.0% average stated
confidence, an 6.4pp gap); after, that same bucket's gap shrank to 2.9pp.

**Additional honesty check, reported separately and clearly labeled (not
the set T was fit on):** ran the same deployed model + fitted T against the
clean-split test set (player-disjoint). Surprising result, flagged rather
than glossed over: 92.7% accuracy there, HIGHER than the leaky-split test's
own 84.5% -- not evidence the model generalizes well to new players, since
this deployed model's own training data already contains many other clips
from those same clean-split-test players (that's the exact leakage the
clean-split retrain exists to fix), so this isn't a leakage-free check for
THIS model. On this set, T=1.4474 actually made ECE worse (1.53% -> 5.98%)
-- the leaky-split's overconfidence pattern that motivated fitting T doesn't
hold on this different-distribution set, where the model runs slightly
underconfident already. Reported honestly as a real complication, not
hidden.

**Re-checked the two specific real-footage cases by name, raw vs.
calibrated side by side** (`training/stroke_classification/recheck_known_cases.py`,
reusing the actual `decode_and_resample_video`/`run_pose_extraction`
pipeline functions from `backend/app.py`):
- `clip5_serve_setup.mp4` (true label serve, confirmed via ball-toss +
  trophy position): predicted backhand, confidence **99.3% -> 96.1%**
  calibrated. Still wrong, still very overconfident -- temperature scaling
  measurably improves aggregate calibration but does NOT fix this specific
  case; reported as-is, not spun as a win.
- `clip1_serve_practice.mp4` (whole-clip upload, volley/72% confidence):
  **no ground-truth grading exists on record for this exact case** -- both
  prior log mentions (2026-07-20 entries) were incidental readouts during
  unrelated UI-failure-mode tests, not an accuracy check. Confidence
  **72.3% -> 61.8%** calibrated; flagged this gap to the user rather than
  assuming which case they meant.

**Wired into the live pipeline**, not left as an offline analysis.
`backend/app.py`: added `STROKE_TEMPERATURE_PATH`, loaded once at startup
into `_state["stroke_temperature"]` (falls back to T=1.0/no-op if the
artifact is missing, so a fresh checkout still runs). `run_stroke_classification()`
gained a `temperature=1.0` default parameter (argmax-preserving softmax
scaling) -- default keeps every `ir/run_*` diagnostic script's existing
raw-softmax behavior unchanged (5 call sites checked), only the live
pipeline's call site passes the real fitted value explicitly.

**Verified end-to-end, twice, independently:**
1. Direct API call (`curl` through the real `/api/v1/analyze-stroke`
   endpoint) with `clip5_serve_setup.mp4`: returned confidence
   **0.9608817...** -- matches the standalone recheck script's 96.1%
   exactly, confirming the same code path.
2. Real on-device run through the actual Android app UI (same clip used
   throughout this session's theme-conversion pass, `theme_test_clip.mp4`):
   this exact clip previously read **52% confidence** (screenshotted
   earlier this session, pre-calibration); after wiring calibration in, the
   same clip through the same UI now reads **46% confidence** -- the
   expected direction/magnitude for temperature softening a mid-confidence
   prediction. Results screen's caveat text also correctly recomputed
   downstream ("it could also be a backhand"), confirming the whole chain
   (API -> StrokeInfo -> Results screen -> caveat logic) reads the
   calibrated value, not a cached raw one.

### Task 2: coaching-text prompt refinements (`rag/prompt.py`)

**2.1 -- coach-tone hedging.** `build_tone_instruction`'s coach branch
(shared by session-level, stroke-level, and chat-followup prompts -- all
three call sites checked) now explicitly instructs the model to state cited
values plainly and trust the reader's expertise, name the two specific
hedge patterns to avoid, and explains WHY they appear: the KB reference
snippets (`rag/kb_biomechanics.py`) are written for a PLAYER audience and
routinely defer to "get a coach's eye on this" -- nonsensical when the
reader already IS that coach. Reserves hedging for genuine data-quality
caveats (single-instant measurement, camera-framing dependence, thin
sample) that hold regardless of audience.

Verified with real before/after generations against `stroke_event_biomech_volley.json`
(coach tone_profile), "before" reconstructed via monkeypatch (backend/rag
aren't git-tracked in this repo, so a stash-based revert wasn't available)
rather than assumed:
- BEFORE (2 samples): *"...consider checking if this aligns with your
  intended shot shape."* / *"...no immediate adjustments are needed unless
  you're experiencing specific balance issues."* / *"...no specific
  adjustment is needed based on this measurement alone."*
- AFTER (2 samples + 1 cross-stroke forehand check): *"Consider adjusting
  your stance or timing to ensure consistent contact."* / *"Focus on
  maintaining this momentum through contact to enhance power and
  consistency."* -- direct technical directives on the genuinely-deviant
  features, not deferrals. One mild residual softening remained on a
  near-median (72.8th percentile, not actually notable) feature ("No
  specific adjustment is needed here") -- judged appropriate, not a
  hedge-on-a-real-finding, since that specific value genuinely isn't
  deviant. Not a 100% elimination, reported honestly as a real, measured
  improvement in directness on the findings that matter, not a claimed-perfect fix.

**2.2 -- thin/generic bare-restatement pattern.** Added a new numbered hard
rule to `build_schema_instructions()`'s shared schema text (used by every
prompt built through `rag/prompt.py`/`rag/stroke_prompt.py`): a
`grounded_observation` must never stop at restating the cited number and
its deviation alone, must connect it to a concrete implication, and must
still do so (cautiously) even when no closely-relevant KB snippet was
retrieved. Renumbered the conditional percentile rule (8 -> 9) and its one
internal self-reference accordingly.

Verified by reproducing the exact original semantic-miss case named in
PROJECT_LOG.md's 2026-07-18 entry (`second_serve_ace_decline`, the
`ace_rate` retrieval near-miss that produced bare number restatement) plus
one second diverse session, both through the real gated pipeline:
- BEFORE (on record from 2026-07-18): *"Your second serve points won
  percentage was 9.07 (6.0 std below average), indicating a significant
  drop compared to typical performance."* -- number + generic label, no
  action.
- AFTER (reproduced fresh): *"Your second serve points won percentage was
  9.07%, which is 6.0 standard deviations below average. Consider reviewing
  your serve placement and strategy to improve consistency."* -- same
  citation, now with a concrete follow-up. All 3 items in the re-run showed
  the same pattern (number -> named implication -> suggested check/action).
  100% grounded, unchanged.

### Cleanup

`mobile/lib/config.js`'s `BACKEND_URL` was temporarily repointed at the
local backend twice during this task (once to test calibration against a
flaky Funnel connection from a freshly-cold-booted emulator, isolated as a
Funnel-specific reachability issue -- confirmed working from the host via
direct `curl` both before and after -- not an app bug) and reverted to the
real Tailscale Funnel URL both times, byte-identical. Backend (port 8001)
and Metro left running, since the user separately asked to connect their
own phone to the app this same session (LAN URL `exp://192.168.1.7:8081`)
and that request was still open.

**Follow-up, same session: real phone connect hit the same Funnel failure
mode.** The user connected via Expo Go over LAN and got a real
"Network request failed" on Sign Up against the Tailscale Funnel URL --
confirming this wasn't an emulator-only artifact (curl from the host kept
succeeding against the same URL throughout, both times). Switched
`BACKEND_URL` to the PC's LAN IP (`http://192.168.1.7:8001`) instead --
matches `backend/app.py`'s own docstring, which already documented this as
the intended same-Wi-Fi testing path, not a workaround. Found and reported
(not fixed -- prohibited from modifying security settings) a real
contributing issue: the existing "VisioTennis Backend" Windows Firewall
rule only allow-lists port 8000, but the backend has run on 8001 since the
2026-08-07 port conflict -- gave the user the exact `New-NetFirewallRule`
command to run themselves. Root cause of the Funnel-specific failure itself
(works via curl and via the emulator's earlier LAN-local path, fails from
real RN fetch over the public tunnel on two separate devices) was not
diagnosed further this session -- flagged as a real, reproducible issue
worth investigating before relying on the Funnel path for off-LAN testing
again, not just worked around silently.

---

## 2026-08-21: Clean-split (player-disjoint) StrokeGRU deployed, replacing the leaky-split checkpoint -- explicit user decision, reversing the choice made earlier this same session

Direct follow-up to the calibration entry above, where the user had
initially chosen to calibrate the model actually live (leaky-split) rather
than swap in the weaker-but-honest clean-split checkpoint as an
undiscussed side effect. This entry is that same user explicitly reversing
that decision before starting stroke-classification improvement work,
wanting the paper's corrected 68.8%/0.647 number to be what the app
actually runs.

**Exported and deployed.** `checkpoints/clean_split/stroke_gru_clean_best.pt`
-> `checkpoints/clean_split/stroke_gru_clean.onnx` via the existing
`export_onnx.py` (same onnxruntime-inference sanity check as the original
export, val_macro_f1=0.741 matches `evaluate_clean_split.py`'s own
checkpoint-selection record). `backend/app.py`'s `STROKE_GRU_ONNX_PATH`
repointed at the new file (original `stroke_gru.onnx` left on disk,
untouched, for provenance/rollback -- not deleted).

**Temperature refit against the newly-deployed model's own held-out
split, not reused from the leaky-split fit** (`calibrate_temperature_clean_split.py`)
-- confirmed the old T=1.4474 does not transfer, exactly the risk flagged
before starting: this checkpoint needed **T=2.0046**, a materially larger
correction.

**Before calibration, this model is dramatically more miscalibrated than
the leaky-split one was:** clean-split test set ECE = **19.93%** (vs the
leaky model's 7.01%) -- the 90-100% confidence bucket alone showed a 20.7pp
gap (97.4% average stated confidence vs only 76.8% actual accuracy).
Consistent with training on 68.8%-accuracy data while the softmax head
still learned to produce sharp, leaky-split-habituated-looking
distributions. **After calibration: ECE = 9.55%** -- a real, large
improvement, but notably still more than double the leaky model's
post-calibration 3.83%; a single global scalar could not fully flatten this
checkpoint's more irregular reliability curve (one bucket, 0.8-0.9, stayed
non-monotonic even after scaling). Reported as a real, incomplete
improvement, not oversold.

**Genuine held-out cross-check this time (unlike the reverse check done
for the leaky model):** ran the clean-split model + its new T against the
OLD leaky-split test set -- a legitimate check here, since THIS
checkpoint's training data never included those clips at all. 87.2%
accuracy, ECE 4.46% -> 6.68% (T calibrated for the clean-split's own harder
distribution over-corrects slightly on this easier one, same asymmetry
pattern as before, direction reversed).

**The two known cases, re-run against the new model:**
- `clip5_serve_setup.mp4` (true serve): still predicted **backhand**
  (wrong, same failure mode survives the retrain) -- raw confidence dropped
  from the leaky model's 99.3% to **92.7%**, calibrated down further to
  **72.3%**. More honest, still wrong.
- `clip1_serve_practice.mp4` (ungraded): predicted class actually
  **changed** from volley (leaky model) to **forehand** (clean-split
  model) -- the two checkpoints genuinely disagree on this ambiguous case.
  Confidence 79.2% raw -> 59.7% calibrated.

**Verified end-to-end, three independent ways:**
1. Direct API call for `clip5_serve_setup.mp4` through the real
   `/api/v1/analyze-stroke` endpoint after restart: confidence
   **0.72320449...**, full distribution matches the standalone recheck
   script exactly (forehand 2.9%/serve 4.5%/volley 19.6%/smash 0.7%).
2. Real on-device run, same clip (`theme_test_clip.mp4`) used throughout
   this session's verification passes: **46% (old leaky+T) -> 70% (new
   clean-split+T)**, same predicted class (volley), genuinely different
   number confirming a different model, not a cache artifact -- confidence
   pill crossed the caveat threshold into the "confident" (non-amber)
   styling this time, a visible UI difference from every prior screenshot
   of this exact clip this session.
3. Startup log / health endpoint confirmed the new ONNX path loaded
   without error after restart.

Coaching generation itself failed on this specific on-device run (Ollama
retry-gate exhausted after 307.2s, "Coaching feedback isn't available right
now" fallback correctly shown) -- unrelated to this change, the same class
of transient generation failure documented elsewhere in this log, not a
regression from the model swap.

---

## 2026-08-21: Two parallel data-strategy investigations for stroke classification -- data augmentation (built and tested for real) and TenniSet as a training source (scoped, not extracted). Neither touched the live pipeline.

Explicit instruction: investigate and report on both before building anything
further; if either looked promising, that becomes the next real fine-tuning
attempt. Both done in parallel this session -- Task A run directly, Task B
delegated to a forked agent with the same rigor expectations (verify real
files/numbers, don't assume, flag thinness plainly).

**A real held-out validation baseline had to be built first, and didn't
exist yet.** The 14-instance "real-footage validation set"
(`evaluate_real_footage.py`) only ever validated Path B's VIDEO model.
Nothing evaluated the actually-deployed POSE-based classifier against it.
Built `evaluate_real_footage_pose_based.py` reusing the exact same 14
ground-truth instances and, for the 11 candidate-window instances
(clip1 x2, clip2 x9), the exact `window_start`/`window_end` bounds already
computed and hand-verified in `ir/example_output/classifier_diagnostic/*/result.json`
-- not re-detected, so this evaluation is pinned to the same windows
already checked against the real footage. **Baseline result for the
currently-deployed clean-split model: 5/14 correct (35.7%), 0/3 on serve.**
Materially worse than that model's own THETIS-derived test accuracy
(68.8%) -- the real-world gap is confirmed severe even after the clean-split
fix, and now has a real number attached instead of just the 3 anecdotal
serve misses on record.

### Task A: data augmentation -- built, retrained, evaluated, real result

**Augmentations targeted the specific diagnosed gap** (2026-08-12 entry
above: THETIS's fixed ~1.5m indoor Kinect capture vs. this project's
further/variable-distance, outdoor, consumer-camera real footage), not a
generic augmentation-library default set:
- `zoom_out` -- shrinks frame content, letterboxes on a neutral gray
  canvas (scale 0.45-0.75 per clip) -- simulates a subject occupying less
  of the frame at greater camera distance, the single biggest diagnosed
  factor.
- `color_jitter` -- per-clip brightness/contrast/saturation shift.
- `motion_blur` -- mild per-clip linear blur, handheld/consumer camera vs.
  THETIS's static rig.
- `noise` -- light additive Gaussian noise, consumer sensor/compression.
- `background_clutter` -- explicitly labeled a PROXY, not a real
  simulation: 2-3 semi-transparent blobs on a smooth per-clip random-walk
  trajectory, confined to the outer thirds of frame. No real
  background-actor footage exists anywhere in this repo to composite a
  convincing second person from; synthesizing one was out of scope for
  this pass. Documented honestly as a simplification, not oversold as
  "simulates other people moving."

**Built via the exact production code path, not a separately-maintained
extraction pipeline.** `build_augmented_train_set.py` runs each raw THETIS
video (`data/.../raw_rgb/<class>/*.avi`, confirmed present, 1980 files/
3.7GB) through `backend/app.py`'s own `decode_and_resample_video` +
`run_pose_extraction` -- the SAME functions a real upload goes through in
production -- with the augmentation applied to the resampled 60-frame
sequence before pose extraction. Augmented ONLY the clean-split TRAIN
partition (1404 clips); val/test copied unchanged from `processed_clean` --
synthetic data has no business in a held-out evaluation split.

**Real numbers, not projected.** 1404 clips x1 augmented copy each -> 1141
successfully extracted, **263 dropped (18.7%) for pose detection falling
below 50% after augmentation** -- itself an honest data point: the
augmentation is aggressive enough to meaningfully degrade pose tracking on
nearly a fifth of clips, consistent with (not contradicting) the real
domain-shift problem being simulated. Combined train set: 2545 clips
(1404 original + 1141 augmented). Retrained StrokeGRU from scratch, same
architecture/hyperparameters as the clean-split retrain, 40 epochs, best
val macro-F1=0.785 (epoch 36) -- higher than the original clean-split
model's own val macro-F1 (0.741), on THETIS's own (still indoor,
still close-range) val distribution.

**On the clean-split TEST set (THETIS distribution): 67.7% accuracy /
0.643 macro-F1** -- statistically indistinguishable from the un-augmented
clean-split model's 68.8%/0.647. Expected and unsurprising: this metric
was never the target: augmentation doesn't teach the model anything new
about the SAME indoor/close-range distribution it's already measuring.

**On the real 14-instance validation set -- the number that actually
matters here: 8/14 correct (57.1%), up from the baseline's 5/14 (35.7%).**
A real, substantial improvement, not noise -- driven almost entirely by
backhand (4/10 -> 7/10). **Serve: still 0/3, unchanged.** `clip5_serve_setup.mp4`
now predicted backhand at 100.0% raw confidence (if anything more
confidently wrong than the clean-split baseline's 72.3%); both clip1
candidate serve windows also still wrong (volley, forehand). **Reported
plainly, per explicit instruction: this augmentation approach helps overall
real-footage accuracy meaningfully, but does NOT touch the serve-specific
failure** -- the same failure mode that has now survived three
independently-sourced real serve clips, a clean-split retrain, AND this
augmentation pass. Whatever is actually wrong with serve classification on
real footage is not fixed by zoom/lighting/blur/noise/clutter simulation on
THETIS's own indoor serve clips -- worth treating as a finding in its own
right (points at something more structural: framing/camera-angle
difference specific to serve's overhead motion, or an actual labeling/
technique difference between THETIS's serve style and real amateur serves,
neither investigated further this pass).

**Not deployed.** Checkpoint saved to `checkpoints/augmented/stroke_gru_augmented_best.pt`
+ exported ONNX, `backend/app.py` untouched, exactly as instructed
("neither task touches the live pipeline yet").

### Task B: TenniSet as a real-footage training source -- scoped, not extracted (forked investigation)

Delegated to a forked agent with the project's established rigor
requirements (verify real files, real counts, real license text; flag
thinness/leakage risk plainly). Full findings, not summarized further here
since the fork's own report already meets this project's reporting bar:

**1. Annotation format -- real per-shot labels exist, but split across two
inconsistent fields.** Real excerpt confirmed from `V006.json`:
`{"custom": {"Player": "Sharapova", "Type": "Topspin", "Side": "Forehand"}, ...}`.
`Side` gives forehand/backhand; `Type` is mostly technique (topspin/flat/
slice) but ALSO carries Volley (62 instances) and Smash (17) as values
instead of `Side` -- mapping to this project's 5-class taxonomy needs
`Type in (Volley, Smash)` checked first, falling back to `Side` otherwise.
Serve events are separate, with a `Result` (In/Fault/Let) field.

**2. Real per-class counts (5 matches combined):** backhand=1290,
forehand=1182, serve=1017 total (727 "In" only), volley=62, smash=17.
**Serve specifically is genuinely strong** -- 727-1017 instances vs.
THETIS's 495 -- directly targeting this session's diagnosed weak point.
Volley/smash are thin, smash (17) far below THETIS's own 165 -- not a
useful rebalancing source for those two.

**3. License -- real ambiguity, same category flagged for THETIS's own raw
video (2026-08-12 entry).** Repo's MIT license (real LICENSE file, quoted)
covers the annotation/training CODE, not the video content -- real
broadcast match footage (Sharapova/Williams/Djokovic/Federer/etc.) hosted
as MP4s on the author's Google Drive, no stated license for the footage
itself. Fine for this project's existing academic/FYP practice, genuinely
unresolved for any future product use.

**4. A real leaky-split risk, caught before extraction, not after.**
Serena Williams appears in 3 of the 5 matches (V006, V008, V010) -- a
genuinely player-disjoint split can only form 3 groups
({V006,V008,V010}/{V007}/{V009}), small and lopsided. **Checked the
repo's own documented split scheme directly -- it does NOT avoid this
leakage either** (their scheme's val/test video V010 shares Williams with
their own train videos). Exactly the class of mistake this project has now
caught and fixed twice this session (THETIS's original split, this
session's calibration-vs-deployed-model mismatch) -- surfaced here before
anyone assumed the authors' split was safe to reuse.

**Step 4 (extraction) correctly NOT done.** Download (~8.5GB across 5
matches) was still in progress when the investigation was scoped (2 of 5
videos fully downloaded, 1 partial, 2 not started) -- extraction was
gated on both the download finishing and a decision on how to handle the
Williams overlap, per explicit instruction not to force it if not ready.

**Verdict (fork's own): promising, not yet actionable.** Real volume for
forehand/backhand/serve is substantial and targets the diagnosed weak
point; volley/smash are not usable from this source; the player-overlap
constraint means any resulting split is small (effectively 3 usable
groups from 5 videos) -- a real, honest limitation to weigh against the
volume gain, not a reason to abandon it outright.

### Combined verdict for next steps

Task A is the more immediately actionable result: real, working code,
retrained checkpoint, and a real (if partial) improvement already measured
-- but it does not fix serve, the originally-diagnosed problem this whole
investigation traces back to. Task B's real per-class serve volume (727+
"In" instances) is exactly the kind of data this project doesn't have
enough of from THETIS alone, but extraction is blocked on finishing an
in-progress download and a split-design decision, not yet a usable
dataset. Neither task is a complete fix on its own; task B's serve volume
combined with task A's already-built augmentation pipeline (applied to
real broadcast serve clips instead of/in addition to THETIS's own) is the
most promising next direction, not started this session.

---

## 2026-08-22: The combined-hypothesis fine-tune -- built, tested for real, RESULT: no improvement, serve still unfixed. A real negative result, reported as instructed.

Direct follow-up: TenniSet download finished, extraction + a genuinely
player-disjoint split + fine-tuning + real-footage evaluation, exactly per
explicit instruction ("this is the real test of the hypothesis... report
back with real numbers before deploying anything"). Delegated to a forked
agent for the download/extraction/split work (same rigor standard), fine-
tuning and evaluation run directly.

**Split design -- real tradeoff resolved, better than either option
originally proposed.** Player overlap by match: V006/V008/V010 share
Williams (3 matches); V007 (Djokovic/Tsonga) and V009 (Federer/Del Potro)
share no players with anything else. The fork found a genuinely
player-disjoint assignment that maximizes volume better than a strict 1:1
bijection: **TRAIN = V006+V008+V009+V010 (4 matches -- Williams repeats
across 3 of them, but repeating WITHIN one split isn't leakage, only
crossing BETWEEN splits is), VAL/TEST = V007 alone (Djokovic/Tsonga,
shares zero players with train), split by event-index parity.** Verified
programmatically (`verify_split_disjoint()`, asserts zero player-set
intersection) before any extraction ran, not just reasoned about.

**Annotation units confirmed empirically before use, not assumed:** ~21
raw frames per Hit event at the video's own ~25fps -- 0.84s, consistent
with a real single-shot duration, confirming start/end are frame indices.

**Extraction: 3084 clips, zero failures** (0 too-short/edge, 0 write
failures) -- train: backhand=1097, forehand=924, serve=599; val/test (from
V007 alone): ~150 each, real per-class counts, not projected.

**A major, unplanned finding surfaced during pose extraction: real
broadcast footage itself defeats MediaPipe pose detection most of the
time.** Running the SAME production pose-extraction path
(`decode_and_resample_video` + `run_pose_extraction`) against the 2620
train clips: **1974/2620 (75.3%) of the RAW, unaugmented real clips failed
the same 50%-pose-detection-survival bar** used throughout this project.
Only 646 real clips (24.7%) survived; of those, only 167 augmented copies
also survived (consistent with augmentation degrading detection further,
same pattern seen on THETIS). This is itself real evidence for the
distance/framing domain-shift hypothesis this whole investigation traces
back to -- broadcast TV camera framing is often too wide/distant for
MediaPipe's landmarker to reliably track a player, a DIFFERENT failure
mode than "the classifier gets it wrong" (this is pose extraction failing
before classification is even attempted). Surviving real+augmented volume
actually added to training: serve=302 (224 real + 78 aug), backhand=261
(206+55), forehand=250 (216+34) -- still substantial, serve's survival
rate was proportionally the best of the three classes (~37% vs ~19-23%
for backhand/forehand), plausibly because serve's overhead motion sits
more centrally in typical broadcast framing than baseline rally shots.

**Fine-tuned from the deployed clean-split checkpoint** (not from
scratch), combined training set = 2545 THETIS(+augmented) + 813
TenniSet(+augmented) = 3358 clips, lr=3e-4 (lower than the from-scratch
default, standard practice continuing from a converged checkpoint), 40
epochs. **Best val macro-F1 = 0.798 -- the highest of all three checkpoints
built this investigation** (clean-split baseline 0.741, THETIS-augmentation-only
0.785, this fine-tune 0.798).

**The real 14-instance evaluation -- the number that actually matters, and
it's a genuine negative result: 5/14 correct (35.7%), tied with the
original pre-augmentation baseline, WORSE than the augmentation-only
checkpoint's 8/14 (57.1%). Serve: 0/3, unchanged again.**
`clip5_serve_setup.mp4` still predicted backhand, now at 99.0% confidence.
Backhand regressed specifically (7/10 under augmentation-only -> 4/10
here) -- clip2's contact_frame=34/213/385/493 all newly wrong under this
checkpoint despite being correct or unchanged under the augmentation-only
one.

| checkpoint | val macro-F1 (internal) | real 14-instance accuracy | serve (real) |
|---|---|---|---|
| clean-split baseline | 0.741 | 5/14 (35.7%) | 0/3 |
| + THETIS augmentation (Task A alone) | 0.785 | **8/14 (57.1%)** | 0/3 |
| + TenniSet fine-tune (this entry) | **0.798** | 5/14 (35.7%) | 0/3 |

**The single most important finding of this whole investigation may be
this table's own shape, not any one checkpoint's score: internal val
macro-F1 on THETIS/TenniSet-derived data INCREASED monotonically across
all three checkpoints while real-footage accuracy was NOT monotonic --
it went up then back down.** The checkpoint with the best internal
validation score is tied for the worst real-world result measured this
investigation. This is a direct, concrete demonstration that the internal
val metric is not a trustworthy proxy for real-world generalization here,
independent of any one technique's success or failure -- any future
stroke-classification work needs to treat the external 14-instance
real-footage check as the primary signal, not a confirmatory afterthought,
exactly the discipline this evaluation script was built to enforce.

**Plausible (not confirmed) explanation for why real broadcast footage
didn't help, offered honestly as a hypothesis, not a proven cause:**
TenniSet is broadcast TV footage -- professional camera work, likely
higher effective resolution on the player despite wide framing,
broadcast-grade compression and color grading -- which may be its own
distinct distribution, no closer to (and possibly further from) the
amateur/consumer-camera footage the app actually needs to handle than
THETIS's indoor Kinect footage is. "Real, non-THETIS video" was the
hypothesis; "real video representative of this app's actual real-world
input" was the implicit assumption underneath it, and this result suggests
that assumption may not hold for broadcast tennis footage specifically.
Not verified further this session -- would need real amateur/consumer-
camera serve footage at volume to actually test, which does not exist in
this project (this is the same finding as Path B's original feasibility
report, restated with a concrete number behind it now).

**Not deployed.** Checkpoint and ONNX saved to `checkpoints/tennisette_finetune/`,
`backend/app.py` untouched -- still the augmentation-only checkpoint's
result (8/14) is the best real-footage number produced across this whole
investigation, and even that doesn't touch serve. No checkpoint from this
investigation is recommended for deployment on serve grounds; the
augmentation-only one is the only one that measurably helped anything, and
only on backhand/forehand.

---

## 2026-08-22: THETIS-augmentation checkpoint DEPLOYED (replacing clean-split), serve investigation formally closed out as a documented limitation

Explicit user decision following the negative TenniSet-fine-tune result
above: deploy the one checkpoint from the whole investigation that
measurably helped (intervention #3, THETIS augmentation, 8/14 real-footage
accuracy), on the explicit understanding that it does not fix serve and
is not being deployed on the pretense that it does.

**Swapped**, same process as the 2026-08-21 clean-split swap:
`backend/app.py`'s `STROKE_GRU_ONNX_PATH` repointed at
`checkpoints/augmented/stroke_gru_augmented.onnx` (already exported this
session). Clean-split checkpoint left untouched on disk at
`checkpoints/clean_split/` for rollback.

**Temperature refit against this specific checkpoint** (`calibrate_temperature_augmented.py`,
using processed_clean's own unaugmented val/test -- synthetic data has no
business in a calibration fit or its held-out eval either). Confirmed
again, a third time this investigation, that T does not transfer between
checkpoints: **this checkpoint needed T=2.4741**, larger than either prior
value (leaky-split 1.4474, clean-split 2.0046). **This is also the most
miscalibrated checkpoint deployed so far: ECE 25.69% raw (T=1) -- worse
than the clean-split model's own 19.93%** -- plausibly because 40 epochs
of training on a mix of clean + noisier augmented data pushed the softmax
head toward sharper, more extreme outputs generally. After calibration:
**ECE 14.80%** -- a real, large improvement (25.69%->14.80%) but still
notably higher than either prior checkpoint's post-calibration number
(leaky 3.83%, clean-split 9.55%) -- reported plainly as a real, unresolved
residual, not smoothed over.

**Verified end-to-end, three independent ways, same discipline as the
last swap:**
1. Direct API call for `clip5_serve_setup.mp4` through the real
   `/api/v1/analyze-stroke` endpoint after restart: confidence
   **0.9596855...**, matches the standalone recheck script's 96.0% exactly.
2. `recheck_known_cases.py` cross-check independently confirmed the same
   number via a separate code path (loads T from the same file, runs
   inference directly rather than through the API).
3. Real on-device run, the same `theme_test_clip.mp4` used for every prior
   on-device check this session: **67% confidence** -- a fourth distinct
   value on this exact clip across the four checkpoints deployed this
   session (52% leaky+cal -> 46% clean-split+cal -> 70% TenniSet-finetune+cal,
   not deployed -> 67% this deployment), confirming a genuinely different
   model+calibration is live, not a cached/stale result. Same predicted
   class (volley) throughout -- only the confidence value moves, as
   expected for a swap that changes weights+calibration but not
   necessarily this specific clip's argmax.

Backend/Metro/emulator had all died again between turns (same recurring
environment-reset pattern documented throughout this session) -- all three
relaunched from scratch before verification could run.

**Documentation updated to reflect this as the final result, not an
interim step:** `training/stroke_classification/README.md` gained a full
"Serve investigation" section (see that file for the complete writeup) --
a clean, paper-ready summary of all 5 attempted interventions, what each
one ruled out, the real per-checkpoint comparison table, and an explicit
closing statement: **serve classification on real footage is a
documented, accepted limitation, not an open bug** -- 0/3 across every
checkpoint tried, including the two attempts (augmentation, real broadcast
fine-tuning) specifically designed to fix it. No further serve-fixing
work is planned unless new real, deployment-representative serve footage
becomes available. The README's "Clean-split retrain results" section
header was updated to clarify it's the base checkpoint fine-tuning started
from, not the deployed model.

**Cleanup:** backend (port 8001), Metro, and the emulator left running
(no explicit instruction to tear them down, and the phone-connect request
from earlier in this session may still be in use).

---

## 2026-08-22: Sixth and final intervention -- simple-averaging ensemble (clean-split + augmented), NO improvement, not deployed. Serve investigation closed.

The one remaining untested idea before closing out the investigation:
does combining the two most different checkpoints' predictions help,
even though neither alone fixes serve. `evaluate_ensemble_real_footage.py`
(new) runs both ONNX models on the identical pose-sequence input per
instance, averages their raw softmax class_probabilities elementwise,
argmaxes the average -- against the same 14-instance set, same window
boundaries, directly comparable to every number already on record.

**Result: 8/14 (57.1%), exactly tied with the augmented checkpoint alone
(currently deployed). Backhand 7/10 (same), forehand 1/1 (same), serve 0/3
(same, again).** Not a coincidence of aggregate counts -- checked per-
instance: the ensemble's correct/wrong pattern is IDENTICAL to the
augmented-alone checkpoint on 13 of 14 instances; the one difference
(clip2 contact_frame=493) is wrong under both anyway, just a different
wrong class (forehand under augmented-alone, serve under the ensemble).
The augmented checkpoint's generally sharper/more confident distribution
dominates the average whenever the two models disagree, so the ensemble
behaves as the augmented model alone in every case that matters for
correctness.

**Decision, per explicit instruction ("if it's not better... don't deploy
it"): NOT deployed.** Zero measured benefit against real, non-trivial
added cost (two ONNX inference passes per classification instead of one,
roughly double the compute for the stroke-classification stage). The
augmentation-only checkpoint deployed in the entry above remains the
final, deployed result of this investigation.

**Serve investigation formally closed, six interventions total, 0/3 on
real serve footage in every single one:** clean-split retrain, temperature
calibration, THETIS augmentation, TenniSet broadcast fine-tuning, the
combined augmentation+broadcast fine-tune, and now a two-model ensemble.
`training/stroke_classification/README.md`'s "Serve investigation" section
is the canonical writeup for the paper; no numbered-intervention update
was needed there since the ensemble doesn't change the deployed
checkpoint or its results -- this log entry is the record of the ensemble
attempt specifically. No further serve-fixing work planned unless new
real, deployment-representative serve footage becomes available.

---

## 2026-08-22: Stroke segmentation resumed -- kinetic-chain TIMING (not amplitude) tested, one real attempt, time-boxed, mixed/informative result

Confirmed starting state before building anything: wrist-speed magnitude
and trunk-rotation magnitude (2026-08-02 entries above) both independently
fail on the same clip1 case -- the frame-108 walking-with-racket false
positive outscores the frame-155 true-positive serve on BOTH signals
(speed 0.115 vs 0.032; rotation 10.70 vs 9.20 deg/frame). No amplitude
threshold on either signal, individually or AND-ed, can fix an inverted
ranking.

**Proposed and picked one new candidate approach before building
anything, per instruction: kinetic-chain TIMING coincidence, not a third
amplitude signal.** Reasoning: real strokes exhibit proximal-to-distal
segment sequencing (hip/shoulder rotation initiates and is temporally
coupled to the wrist's peak speed, a well-established biomechanical
signature) -- walking-with-racket has an arm swinging independently of
gait, no such coupling. This is a WHEN question, not a HOW-BIG question,
which is exactly the axis an amplitude threshold can't touch. Rejected the
other two candidates before building: motion-periodicity (weaker fit --
the false positive is one isolated walking-arm-swing instance, not
visibly periodic within a single candidate's local window) and a learned
classifier (highest risk -- only ~13 hand-labeled real instances exist
total across the 2 usable clips, not enough to train or validate without
repeating this session's own small-sample-overfitting lesson from
stroke-classification).

**Metric: |lag| between the wrist-speed peak and trunk-rotation's own
local peak (searched over the full +/-30-frame candidate window, not the
existing filter's narrower amplitude-gated +/-10), tested as a
replacement filter, not an additional AND-ed threshold.**

**Result on clip1 (the originally diagnosed case): clean, 4/4 correct.**
lag(frame 48, FP)=+28, lag(frame 108, FP)=+18, lag(frame 155, TP)=-8,
lag(frame 235, TP)=+1. Any threshold in the range 8 < |lag| < 18 (e.g.
15) perfectly separates the false positives (large lag) from the true
positives (small lag) -- confirmed the hypothesis for this exact case.

**Result on clip2 (the clip the EXISTING filter already handles well, 9/9
precision): a real regression, confirmed across the entire feasible
threshold range (10/12/15/17), not one bad threshold pick.** 3 of clip2's
9 genuine true positives (frames 213, 493, 574 -- lag -20, -25, +20) have
lag magnitudes AS LARGE AS OR LARGER than clip1's false positive (+18) --
meaning this new filter would incorrectly reject them at any threshold
tight enough to still catch clip1's false positive. `clip2` frame 574
specifically (lag=+20, same sign and comparable magnitude to clip1's
frame-108 false positive's +18) is a genuine, confirmed instance of a
real stroke and a real non-stroke producing near-identical lag values --
the same "no single threshold separates an inverted pair" problem this
project already documented within one clip, now shown to recur ACROSS
clips at the cross-clip level too.

**clip3: unchanged assessment, as expected.** Lag values are essentially
scattered/uninformative there too (e.g. the two highest-speed candidates,
frames 191 and 238, have lags -9 and -6 -- inside any reasonable
coincidence window, so this filter would not have rejected them either) --
consistent with the already-documented finding that clip3 doesn't
validate segmentation logic at all (multi-person pose-identity
instability, a hard prerequisite gap unrelated to peak-detection or
filtering).

**Honest conclusion, time-boxed as instructed -- not deployed, not
iterated further:** kinetic-chain timing coincidence is a real,
theoretically well-grounded signal that DOES cleanly solve the originally
diagnosed clip1 problem in isolation, but does NOT generalize as a
universal filter -- it trades clip1's fix for 3 new false rejections on
clip2's previously-clean detection. This is reported as the genuine,
informative result of one real, properly-validated attempt, not smoothed
over or chased with further threshold tuning (a quick asymmetric-window
variant was sanity-checked and still could not resolve the clip2 frame-574
conflict without either losing clip1's fix or accepting the same
regression in a smaller form -- not pursued further, per the explicit
time-box). `ir/stroke_segmentation.py`'s existing filters (wrist-speed +
trunk-rotation amplitude) remain as-is, unmodified -- this was a real
attempt at a fix, confirmed not to be a clean win, not deployed as a
replacement. Stroke segmentation over-detection on walking-with-racket
remains a documented, unresolved limitation on this specific clip1 case,
same status as before this investigation, now with one more ruled-out
approach on record.

**Standing conclusion across all three attempts to date, stated plainly
rather than left as three separate incident logs:** wrist-speed magnitude
(first pass), trunk-rotation magnitude (2026-08-02), and kinetic-chain
timing coincidence (this entry) are three genuinely different signals --
not three tunings of the same idea -- and all three have now hit the
identical failure shape: a real stroke instance and a non-stroke instance
producing overlapping or inverted values on that signal, with no
threshold able to separate them without breaking a different clip's
correct detections. This is no longer three isolated findings; it is a
pattern. **Single hand-tuned scalar features appear structurally
insufficient for this discrimination task, given the data actually
available** -- not for lack of trying reasonable, theoretically-motivated
signals (amplitude on two independent kinematic channels, then a
temporal-coupling reformulation of the same two channels), but because
real strokes and real false positives on this project's actual footage
occupy overlapping ranges on every single scalar tried so far.

**A learned classifier over short pose-sequence windows (option 3 from
this entry's proposal, reusing the existing biomechanics features as
input) remains the most promising untried approach** -- it wasn't picked
this round specifically because it was already correctly judged too
high-risk against the available data, and that same constraint is the
reason nothing else has worked either: **~13 hand-labeled real instances
(4 on clip1, 9 on clip2, clip3 unusable) is the confirmed bottleneck for
stroke segmentation, not a lack of feature-engineering effort.** Three
independently-motivated, real attempts is enough evidence to say this
plainly rather than attempt a fourth scalar feature. Any future
segmentation work should start from acquiring materially more labeled
real-footage instances (the same underlying scarcity already identified
as the bottleneck for the stroke-CLASSIFICATION serve investigation
above, now confirmed as the bottleneck for stroke SEGMENTATION too, via
independent reasoning and independent attempts) -- not from a fourth
hand-tuned signal.

**No further segmentation attempts planned, per the time-box.** This
closes the segmentation investigation in its current state: over-detection
on walking-with-racket-swinging remains open, three approaches ruled out
with reasons, and the path forward (more labeled data, then revisit a
learned classifier) identified but not started.

---

## 2026-08-23: Model 3 (momentum tracking) -- real training pipeline built, smoke-tested, working. Full run not yet started, per instruction.

Confirmed starting state: point data ingested and EWMA-labeled
(`points_ewma_momentum.parquet`, 4,996 real-ATP-linked matches, sanity-
checked against 3 hand-picked real matches including a negative control --
2026-07-28 entries). `TennisMomentumLSTM` (`model.py`) was architecture-
only, explicitly never trained (its own docstring: "do not wire this into
a training script until real data is acquired"). This entry is that
wiring.

**Feature engineering against the REAL confirmed schema, not the original
provisional 13-dim spec** (`dataset.py`, new) -- same "inspect before
assuming" discipline as every other model here. The real
`charting-m-points-*.csv` columns don't include ready-made break/set/
match-point flags, rally length, or outcome-type columns; all derived:

- Everything kept in PLAYER 1's frame and SIGNED (+1 favors P1, -1 favors
  P2, 0 neither) throughout -- matches how `momentum_ewma_p1` itself is
  defined, more informative than the original spec's flat 0/1, no extra
  cost once the P1/P2 framing is fixed anyway.
- `point_outcome_ace` / `_winner` / `_unforced_error`: derived from the
  last character of the `1st`/`2nd` shot-notation column, EMPIRICALLY
  VALIDATED against real `PtWinner` before trusting it (not assumed from
  memory of the MCP notation spec): `#` correlates with the server winning
  77.5% of the time (n=272,465, consistent with unreturned/ace-type),
  `@` at 48.4% (n=261,538, near coin-flip, consistent with unforced
  error), `*` at 72.9% (n=257,491, consistent with a clean winner) --
  these 3 codes cover ~96% of all terminal characters.
- `point_outcome_forced_error` DROPPED, not guessed -- no terminal
  character cleanly separated out as a distinct forced-error marker
  without further unverified assumption.
- `stroke_type_onehot` not included -- explicitly optional in the
  original spec, needs per-point video alignment with Model 2 that
  doesn't exist for this (completely different) charted-match data source.
- `rally_length`: no exact shot-count parser built -- used an honestly-
  labeled PROXY instead (character length of the shot-notation string,
  normalized), not an exact count.
- **Net: input_dim=12, not the original spec's 13** -- `model.py`'s
  default updated to match, with the deviation documented directly in the
  class docstring, not just this log.

**A real bug, found and fixed before it could corrupt training, not
after.** Initial break/set/match-point derivation used the `TbSet` column
to decide whether to parse `Pts` as standard 0/15/30/40/AD scoring or raw
tiebreak point counts. First full-corpus run produced an implausible
55.6% "is_set_point" rate -- checked by hand against one real match's
first 20 points rather than accepted, which showed `game_point=1` firing
on ordinary early-game scores like "15-0". Root cause: **`TbSet` is a
SET-level flag ("did this set end in a tiebreak"), not a point-level
"is this point currently inside the breaker" flag** -- it reads `True` for
every point in a set from the very first 0-0 point onward whenever that
set eventually reaches 6-6, so trusting it point-by-point silently
misparsed "15-0" as "15 points to 0" tiebreak scoring. Fixed by detecting
tiebreak-style scoring directly from whether `Pts`' values fall inside the
standard vocabulary at all (no dependency on correctly interpreting what
`TbSet` means), and by using `Gm1==6 and Gm2==6` (confirmed clean columns)
to detect the tiebreak GAME for the separate set-point-clinching check.
Re-verified against the same real match by hand after the fix: rates
dropped to break_point=8.5%, set_point=3.0%, match_point=1.1% of all
points -- all now consistent with known real-tennis base rates, unlike
the original 55.6%/17.7%/17.4% numbers the bug produced.

**Match-level split** (`dataset.py:match_level_split`), same lesson as
every other split-leakage fix this project has made (THETIS clean-split,
TenniSet player-disjoint split): shuffled by `match_id`, not by row, with
an explicit assertion that no match_id appears in more than one split --
not just documented as leak-free, verified programmatically every run.
3,995 train / 499 val / 500 test matches (4,994 total, 2 dropped below the
20-point minimum-length filter -- almost certainly retirements/walkovers
charted as a handful of points).

**Smoke test** (`train.py --smoke-test`, 64 train / 16 val matches
subsampled with a fixed seed, 5 epochs): **loss decreases every single
epoch** -- train MSE 0.0836 -> 0.0728, val MSE 0.0850 -> 0.0702 (train MAE
0.208 -> 0.190, val MAE 0.207 -> 0.183) -- confirms the model is actually
learning something real from the signal, not stuck or diverging.
**Resource usage, measured not assumed:** 0.055M params, 0.1-0.6s/epoch
on this subset, 0.05GB peak GPU memory -- trivially cheap, consistent
with StrokeGRU's own similarly-tiny-and-fast profile earlier in this
project. Full feature-engineering pass over the whole real corpus
(4,994 matches, 823,874 points) took 20.2s, a one-time fixed cost paid
once per training run regardless of epoch count.

**Architecture:** confirmed as originally scoped otherwise -- 2-layer
LSTM (hidden_dim=64) -> small FC head -> Tanh-bounded [-1,1] regression
output, masked-MSE loss added (padded timesteps, from variable per-match
sequence length, must not contribute to the loss or the model would be
penalized on positions with no real point) via `pack_padded_sequence` --
the model's own `forward()` had this parameter stubbed out from the start
("wiring this up is part of writing the training script"); now wired for
real, used identically in training and at inference (a single real match,
no padding needed, still works via the same `forward()` call with
`lengths=None`).

**Not yet done, explicitly, per instruction:** the full training run
(30 epochs on the full 3,995-match train set, checkpointed every epoch),
held-out MSE/MAE on the real test split, and the qualitative check against
the Gaudio comeback's known plausible-momentum-curve shape. Reporting back
now, before starting that run, as explicitly instructed.

---

## 2026-08-23: Model 3 (momentum tracking) -- full 30-epoch run complete, evaluated, working. First genuinely new trained model this project has produced (not a retrain/fix of an existing one).

**A second real bug, caught immediately rather than let run for 30 epochs
blind.** The full run (touching every one of the 3,995 train matches,
unlike the smoke test's random 64-match subsample) produced `nan` loss
from epoch 1 -- killed the run at epoch ~9 rather than let it keep
burning compute once the pattern was clear, and root-caused before
retrying. Found exactly 1 NaN value (`Gm2`) out of 823,874 real points,
in match `20160123-M-Australian_Open-R32-Stan_Wawrinka-Lukas_Rosol`,
Pt=59 -- a genuine gap in the raw MCP source file, not a derivation bug.
Verified the fix was correct, not a guess: the neighboring points (Pt=58
and Pt=60, same game, `Gm#`=9.0) both read `Gm2=0.0`, so a within-match
forward/back-fill is the same value the point immediately before and
after both actually have, not an assumption. Fixed in `dataset.py`,
documented inline. Re-verified zero NaN/Inf across the full engineered
feature matrix before restarting.

**Full run: 30 epochs, full 3,995/499/500 match split, checkpointed every
epoch.** Loss converges fast and stays stable -- val_mse drops from 0.0337
(epoch 1) to effectively 0 within ~10 epochs, consistent with the smoke
test's own trend just continuing further with more data. **Best checkpoint
selected by val_mae, not last epoch** (epoch 30's val_mae=0.0051 is
actually slightly worse than the best): **epoch 22, val_mae=0.0040**,
saved separately from the last-epoch checkpoint (`momentum_lstm_best.pt`
vs `momentum_lstm_last.pt`), same overfitting-awareness convention as
every other model here.

**Held-out TEST set (500 matches, 80,271 points, genuinely unseen by
training -- match-level split verified zero-overlap by assertion):
MSE=0.00004, MAE=0.00401, RMSE=0.00624.** An honest interpretive note,
not just the number: this is a VERY tight fit, and that's expected, not
suspicious -- `point_winner` (the exact underlying signal the EWMA target
is itself computed from) is a direct input feature, so the task is
"learn to reconstruct a known, deterministic smoothing function of a
signal you can already see," not predicting an independent unknown. A
near-zero MSE here confirms the model correctly learned that recurrence
relation; it is not evidence of some harder, more surprising capability
being solved. Worth stating plainly rather than letting a small number
imply more than it does.

**Qualitative check, same 3 hand-picked matches `sanity_check_ewma.py`
already validated the LABEL against** (`evaluate.py`, new,
`figures/trained_model_vs_ewma_examples.png`): the trained model's own
output line is visually near-indistinguishable from the EWMA ground-truth
line on all 3.
- **Gaudio comeback: reproduces the exact confirmed shape.** Sustained
  negative momentum from the match start through roughly point 100-140
  (Coria's 2-sets-to-love lead), crosses positive around point 140-150,
  stays mostly positive with realistic point-to-point noise through to
  the end (Gaudio wins) -- the same shape already confirmed for the label
  itself, now confirmed for the TRAINED MODEL's independent output.
- **Negative control (Djokovic/Cilic blowout): stays positive throughout,
  no invented drama.** No sustained reversal, no fabricated crossing --
  the model didn't hallucinate a comeback narrative onto a match that
  never had one. This was the specific failure mode worth checking for
  (a model that gets the dramatic case right but invents drama on a
  boring one would be a real problem), and it did not occur.
- Federer/Wawrinka: also reproduces the confirmed early-Wawrinka-lead,
  Federer-turnaround shape.

**Honest split disclosure for the qualitative check, not glossed over:**
Gaudio and Federer/Wawrinka both landed in the TRAIN split (the model has
seen these exact matches' point sequences during training) -- only the
Djokovic/Cilic negative control landed in VAL, genuinely unseen. Given the
near-perfect-reconstruction nature of this task (see the interpretive note
above), train-vs-held-out membership matters less here than it would for
a genuinely predictive task, but it's disclosed plainly rather than
implying all 3 were held out when 2 of 3 weren't.

**Resource cost, measured:** ~4.0-4.8s/epoch on the full corpus, ~130s
total for all 30 epochs -- trivially cheap, consistent with the smoke
test's own projection.

**Result: Model 3 (momentum tracking) is now a real, trained, working
model** -- the first genuinely NEW capability produced among this
project's deferred items, not a retrain or fix of something already
built. Checkpoint: `checkpoints/momentum_lstm_best.pt` (epoch 22,
val_mae=0.0040). Not wired into any live pipeline or product surface --
this was scoped as building and validating the model itself, not a
deployment.

---

## 2026-08-23: Momentum tracking closed out -- scope of what MAE=0.00401 actually demonstrates stated precisely, at the same weight as the number, not below it

Explicit instruction: log the following as plainly as the MAE number
itself, not as a caveat buried underneath it, and flag it in the research
materials (a `README.md` for this model didn't exist yet -- every other
model in this project has one; momentum tracking didn't, now does).

**The headline number and its scope, stated together, same weight:**

> Held-out test MSE=0.00004, MAE=0.00401, RMSE=0.00624 (500 test matches,
> 80,271 points, match-level split verified zero-overlap).
> **This is a validated reconstruction of the EWMA momentum formula, not
> yet evidence of independent predictive power.** `point_winner` is BOTH
> a per-timestep input feature the model sees AND the exact raw signal
> the training target (`momentum_ewma_p1`) is itself computed from (a
> plain EWMA of that same +1/-1 series, `ingest_points.py:compute_ewma_momentum`).
> Given that input, the model is reconstructing a known deterministic
> smoothing function of information already in its own input vector --
> confirming the LSTM learned that recurrence correctly, not demonstrating
> a new capability to infer momentum from anything the formula doesn't
> already have.

This is not new information relative to the prior entry (which already
carried an "honest interpretive note" immediately after the metric) --
this entry exists because the user explicitly asked for the limitation to
be logged with the SAME PROMINENCE as the number, in the research
materials specifically, not just inline in a session log entry, and for
the record to be unambiguous and quotable on its own rather than requiring
the full entry's context to find it.

**What would actually demonstrate independent predictive power, named
explicitly, not started:** predicting momentum (or point/game outcomes)
ahead of the current point using only PAST information (no same-timestep
`point_winner`), or training against a target not directly derived from
the same per-point signal the model already sees as input. Real, scoped,
next steps if this model is ever revisited -- not attempted this session.

**`training/momentum_tracking/README.md` created** (new -- this model was
the one exception to every other model in this project having a
research-materials doc of its own) with this exact framing as its lead
section, ahead of the architecture/pipeline description, so anyone
reading it hits the scope limitation before the number invites the wrong
conclusion.

**Confirmed still correctly scoped as model-building and pipeline
validation, not deployment.** The trained checkpoint remains unwired from
`backend/app.py` and every other live surface, exactly as before this
entry -- nothing changed operationally, this entry is a documentation-
precision fix, not a code or pipeline change.

**Momentum tracking is closed out for this session, per explicit
instruction.** No further work planned unless deliberately revisited.

---

## 2026-08-23: Coach Avatar for StrokeChat — nameable persona + illustrated expression states, verified end-to-end in both themes

**What was done:** Designed and shipped a small illustrated coach character
for StrokeChat (idle / listening / speaking expression states, inline
`react-native-svg`, duotone navy linework matching the existing design
system — no new color tokens, no image assets) plus per-user naming.
Design mockup reviewed and approved before any implementation (Artifact:
`Coach Avatar`, favicon 🎾) — same design-first process as every other
feature this session.

**Backend:** `coach_name` column on `User` (nullable, no migration risk —
picked up by the existing idempotent `_migrate_add_missing_columns`
pattern). New, deliberately separate `PATCH /api/v1/auth/coach-name`
endpoint (not merged into the role-gated `ProfileUpdateRequest`, since
`coach_name` must not depend on role being set). Blank/whitespace input
normalizes server-side to `null` — never stored as a literal empty name.

**Mobile:** Settings' existing "Coaching profile" card gained a Coach Name
field (no new screen). `StrokeChat.js` gained a `CoachAvatar` component (3
SVG expression states sharing one base illustration) and a `CoachBubble`
wrapper around the original, byte-identical `CoachBubbleContent` — bubble
styling and the retry-gate logic were not touched. A persistent header
avatar next to "Ask about this stroke" gives Idle/Listening real UI
presence beyond the per-message bubbles.

**Verified for real, not assumed, per explicit instruction:**
- Backend edge cases via direct `curl`: `coach_name` defaults to `null`; a
  `PATCH` with a real name stores/returns it; a `PATCH` with
  whitespace-only input normalizes to `null`; the account's own `name`
  field never gets touched by any of this (no leak path from player name
  into coach name).
- On-device (emulator), light theme: Settings field saves on blur and
  persists server-side (confirmed via a fresh, independent API call, not
  local React state); StrokeChat header avatar shows Idle; sending a
  question flips it to Listening (header + loading-row mini-avatar, with
  the original untouched spinner/"Thinking..." content inside); the
  delivered response shows the Speaking-state avatar with the custom name
  above it.
- Same full cycle repeated in dark theme with a second question — both
  avatars re-theme correctly (light navy linework on dark surface), name
  label correct on both exchanges.
- Name-label "run" behavior: each coach response is always immediately
  preceded by the player's own message in the message array (one
  `setMessages` call per turn), so the label shows on every coach bubble
  by construction — never suppressed, never duplicated within a bubble.
  Verified directly across two consecutive exchanges rather than assumed
  from the design doc.

**Coach Avatar feature is complete and verified working in both themes.**

---

## 2026-08-25: Coach review doc — two content fixes from a technical read (E27, E5), re-verified at 100% grounding

Status check on `rag/KB_COACH_REVIEW.md` (see 2026-08-01/08-02 entries) found
it current and unsent -- no record anywhere in this log of a real coach
having actually reviewed it yet. That same check surfaced two specific,
fixable content problems on a careful technical read of the document
(not a coach's opinion, and not treated as one) -- fixed directly, per
instruction, rather than left as a "someday, pending the coach" item.

**E27 (`weight_transfer_ratio`, `rag/kb_biomechanics.py`).** The
2026-08-02 tone rewrite had turned a pure data-quality caveat entry into
advice-only text, dropping a real caveat in the process: this
measurement's sign depends on camera framing and hasn't been
independently verified. Restored a short version of it alongside the
existing advice, modeled directly on E34's already-correct pattern
(advice + a real measurement caveat, both present, not one traded for
the other):
- Before: "How much your weight shifts forward into the shot -- from
  your back foot toward the ball -- shows up here. If yours looks
  light, try driving more off your back foot as you swing, so your
  weight is moving into the shot at contact rather than staying back."
- After: "...try driving more off your back foot as you swing, so your
  weight is moving into the shot at contact rather than staying back.
  One caveat: this reading's direction can vary with how your camera
  was angled, so treat it as a starting point, not a certainty."

**E5 (`second_serve_points_won_pct`, `rag/kb_placeholder.py`).** The
stated cause didn't match the stated symptom: rising double faults
alongside a falling second-serve win rate points to overhitting or
pressing under pressure, not a serve that's "become too safe and
predictable" (a too-safe serve reads as low double faults with low
effectiveness, the opposite pattern). Rewrote the interpretation to
match the actual pattern, same tone/length/action-oriented style as the
rest of the entries:
- Before: "...that combination usually means it's become too safe and
  predictable. Try adding some pace or placement variety in practice,
  not just hitting more of the same serve."
- After: "...that combination usually points to overhitting or pressing
  on the second serve, not playing it too safe. Try taking a little
  pace off and aiming for more margin over the net, rather than going
  for more on that ball."

Both fixes mirrored into `rag/KB_COACH_REVIEW.md` (E27's ⚠ NEW ADVICE
annotation updated to describe the restored caveat instead of its
absence).

**Re-verified via the same two grounding harnesses used for the
2026-08-02 rewrite, not assumed:**
- `rag/run_gated_stroke_trials.py`: **4/4 (100%) gated success**, one
  clean pass. Trial 4 (volley) retrieved and grounded on the edited
  `weight_transfer_ratio` entry directly (cited at the 88.4th
  percentile, advice text matches the new wording) -- the E27 edit
  exercised for real, not just inspected.
- `rag/run_gated_diverse_trials.py`: first run came back 5/6 (83.3%) --
  one trial (`serve_breakdown`, which had retrieved the edited
  `second_serve_points_won_pct`/E5 entry as context) failed after
  exhausting 3 attempts on malformed-JSON/uncited-value grounds, not on
  anything specific to the new KB text. Rather than write this off,
  re-ran the full harness a second time to check reproducibility: this
  time **6/6 (100%)**, with `serve_breakdown` passing cleanly on its
  2nd attempt (6/6 grounded, including a correct citation of
  `second_serve_points_won_pct=50.65%` in the generated text). Confirms
  the first run's single failure was ordinary local-LLM (Qwen3
  8B/Ollama) run-to-run variance -- consistent with flakiness already
  documented earlier this project (2026-08-02 network stalls,
  intermittent parse failures elsewhere in these same harnesses) -- not
  a regression caused by either content edit.

**Both content fixes are applied and confirmed grounded at the same
standard as the original rewrite. The document is still unsent to a
real coach** -- that remains a separate, outstanding step, not resolved
by this content pass.

## 2026-08-25: Coach Avatar redesign -- distinctive silhouette + unified demeanor, mockup approved and shipped, verified on-device in both themes at true size

**What was done:** Revised the Coach Avatar character (approved 2026-08-23)
to read as a distinct character rather than a generic template, per
explicit feedback that the original hair/jaw shape and per-state mouth
lines were too generic and too interchangeable across idle/listening/
speaking. Design-first process, same as every other feature this
session: mockup shown as an Artifact (`Coach Avatar Redesign`, favicon
🎾, both themes, all 3 states, true 22px/26px production scale) and
explicitly approved -- including one flagged addition beyond the
original two-item request (collar now shown in idle too, not just
listening/speaking) -- before any code was touched.

**Two changes ported into `CoachAvatar` in `StrokeChat.js`:**
- **Silhouette:** the old symmetric hair "dome" replaced with an
  asymmetric side-swept hairstyle (off-center part, a diagonal
  part-line stroke, uneven sideburn tabs -- longer on the left, tighter
  on the right), plus a slightly tapered jaw (chin point pulled from
  y92 to y95 with tighter control points). Reads as one particular head
  in outline, not a generic short-crop shape, even at avatar size.
- **Demeanor:** idle and listening now share one calm closed-mouth
  curve family (`M50 76.5 Q60 80 70 76.5` idle, a shallower
  `M52 77.3 Q60 78.6 68 77.3` for listening) instead of idle's curve and
  listening's unrelated flat line. Speaking's open-mouth ellipses stay
  anchored at the same x-center (60) for continuity. Brows/eyes/teal
  status dots left unchanged -- already judged consistent enough across
  states.
- **Approved extra:** the collar path now renders in all 3 states,
  including idle (previously hidden there) -- part of the same
  demeanor-consistency goal, called out separately for sign-off in the
  mockup and kept as-is per the user's explicit instruction not to
  revert it.

**Verified end-to-end on the `vt_test` emulator, not assumed from the
mockup render** -- real backend (`uvicorn`, port 8001), real Metro,
real Expo Go, logged in as `test1@example.com` against that account's
real 7-session history (its bcrypt password hash was reset locally via
a one-off script since the original wasn't recorded anywhere; this is
local trial dev data, not a real credential). `BACKEND_URL` was
temporarily pointed at `10.0.2.2:8001` (the standard emulator-to-host
alias) instead of the LAN IP already recorded as stale/temporary in
`config.js`, and reverted to the original line exactly as found once
verification was done.
- **Light theme, all 3 states at true size:** idle (22px, header next
  to "Ask about this stroke"), listening (26px, loading row, real
  in-flight request -- teal status dots visible), and speaking (26px,
  delivered bubble from a real grounded chat response, e.g. "Your
  shoulder hip separation has improved significantly...") all captured
  via `adb screencap`, cropped and inspected close-up. Asymmetric hair,
  tapered jaw, new collar, and the unified mouth curve all read clearly
  at true production scale, not just in the larger mockup render.
- **Dark theme, same 3 states, same real flow (second live chat
  question sent after toggling `adb shell cmd uimode night yes`):**
  re-themes correctly -- light navy-on-dark-surface linework, brighter
  teal status dots -- with the same silhouette and demeanor legible at
  22px/26px.

**Coach Avatar redesign is complete, approved, implemented, and
verified working at true production size in both themes -- not just in
the design mockup.**

## 2026-08-25: De-genericizing pass -- full-app structural audit, 5-part redesign proposal approved and shipped, verified on-device in both themes

**Audit first, as instructed.** Went through all 16 screen/component files
cataloguing 4 specific tells: icon-in-circle+card template, numbered-circle
step indicators, centered/symmetric vertical-stack layouts, stock Ionicons
without custom treatment. Reported to the user before any proposal:
- Icon-in-circle: only the Coach Avatar (the *good* version) uses it; zero
  stock-icon-in-circle instances anywhere. The real issue was the "card"
  recipe (border + radii.md + surface + border color) hand-copied
  independently 6 times across 4 files with no shared component.
- Numbered-circle steps: **zero instances** -- Home's capture flow and
  Profile-setup both have real step state but render no progress
  affordance at all. Worth stating plainly rather than "fixing" a
  non-problem.
- Centered/symmetric stacks: **13 instances across 9 files** -- the app's
  actual default template. Headline finding: signup.js and login.js are
  structurally byte-identical (same `makeStyles` object), and
  Signup -> Profile-setup -> Home is the real first-run path, all three
  visually the same screen with different copy.
- Stock icons: Ionicons confined to the 3 tab-bar icons only. The sharper
  tell was `ActivityIndicator` used in 4 places, including right next to
  the custom Coach Avatar in StrokeChat's own loading row.

**Proposal (mockup, no code) -- Artifact "De-Genericizing Pass", favicon
📐, dark crit-room/lightbox chrome with the app's real light-theme tokens
inside every phone mockup, not lorem/invented content.** Five sections:
(01) Home -- hero/secondary restructure of Practice vs Friendly Match,
typography-led stroke-picker list replacing the 5-card grid + blank
placeholder swatch; (02) Results -- stroke name as real headline,
pull-quote treatment for the top coaching item, other items demoted but
not dropped; (03) step indicator -- reuse the existing eyebrow-caption
typography instead of numbered circles; (04) icons -- drop the redundant
spinner from StrokeChat's loading row, keep the plain spinner on
Processing (flagged as an open call, not decided for the user); (05) card
weight -- History's most-recent session promoted to a headline treatment,
older sessions compressed into a denser list. All 5 approved as proposed,
including the exact call on item 04.

**Implementation, with two honest deviations from the mockup's own
illustrative shortcuts (caught before shipping, not after):**
- The mockup's step-indicator demo used a literal "Step 2 of 3" fraction.
  Home's capture flow branches (match / roadmap / by-stroke have
  different total step counts), so a hardcoded fraction would sometimes
  be wrong -- shipped as a contextual breadcrumb ("Practice", "Practice ·
  By Stroke") instead, with a real "Step 2 of 2" fraction only where it's
  actually fixed and honest (Profile-setup).
- The mockup's Results "after" state summarized the two secondary
  coaching items as one joined line with no citations shown. Given how
  much this project's grounding work has emphasized citations being real
  and traceable, that would have been a genuine data-loss regression, not
  just a style change -- shipped with every item's full citation intact,
  just visually smaller/quieter than the pull-quote hero item.
- Stroke-picker's "last practiced" marker uses real `listSessions()` data
  (same call History already makes), not a fabricated date like the
  mockup's illustrative "3d ago" -- confirmed against test1@example.com's
  real 7-session account (4 volley, 3 backhand) before calling it done.

**Verified end-to-end on the `vt_test` emulator, both themes, not
assumed from the mockup** -- logged in as `test1@example.com`, walked the
real flow: Home's hero card + stroke list (real "last: 2d ago" markers,
Volley correctly marked most recent) -> History (most-recent Volley
session promoted, Backhand/Volley compressed into the "Earlier" list) ->
Results (large "Volley" headline, pull-quote weight-transfer insight with
citation, knee/elbow items demoted but citations intact) -> StrokeChat (a
real "How am I trending?" question sent through the live backend/Ollama
pipeline -- loading row shows the avatar's listening expression + teal
dots alone, spinner confirmed gone) -> Settings -> Edit Coaching Profile
("STEP 2 OF 2" breadcrumb on the fixed 2-step flow). Full sequence
repeated after `adb shell cmd uimode night yes` -- every screen re-themes
correctly (navy-on-cream hero card flips to the dark palette's light-slate-
on-navy-ink, same for the confidence/quality pills and pull-quote border).

**All 5 items are live and verified in both themes.** `mobile/lib/config.js`'s
`BACKEND_URL` was temporarily pointed at the emulator alias again for this
pass and reverted to the original LAN-IP line exactly as found afterward.

## 2026-08-26: Externally pretrained ball-tracking + court-detection checkpoints (yastrebksv/TrackNet, yastrebksv/TennisCourtDetector) tested on the same real footage -- RULED OUT, worse than this project's own models, not better

Follow-up to the 2026-08-12 confirmed generalization failure (1.22m-6.27m
court-homography reprojection error on the 3 real clips). Cheap,
no-training-required test: do externally pretrained checkpoints, trained on
real broadcast footage rather than this project's own narrower curated
dataset, generalize any better? Explicitly out of scope, per instruction:
stroke classification (no better pretrained alternative found in the earlier
2026-08-12 "Path B" literature search) and segmentation (stays closed per the
standing decision).

**License check, done before downloading anything, not glossed over:**
neither repo ships a `LICENSE` file or any license statement in its README --
confirmed via the GitHub API (`license: null` on both
`yastrebksv/TrackNet` and `yastrebksv/TennisCourtDetector`) and by listing
repo contents directly. Default copyright applies -- no explicit grant to
use, modify, or redistribute, even though both are public. Flagged to the
user before downloading; approved for LOCAL VALIDATION USE ONLY. **Real
constraint that survives this entry regardless of the result below: even a
good number here would not by itself clear these checkpoints for shipping in
the product** -- that would need the author's explicit permission first, or a
differently-licensed replacement, same class of caveat already logged for
the unofficial MoViNet-A0 port in the 2026-08-12 "Path B" entry.

**Downloaded** (~85MB total): `model_best.pt` (TrackNet, 42.9MB) and
`model_tennis_court_det.pt` (TennisCourtDetector, 42.3MB), both from the
Google Drive links in each repo's own README, plus the small `.py` source
files needed to construct the architectures the checkpoints load into. New
directory: `ir/external_pretrained_validation/` (source mirrors, checkpoints,
new script `run_external_pretrained_validation.py`, debug output).

**Reused the exact same measurement as the 2026-08-12 pass** -- same 3 real
clips (`mvp/real_footage_test/`), same `ir/geometry.py` `compute_homography`
reprojection-error metric, same visual-confirmation discipline (annotated
debug frames, not just trusted numbers). Adapted preprocessing to match how
these specific checkpoints were actually trained (confirmed by reading their
source, not assumed): BGR input, not RGB (their own code never converts);
plain /255 normalization, no ImageNet mean/std; fixed 640x360 model input
with real per-clip native-resolution scale factors computed fresh (their own
reference scripts hardcode a "scale=2" that silently assumes 1280x720 native
-- would have been a hidden bug + an unfair thumb on the scale for these
1920x1080 clips). Confirmed by reading `court_reference.py`'s key-point
construction (and checking real-world ratios) that this repo's 14-keypoint
order is IDENTICAL to `ir/geometry.py`'s `COURT_KEYPOINTS_REAL_WORLD_M`
order -- the two plug together with no reordering.

**Sanity-checked the pipeline on genuine in-distribution broadcast frames
before trusting any real-clip number** (`ir/external_pretrained_validation/sanity_check/`,
images from the TennisCourtDetector repo's own README) -- on those, all 15
heatmap channels produce sharp, confident peaks (max ~1.0) at spatially sane
locations matching the visible court geometry. This matters because the
real-clip result below looks, at first glance, like a bug; the sanity check
is what confirms it isn't one.

**Ball tracking -- worse than this project's own model on all 3 clips, not
better:**

| clip | own model: visible rate / conf mean | external: visible rate / conf mean |
|---|---|---|
| clip1_serve_practice | 75% (45/60) / 0.715 | 37% (22/60) / 0.347 |
| clip2_teen_practice | 65% (39/60) / 0.609 | 33% (20/60) / 0.327 |
| clip3_group_distant | 25% (15/60) / 0.265 | 20% (12/60) / 0.187 |

Same relative difficulty ordering as every other component tested on these 3
clips (clip1 easiest, clip3 hardest), so the external model isn't behaving
randomly -- it's just uniformly weaker here. clip1's same stray-tennis-balls
confound noted on 2026-08-12 (ball-basket drill balls littering the ground)
applies equally to both models, not a new problem introduced by this test.

**Court homography -- categorically worse, not just a worse number:**
all 3 clips FAILED outright (`cv2.findHomography`: "degenerate keypoints").
Root-caused, not just logged as a crash: 13-14 of the 15 output heatmap
channels collapse to an exactly-constant value across the entire spatial
map on all 3 real clips (confirmed by printing per-channel min/max -- min
== max to float precision), so the "keypoint" location becomes an
arbitrary tie-break at the array's first or last index (every dead
channel lands at exactly (0,0) or exactly the frame's bottom-right
corner). Traced to the checkpoint's BatchNorm running statistics: stored
`running_var` values are enormous (hundreds to tens of thousands) relative
to the checkpoint's `weight` (~4.37) and real pre-BN activation scale, so
at eval time almost all real spatial variation gets divided down to
near-zero before the affine BN transform, collapsing the sigmoid output to
the bias term everywhere except the rare pixel with an activation extreme
enough to survive it. This is the SAME class of distribution-shift failure
already confirmed for this project's own CourtKeypointNet on 2026-08-12
(camera-framing convention mismatch), just manifesting through BatchNorm
collapse instead of "garbled-but-present" keypoints -- a MORE total failure
mode, not a milder one.

**Verdict: RULED OUT, honestly, per instruction to report either way.**
Neither externally pretrained checkpoint generalizes better to this
project's real amateur footage than this project's own from-scratch models
-- ball tracking is uniformly weaker on all 3 clips, court detection fails
outright rather than merely producing a worse number. The heatmap feature
stays paused; this closes one more real, externally-sourced option, not a
half-checked one -- both checkpoints were genuinely downloaded, genuinely run
through the same measurement, and the court-detection failure was root-
caused (BatchNorm collapse) rather than left as an unexplained crash. Debug
overlays for all 3 clips, both models, are at
`ir/external_pretrained_validation/example_output/` for visual inspection.

## 2026-08-26: Momentum tracking upgraded from reconstruction to genuine prediction -- leak-free feature set + horizon-shifted target, smoke-tested, working. Full run not yet started, per instruction.

Direct follow-up to the explicit gap named in the 2026-08-23 closeout entry:
"What would actually demonstrate independent predictive power, named
explicitly, not started: predicting momentum ... ahead of the current point
using only PAST information (no same-timestep `point_winner`), or training
against a target not directly derived from the same per-point signal the
model already sees as input." This entry is that work.

**1. Rebuilt the input feature set (`dataset.py`), leak-free.** Split the
old 12-dim `FEATURE_NAMES` into two groups:
- **Kept (`FEATURE_NAMES`, now 6-dim):** `server`, `game_score_diff`,
  `set_score_diff`, `is_break_point`, `is_set_point`, `is_match_point` --
  confirmed (re-checked `DATA_ACQUISITION_CHECKLIST.md`, not re-assumed)
  that `Set1`/`Set2`/`Gm1`/`Gm2`/`Pts` "record the score entering each
  point... not the score after it," and `Svr` is the fixed alternating-
  service assignment -- none of these 6 are derived from the point they
  describe having been played yet.
- **Dropped from the model's input (`LEAKY_FEATURE_NAMES`, still computed
  for diagnostics, never passed to `MomentumDataset`):** `point_winner` (the
  exact +1/-1 series `momentum_ewma_p1` is a plain EWMA of -- this was the
  entire problem), `rally_length_norm`, `server_streak_norm`, and the 3
  `point_outcome_*` flags -- all derived from how the point itself was
  played (shot notation, terminal character), only knowable after the fact.
  Fixed a related bug while in this code: `server_streak_norm` was
  previously recorded AFTER folding in the current point's own outcome
  (`streak[i] = running` computed post-update) -- moot now that it's
  excluded from the model's input, but fixed anyway so the diagnostic
  column itself is honest, not just unused-and-still-wrong.
- `TennisMomentumLSTM.__init__`'s `input_dim` default: 12 -> 6.

**2. Reframed the target (`MomentumDataset`, new `horizon` parameter,
default 5).** Position i's target is no longer `momentum_ewma_p1[i]`
(same-point reconstruction) -- it's `momentum_ewma_p1[i + horizon]`,
1-indexed: horizon=1 asks "what will this point's own resulting momentum
be, having seen only the state entering it" (the literal example given);
horizon=5 (the default used here) asks the same question 4 points further
out. Picked a multi-point default deliberately, not just horizon=1: with
`point_winner` now excluded at EVERY lag (not just the current point), even
horizon=1 is no longer literal reconstruction, but a single EWMA step's own
contribution is small (halflife=10 -> alpha=1-0.5^0.1=0.067, so one point's
own outcome only ever moves the value by ~6.7% of the full range) --
horizon=5 compounds that irreducible per-point uncertainty over more
unseen points, making the "genuinely forecasting ahead" framing
unambiguous rather than borderline. The last `horizon-1` points of each
match have no valid future target and are truncated (not masked with a
sentinel -- simpler, no second mask semantic needed on top of the existing
padding mask); matches too short to have even 1 valid position after
truncation are dropped and counted, not silently skipped (0 dropped at
`min_points=20`, `horizon=5`, confirmed in the smoke-test run below).

**`evaluate.py` updated to match**, not left stale: `qualitative_check` now
plots `pred[i]` against the EWMA value at the point it actually forecasts
(`Pt[i] + horizon - 1`), not against `true_ewma[i]` itself -- comparing a
forecast to the wrong point would have silently understated error on a
trending match and overstated it on a flat one. `horizon` is read back from
the checkpoint (`ckpt["horizon"]`, now saved by `train.py`) rather than
assumed, so evaluation always matches whatever the checkpoint was actually
trained with.

**Verified the pipeline before trusting the smoke test, not just run and
read**: `dataset.py`'s own `__main__` check still reports the same
4,994 matches / 823,874 points / 3,995-499-500 split as before this change
(feature-engineering and split logic untouched, only which columns get
selected into `X` and how the target is shifted) -- confirms this is a
real feature/target reframe, not an accidental corpus change. Spot-checked
`MomentumDataset` shapes directly on 5 real matches before running
`train.py`: horizon=5 sequences are consistently 4 points shorter than the
same matches' horizon=1 sequences (e.g. 130->126, 211->207, 94->90),
matching the truncation math exactly.

**Smoke test** (`train.py --smoke-test`, same 64 train / 16 val matches
subsample, fixed seed, 5 epochs, horizon=5, input_dim=6): **loss decreases
every single epoch, both train and val** -- train MSE 0.0988 -> 0.0688, val
MSE 0.1147 -> 0.0733 (train MAE 0.2477 -> 0.1962, val MAE 0.2687 -> 0.1995).
Confirms the model is learning real signal from score-state/server/
game-context ALONE -- no point-outcome history at any lag -- not stuck or
diverging. Resource cost, measured: 0.054M params (vs 0.055M for the old
12-dim input -- input layer is a tiny fraction of total params, negligible
change), 0.05GB peak GPU memory, ~0.1-0.6s/epoch on the smoke subset,
20.4s one-time full-corpus feature-engineering cost -- all consistent with
the original smoke test's own resource profile; this reframe changed what
the model can see, not its cost.

**Honest framing of the smoke-test numbers, upfront, not deferred to the
full-run report:** these values are naturally higher than the OLD
reconstruction task's smoke test (val MAE 0.183 after 5 epochs on the same
64/16-match subsample) -- expected, not a regression, since the model can
no longer see any point outcome at all, only pre-point score/server/game
context. This smoke test's only job is confirming the pipeline is wired
correctly and the model can learn from the new, harder, honestly-scoped
input -- it does. The real held-out number, and whether it's still small
enough to be behaviorally useful (Gaudio comeback shape, negative-control
flatness), is exactly what the full run + `evaluate.py` will show, and
per instruction, reporting back now, before starting it.

**Not yet done, explicitly, per instruction:** the full training run
(30 epochs, full 3,995-match train set, checkpointed every epoch), held-out
test MSE/MAE/RMSE on the real test split, and the qualitative check against
the Gaudio comeback / Djokovic-Cilic negative control shapes -- now
comparing the model's forecast against the correctly-shifted target, not
the same-point value. Reporting back now, before starting that run.

## 2026-08-26: Momentum tracking full run complete, evaluated against a real baseline -- genuinely predictive, but does NOT yet beat naive persistence; reported plainly, not oversold

**Full run: 30 epochs, full 3,995/499/500 match split, checkpointed every
epoch, horizon=5, input_dim=6 (leak-free feature set from the smoke-test
entry).** No NaN issues this time (the 2026-08-23 Gm2 forward/back-fill fix
already covers the one known real gap). Converges smoothly: val_mse
0.0479 (epoch 1) -> 0.0375 (epoch 28), val_mae 0.1600 -> 0.1357. Train and
val track closely throughout (epoch 28: train_mae=0.1366, val_mae=0.1357 --
val slightly BELOW train, no overfitting gap) -- a healthy training curve,
consistent with the task being harder for THIS feature set specifically
rather than the model being undertrained or badly regularized. **Best
checkpoint selected by val_mae, not last epoch** (epoch 30's val_mae=0.1359
is marginally worse than the best): **epoch 28, val_mae=0.1357**, saved to
`momentum_lstm_best.pt`, same overfitting-awareness convention as every
other model here. ~3.7-4.7s/epoch, ~120s total for all 30 epochs --
consistent with the smoke test's own projection and the old reconstruction
run's cost (this reframe changed what the model can see, not its cost).

**Held-out TEST set (500 matches, 78,271 points -- 80,271 minus 500*4
horizon-truncated trailing points, exactly accounted for): MSE=0.03745,
MAE=0.13520, RMSE=0.19352.** ~34x the old reconstruction task's MAE
(0.00401) and ~936x its MSE (0.00004) -- exactly the order-of-magnitude
jump expected once `point_winner` and every other outcome-derived feature
were removed and the target was shifted 5 points ahead, per the smoke-test
entry's own prediction. Confirms this is measuring real forecasting
difficulty now, not a near-zero reconstruction floor.

**The number alone doesn't say whether 0.135 MAE is actually GOOD for this
task -- so two real baselines were computed on the identical test set
before calling this a useful result, not after:**

| approach | MAE | MSE | RMSE |
|---|---|---|---|
| **trained model** | 0.13520 | 0.03745 | 0.19352 |
| persistence (predict "momentum hasn't changed in 5 points") | **0.13048** | **0.02837** | **0.16844** |
| neutral (always predict 0) | 0.18669 | 0.06238 | 0.24975 |

**Honest finding, not glossed over: the trained model is slightly WORSE
than the naive persistence baseline on every metric** (MAE +3.6%, MSE
+32% relative to persistence), while clearly beating the neutral/predict-0
baseline (MAE -28%, MSE -40%). So the model has genuinely learned to use
score-context (server, game/set score, break/set/match-point flags) to
make informed forecasts rather than defaulting to a neutral guess -- that
part is real. But it has NOT learned to forecast momentum better than the
simplest sensible heuristic for this task ("assume it hasn't moved much
in 5 points"). Given the healthy train/val curve (no overfitting gap),
this reads as a genuine ceiling for what these 6 score-context features
predict about momentum 5 points out, not an undertrained model -- worth
stating exactly this plainly, per instruction, rather than letting the
"beats neutral" comparison alone imply more capability than is actually
demonstrated here.

**Qualitative check (`evaluate.py`, `figures/trained_model_vs_ewma_examples.png`,
updated to plot each forecast against the point it actually targets --
`Pt[i]+horizon-1`, not `Pt[i]` -- so the comparison is apples-to-apples):**
- **Gaudio comeback: right shape, forecast 5 points ahead of the real
  swing, damped in amplitude.** Model output stays negative through
  roughly the first 110-120 points (true label's sustained-negative
  span runs through ~140), trends positive from there, and stays mostly
  positive afterward with realistic noise -- the comeback's DIRECTION and
  rough TIMING both come through in the forecast, genuinely predicted
  ahead rather than copied from same-point input. Amplitude is
  consistently damped relative to the true EWMA (model's peak ~0.4-0.65
  vs the label's ~0.6-0.65, and the model never reaches the label's
  early swings to -1.0/+0.85) -- expected, not a defect: the true EWMA
  can swing to near the extremes early in a match off a handful of point
  outcomes the model never sees; a model using only score-context has no
  way to replicate that same extremity and shouldn't be expected to.
- **Negative control (Djokovic/Cilic blowout, genuinely held out -- VAL
  split, confirmed by re-running `match_level_split` on this run's own
  data, unchanged from the 2026-08-23 assignment): stays positive
  throughout, no invented drama, mostly.** Model output ranges roughly
  0.15-0.45 for nearly the entire match, tracking the label's real
  (smaller, noisier) positive fluctuations without fabricating a
  reversal -- the specific failure mode this check exists to catch did
  not occur. One honest exception, not smoothed over: a brief, shallow
  dip to slightly negative (~-0.05) around points 105-112, before
  recovering back to clearly positive -- not a sustained or dramatic
  reversal, but not perfectly flat either.
- Federer/Wawrinka: correct DIRECTION of the turnaround (starts positive/
  Wawrinka-favoring, ends negative/Federer-favoring, forecast ahead of
  the real swing) but, like Gaudio, damped relative to the label's own
  much larger early swing (model starts ~0.3 vs the label's ~1.0).

**Split disclosure, same as 2026-08-23, re-verified rather than assumed
carried over:** Gaudio and Federer/Wawrinka are both in TRAIN (re-checked
against this run's own `match_level_split` call, unchanged from before --
feature/target changes don't touch match-id assignment); only Djokovic/
Cilic is genuinely held out (VAL). Matters more here than it did for the
old reconstruction task, since this result is no longer near-ceiling --
the Gaudio/Federer shape checks are qualitative-on-seen-matches, only the
negative control and the quantitative TEST-set numbers above are on
genuinely unseen data.

**Answering the actual question this upgrade was for, plainly: partially
useful, not fully.** The upgrade unambiguously did what it was supposed to
-- this is no longer a validated reconstruction of a formula the model
could already see the exact input to; `point_winner` and every other
point-outcome feature are gone from the input at every lag, and the held-
out error moved by 2-3 orders of magnitude exactly as that change would
predict. The model demonstrably learned real, non-trivial signal from
score-context (clear win over the neutral baseline; qualitatively correct
comeback/turnaround direction and rough timing on both positive examples;
no fabricated drama on the negative control). But by the most relevant
real-world bar -- does it forecast momentum better than assuming momentum
hasn't changed much in 5 points -- it does not, on this test set, with
this feature set. **Not a case for wiring this into the product as a
"prediction" surface as-is.** A real next step, if revisited: either a
richer leak-free feature set (recent SERVER-level game/set-transition
patterns are all that's here now; something like rest/fatigue proxies or
tournament/round context could plausibly help) or accepting that
persistence is the honest baseline to beat and reporting future attempts
against it explicitly, not just against a neutral guess.

`training/momentum_tracking/README.md` rewritten to match: new lead section
states this entry's result (numbers, baseline table, qualitative findings,
bottom-line verdict) with the same directness the old lead section used
for its own reconstruction-task limitation; the 2026-08-23 version is kept
further down as explicit history, not deleted.

## 2026-08-26: LLM-quality-lever test -- Qwen3 14B, few-shot examples, schema-tightened "implication" field. One real improvement, two mixed/not-shipped, all three re-verified against the grounding harness

Three independent quality levers, tested with real before/after generations
against the real grounding harness on every trial -- per instruction, no
change ships without re-confirming 100% grounding. Verdict up front: item 1
(Qwen3 14B) is a genuine improvement; items 2 and 3 show real but MIXED
results with a real, measured downside each -- neither is shipped (neither
default was flipped), reported honestly rather than declared a win on a
small sample.

### Item 1: Qwen3 14B feasibility -- genuinely fits, genuinely better, real cost

**VRAM headroom, measured before pulling anything, not assumed:** 11,136
MiB free (RTX 3060, 12,288 MiB total) with nothing else GPU-resident --
`backend/app.py` runs its stroke classifier on `CPUExecutionProvider`
(confirmed by reading the code, not assumed), so the backend contributes
~0 GPU memory regardless of whether it's running alongside. Qwen3 14B's
Q4_K_M tag confirmed via the Ollama registry manifest (not downloaded
blind first): 9.276 GB, 14.8B params. Pulled (user-authorized in the
task instruction itself, per its own "if it genuinely fits, pull it"
wording) and measured for real, not assumed to fit from the file-size math
alone: `ollama ps` during a live generate call shows **100% of the model
resident in VRAM** (`size_vram` == full model size, no CPU offload),
**10,256 MiB used, 1,858 MiB free** -- genuinely fits, with real headroom,
not a forced squeeze.

**Same trial script used for both models** (`rag/run_qwen_multitrial.py`,
generalized with a `model` parameter rather than duplicated -- byte-
identical default behavior for existing callers), same real SessionRecord
(`ir/example_output/session_record_0.json`), same 6 trials as the ORIGINAL
Qwen3 8B evaluation (2026-07-17) it's being compared against:

| | Qwen3 8B (2026-07-17, original) | Qwen3 14B (2026-08-26) |
|---|---|---|
| clean trials | 4/6 (100% grounded each) | **6/6** (100% grounded each) |
| failure modes | 1/6 complete hallucination (0/6 grounded, fabricated record_id + 2 fake feature names); 1/6 schema-malformed (missing "type" on a trailing item) | 0 hallucinations, 0 malformed |
| pooled grounding rate | 4 clean trials only (no single pooled denominator reported originally) | **19/19 grounded_observation items (100.0%)** |
| latency | mean 26.8s on the 4 clean trials (bimodal, 2 early ad-hoc runs at 64-73s) | mean 60.4s (range 47.9-76.8s) |

**Real finding, not glossed over: one 14B trial (#2) shows a real rule
violation the pooled 100% number doesn't capture.** `check_grounding()`
flagged `SUSPECT_GENERAL_NOTE` on a `general_note` item containing an
uncited digit -- verified against the real record: NOT a hallucination
(the cited "51.0%" matches `features.first_serve_pct=50.989...` exactly),
but a real rule-4 violation (a numeric claim must be a `grounded_observation`
with a citation, never a bare `general_note`) that the retry-gate's current
accept logic (`gated_pipeline.py`, `fully_grounded` computed only over
`grounded_observation` items) does not check and would have let through.
This is a **pre-existing gap in the retry-gate itself**, not something 14B
introduced -- it could equally happen on 8B -- surfaced here because 14B's
run happened to trigger it. Flagged as a real follow-up item (tightening
the gate to also reject on `SUSPECT_GENERAL_NOTE`), not fixed in this pass
(out of the 3 levers this task scoped).

**Response quality, spot-checked, not just the grounding number** (real
text from `rag/example_output/qwen3_14b_trials/`): 14B's coaching text
reads more naturally and more often names a SPECIFIC diagnostic distinction
rather than a generic deferral -- e.g. 14B: "Check if these clustered
during break points or late in games, as that suggests pressure management
rather than a toss issue" vs. 8B's own clean-trial text on the same feature:
"suggests pressure-related second serve errors, particularly on break
points or late in service games." Both are real, grounded, coherent --14B
reads as a genuine (if modest) step up, not a dramatic one.

**Verdict: real improvement.** Strictly better reliability (6/6 vs 4/6
clean on the same trial count), comparable-or-better text quality, at
~2.3x the latency cost (60.4s vs 26.8s mean) -- and it genuinely fits the
hardware with real headroom to spare. **Not switched as the live default**
in this pass -- that's a production-config change (`backend/app.py`'s
`llm_backend`) with real consequences (every coaching generation call gets
~2.3x slower) that should be a deliberate decision, not a side effect of a
feasibility test; recommending it, not silently shipping it.

### Item 2: few-shot examples -- real improvement in scoping, real new risk in phrasing reuse

**2 real, already-grounded examples** (`rag/few_shot_examples.py`), NOT
invented: reconstructed exactly from `rag/example_output/qwen3_gated_diverse_trials/{clutch_breakdown,movement_fatigue}_report.txt`'s
already-verified `[GROUNDED]` text + field path + cited value, paired with
the real `record_id` from each source SessionRecord -- re-validated against
`rag/schema.py`/`rag/grounding.py` before use (3/3, 3/3, both 100%), not
just copied and trusted. Picked as the 2 cleanest available: both
succeeded in 1 attempt at full (1.0) retrieval recall in the original
diverse-session trial. Wired as `build_prompt(..., use_few_shot=True)`,
default `False` -- byte-identical existing behavior for every caller that
doesn't pass it.

**A/B'd on 2 held-out sessions** (`rag/run_few_shot_ab_trial.py`) --
deliberately NOT the 2 sessions used as examples, and deliberately
including `second_serve_ace_decline`, the ORIGINAL documented thin/generic
case (2026-07-18) already improved once by the 2026-08-21 rule-8 fix, as
the sharpest real regression test:

- **Real improvement: tighter scoping.** BEFORE, `second_serve_ace_decline`
  produced 5 items, 2 of which cited real-but-tangential fields
  (`avg_reaction_time_ms`, `total_distance_covered_m`) never named in the
  session's actual `explanation`. AFTER, exactly 3 items, matching exactly
  the 3 flagged deviations -- no scope creep. Also richer on genuinely
  novel-to-the-examples features (`ace_rate`'s AFTER implication names 3
  distinct possible causes vs. BEFORE's single generic suggestion).
- **Real new risk, not glossed over: near-verbatim phrasing reuse when a
  feature name overlaps a few-shot example.** Both held-out test sessions
  happen to include `unforced_error_rate` (also present in the
  `clutch_breakdown` example). AFTER's text for that feature, in BOTH
  sessions, is close to word-for-word identical to the example's own
  phrasing ("Check when they happened/occurred -- a cluster late in the
  session points to fatigue, so rest more between reps; errors on one
  specific shot point to that shot's technique...") -- the model is
  templating off the example rather than generating session-specific
  reasoning whenever the feature name matches. Both instances still pass
  grounding (the CITATION correctly points at each session's own real
  value; the prose itself doesn't restate a wrong number), so this is a
  diversity/quality risk, not a grounding failure -- but a real one, only
  visible because both my held-out test sessions happened to share this
  feature with an example.

**Verdict: mixed, not shipped.** Real, measurable improvement in scoping
discipline; a real, only-just-surfaced risk in phrasing diversity for
recurring feature names. 2 held-out sessions is too small a sample to know
whether the templating risk is narrow (an artifact of picking examples
that happen to share `unforced_error_rate` with common anomaly profiles)
or systemic. `use_few_shot` stays `False` by default -- not a validated
net improvement yet, per instruction.

### Item 3: schema-tightened "implication" field -- mixed, real retry cost on one test case

**New required field, kept OUT of the production schema** --
`rag/schema.py:CoachingFeedbackItem.implication` is additive-only
(`str | None = None`, no validator), so every existing caller keeps
parsing exactly as before whether or not a response includes this key.
The actual hard requirement (non-empty, at least 15 chars, not identical
to `text`) lives in a separate subclass,
`rag/implication_schema.py:StrictCoachingResponse`/`StrictCoachingFeedbackItem`
-- used only by this lever's own trial, so testing a stricter schema could
never accidentally break every other already-working caller of the
production `CoachingResponse`. `build_schema_instructions(require_implication_field=True)`
describes the new required key to the model; confirmed byte-identical
output at the default `False` before touching anything live.

**A/B'd on the same 2 held-out sessions** (`rag/run_implication_field_ab_trial.py`),
BEFORE = current production schema/prompt exactly, AFTER = the strict
schema + matching prompt:

- **`rally_style_shift`: clean win.** Same attempt count (1 attempt both
  before and after) -- zero added cost. AFTER's `text`/`implication` split
  reads noticeably better organized: e.g. `text`="Your average rally
  length was 1.0, shorter than usual, indicating a more defensive
  approach." / `implication`="Shorter rallies may signal fatigue or a
  tactical shift -- check if this was intentional or if rest between
  points is needed." -- a real, clean separation of finding from
  takeaway.
- **`second_serve_ace_decline`: real cost, not glossed over.** BEFORE
  needed 2 attempts (1 parse failure) in 105.2s; AFTER needed 3 attempts
  (**2 of 3 failed to parse**) in 151.2s -- the stricter schema
  measurably increased retry churn and latency on this session. Quality
  on the accepted AFTER response is good, but 2 of its `text`/
  `implication` pairs are semantically redundant even though mechanically
  distinct (e.g. `text` already says "may be the cause" of a fatigue
  pattern, and `implication` restates "check for fatigue patterns" in
  different words) -- the two deliberately-mechanical checks (non-empty,
  not string-identical to `text`) can't catch a PARAPHRASED duplicate,
  only a literal one, by design (no NLP judgment call) -- so this is an
  expected, disclosed limitation, not a bug.

**Verdict: mixed, not shipped.** Genuinely better structure on 1 of 2
test sessions at zero cost; a real, measured 3x-latency/2-parse-failure
cost on the other, plus a known (by design) blind spot for paraphrased
redundancy. `require_implication_field` stays `False` by default -- not a
validated net improvement yet, per instruction.

### Grounding re-confirmed on every trial, all three levers, per instruction

Every single trial across all three levers -- 6 for Qwen3 14B, 2x2 for
few-shot, 2x2 for the implication field -- ran through the SAME real
`rag/grounding.py:check_grounding()` the production pipeline uses, via the
SAME `rag/gated_pipeline.py:generate_grounded_response()` retry-gate
(accept only at exactly 100% grounding on every `grounded_observation`
item). No change was accepted as "working" on anything short of that bar.
The one real gap found (SUSPECT_GENERAL_NOTE not currently blocking
acceptance) is a property of the EXISTING gate, confirmed present
regardless of which of these 3 levers is used -- not introduced by any of
this session's changes, and not fixed here (out of scope for this pass,
flagged as a real follow-up).

**Net result: 1 of 3 levers recommended (Qwen3 14B, pending a deliberate
decision to switch the live default), 2 of 3 built, tested, and explicitly
NOT shipped** (few-shot examples, implication-field schema tightening) --
both stay available as opt-in parameters (`use_few_shot`,
`require_implication_field`) for further iteration, not deleted, but
neither is the production default. All new/changed files:
`rag/few_shot_examples.py` (new), `rag/implication_schema.py` (new),
`rag/run_few_shot_ab_trial.py` (new), `rag/run_implication_field_ab_trial.py`
(new), `rag/run_qwen_multitrial.py` (generalized with a `model` param),
`rag/prompt.py` (`use_few_shot`/`require_implication_field` params, both
default `False`), `rag/schema.py` (additive-only `implication` field),
`rag/gated_pipeline.py` (`response_schema` param, default `CoachingResponse`).

## Qwen3 14B deployed as the live default; SUSPECT_GENERAL_NOTE gate closed (2026-08-26)

Follow-up to the lever test above: the user asked for two specific, separately-tracked
actions -- (1) switch Qwen3 14B from "recommended, not yet deployed" to the live
default, verified end-to-end on the emulator in both themes, and (2) close the
SUSPECT_GENERAL_NOTE gap flagged in that same test (an accurate value appearing
uncited in a `general_note` item was only a WARNING, not a rejection reason) --
tested against the real case that surfaced it plus constructed variants, same
methodology as the 2026-08-12 skill-tier backstop. Few-shot examples and the
implication-field schema stayed off, per explicit instruction -- untouched here.

### Temperature-calibration premise check, before touching anything

The instruction said to "refit temperature calibration for this model specifically."
Checked the codebase first rather than guessing: the only *fitted* temperature
calibration that exists is `training/stroke_classification/calibrate_temperature.py`'s
Guo et al. 2017-style calibration for the StrokeGRU classifier's confidence output
(`backend/app.py`'s `STROKE_TEMPERATURE_PATH`/`_state["stroke_temperature"]`) --
entirely independent of which Ollama model generates coaching text. The LLM's only
"temperature" is `rag/llm.py:OllamaBackend.generate()`'s hardcoded
`options.temperature = 0.2`, a decoding/sampling knob that was never itself a fitted
quantity for either 8B or 14B. Flagged this mismatch to the user rather than silently
acting on either interpretation; confirmed there was nothing to refit -- proceeded with
just the model swap.

### SUSPECT_GENERAL_NOTE gate fix

`rag/grounding.py`: added `GroundingReport.has_suspect_general_note` (true iff any
`general_note` item's verdict is `SUSPECT_GENERAL_NOTE`), deliberately kept out of
`grounding_rate`'s own denominator so that metric's existing meaning is unchanged.
`rag/gated_pipeline.py`: `_try_one_generation` now requires `not report.has_suspect_general_note`
in addition to 100% `grounding_rate` for `accepted`; a response that is 100% grounded on
every `grounded_observation` item but has a `general_note` with an uncited digit is now
rejected and retried, not accepted with a warning. Also fixed a message bug caught while
implementing this: the old failure-reason text always said "grounded X/Y items, below the
required 100%" even when X/Y was 100% and the actual rejection reason was
SUSPECT_GENERAL_NOTE -- `generate_grounded_response`'s failure message now branches
three ways (parse failure / genuinely below 100% / 100% but suspect general_note).

**Verified (`rag/run_suspect_general_note_gate_trials.py`, new, 7/7 checks passed):**
- **Part A, the real case:** replayed the actual trial-2 raw output
  (`rag/example_output/qwen3_14b_trials/trial_2_raw.txt`) that originally surfaced this
  gap against its real source `SessionRecord` (record id verified to match, not assumed).
  Old gate would have accepted it (3/3 grounded_observation items, 100%); new gate
  correctly rejects it (general_note: *"Your first-serve percentage was 51.0%..."* --
  an uncited real value).
- **Part B, 4 constructed variants:** 2 true positives (single-digit general_note;
  a digit mixed among otherwise-clean general_notes) both correctly rejected; 2 true
  negatives (general_note with no digit; no general_note at all) both still correctly
  accepted -- confirms the fix doesn't start rejecting fine responses.
- **Part C, live:** 2 fresh generations through the real `generate_grounded_response()`
  retry loop against real `qwen3:14b`/`session_record_0.json` -- both succeeded, neither
  accepted response ever had `has_suspect_general_note=True`.

### Qwen3 14B deployment

`backend/app.py`: `OllamaBackend(model=...)` changed from `"qwen3:8b"` to `"qwen3:14b"`.

**End-to-end emulator verification, both themes, real sessions (not just a health
check):** started the real backend (`uvicorn backend.app:app --host 0.0.0.0 --port 8000`,
confirmed startup log reads `configuring Ollama backend (qwen3:14b)`) and a real Expo/Metro
session against a fresh AVD (`vt_test`). Found and fixed two stale dev-config issues along
the way (not part of the LLM change itself, but blocking this verification): `mobile/lib/config.js`'s
`BACKEND_URL` LAN IP was stale from 2026-08-21 (DHCP had re-leased the PC's address since),
and it pointed at port 8001 even though `backend/app.py`'s own docstring and the
"VisioTennis Backend" Windows Firewall rule both only recognize port 8000 (the 8001 vs.
8000 mismatch was already known, PROJECT_LOG.md 2026-08-21) -- repointed at the current LAN
IP on port 8000 to match both.
- **Light theme:** signed up a fresh account, completed profile setup (Player/Intermediate),
  picked a real serve clip from the emulator's library, ran it through
  `/api/v1/analyze-stroke` end-to-end (pose -> StrokeGRU -> biomechanics -> percentiles ->
  qwen3:14b gated generation) -- classified Backhand at 96% confidence, 100% tracking
  quality, 5 coaching feedback items with real percentile citations
  (`wrist_height_ratio`, `stance_width_ratio`, `knee_flex_left_deg`, etc.), no thin/bare
  restatements.
- **Dark theme:** toggled via `adb shell cmd uimode night yes` (the app has no in-app
  theme toggle -- it follows the OS `prefers-color-scheme`/`Appearance` setting, confirmed
  in `mobile/lib/theme.js`/`app/_layout.js`). Settings and Home screens re-rendered
  correctly with no unreadable/wrong-theme elements. Ran a second real session (a different
  clip, forehand this time) fully through the same pipeline -- classified Forehand at 90%
  confidence, 97% tracking quality, 5 grounded coaching items
  (`weight_transfer_ratio: 100th percentile`, etc.). Reverted the emulator to
  `uimode night no` afterward.

Both sessions' backend logs confirm real `POST /api/v1/analyze-stroke ... 200 OK` against
the live `qwen3:14b` backend -- not mocked, not a dry run.

**Deployed and verified**, both tracked items closed. `run_suspect_general_note_gate_trials.py`
kept as a permanent regression check (same pattern as the skill-tier backstop's own test file).

## qwen3:14b latency investigation: real stage breakdown, 4 levers checked, 1 shipped (2026-08-26)

With 14B now live, investigated where generation time actually goes before assuming nothing
could be done -- real measurement throughout (`rag/run_latency_investigation.py`, new, kept
as a record of how these numbers were produced), no change shipped without a real before/after,
same standard as every other change this session.

### Stage breakdown -- LLM generation confirmed dominant, not assumed

Real `stage_timings_s` from the two real emulator sessions run for the 14B deploy verification
above: **60.2s/65.8s total (91.5%)** and **111.8s/127.5s total (87.6%)** spent in
`llm_generation`. Every other stage is small and roughly fixed regardless of model:
`video_decode` ~0.9-1.4s, `tennis_classification` ~0.3-6.3s, `pose_extraction` ~1.3s,
`stroke_classification`/`biomechanics_extraction` ~0s (both pure numpy on an already-computed
pose sequence), `prompt_build_and_retrieval` ~3-7s. The grounding check itself
(`rag/grounding.py:check_grounding()`) isn't separately timed but is confirmed by code
inspection to be pure in-memory dict/field-path lookups with no I/O -- negligible, not a
real cost center. Retry overhead IS real cost, captured in `attempt_latencies_s` (a 2nd
attempt is a full 2nd generation, not a cheap correction).

### Four levers checked; three don't help, one does

**1. Prompt size -- already lean, not trimmed further.** Real prompt for stroke-level
coaching: 4,976 + 3,075 = 8,051 chars (~2,012 tokens estimated). `rag/stroke_prompt.py`
already deliberately trims what's shown to the model (`_build_citable_view`, a documented
fix for an earlier 3.8x-prompt-size regression) -- nowhere near the model's 40,960-token
context, and prompt processing isn't the bottleneck (generation is). No further trimming
shown to help; none attempted.

**2. Ollama context window -- confirmed at default (4096), explicitly raising it made
things WORSE, not left alone out of neglect.** `ollama ps` during a live generation
confirms `CONTEXT 4096` (Ollama's runtime default -- `rag/llm.py:OllamaBackend.generate()`
never sets `num_ctx`). Real paired test, same prompt, same model: default 4096 →
**45.6s**, `num_ctx=8192` explicit → **108.0s** (2.4x SLOWER), both landing at 100%
grounding either way. The model's "thinking" length scaled up to fill the extra headroom
(1,820→10,317 chars) rather than staying fixed -- more room didn't buy anything, it just
gave the model more space to think longer for the same answer. **Left at the default 4096
-- raising it is a real, measured regression, not a neutral no-op.**

**3. Thinking mode -- real but modest/unreplicated signal, deliberately NOT flipped.**
Ollama 0.33 returns Qwen3's reasoning in a separate `message.thinking` field (not inline
`<think>` tags in `content`, which is why `rag/run_qwen_eval.py`'s `clean_json_text()`
regex strip never had anything to strip -- it was already a no-op, just not previously
noticed). One clean paired trial: thinking-on **34.9s** vs `think=False` **28.8s** -- a
real but modest 1.21x, and both fully grounded (5/5 items, `think=False` output read as
coherent, well-reasoned text, not degraded). This is a genuine signal worth a dedicated
follow-up, but n=1 paired trial is too thin to act on, and -- more importantly -- disabling
a model's reasoning step is a change to how it arrives at an answer, arguably crossing from
"latency lever" into "quality-relevant" territory the task explicitly scoped out ("don't
touch model choice or quality"). **Flagged as a follow-up, not shipped in this pass.**

**4. `max_attempts=3` → `2` -- real evidence, shipped.** Pooled every real end-to-end
qwen3:14b generation from today (2 emulator sessions above + 5 fresh trials through the
real `generate_grounded_response()` retry loop, same stroke-level prompt/record): **7/7
real trials succeeded, and 0/7 ever needed a 3rd attempt** -- 5 succeeded on attempt 1, 2
needed exactly attempt 2 (one of those 2 was rejected on attempt 1 for `SUSPECT_GENERAL_NOTE`,
confirming the new gate from the section above is actively firing in practice, not just in
its own test suite). At 14B's ~35-110s per-attempt cost, a 3rd attempt is pure worst-case
latency with zero observed reliability payoff on this sample. `backend/app.py`'s
`/api/v1/analyze-stroke` call site changed from `max_attempts=3` to `max_attempts=2`
(only this call site -- `rag/gated_pipeline.py`'s own function default and
`rag/chat_pipeline.py`'s separate chat-turn default are untouched; no real data was
collected for the chat path, which is a different UX context, so nothing there was
assumed). **Caveat stated plainly, not glossed over: n=7 is a modest sample** -- if a
genuine need for a 3rd attempt shows up later, this should be revisited, not assumed
permanently safe. Backend restarted with the change; re-verified with one more real
`POST /api/v1/analyze-stroke` call end-to-end (real clip, real qwen3:14b) --
`n_attempts: 1`, 100% grounded, `200 OK`.

**Net result: 1 of 4 levers shipped** (`max_attempts=2`), backed by real pooled evidence
across 7 real generations; the other 3 either showed no benefit (prompt trimming had
nothing to trim that mattered), a real measured regression (raising `num_ctx` made things
2.4x slower), or too thin/out-of-scope a signal to act on (thinking-mode toggle -- flagged
for a dedicated follow-up test, not flipped here).

## qwen3:14b thinking-mode A/B test: real quality evidence, not just the timing signal (2026-08-26)

Tracked as its own item per instruction, not folded into the latency-lever pass above --
that pass found a real-but-thin 1.21x speedup on a single paired trial and explicitly
deferred a recommendation pending genuine quality evidence, not just a speed number. This
is that follow-up: **8 paired trials** (thinking on vs. `think=False`, same prompt each
pair), across **6 diverse real SessionRecords + 2 real StrokeEvents** (`ir/example_output/
session_record_diverse_*.json` + `stroke_event_biomech_{forehand,backhand}.json` -- the
same diverse fixture sets other lever evaluations this session already used, not one
prompt repeated 8 times), via a new script kept as a permanent record
(`rag/run_thinking_mode_ab_trial.py`; run pair-by-pair as 8 short foreground calls after
an unattended background version was killed 3 times in a row for unclear reasons -- the
user asked for a different execution shape rather than a 4th retry, and that worked cleanly).

### Real numbers, both latency and grounding

| | thinking ON (default) | `think=False` |
|---|---|---|
| mean latency | 64.1s (range 37.5-121.0s) | 28.9s (range 20.5-34.0s) |
| OFF faster | -- | **8/8 pairs** |
| fully grounded (would pass the retry-gate) | 6/8 | **8/8** |

**Not a pure speed-for-quality tradeoff -- `think=False` won on BOTH axes.** Pair 4
(`session_record_diverse_second_serve_ace_decline.json`) is the clearest case: thinking-ON
produced a **complete grounding failure (0.0)** on that exact prompt, while `think=False`
on the identical prompt was 100% grounded. Pair 1 similarly: thinking-ON tripped the
just-shipped `SUSPECT_GENERAL_NOTE` gate (a real digit -- 52.21%, 43.93% -- leaked into
uncited `general_note` items), `think=False` on the same prompt didn't. This means
thinking-on's real-world cost is understated by the raw latency numbers above: those two
failures would each trigger a real retry-gate regeneration in production (another full
~60-120s attempt), so thinking-on's true end-to-end cost gap is larger than 64.1s vs.
28.9s suggests.

### The real quality caveat -- found, verified, and disclosed, not glossed over

Read all 8 pairs' full text side-by-side (not just the grounding pass/fail) for the
thin/generic-vs-specific distinction already characterized earlier this session
(rag/DESIGN.md's semantic-miss case; the few-shot/schema-tightening lever entries above).
Individual `think=False` responses were NOT thin or vague -- they were specific,
appropriately number-forward, and gave concrete follow-up actions, same as thinking-on.
But a **quantified, different** genericness problem turned up: searching all 83 generated
items for exact duplicate text found **0/36 thinking-on items repeated verbatim across
different sessions, vs. 7/47 (15%) `think=False` items reusing IDENTICAL boilerplate text
across sessions with a similar anomaly shape** -- e.g. the exact sentence *"Your reaction
time was slower than usual today, especially later in the session -- that's a classic
fatigue marker, not a technical issue..."* appeared word-for-word in 3 different sessions'
`think=False` output. This is real evidence `think=False` falls back to a fixed template
for common anomaly patterns rather than composing session-specific text every time --
distinct from the earlier-characterized "vague single-instance" thin-response failure
mode, but a genuine quality cost worth tracking, not hidden because it complicates a
otherwise-clean "ship it" story.

### Verdict: shipped, with the caveat on record

Given this project's own stated priority order (grounding first, latency second, and
`think=False` won BOTH), and that the caveat found is real but different in kind from a
blocking failure (no ungrounded claims, no thin/vague text, only literal phrase reuse
across separate sessions a given player wouldn't see side-by-side), the call is to ship
it and track the template-reuse finding as a named follow-up rather than let it block a
measured double win. `rag/llm.py:OllamaBackend` gained a `think: bool | None = None`
param (default `None` -- byte-identical to every other existing caller, only sends
Ollama's `"think"` key when explicitly set). `backend/app.py`'s `_state["llm_backend"]`
construction now passes `think=False` -- this is a SHARED instance also used by stroke
chat (line ~617), which wasn't separately tested with chat-shaped prompts but shares the
same citation-discipline prompt structure as what was tested; flagged as an inference, not
a directly-tested claim.

**Re-verified end-to-end through the real production path** (not just the A/B script):
backend restarted, startup log confirms `configuring Ollama backend (qwen3:14b,
thinking=off)`, one real `POST /api/v1/analyze-stroke` call with a real clip ->
`200 OK` in 37.4s total, `n_attempts: 1`, `llm_generation: 23.7s` (down from the
~78-92s range seen earlier this session with thinking on) -- fully grounded, real
percentile-cited coaching text, no regression.

### Follow-up: StrokeChat spot-check with `think=False` live

The `think=False` change above applies to `_state["llm_backend"]`, a SHARED instance also
used by `/api/v1/history/{id}/chat` (StrokeChat) -- not separately tested by the 8-pair
A/B (which only exercised the initial-coaching-generation prompt shape). Targeted check,
not a full A/B: 4 real chat exchanges through the real endpoint against a real stored
session (`bdd064e7...`, a volley), covering all 3 requested response types plus one extra:

1. **`existing_citation`** (tier 1, no generation) -- "Can you say more about what you
   meant about my wrist height at contact?" -> correctly replayed the stored
   `wrist_height_ratio` citation verbatim, `n_attempts: 0` (tier 1 never generates).
2. **`new_grounded_claim`** (tier 2) -- asked about `shoulder_hip_separation_deg`, a real
   field NOT in the original coaching items -> correctly grounded, real citation
   (`biomechanics.shoulder_hip_separation_deg = 30.145...`), `n_attempts: 1`, 10.0s.
3. **`general_note`** (tier 3) -- generic "what makes a volley technically sound?" with no
   reference to the player's own data -> correctly routed, `citation: null`, no stray
   digits, 6.5s.
4. **`new_grounded_claim` again** (different field, `elbow_angle_deg`) -- correctly
   grounded (real value `35.147...` matched, cited), but took `n_attempts: 3` (the full
   budget) and 15.99s -- direct real evidence the `SUSPECT_GENERAL_NOTE` gate is actively
   rejecting attempts in the chat path too, not just session/stroke coaching, consistent
   with everything else confirmed this session.

All 4 exchanges: `200 OK`, `success: true`, grounding held. **Inference confirmed --
`think=False` works correctly through the real chat path**, nothing chat-specific broke.
One incidental, pre-existing (not caused by today's changes) observation while reading
turn 4's raw response: `backend/app.py`'s `/chat` response only surfaces
`turn.items[0].citation` as the top-level `"citation"` field, but a multi-item tier-2
response can have its first item be an uncited `general_note` framing sentence followed
by the actual cited claim -- the returned `citation` field can be `null` even though the
joined `text` does contain a grounded, cited number. Not a grounding bug (the number IS
backed by a real citation on a later item) and not something this task's scope covers
fixing -- noted here only so it isn't mistaken for a new regression if it's ever
re-encountered.

## Named follow-up, tracked: `think=False` boilerplate/template reuse across sessions (2026-08-26)

**Not investigated further per instruction -- captured here precisely enough to resume
without rediscovery.** Distinct from two earlier, different findings this session already
tracked: (a) the "thin/generic" failure mode (a single response giving a bare
number-plus-generic-label restatement with no specific implication -- see the
few-shot/schema-tightening lever entries above) and (b) the skill-tier/percentile
mislabeling backstop (2026-08-12 entry -- a citation technically grounded but wrong to
show given how the question was framed). This is a third, different shape of "generic":

**Finding:** in the 8-pair thinking-mode A/B test above
(`rag/example_output/thinking_mode_ab_results.json`), searching all 83 generated
coaching items for exact duplicate text found **0/36 `think=on` items repeated verbatim
across different sessions, vs. 7/47 (15%) `think=False` items reusing IDENTICAL
boilerplate text word-for-word across DIFFERENT sessions that happened to share a similar
anomaly shape** (e.g. same feature flagged, same direction of deviation). Concrete
example: the exact sentence *"Your reaction time was slower than usual today, especially
later in the session -- that's a classic fatigue marker, not a technical issue. Check how
long you'd been playing and what time of day it was before reading too much into it."*
appeared verbatim in 3 separate sessions' `think=False` output
(`session_record_diverse_movement_fatigue.json` and `session_record_diverse_
unforced_error_spike.json`, among others -- see the saved JSON for exact pairing). Each
individual instance is real, grounded, and not itself vague -- the concern is CROSS-SESSION
identical phrasing for a recurring pattern-shape, not any single response's quality.

**Why this wasn't blocking:** `think=False` was still shipped as the new default (see
above) because it won on both axes this project prioritizes first (grounding: 8/8 vs 6/8;
latency: 2.22x) -- the templating finding doesn't make any individual output wrong or
ungrounded, and a player only ever sees their own single result, not a side-by-side
comparison across sessions where the repetition would be visible.

**Why it's worth tracking anyway:** removing `thinking` from generation appears to reduce
the model's tendency to compose fresh phrasing per-session for common anomaly shapes,
falling back to a fixed template instead -- this could compound with other levers already
built this session (few-shot examples, `require_implication_field`) in ways worth
re-testing together rather than assuming they stack additively. Possible future angles,
NOT started: (1) re-run the few-shot A/B (`rag/run_few_shot_ab_trial.py`, currently off)
specifically against `think=False` to see if real examples reduce template convergence;
(2) check whether a slightly higher `temperature` (currently a fixed `0.2` in
`rag/llm.py`, itself never tuned per model) increases per-session phrasing variety without
costing grounding; (3) re-run this exact 8-pair harness with more diverse records (n=8 is
still a modest sample for a 15% observed rate) to see if the reuse rate holds, grows, or
was partly an artifact of these specific 8 fixtures sharing overlapping anomaly shapes by
construction.

## Follow-up: templating-rate recheck -- temp=0.3 shipped as a modest mitigation, NOT a resolution (2026-08-27)

Picks up the three "possible future angles" logged above, worked cheapest-first per
instruction. **Status: two of three angles now attempted (temperature done and shipped,
few-shot correctly still untried as last-resort). Stated plainly: the underlying
cross-session templating tendency remains -- this is a partial mitigation, not a fix.**

**Angle 3 (larger sample) -- done first, confirms the finding is real, not a small-n
artifact.** Added 16 new, genuinely distinct think=False trials
(`rag/run_templating_rate_recheck.py`): 5 real fixtures not used in the original 8-pair
test (`session_record_0/1/2.json`, `stroke_event_biomech_serve.json`,
`stroke_event_biomech_volley.json`; `stroke_event_0/1/2.json` and
`stroke_event_end_to_end.json` were excluded -- pre-`biomechanics`-field schema,
`KeyError` on `build_stroke_prompt`), plus 11 real, distinct StrokeEvents pulled from
`backend/app.db`'s `stroke_history` table (27 stored rows, deduped by `source_video` to 11
distinct clips -- genuine production data from real analyze-stroke calls made throughout
this project). Short of the "20-30 more" target (only 16 genuinely distinct new sources
existed without repeating an already-used source, which would trivially "duplicate" at
temp=0.2 for the boring reason that identical input -> similar output, telling us nothing
new) -- reported honestly rather than padded.

Combined with the original 8 sessions: **n=24 sessions, 143 items total, duplicate rate
15/143 (10.5%)** -- down somewhat from the original 15% (7/47) but confirmed as a real,
generalizing effect at 3x the sample size, not a fluke: duplicates now span new AND
original fixtures, and one duplicate pair involves two genuinely different REAL
production history sessions (`real_history::1000000261.mp4` and
`real_history::clip5_serve_setup.mp4`) -- independent evidence this isn't an artifact of
the original 8 fixtures' construction. **Verdict: real, worth addressing per instruction,
proceeded to angle 1 (temperature).** Caveat that still applies even after angle 1: n=24
sessions is a real improvement over n=8 but is still a relatively small base for the
9-13% dupe-rate figures below -- these are honest numbers on the sample collected, not a
claim of tight statistical confidence.

**Angle 1 (temperature tuning) -- tested two points, paired design.** Same 24 sources
re-run at each temperature (`rag/run_temperature_ab_trial.py`), think=False, only the
temperature changed, compared directly against the temp=0.2 baseline recomputed on the
identical 24 sources for a clean paired comparison. Baseline (temp=0.2, currently the only
value ever used in `rag/llm.py`): 10.5% dup (15/143), grounding not separately
re-verified at this step since it's just the already-collected baseline data.

- **temp=0.5**: dup rate dropped to 4.3% (real reduction) but grounding regressed --
  22/24 fully-grounded-would-pass-gate vs 24/24 at baseline, 2 new `SUSPECT_GENERAL_NOTE`
  flags never seen at temp=0.2 (`session_record_diverse_clutch_breakdown.json`,
  `session_record_diverse_rally_style_shift.json`). **Fails the stated bar** ("reduces
  verbatim repetition without hurting grounding") -- not shipped.
- **temp=0.3** (more genuinely "modest" bump, tested next): **24/24 parse_ok, 24/24
  fully-grounded, 0 suspect flags** -- no grounding cost at all, including both sessions
  that regressed at 0.5. Duplicate rate: **13/143 (9.1%)**, down from the 10.5% baseline
  on the identical 24 sources. A real reduction (dup groups dropped from a broader set to
  4 recurring templates, still all real/grounded individually), but a modest one -- 2
  fewer duplicate items out of 143, not a resolution of the underlying tendency.

**Overall verdict on temperature as a lever:** temp=0.3 is safe to ship by the
no-grounding-cost bar (matches baseline exactly, 24/24) and gives a small, real
improvement, unlike temp=0.5 which trades grounding for a bigger reduction and therefore
fails the bar outright.

**Shipped 2026-08-27:** `rag/llm.py`'s `OllamaBackend.generate()` now sends
`temperature=0.3` (was a fixed `0.2`, never tuned per model before this). **Stated
plainly, not softened: this is a modest mitigation, not a resolution.** The duplicate
rate moved from 10.5% to 9.1% on the same 24 sources -- two fewer duplicate items out of
143, not evidence the underlying tendency toward cross-session boilerplate reuse for
recurring anomaly shapes is gone. Three angles were logged as possible follow-ups; two
have now been attempted (larger sample, temperature). The third -- few-shot -- remains
correctly untried, deliberately: it's last-resort per instruction, given the
separately-confirmed risk (this session, different investigation) that few-shot examples
cause a related but different templating problem, near-verbatim reuse when a feature name
matches an example. If the residual ~9% rate needs to close further, few-shot is the next
and only remaining logged angle -- attempt it with the same caution already flagged, and
re-verify grounding at the same 100% standard before considering it for shipping.

Raw data: `rag/example_output/templating_rate_recheck_results.json` (angle 3, 16 new
trials), `rag/example_output/temperature_ab_results_t05.json` (24 trials @ 0.5, rejected),
`rag/example_output/temperature_ab_results_t03.json` (24 trials @ 0.3, shipped).

## Momentum tracking: biomechanics-feature join -- CLOSED, confirmed-negative (2026-08-27)

**Settled, not open.** No further action planned on this angle unless a future data
source closes the identifier gap described below.

Investigated whether momentum tracking (see status above -- score-context-only model sits
below the persistence baseline) could be extended with the 7 validated, percentile-ranked
biomechanics fields from the stroke-classification work, to test whether real shot-quality
signal closes that gap. Per instruction, checked join feasibility BEFORE any training,
and confirmed independently rather than trusting `dataset.py`'s own prior docstring note
that already flagged this as unsupported.

**Not feasible -- no training attempted.** The two data sources share no common
identifier of any kind:
- Momentum's source (`charting-m-points-*.csv`, Jeff Sackmann's Match Charting Project) is
  keyed by `match_id` and per-point sequence, against real, named, historical ATP matches
  (e.g. Gaudio/Coria, Djokovic/Cilic, Federer/Wawrinka).
- Biomechanics' source (`ir/example_output/biomechanics_full_dataset.csv`, 1980 rows) is
  keyed by anonymized `filename` (e.g. `p53_bvolley_s3.avi`) from the stroke-classification
  video corpus -- no `match_id`, no point index, no real player identity (players appear
  only as opaque IDs like `p3`/`p36`/`p53`), no timestamp that could line up with a
  charted match.
- Checked every StrokeEvent fixture directly for `match_id`/`point_id`/`player_id` fields
  that might bridge the two: all are either `None` or synthetic/demo values, not real
  match references.

This is the same conclusion `dataset.py`'s own docstring already stated (`stroke_type_onehot`
excluded because it "requires per-point video alignment with Model 2's stroke classifier,
which does not exist for this charted-match corpus") -- independently re-confirmed here
rather than assumed. **No proxy/approximation attempted per instruction ("don't force a
bad approximation just to have something to train on")** -- a fabricated join (e.g.
randomly assigning biomechanics rows to points) would not test the real hypothesis
(whether real shot-quality signal closes the gap) and was correctly not built. Momentum
tracking remains at its previously-reported state (6-feature score-context-only model,
TEST MAE=0.13520, still above the persistence baseline of 0.13048) -- no changes made.

---

## Tailscale Funnel -- real diagnosis, permanent-default startup script, shipped (2026-08-28)

Direct follow-up to the recurring 502s/RN-fetch scares logged 2026-08-07, 2026-08-08,
2026-08-21. Explicit instruction: diagnose for real before changing anything permanently,
don't assume fixed or still broken, don't silently re-break the app the way this was
reverted before.

### 1. The current 502, actually diagnosed

Before touching anything: backend wasn't running at all, and Funnel was still pointed at
port **8001** (leftover from the 2026-08-07 port-convention entry), while
`backend/app.py`'s documented default is port **8000**. `curl` against the Funnel URL
returned a clean `502` -- a tunnel proxying to nothing. **Not a Funnel-side failure.**
Fixed: started the backend on :8000, confirmed Ollama has `qwen3:14b` loaded, repointed
Funnel (`tailscale funnel --bg --yes 8000`) to match. Verified via `curl`: `401` on
`/docs` (auth-gated, reachable) and a real `200` on a live `POST /api/v1/auth/signup` --
full round trip through Funnel -> backend -> DB confirmed working via curl.

### 2. The older, unresolved 2026-08-21 "RN fetch fails, curl succeeds" scare -- re-tested for real, did NOT reproduce

That entry recorded real RN `fetch` Sign Up calls failing with `"Network request failed"`
against the Funnel URL on two separate devices in one session, while curl kept succeeding
throughout -- and it was explicitly never root-caused, just reverted to the LAN IP for
safety. Re-tested this properly this session with a real paired/connected Pixel 6a
(Android 17) rather than assuming either "fixed" or "still broken":

- **Chrome on the same physical device** loaded `https://visiotennis-trial.tail458008.ts.net/docs`
  cleanly -- real TLS handshake, real 401 response. Rules out a device-side cert-trust
  problem (the leading hypothesis going in: Funnel's TLS chain terminates at a self-signed
  `ISRG Root X2` with no cross-signed fallback, which some older Android trust stores
  reject even though curl's CA bundle accepts it -- confirmed NOT the cause here).
- **A genuinely stuck splash screen was hit mid-session** -- but root-caused via `adb
  logcat` to Metro's own dev-tooling WebSocket (`ws://192.168.1.3:8081/message`, `java.net.SocketException:
  Software caused connection abort`), completely unrelated to `BACKEND_URL`. A clean
  `am force-stop` + fresh relaunch (bypassing the stale dev-client state) resolved it.
  Documenting so this isn't mistaken for a Funnel problem again.
- **Controlled, on-device A/B, same build**: LAN IP loaded Home instantly. Funnel URL,
  freshly relaunched, ALSO loaded Home instantly -- same speed, no error.
- **Three distinct real RN fetch types, all succeeded over Funnel on the real device**:
  `GET /api/v1/auth/me` (validated a stored token), `POST /api/v1/auth/signup` (fresh
  throwaway account, `funnel_diag_realdevice@example.com` -- the exact endpoint that
  failed in 2026-08-21), `PATCH /api/v1/auth/profile` (coaching-profile save). All fast,
  no hangs, no errors.
- **Off-Wi-Fi leg**: could not be driven via `adb` myself -- wireless debugging runs over
  Wi-Fi, so the phone going off Wi-Fi drops the adb connection too (confirmed: device
  showed `offline` the moment Wi-Fi was disabled). User completed this leg directly:
  Settings screen (a real `GET /api/v1/auth/me` call) loaded correctly on cellular data,
  off home Wi-Fi.

**Conclusion: the 2026-08-21 failure did not reproduce under controlled conditions.** Most
likely explanation, not provable in hindsight: the same port-mismatch/backend-down class
of bug as the 502s above, hit during a session already flagged as having flaky
backend/emulator/Metro state, and never isolated with a controlled curl-vs-RN test at the
time. Reported honestly as "did not reproduce," not "proven impossible" -- if it recurs,
the next step is capturing `adb logcat`'s okhttp lines at the exact moment of failure
(zero such lines appeared during any of this session's successful requests, which itself
is informative: a real hang or TLS failure would leave a trace there).

**Real, unrelated mistake made and disclosed during this session, not swept in with the
Funnel findings:** a mistyped tap during on-device testing hit "Log Out" instead of the
Coach Name field, signing the real `[user-email]` account out on the test device.
Their password isn't known to this session -- flagged directly, account data itself
(sessions, coach profile) is untouched server-side, only the local session token cleared.

### 3. `mobile/lib/config.js` -- Funnel URL shipped as the permanent default

```js
export const BACKEND_URL = "https://visiotennis-trial.tail458008.ts.net";
```
LAN IP kept commented alongside as an explicit fallback (same-Wi-Fi-only, needs manual
refresh after DHCP re-leases -- the config-drift class of bug that has cost real time
across multiple sessions).

### 4. `start_visiotennis.ps1` -- one-shot startup routine, built and tested cold + warm

New file, project root. Four idempotent steps: confirm/start Ollama (checks `qwen3:14b`
is loaded, not just that Ollama responds), confirm/start the backend on :8000, confirm/fix
Funnel's target port, verify the public URL actually responds, then print it. Tested both
paths for real, not just written and assumed: warm (everything already up -- all 4 steps
report OK, no duplicate processes started) and cold (backend killed first -- script
detected it down, started it, confirmed reachable end-to-end). Run it each session instead
of remembering the manual sequence.

### 5. Cloudflare Tunnel fallback -- not needed, not built

Item 4 of the instruction ("if Funnel genuinely can't be made reliable, propose
Cloudflare Tunnel instead") was conditional on Funnel failing this real diagnosis. It
didn't -- every real test this session passed, including the off-Wi-Fi leg the user ran
directly. Not switching away from a working, already-integrated, zero-cost option on a
hypothetical; noting Cloudflare Tunnel (quick tunnel or a named tunnel with a stable
subdomain, already evaluated once before -- see 2026-08-04 entry -- and known to work for
plain HTTP where Funnel's TLS-only serving was the blocker back then) as the documented
next option if Funnel regresses again.

