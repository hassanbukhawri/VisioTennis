# MVP slice — design/scoping pass (not yet built)

**Status: design/scoping only, per explicit instruction — no backend or
mobile code written yet.** This document covers Steps 1-4 of the MVP
scoping request: what's deployable today, the minimal backend, the
minimal mobile client, and the real blockers before writing code.

## Purpose

Everything built so far (5 models, the IR layer, RAG session + stroke
coaching, the gated pipeline) is the ML core. This phase asks a different
question: can this be a real product, end to end, on a real device? The
target is the smallest possible slice that proves it — one phone screen,
one video, one path to one piece of grounded coaching feedback. No dual
tiers, no match analytics, no momentum tracking, no opponent scouting.

---

## Step 1: Inventory — what's deployable today

**Independently re-verified via `ir/verify_onnx_exports.py`, run just now
— not trusting each `training/*/export_onnx.py` script's own export-time
sanity check** (those ran once, at export time, possibly on a different
onnxruntime/onnx version; the files themselves could have drifted since).
Each model's input/output shape below was read directly from the `.onnx`
graph itself via `onnxruntime`, not assumed from training args.

| Component | File | Input shape (from graph) | Output shape (from graph) | Structural check | Real-data check |
|---|---|---|---|---|---|
| Model 1A — ball tracking (TrackNet) | `training/ball_tracking/checkpoints/tracknet.onnx` | `(batch, 9, 288, 512)` — 3 frames × 3 channels, **seq_len=3 and 512×288 are hard-coded, not dynamic** | `(batch, 3, 288, 512)` heatmaps | OK — loads, runs, no NaN/Inf | OK — real clip (game10/Clip10), heatmap peaks 0.93–0.95 confidence, vary sensibly per frame |
| Model 1B — court keypoints (CourtKeypointNet) | `training/court_keypoints/checkpoints/court_kp.onnx` | `(batch, 3, 224, 224)` — **224×224 hard-coded** | `(batch, 28)` — 14 points × (x,y) | OK | OK — real validation image, predicted points in plausible 0–1 normalized range |
| Model 2 — stroke classification (StrokeGRU) | `training/stroke_classification/checkpoints/stroke_gru.onnx` | `(batch, 60, 99)` — **seq_len=60 hard-coded**, only batch is dynamic | `(batch, 5)` logits | OK | OK — 18/20 (90%) correct on real THETIS test clips, consistent with the reported 0.834 macro-F1 |
| Model 1B — court detection/segmentation (YOLOv8-seg, `court_seg`) | `training/court_detection/runs/court_seg/weights/best.onnx` | `(1, 3, 640, 640)` | `(1,37,8400)` boxes/cls/mask-coeffs + `(1,32,160,160)` mask prototypes | OK | Not run (see finding below) |
| Model 1B — court detection/segmentation (YOLOv8-seg, `court_seg_grouped`, leakage-fixed split) | `training/court_detection/runs/court_seg_grouped/weights/best.onnx` | same as above | same as above | OK | Not run (see finding below) |

**All 5 ONNX files load and run cleanly via `onnxruntime` in isolation,
right now, on this machine.** No drift found.

**Real finding, not assumed: `court_detection` (the YOLO segmentation
model) is not wired into the actual IR pipeline at all.** Grepped the
whole repo — `ir/populate.py` and `ir/populate_e2e.py` only ever import
`training/court_keypoints/model.py` (`CourtKeypointNet`, direct keypoint
regression on a resized raw frame); nothing anywhere references
`court_detection`/`court_seg` outside its own training directory. It's a
real, trained, exportable model with its own README (documenting a
split-leakage fix), but it appears to be a parallel/exploratory Model 1B
approach that was never actually integrated — court position in every
real `StrokeEvent` built so far comes from `CourtKeypointNet` alone, no
segmentation step. **Flagging this rather than assuming either "it's
needed" or "it's dead code"** — Step 2 below treats it as NOT required
for the MVP slice pending confirmation.

**Not exported, and why (not just "missing"):**

- **Model 3 (momentum tracking)** — no trained weights exist anywhere
  (confirmed: no `training/momentum_tracking/checkpoints/` content). Known
  gap, logged since the IR layer's original design
  (`ir/DESIGN.md` "Not yet built"). Irrelevant to this MVP slice regardless
  (momentum was never part of the target output).
- **Model 4 (anomaly detection / IsolationForest)** — `training/anomaly_detection/checkpoints/`
  is empty; per this project's own design, this model is "fit at
  population time, not persisted" (session-level statistics, refit against
  whatever population exists at generation time — not a checkpoint you
  export once). More fundamentally: Model 4 operates on session-level
  AGGREGATE match statistics (`first_serve_pct`, `unforced_error_rate`,
  etc. — 12 features derived from many strokes/points across a session),
  which a single uploaded stroke video cannot produce at all. Out of scope
  for this MVP slice by definition, not merely "not exported yet."
- **MediaPipe Pose (BlazePose landmarker)** — not an ONNX export and never
  will be in this project; it's a separate bundled runtime
  (`training/stroke_classification/pose_landmarker_full.task`, loaded via
  the `mediapipe` Python package, already used successfully in
  `ir/populate_e2e.py` and the biomechanics extraction work). **Flagging
  explicitly: the MVP backend needs both `onnxruntime` AND `mediapipe` as
  inference runtimes, not just one.**
- **The coaching LLM (Qwen3 8B)** — not an export either; it's a separate
  local service (Ollama, HTTP API on `localhost:11434`), already the
  adopted choice for the RAG layer. **Flagging explicitly: the MVP backend
  needs Ollama running and reachable, a real deployment dependency, not
  just a Python import.**
- **Biomechanics feature extraction (`ir/biomechanics.py`) and percentile
  ranking (`ir/biomechanics_percentiles.py`)** — not models at all
  (deterministic geometry + a CSV-backed percentile lookup), nothing to
  export, just Python code to call directly.

---

## Step 2: Minimal backend proposal

**One FastAPI service, one primary endpoint.** Pipeline stages, in order:

1. Decode the uploaded video into frames (new code — see Step 4, gap #1;
   nothing in this repo currently reads a raw video container).
2. Normalize: correct rotation metadata, resample to a fixed FPS, resize
   per-model as needed (new code — Step 4, gap #2).
3. Extract a 60-frame MediaPipe Pose sequence over the (assumed
   pre-trimmed, see Step 4 gap #4) stroke window.
4. Run `StrokeGRU` (ONNX) → `predicted_class`, `confidence`,
   `class_probabilities`.
5. Run `ir/biomechanics.py` on the same pose sequence → the 10
   `BiomechanicsInfo` fields (deterministic, no model call).
6. Construct a real `StrokeEvent` (`ir/schema.py`) — schema-validated by
   construction, same principle as every other phase.
7. Look up percentiles for the 7 usable fields via a
   `BiomechanicsPercentileBenchmark` instance loaded ONCE at app startup
   (it's a static CSV-backed lookup — no reason to reload per request).
8. Build the stroke coaching prompt (`rag/stroke_prompt.py`) and run it
   through the retry-gate (`rag/gated_pipeline.py:generate_grounded_response`,
   `max_attempts=3`) against Qwen3 via Ollama.
9. Return `StrokeEvent` + coaching result as one JSON response.

**Ball tracking (TrackNet) and court keypoints are listed here as
OPTIONAL, not included in the default path — see the flagged scope
question below**, because the coaching prompt built in the previous phase
already never references `ball`/`court` (only `biomechanics` and
`stroke.predicted_class` are shown to the LLM or citable — confirmed by
re-reading `rag/stroke_prompt.py:_build_citable_view()`). Running them
would only be for richer display, not for anything the coaching output
actually depends on.

### Proposed endpoint shape

```
POST /api/v1/analyze-stroke
  Content-Type: multipart/form-data
  fields:
    video: file (required) -- short clip, ideally one stroke, see Step 4 gap #4
    player_id: string (optional)

  200 response (coaching succeeded):
  {
    "stroke_event": { ...StrokeEvent.model_dump(), including biomechanics... },
    "coaching": {
      "success": true,
      "items": [ {"text": ..., "type": ..., "citation": {...}} ],
      "n_attempts": 1,
      "total_latency_s": 41.5
    }
  }

  200 response (pipeline ran, coaching gate exhausted all retries):
  {
    "stroke_event": { ... },
    "coaching": { "success": false, "failure_reason": "..." }
  }

  422 response: video rejected (no pose detected in any frame, video
    unreadable/corrupt, etc.) -- an explicit failure state, never a
    fabricated/empty result

GET /api/v1/health
  -- confirms: ONNX sessions loaded, Ollama reachable, percentile
     benchmark loaded. The first thing to check when a real-device test
     doesn't work, before assuming the pipeline itself is broken.
```

### Blocking vs. async sequencing

**Proposed: fully synchronous, single request/response — no job queue,
no polling, for this MVP.** Rationale: this project's own gated-pipeline
trials already measured real latency —individual LLM+gate attempts ran
21–83s, with a 3-attempt worst case around 2–4 minutes. Building a
job-queue/polling system (submit → poll status → fetch result) would add
real infrastructure that isn't needed to prove the path exists, and
directly conflicts with the "no dual-tier architecture" instruction for
this phase. The CV/pose/biomechanics stages (2–7 above) are fast
(low seconds on CPU, per this project's own model-eval numbers); the LLM
gate (step 8) is the only slow part, and it's already bounded by
`max_attempts`.

**Consequence that must be handled, not silently assumed away:** the
mobile client's HTTP timeout must be set well above typical defaults
(propose 5 minutes), and the UI must show a "this can take a couple of
minutes" message rather than a generic spinner — this is a known,
accepted MVP tradeoff, not a bug to fix now.

---

## Step 3: Minimal mobile client proposal

**Recommend Expo (React Native), not a bare-native prototype.**
Justification: the fastest path to something running on a REAL phone,
today, is `expo start` + scanning a QR code with the Expo Go app — no
Xcode/Android Studio project setup, no code-signing, no native build
step for a first cut. Camera/video recording (`expo-camera` or
`expo-image-picker`) and multipart file upload are both well-documented,
common recipes. A bare-native (Swift/Kotlin) prototype would take
meaningfully longer to get a first build onto a real device with no
corresponding benefit at this stage (no App Store distribution is needed,
no native-module requirement exists yet). Flutter is a comparably fast
alternative but was not chosen since nothing else in this project uses
Dart, and RN/JS is closer to this project's existing tooling.

### Single-screen flow

1. Two entry actions: **Record Video** (`expo-camera`, capped at ~5–10s)
   and **Pick from Library** (`expo-image-picker`) — the second matters
   for development/testing (re-using the same clip repeatedly without
   re-recording).
2. On capture/pick: show a "Processing… this can take a couple of
   minutes" state, `POST` the file to `/api/v1/analyze-stroke` as
   multipart form data, with a long client-side timeout (5 min).
3. On success: display `stroke.predicted_class` + confidence, the
   coaching `items[].text` list, and (optionally) 1–2 headline
   biomechanics numbers with their percentile (e.g. "Contact height: 93rd
   percentile for serve").
4. On failure (`422`, `coaching.success == false`, network error, or
   timeout): a plain error message naming what happened — no polish
   needed, no retry-UI required for v0.

No navigation stack, no auth, no local persistence — literally the
minimum needed to prove "video in → coaching out" works over a real
network call from a real device.

---

## Step 4: Real blockers before writing any code

Ordered roughly by how fundamental each gap is, not by file/component.

### 1. No "find the stroke in a longer clip" step — confirmed real, matches the stated concern

Every model in this pipeline was trained on **pre-segmented** input:
`StrokeGRU` on THETIS's already-trimmed single-action clips; `TrackNet` on
already-trimmed rally windows. There is no stroke-boundary-detection model
anywhere in this project (already logged as "Not yet built" in
`ir/DESIGN.md` decision #2, well before this MVP phase). A real user's
self-shot phone video will have idle setup time before/after the actual
swing.

**Smallest fix, deliberately NOT a new ML component (per "not adding more
ML depth"):** push this onto the product/UX constraint instead of the
model — instruct the user to record a short clip of ONE stroke (e.g. "≤3
seconds, one swing"). Server-side, apply a purely mechanical (non-ML)
temporal resampling: whatever frame count the trimmed upload actually has,
uniformly resample to exactly the frame count each model requires (60 for
`StrokeGRU`, confirmed hard-coded in the ONNX graph — see Step 1). This is
real, necessary glue code (not yet written anywhere in this repo), but not
a new model. True automatic stroke-finding in an arbitrary-length video
remains explicit future work, not attempted here.

### 2. Video/frame preprocessing — no code in this repo reads a raw video file at all

Checked, not assumed: every population/demo script so far
(`ir/populate.py`, `ir/populate_e2e.py`, `ir/populate_biomechanics*.py`)
reads from **pre-extracted frame files** (`clip.frame_files[i]`) or
**pre-extracted numpy arrays** (`X_test.npy`). Nothing decodes an actual
`.mp4`/`.mov` container anywhere in this project yet. A real phone video
also brings real-world gotchas none of the existing code has ever had to
handle: portrait orientation (courts/strokes are typically framed
landscape; a portrait phone video has a very different aspect ratio),
rotation stored as container/EXIF metadata rather than baked into pixels,
and variable FPS. **This is new code, not a reuse of anything existing.**
Smallest fix: a single preprocessing module (OpenCV or ffmpeg-backed)
that decodes respecting rotation metadata, resamples to a fixed FPS, and
resizes to each model's exact required input size (confirmed via Step 1:
512×288 for `TrackNet`, 224×224 for `CourtKeypointNet`, both hard-coded in
the exported graphs — this preprocessing must hit those exact sizes, not
"roughly" match them).

### 3. Untested: does raw-frame MediaPipe Pose detection even work on a real self-shot clip?

`ir/populate_e2e.py`'s docstring already documents that MediaPipe Pose
failed completely (0/60 frames) on a raw WIDE BROADCAST frame, needing a
crude, explicitly-throwaway HSV red-shirt crop hack to work at all. A
self-shot phone video is plausibly framed much tighter (closer to
THETIS's tight single-player crops, which needed no such hack) — but this
is a real, untested assumption, not a verified fact. **Smallest fix:**
try raw-frame detection first (no crop hack — that hack is explicitly
documented as non-reusable, clip-specific scaffolding); if it fails,
fall back to a simple fixed center-crop, or surface a clear "couldn't
detect a person, try filming closer" error rather than silently producing
garbage. **This needs to be manually tested with an actual self-shot
phone clip before trusting it** — nothing in this project's existing
datasets (THETIS: pre-cropped; broadcast: wide, needed a hack) can answer
this question in advance.

### 4. Untested: does TrackNet generalize to phone-shot footage at all? (only matters if ball tracking is kept — see the scope question below)

`TrackNet` was trained exclusively on broadcast-style Ball Tracker+Bounce
footage (consistent camera angle, full court visible, small but
consistently-lit ball). A phone video shot by an amateur player has no
verified relationship to that distribution — different camera height/angle,
likely worse ball visibility (motion blur, sensor quality). This is an
out-of-distribution generalization risk that preprocessing cannot fix;
flagged as something to empirically check with 1–2 real clips, not
assumed to work.

### 5. Scope question this design pass surfaced, not just a blocker — does the MVP even need ball tracking / court detection?

Re-reading `rag/stroke_prompt.py`'s actual prompt construction: the
stroke-level coaching path built in the previous phase shows the LLM
**only** `event_id`, `player_side`, `stroke.predicted_class`/`confidence`,
and `biomechanics.*` (`_build_citable_view()`) — it never references
`ball` or `court` at all, and nothing in the KB or grounding rules allows
citing them. **The coaching output this MVP is meant to prove does not
actually depend on ball tracking or court detection.** Running them would
only add richer *display* data (e.g. a ball trajectory overlay), not
anything the "one piece of grounded coaching feedback" target needs.

Cutting them would also remove two of the real risks above (gap #4's
TrackNet generalization risk, and half of gap #2's preprocessing
surface). **This is a genuine scope fork the original task description
didn't resolve** (it named "ball tracking, court/pose extraction, stroke
classification" as if all were needed) — flagging it explicitly rather
than silently deciding either way.

### 6. Deployment dependency: Ollama must be running and reachable from the phone's network

The coaching LLM is a local Ollama service, not a library import. For a
real-device test, the FastAPI backend (and Ollama) need to run on some
reachable machine — a dev laptop on the same Wi-Fi as the test phone, or
a tunnel (e.g. ngrok) for a demo away from that network. Not code, but a
real setup step that will block a real-device test if not arranged.

### 7. Percentile comparability: the reference population and the new stroke must be measured the same way

`ir/biomechanics_percentiles.py`'s reference population was built from
THETIS clips — pre-cropped, consistent recording conditions, always
exactly 60 real frames (mostly). A percentile comparison is only
meaningful if the new stroke's biomechanics were extracted under
comparable conditions (same landmark scheme, same effective temporal
density after resampling). This doesn't block the pipeline from running,
but it's a real caveat on whether the output number MEANS anything for a
video shot under very different conditions than the reference population
— worth stating in the MVP's own output/disclaimer, not silently assumed
fine.

### 8. Latency: the gated pipeline's measured worst case is real, not hypothetical

Already measured in this project's own trials: individual attempts
21–83s, up to 3 attempts. A synchronous mobile request needs a long
client timeout and a UI that sets the right expectation — flagged in
Step 2/3 above, restated here because it's a genuine blocker to a
believable real-device demo if not handled (a client that times out at
30s will make a working backend look broken).

---

## Decisions (2026-07-19)

1. **Ball tracking / court detection: OUT of the MVP slice.** Confirmed —
   the backend pipeline is video → decode/normalize → MediaPipe Pose
   (60-frame sequence) → `StrokeGRU` → `ir/biomechanics.py` → percentile
   lookup → gated coaching generation → `StrokeEvent` + coaching JSON.
   `TrackNet` and `CourtKeypointNet` are NOT run; the response carries no
   ball trajectory or court position. This removes gap #4's untested
   TrackNet-on-phone-footage risk entirely and cuts half of gap #2's
   preprocessing surface (no 512×288 / 224×224 resize paths needed, only
   whatever MediaPipe Pose itself requires).
2. **Stroke pre-trimming: instruction-only for v0.** Confirmed — no
   motion-energy or ML-based auto-trim. The recording UI asks for "one
   swing, ≤3 seconds"; the server's only job is uniformly resampling
   whatever frame count the trimmed upload has down to exactly 60 frames
   (confirmed hard-coded in `StrokeGRU`'s ONNX graph, Step 1) before
   `StrokeGRU` and `ir/biomechanics.py` see it. Revisit only if this proves
   unreliable once tested against real self-shot clips.
3. **Synchronous request/response** — kept as proposed in Step 2, not
   revisited yet (no combined end-to-end latency measurement exists yet
   to revisit it against).

With ball/court cut, Step 4's gap list narrows to: #1 (pre-trimming,
resolved above as instruction-only), #2 (video decode/normalize — still
needed, but only for MediaPipe Pose's input, not three different
model-specific resolutions), #3 (MediaPipe Pose raw-frame reliability on
a real phone clip — now the SINGLE biggest untested risk in the whole
slice, since nothing else upstream of it remains), #6 (Ollama
reachability), #7 (percentile comparability caveat), and #8 (gate
latency/timeout handling). Gap #4 (TrackNet generalization) and half of
gap #2 (ball/court-specific resizing) no longer apply.

---

## Gap #3 stress test (2026-07-19): MediaPipe Pose on real, non-dataset footage — result: viable

**Before any backend/mobile code**, per explicit instruction, tested the
single biggest remaining untested risk directly.

### Footage used, and why

**No genuinely self-shot phone footage exists anywhere accessible to
this session** — checked first, not assumed: no video files anywhere in
this repo (confirmed earlier — every dataset here ships pre-extracted
frames/arrays) or in common user directories (Downloads, Desktop,
Pictures, Videos) on this machine. One video file was found elsewhere on
the machine (`C:\Users\PC\tennis_analysis\...\input_video.mp4`), but
inspection showed it's professional ATP broadcast footage (TV camera,
scoreboard overlay, full crowd) belonging to an unrelated separate
project — the same wide-broadcast condition already known to fail
(`ir/populate_e2e.py`'s red-shirt hack), so using it would only
re-confirm something already documented, not test the actual open
question. Per user direction, fetched 3 free/no-attribution stock clips
instead as real-footage stand-ins — genuinely amateur/practice footage,
not curated dataset clips or broadcast TV, each **explicitly flagged as a
stand-in, not literally self-shot**:

| Clip | Source | Framing | Lighting |
|---|---|---|---|
| `clip1_serve_practice.mp4` | [Pexels](https://www.pexels.com/video/a-tennis-player-practicing-his-serve-4902139/) | solo, close/full-body | bright sun, outdoor clay |
| `clip2_teen_practice.mp4` | [Pexels](https://www.pexels.com/video/a-teenage-girl-practising-tennis-11154706/) | solo, medium | overcast/hazy, outdoor hard court |
| `clip3_group_distant.mp4` | [Pexels](https://www.pexels.com/video/a-group-of-children-practising-tennis-with-their-teachers-11154490/) | multi-person, distant/wide | hazy, real amateur group lesson |

### Method

`mvp/stress_test_pose.py`: 90 real consecutive frames per clip, MediaPipe
Pose run on the **raw frame directly — no crop, no red-shirt-style
workaround**. For each clip: (1) per-frame detection rate; (2) a subject
CONTINUITY check (frame-to-frame hip-center displacement) — detecting *a*
pose every frame isn't the same as tracking *one* pose the whole window,
since `num_poses=1` picks whichever detection is most confident
independently each frame, with no re-identification; (3) the real,
already-validated `ir/biomechanics.py:extract_biomechanics_features()`
run on the resulting sequence, joint angles spot-checked for anatomical
plausibility (same fields checked in the earlier biomechanics work); (4)
one visual spot-check — landmarks drawn on a real frame
(`mvp/real_footage_test/clip1_pose_spotcheck.png`).

### Result

| Clip | Detection rate | Subject continuity | Joint angles |
|---|---|---|---|
| clip1 (solo, close, bright) | **100.0%** (90/90) | Clean — 0/89 large jumps | All in valid range (knee 172°/179°, elbow 146°, X-factor 28°) |
| clip2 (solo, medium, hazy) | **100.0%** (90/90) | Clean — 0/89 large jumps | All in valid range (knee 161°/169°, elbow 57°, X-factor 28°) |
| clip3 (multi-person, distant, hazy) | **100.0%** (90/90) | **6/89 large jumps — likely identity switches** | All in valid range, but see caveat below |

**The red-shirt-tracker problem does NOT recur on any of these clips —
raw-frame detection worked at 100% with zero crop/workaround, on all
three, including the hazy/overcast one.** This directly answers the
open question from Step 4 gap #3: a genuine broadcast-style wide shot
needed a hack; none of these closer, more deliberate framings did. The
visual spot-check (`clip1_pose_spotcheck.png`) confirms the landmarks are
not just "present" but accurately tracking real joints — shoulders,
elbows, wrists, hips, knees, and ankles all visibly align to the real
body in the frame.

**One real, separate finding, not glossed over: MediaPipe's `num_poses=1`
has no subject-tracking/re-identification, and a multi-person scene can
silently switch which person it follows mid-window while still reporting
"a pose detected" on every frame.** Checked, not assumed: solo clips
1 and 2 show a smooth, tiny frame-to-frame hip-center displacement (max
0.011–0.017, normalized coords) with zero large jumps; clip 3's group
scene shows 6 jumps past 0.15 (up to 0.48) out of 89 transitions —
consistent with the tracked "person" changing partway through. This
would corrupt any feature relying on temporal continuity (weight
transfer, follow-through, the contact-frame speed proxy) even though
per-frame detection looked perfect. **This risk is specific to
multi-person scenes** — the actual MVP use case (one user recording
themselves) is much less likely to trigger it, but it's a real,
demonstrated failure mode worth a product-level mitigation (e.g.
recording guidance: "make sure you're the only person clearly in frame")
rather than being silently assumed away.

### Conclusion: viable as designed — proceed to backend/mobile scaffolding

Pose extraction holds up on real, non-curated footage without the
broadcast-specific workaround. The re-scoped MVP pipeline (video → decode
→ MediaPipe Pose → `StrokeGRU` → `ir/biomechanics.py` → percentiles →
coaching) does not need a new crop/detection step for the single-user
case this MVP targets. The identity-switching finding is real but scoped
to multi-person framing, not a blocker for the primary use case — carried
forward as a product-guidance note (ask users to keep the frame clear of
other people), not a new engineering requirement.

---

## Steps 2/3 implemented (2026-07-19)

**Backend** (`backend/app.py`) built exactly to the Step 2 shape:
`POST /api/v1/analyze-stroke`, `GET /api/v1/health`, orchestration-only,
reusing `ir.biomechanics`, `ir.biomechanics_percentiles`,
`rag.stroke_prompt`, `rag.gated_pipeline`, `rag.llm` unchanged, plus the
verified StrokeGRU ONNX export run via `onnxruntime`. Tested locally
first (per instruction) against one of the pose-stress-test Pexels clips
— real 200 OK, real grounded coaching output, ~39s. Two real structural
fixes required first, both logged in `PROJECT_LOG.md`: `StrokeEvent.ball`/
`.court` had to become Optional (the ball/court-cut decision had a schema
consequence not caught at the time), and a pydantic 2.5.3→2.13.4
transitive upgrade (from installing `fastapi`) broke a `float("nan")`
sentinel pattern, fixed by making `ModelProvenance.metric_value` properly
Optional instead of pinning around the version difference.

**Mobile** (`mobile/`, Expo SDK 57) built exactly to the Step 3 shape:
single screen, record/pick a video (`expo-image-picker`), upload,
display predicted class + coaching text + percentile citations +
diagnostics. Checked the current SDK 57 API surface before writing code
(`MediaTypeOptions` is deprecated in this version) rather than assuming a
remembered older API. One real bug caught before the device test: a
manually-set `Content-Type: multipart/form-data` header on the upload
`fetch` call overrides `FormData`'s auto-generated boundary parameter and
silently breaks server-side parsing — removed.

**Both services running, backend LAN IP and Metro both confirmed
reachable over Wi-Fi (`192.168.1.9`).** Windows Firewall rules could not
be queried/created from this session (needs admin rights not available
here) — flagged for the user to check if a real device can't reach
either port. The true end-to-end device test needs a real phone, which
this session doesn't have — handed off, result to be reported once run.

**Update — mobile SDK-mismatch bundler error (2026-07-19/20):** user
downgraded `expo` to `54` by hand to fix a connection issue but left
`expo-image-picker`/`expo-status-bar`/`react`/`react-native` on their
prior SDK-57-era versions, causing a codegen `TransformError` on
`VirtualViewExperimentalNativeComponent.js`. Confirmed via direct
`package.json` inspection, not guessed. Fixed with `npx expo install
--fix` alone (resolved to `react-native 0.81.5`, `react 19.1.0`,
`expo-image-picker ~17.0.11`, `expo-status-bar ~3.0.9`); a clean
`node_modules` reinstall was not needed (verified no duplicate/
conflicting versions remained). Verified beyond "the bundler starts":
fetched the compiled bundle directly, confirmed 701 modules with no
errors, and manually confirmed the specific previously-broken
`VirtualView` component's compiled output was now well-formed.

---

## Failure-mode hardening pass (2026-07-20)

Per explicit instruction: the true end-to-end path now works on real
hardware (user confirmed the app loads and completes a round trip on
their phone after the SDK-mismatch fix above). Before adding anything
new, systematically tested and hardened the pipeline against realistic
failure modes — each one tested for real against the running backend
(and, where a physical device isn't required, the mobile code path),
not just reasoned about.

### 1. Low/degraded pose detection

**Before this pass:** a floor existed (reject if `n_detected < 30/60`,
i.e. below 50%) but it was never validated against a real degraded
clip — it was a guessed round number.

**Test:** took the real `clip1_serve_practice.mp4` (100% detection
clean) and produced graduated synthetic degradations (darkening +
Gaussian blur at several intensities) to build an actual detection-rate
curve, rather than guessing where MediaPipe would start failing:

| Degradation | Detection rate | Backend response (old 50% floor) |
|---|---|---|
| Clean (baseline) | 100.0% | 200, normal |
| Mild dark+blur | 98.3% | 200, normal |
| **Moderate dark+blur** | **63.3%** | **200, normal — 92.5% stroke-classification confidence, full coaching text, indistinguishable from a clean result** |
| Heavy dark+blur | 1.7% (1/60) | 422 (barely — right at the old floor) |
| Severe dark+blur | 0.0% | 422 |
| 8x downscale+noise (distant/low-res proxy) | 100.0% | 200, normal (MediaPipe proved robust to this specific degradation) |

**Finding, confirmed not assumed:** the moderate-degradation case is
exactly the failure this task asked about — over a third of frames had
no real pose data, yet the response was presented with the same format
and even *higher* apparent confidence (92.5%) than some clean runs,
with the only signal being a `pose_detection_rate` diagnostic field
easy to overlook at the bottom of the mobile screen.

**Fix:** raised `MIN_POSE_DETECTION_RATE` from 50% to **80%** in
`backend/app.py` — clearly above the demonstrated-bad 63.3% case, at
the low end of where real clean footage in this pipeline actually sits
(98–100%, per `mvp/stress_test_pose.py` and this test). Below the
floor, the API now returns 422 with an explicit message including the
actual detection rate achieved: *"couldn't get a reliable read on this
video — a clear pose was only detected in 38/60 frames (63%, need at
least 80%); try better lighting, a closer/clearer view of the player,
and make sure only one person is in frame."*

**Verified:** re-ran the moderate-degradation clip post-fix → 422 with
the message above. Re-ran clean `clip1`/`clip2`/a trimmed `clip3` → all
still 200, unaffected.

### 2. No person / wrong content in frame

**Test:** a synthetic no-person video (blank frames, no human) and the
real multi-person `clip3_group_distant.mp4`.

| Case | Result |
|---|---|
| No person (blank frames) | 422, clear message ("no pose detected in any frame...") — **already correct behavior, confirmed not assumed** |
| Multi-person (clip3, trimmed to fit the new frame cap) | 200, 100% detection, plausible-looking output — no crash/hang, but the earlier stress-test finding stands: MediaPipe's `num_poses=1` has no re-identification, so a multi-person scene can silently track different people frame-to-frame. **Known limitation, not fixed this pass** — flagged as a product-guidance item (ask the user to keep the frame clear of other people) rather than a new engineering requirement, consistent with the original stress-test conclusion. |
| Person present but not clearly mid-stroke (the untrimmed `clip1` case, already seen twice in this project) | Confidently classifies anyway — there is no "not a real stroke" rejection class. **Known, inherent limitation of a fixed 5-class softmax classifier, not fixed this pass** — would need genuine out-of-distribution detection, which is new ML scope, not a hardening bug fix. |

### 3. Bad upload / malformed input

**Before this pass:** no file-size limit, no minimum/maximum
duration/frame-count limit. A long, high-res upload would be decoded
entirely into memory before any check could reject it.

**Tests, all against the real running backend:**

| Input | Result |
|---|---|
| Non-video file (random bytes, `.mp4` extension) | 422, clear message — already correct |
| Non-video file (plain text, `.mp4` extension) | 422, clear message — already correct |
| Corrupted/truncated real video (real `.mp4` cut to 30%) | OpenCV tolerated it and decoded 79 real frames — processed normally, no crash. Not a bug; flagged as "handled either way," not something to force-reject. |
| Extremely short (2-frame synthetic, and a real 4-frame clip) | **Before fix:** passed through to pose extraction and failed there with a slightly misleading message. **After fix:** rejected at decode time with a specific, accurate message ("video is too short (4 frame(s), need at least 5) — record at least about a second"). |
| Extremely long (5400-frame/3-minute synthetic, and real un-trimmed `clip3`) | **Before fix:** no limit — decoded the entire file into memory regardless of length (measured: 5400 low-res frames alone → ~1GB of raw arrays; a real 1080p phone video of the same length would be over an order of magnitude larger — a real resource-exhaustion risk). **After fix:** decode loop now bails out mid-read, not after, as soon as it exceeds 900 raw frames (~30s) — rejected in under 1 second regardless of the file's actual length, with a clear message. |
| Oversized upload (110MB dummy file) | **Before fix:** no limit — would be fully buffered into memory. **After fix:** rejected with 413 in ~0.6s (aborted mid-stream, not after buffering). |

**Fix:** added `MAX_UPLOAD_BYTES` (100MB, checked while streaming the
upload in chunks), `MIN_RAW_FRAMES`/`MAX_RAW_FRAMES` (5–900 frames,
enforced *during* the decode read loop so a huge file is rejected
early, not after being fully read) to `backend/app.py`.

### 4. Backend/network failure paths on the mobile side

**Before this pass:** the mobile `fetch` call had a single generic
`catch` that displayed `err.message` verbatim — for a real network
failure this would have shown React Native's raw `"Network request
failed"` `TypeError` message, not something a real user could act on.

**Investigated concretely** (via Node's fetch against an unreachable
port, and against a real killed-backend scenario) what each failure
actually looks like: a connection-level failure throws `TypeError`
(RN's fetch message: `"Network request failed"`); the 5-minute
`AbortController` timeout throws an error with `name === "AbortError"`;
a non-2xx HTTP response is already handled separately and now carries
the backend's own (much improved, per items 1–3 and 5) `detail` text.

**Fix (`mobile/App.js`):** the catch block now distinguishes these
three cases and shows a specific, actionable message for each —
network-unreachable ("Couldn't reach the server. Make sure your phone
and the backend computer are on the same Wi-Fi network, and that the
backend is running."), timeout ("This is taking too long and was
cancelled..."), and HTTP-error (the backend's own message, now always
present since item 5's fix means the API essentially never 500s on a
known failure path). Also added a "Try Again" button on the error
screen.

**Verified:** the bundle was rebuilt and fetched directly after this
change (701 modules, 0 errors, all three new message strings present
in the compiled output) — confirms the JS is valid and the logic is in
the bundle actually served to the phone. **Not verified: how this
actually renders on the physical device** — this session has no phone;
that step still requires the user.

**Also found and fixed in this step, unrelated to the 5 listed items
but blocking on-device testing entirely:** the backend machine's LAN IP
had changed since the last session (DHCP re-lease, `192.168.1.9` →
`192.168.1.8`) — `BACKEND_URL` in `mobile/App.js` was stale and would
have made the app fail to connect at all, independent of any of this
pass's fixes. Updated and reconfirmed both services reachable at the
new IP.

### 5. The Ollama/gated-pipeline failure path

**Two distinct failure paths exist here, and only one was previously
tested:**

- **Retry-gate exhaustion** (the LLM responds, but never produces a
  fully grounded/well-formed answer within 3 attempts) — already
  returns an explicit `GatedResult(success=False, failure_reason=...)`
  (`rag/gated_pipeline.py`), confirmed to fire in practice during the
  earlier RAG validation work, and the backend already surfaced this
  correctly as `coaching.success: false` + `coaching.failure_reason`.
  Mobile's UI already had a branch for this (`result.coaching.success
  ? ... : <error text>`) — reviewed, confirmed correct, no change
  needed.

- **Ollama unreachable entirely** (the LLM never responds at all) —
  **not previously tested, and a real bug.** Tested for real: pointed
  the backend at an invalid Ollama host (via a new `OLLAMA_HOST` env
  var, added specifically so this could be tested without touching the
  actually-running Ollama service) and sent a real clip through
  `/api/v1/analyze-stroke`. **Confirmed before any fix:** the request
  ran stroke classification and biomechanics successfully, then threw
  an uncaught `urllib.error.URLError` from deep inside
  `generate_grounded_response` → `OllamaBackend.generate`, which
  FastAPI turned into a bare `500 Internal Server Error` with a
  **plain-text, non-JSON body** — silently discarding the
  already-computed, already-correct stroke classification and
  biomechanics, and giving the mobile client nothing usable to parse
  or display (its `body.json().catch(() => ({}))` fallback would have
  shown a generic "Server returned 500").

**Fix:** wrapped the coaching-generation call in `backend/app.py` in a
`try/except Exception`, converting any failure (connection error, HTTP
error, anything else) into the *same* `coaching.success: false /
failure_reason` shape the retry-gate's own exhaustion path already
produces — so the mobile client only ever has to handle one failure
shape, and a coaching failure never discards the stroke
classification/biomechanics that already succeeded.

**Verified:** re-ran the same Ollama-unreachable test post-fix → `200
OK`, full `stroke_event` (classification + biomechanics both present
and correct), `coaching.success: false`, `coaching.failure_reason:
"coaching feedback service unavailable (URLError: ...)"`. This is
structurally identical to the retry-gate's own failure shape, so
mobile's existing UI branch renders it correctly with no mobile-side
change needed for this part.

### Summary table

| # | Failure mode | Before this pass | Fix | Status |
|---|---|---|---|---|
| 1 | Low/degraded pose detection | 50% floor, never validated; a real 63.3%-detection clip passed as a normal, high-confidence result | Raised floor to 80% (evidence-based, see curve above); clear rejection message with actual rate | **Fixed** |
| 2a | No person in frame | Already correctly rejected (422) | none needed | **Acceptable (confirmed, not assumed)** |
| 2b | Multiple people in frame | No crash, but no subject re-identification — can silently track the wrong person | none this pass (product-guidance note, not an engineering fix) | **Known limitation** |
| 2c | Person present, not mid-stroke | Confidently misclassifies (no OOD/rejection class) | none this pass (new ML scope, not a bug fix) | **Known limitation** |
| 3a | Non-video / corrupted upload | Already correctly rejected (422) | none needed | **Acceptable (confirmed, not assumed)** |
| 3b | Extremely short clip | Reached pose extraction, misleading error | Rejected at decode time with accurate message | **Fixed** |
| 3c | Extremely long clip / oversized upload | No limit — full file decoded/buffered into memory regardless of size (real resource-exhaustion risk) | 900-frame cap (bails mid-decode) + 100MB upload cap (bails mid-stream) | **Fixed** |
| 4 | Mobile network/backend failure UX | Generic catch, raw RN error text shown to user | Distinguishes network/timeout/HTTP-error, shows specific actionable message + Try Again button | **Fixed (verified via bundle compile; on-device rendering not verifiable from this session)** |
| 5 | Ollama unreachable | Unhandled exception → bare non-JSON 500, discards already-successful classification | Caught, converted to the same explicit failure shape the retry-gate already uses | **Fixed** |
| 5b | Retry-gate exhaustion | Already handled correctly end-to-end (backend + mobile) | none needed | **Acceptable (previously confirmed, re-verified by code review this pass)** |
| — | Stale backend LAN IP in mobile config | Would have blocked all on-device testing regardless of the above | Found and updated (`192.168.1.9` → `192.168.1.8`) | **Fixed** |

**What still requires the user's physical phone, honestly stated:**
whether the mobile-side error messaging (item 4) actually renders as
intended on-device, and a fresh true end-to-end run now that the LAN
IP has been corrected. Both services are up and reachable at the
current IP for that test.

---

## Phase B scoping (2026-07-28): multi-screen navigation + session history

**Implemented 2026-07-28** — see the "Implemented" note at the end of this
section and `PROJECT_LOG.md`'s matching entry for what was actually built,
one real adjustment from the proposal below, and how it was verified. The
proposal itself is left as originally written (this paragraph is the only
edit) since it's still the accurate record of the reasoning behind the
choices made.

Today's `mobile/App.js` is a single component with no navigation stack at
all — `status` (`idle | uploading | done | error`) just swaps which JSX
block renders in place. Phase B moves this to real screens: Home/Capture,
Processing, Results, History, Settings. This section proposes the
navigation approach and screen structure and where session history should
live, per instruction — **no screens are written yet**, this is scoping
only, to be confirmed before implementation starts.

### Navigation approach: Expo Router, not React Navigation

The installed SDK is Expo 54 (`mobile/package.json`), but
`mobile/AGENTS.md` flags that Expo has changed meaningfully since and
directs to the current versioned docs before writing any code — so this
decision was checked against the actual current docs
(`docs.expo.dev/versions/v57.0.0/` and the Expo Router docs), not assumed
from general React Native familiarity. Two things confirmed there that
matter for this decision and would have been wrong to assume from older
knowledge: **Expo Router is now built directly on `react-native-screens`,
not on top of React Navigation** (an actual architecture change from
earlier versions), and the file-based root has moved to `src/app/`
(`_layout.tsx` inside it), not a bare top-level `app/`.

**Recommend Expo Router.** Real tradeoffs, not just "it's the default":

- **For:** it's the docs' explicit recommendation for new apps ("if you
  are building a new app, we recommend using Expo Router"); file-based
  routing means the file tree mirrors the screen list directly (less
  boilerplate than hand-registering a navigator); typed routes catch
  navigating to a nonexistent screen at compile time, not at runtime on a
  tester's phone; and — the deciding factor here — **today's app has zero
  existing navigation investment**, so there's no migration cost to weigh
  against React Navigation's maturity. This is a green-field choice, not a
  migration.
- **Against:** file-based routing is a different mental model than
  imperative navigator setup, so there's a small learning-curve cost if
  whoever builds the screens hasn't used it before; and its conventions
  have genuinely changed across recent SDK versions (see above), so
  whatever gets built needs to track the *current* docs precisely, and a
  future SDK bump could shift conventions again — a real, if modest,
  ongoing-maintenance surface that a more stable/older API wouldn't carry.
- **React Navigation was not chosen** because neither of its real
  advantages (preserving existing navigation code; more accumulated
  third-party recipes) applies here — there's nothing to preserve, and
  Expo Router's own docs and examples are sufficient for a screen set this
  small.

### Proposed screen structure

```
src/app/
  _layout.tsx              -- root layout: init, wraps app in the result/session
                               state provider (see "passing data" below)
  (tabs)/
    _layout.tsx             -- tab bar: Home, History, Settings
    index.tsx                -- Home/Capture: record or pick a clip, "Analyze" button
    history/
      index.tsx               -- History: list of past sessions (see persistence below)
      [sessionId].tsx          -- past-session detail, reusing the Results screen's
                                  rendering for a historical (not just-computed) result
    settings.tsx              -- Settings (trial passcode/backend URL if ever made
                                  user-configurable; currently hardcoded, no urgent need)
  processing.tsx             -- Processing/loading screen -- NOT a tab, pushed on top
                                 of the stack after "Analyze" is tapped
  results.tsx                -- Results screen -- NOT a tab, replaces Processing once
                                 the backend responds
```

**Flow:** Home (tab) → tap Analyze → `router.push('/processing')` → on
success `router.replace('/results')` (replace, not push, so the back
stack from Results is `[Home, Results]` — pressing back from Results goes
straight to Home, not to a stuck Processing screen) → on failure, stay on
Processing and show the existing inline error + Try Again pattern from
`App.js` today, rather than a separate Error screen (no new state machine
needed, just relocating the existing one). History tab → tap a past
session → pushes `history/[sessionId]`, which renders with the same
Results-rendering component fed historical data instead of a live result.

**Passing data between screens:** per the current docs, Expo Router's URL
params are meant for small serializable values (an id, a short string) —
not the full analysis JSON (stroke + coaching + diagnostics) this app
needs to carry from Processing to Results. Proposed approach: a small
React Context (e.g. `SessionContext`) set by Processing when the backend
responds and read by Results, rather than pushing the whole payload through
`useLocalSearchParams()`. This keeps the router itself simple and avoids
adding a full state-management library (Redux/Zustand) for what's really
just "hand off one object to the next screen."

### Where session history should live: on-device only (proposed), not backend-persisted

| | On-device only | Backend-persisted |
|---|---|---|
| Setup cost | Low — a local store (`expo-sqlite` recommended over a raw AsyncStorage blob once there's a growing list to query/sort by date; final library choice left to implementation) | Real new backend work: today's backend is fully stateless (nothing is stored anywhere, every request computed fresh) — would need a DB, a real session/player identity model (today's `player_id` is a hardcoded `"mobile_test_user"` string), and new list/fetch endpoints |
| Privacy/scope | No additional data leaves the device beyond what's already sent for a single analysis | Testers' analysis history would be stored server-side, retained indefinitely by default unless a retention policy is built |
| Cross-device/central visibility | None — a reinstall or device swap loses history; whoever runs the trial can't review testers' sessions without asking them | Testers keep history across devices; trial runner gets central visibility into usage/results, which can be genuinely useful for a trial |
| Dependency on the deferred consent/data-handling decision | **None** — storing data only on the tester's own device doesn't raise the same retention/access questions | **Direct** — persisting testers' data server-side is exactly the kind of thing the consent/data-handling flow (explicitly deferred, next after this) needs to define first (retention period, who can see it, deletion) |

**Recommend on-device only for Phase B.** The deciding factor is the last
row: backend persistence isn't just more work, it's work that depends on a
decision (consent/data-handling) that's been deliberately deferred —
building it now would mean either building it twice or quietly making
retention/access decisions ahead of that conversation. On-device keeps
Phase B fully unblocked and independent of that pending decision. To keep
a later move to backend-persisted history cheap if the trial's needs
justify it, propose writing the History screen against a small local
"history repository" module (a thin read/write interface) rather than
calling `expo-sqlite` directly from the screen — so swapping the
implementation later is a contained change, not a screen rewrite.

**Not yet decided / explicitly out of scope for this pass:** the concrete
on-device storage library (`expo-sqlite` proposed, not committed), exact
Settings screen contents, and any screens themselves — this is
architecture/approach only, per instruction, before writing code.

### Implemented (2026-07-28)

Built essentially as proposed, with one concrete adjustment: routes live at
plain `app/` (project root), not `src/app/` — the Expo Router installation
docs clarify `src/app/` is only for projects that opt into a `src/`
directory, and this project has never used one, so introducing it here
would've been an unrelated structural change with no benefit. Everything
else matches the proposal: `expo-router` (`Stack` + `Tabs`), `expo-sqlite`
behind `lib/historyRepository.js` as the swappable seam, `SessionContext`
(not router params) for handing the full result from Processing to Results,
and the same push-then-replace navigation flow. Full build log, two real
install snags hit and fixed, and a genuine Metro bundling gotcha (stale root
`index.js` silently shadowing the new `expo-router/entry` main field) are in
`PROJECT_LOG.md`'s 2026-07-28 Phase B entry. Verified via Metro bundle-compile
checks only (all 7 screens' real content confirmed present, old app's content
fully gone, Android + iOS bundles both clean) — no physical device or
emulator was available this session, confirmed with the user before
proceeding on that basis; on-screen rendering and `expo-sqlite`'s runtime
behavior remain unverified pending a real device/simulator pass.
