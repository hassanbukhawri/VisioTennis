"""
MVP backend: a single FastAPI service implementing the pipeline scoped in
mvp/DESIGN.md Step 2 -- video upload -> pose -> StrokeGRU -> biomechanics
-> percentiles -> gated coaching generation -> IR + coaching JSON.

Deliberately orchestration only -- every actual inference/logic step reuses
existing, already-validated code:
  - ir.biomechanics.extract_biomechanics_features  (real, validated on 1980 clips)
  - ir.biomechanics_percentiles.BiomechanicsPercentileBenchmark  (real, checked population)
  - rag.stroke_prompt.build_stroke_prompt / select_notable_biomechanics_features
  - rag.gated_pipeline.generate_grounded_response  (the retry-gate)
  - rag.llm.OllamaBackend
  - the StrokeGRU ONNX export (training/stroke_classification/checkpoints/stroke_gru.onnx),
    independently verified in ir/verify_onnx_exports.py, used here via onnxruntime
    rather than re-loading the training-time torch checkpoint.
  - MediaPipe Pose, run RAW (no crop) -- validated on real non-dataset
    footage in mvp/stress_test_pose.py.

Per the MVP scope decision (mvp/DESIGN.md): ball tracking (TrackNet) and
court detection (CourtKeypointNet) are NOT run. StrokeEvent.ball/.court are
now Optional (ir/schema.py, changed 2026-07-19 for this reason) and left
None here.

Run (from the repo root):
    uvicorn backend.app:app --host 0.0.0.0 --port 8000

--host 0.0.0.0 is required, not optional, for a phone on the same Wi-Fi
network to reach this service via the PC's LAN IP (e.g. http://192.168.x.x:8000)
-- binding to the default 127.0.0.1 only accepts connections from the same
machine. Windows Firewall may also need an inbound rule for this port; see
mvp/DESIGN.md's networking note before assuming a connection failure is a
code bug.
"""
import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from typing import Literal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from ir.schema import StrokeEvent, StrokeInfo, BiomechanicsInfo, ModelProvenance, FrameRange, STROKE_CLASSES
from ir.biomechanics import extract_biomechanics_features, _valid_frame_mask
from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark
from ir.tennis_classifier import classify_frames
from ir.session_digest import DEFAULT_LOOKBACK_K, compute_session_digest
from rag.stroke_prompt import build_stroke_prompt, select_notable_biomechanics_features
from rag.gated_pipeline import generate_grounded_response
from rag.llm import OllamaBackend, OpenAICompatibleBackend
from rag.chat_pipeline import handle_chat_turn
from rag.chat_router import ExistingCitation
from rag.schema import CoachingResponse

from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.db import AuthToken, StrokeHistory, User, get_db_session, init_db, utcnow
from backend.auth import (
    AuthResponse, CoachNameUpdateRequest, LoginRequest, ProfileUpdateRequest, SignupRequest, UserOut,
    get_current_token, hash_password, issue_token, user_to_out, user_tone_profile, verify_password,
)

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/qwen3-8b"

POSE_TASK_PATH = os.path.join(REPO_ROOT, "training/stroke_classification/pose_landmarker_full.task")
# THETIS-augmentation checkpoint (2026-08-22) -- REPLACES the clean-split checkpoint
# (2026-08-21 swap) as the deployed model. Exported from
# checkpoints/augmented/stroke_gru_augmented_best.pt via export_onnx.py, same
# ONNX-Runtime-inference sanity check as every prior export. This is the FINAL
# checkpoint chosen from the serve-investigation (PROJECT_LOG.md, 2026-08-21/22
# entries, and training/stroke_classification/README.md's "Serve investigation"
# section for the full writeup): the only one of 5 attempted interventions that
# measurably improved real-footage accuracy (5/14 -> 8/14 on the hand-labeled
# real-footage set), driven by backhand/forehand -- serve remains 0/3 across
# every checkpoint tried, an accepted, documented limitation, not something this
# swap claims to fix. Clean-split checkpoint left on disk at
# checkpoints/clean_split/ for rollback, untouched.
STROKE_GRU_ONNX_PATH = os.path.join(
    REPO_ROOT, "training/stroke_classification/checkpoints/augmented/stroke_gru_augmented.onnx")
# Post-hoc temperature scaling (2026-08-21) -- a single scalar fit on this exact
# deployed checkpoint's own held-out val split via training/stroke_classification/
# calibrate_temperature.py (Guo et al. 2017 method), softening raw softmax
# confidence toward its true accuracy. Argmax-preserving -- does not change
# which class wins, only how honestly the winning-class probability is
# reported. Falls back to T=1.0 (no-op) if the file is missing, so a fresh
# checkout without the calibration artifact still runs, just uncalibrated --
# not a hard startup dependency. Refit per-checkpoint every time the deployed
# model changes -- confirmed twice now (2026-08-21 clean-split swap, 2026-08-22
# augmented swap) that a T value fit on one checkpoint does NOT transfer to another.
STROKE_TEMPERATURE_PATH = os.path.join(REPO_ROOT, "training/stroke_classification/checkpoints/temperature.json")
SEQ_LEN = 60  # hard-coded in the StrokeGRU ONNX graph -- confirmed via ir/verify_onnx_exports.py

# Phase 1 instrumentation (2026-08-01). Not committed -- see .gitignore.
SESSION_LOG_PATH = os.path.join(REPO_ROOT, "backend", "logs", "sessions.jsonl")
FEEDBACK_LOG_PATH = os.path.join(REPO_ROOT, "backend", "logs", "feedback.jsonl")

# Hardening limits (2026-07-20 failure-mode pass -- see PROJECT_LOG.md).
# All of these were picked from real measurements against this pipeline, not
# guessed round numbers -- see the log entry for the test data behind each.
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100MB -- this MVP wants one short single-stroke clip, not a whole match recording
MAX_RAW_FRAMES = 900  # ~30s at 30fps. Bails out of decoding EARLY (mid-read) rather than reading the whole
# file first and rejecting after -- a long, high-res phone video decoded fully into memory before any check
# is a real resource-exhaustion risk (confirmed: a 320x240 synthetic 3-minute/5400-frame clip alone decoded to
# ~1GB of raw frame arrays; a real 1080p phone video of the same length would be over an order of magnitude
# larger). 900 is generous for "record one stroke" while bounding worst-case decode cost regardless of what a
# malicious or just-too-long upload's container metadata claims.
MIN_RAW_FRAMES = 5  # below this there isn't enough of a temporal window for StrokeGRU/biomechanics to mean
# anything -- resampling e.g. 2 real frames up to 60 via np.linspace just duplicates them, it doesn't invent data.
MIN_POSE_DETECTION_RATE = 0.80  # Raised from an earlier 50% floor after a real test (PROJECT_LOG.md,
# 2026-07-20): a moderately dark/blurry real clip landed at 63.3% detection and STILL produced a fully
# normal-looking response (92.5% stroke-classification confidence, full coaching text) with nothing
# distinguishing it from a clean 100%-detection result -- exactly the "sparse data presented as normal"
# failure this floor exists to prevent. Clean real footage in this pipeline (mvp/stress_test_pose.py,
# clip1/2/3) sits at 98-100%; 80% draws the line clearly above the demonstrated-bad case while leaving
# headroom for minor per-frame misses on otherwise-good footage.

app = FastAPI(title="VisioTennis MVP backend")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Trial-scope gate (2026-07-28): now that the backend is reachable over a public
# tunnel URL, a shared passcode keeps it from being wide open to anyone who
# stumbles on the link -- NOT real auth (one value, shared by every tester,
# checked as a plain string equality), which is deliberately enough for a
# bounded trial with a small number of known testers and nothing more. Applied
# as global middleware (not a per-route dependency) so it covers every request,
# including any future endpoint, without relying on remembering to add it each
# time. See mobile/lib/config.js for the matching header sent on every request.
#
# Read from the environment, not hardcoded (2026-08-28, moved as part of the
# "publish to GitHub" audit -- this value is the ONLY thing gating this
# backend from the open internet now that it's permanently reachable via
# Tailscale Funnel; hardcoding it in tracked source would have made that gate
# meaningless the moment the repo went public). No fallback default on
# purpose -- fails loudly at startup rather than silently running unprotected
# if TRIAL_PASSCODE is never set. Set it via an untracked .env-style mechanism
# or the shell environment before starting uvicorn; mobile/lib/config.js's
# copy must match exactly.
TRIAL_PASSCODE_HEADER = "X-Trial-Passcode"
TRIAL_PASSCODE = os.environ["TRIAL_PASSCODE"]


@app.middleware("http")
async def require_trial_passcode(request: Request, call_next):
    if request.headers.get(TRIAL_PASSCODE_HEADER) != TRIAL_PASSCODE:
        return JSONResponse(status_code=401, content={"detail": "missing or incorrect trial passcode"})
    return await call_next(request)

# Loaded once at process startup, not per-request -- all three are either
# a static reference population (percentile benchmark), a loaded ONNX
# graph, or a loaded MediaPipe task bundle; none of these need re-loading
# per request.
_state = {}


@app.on_event("startup")
def load_models():
    print("[startup] initializing accounts/history DB (backend/app.db)...")
    init_db()

    print("[startup] loading StrokeGRU ONNX session...")
    _state["stroke_session"] = ort.InferenceSession(STROKE_GRU_ONNX_PATH, providers=["CPUExecutionProvider"])

    if os.path.exists(STROKE_TEMPERATURE_PATH):
        with open(STROKE_TEMPERATURE_PATH) as f:
            _state["stroke_temperature"] = json.load(f)["temperature"]
        print(f"[startup] loaded stroke-confidence temperature T={_state['stroke_temperature']:.4f}")
    else:
        _state["stroke_temperature"] = 1.0
        print("[startup] no temperature.json found -- stroke confidence reported uncalibrated (T=1.0)")

    print("[startup] loading MediaPipe Pose landmarker...")
    base_options = mp_python.BaseOptions(model_asset_path=POSE_TASK_PATH)
    pose_options = vision.PoseLandmarkerOptions(
        base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_poses=1)
    _state["landmarker"] = vision.PoseLandmarker.create_from_options(pose_options)

    print("[startup] loading biomechanics percentile benchmark...")
    _state["benchmark"] = BiomechanicsPercentileBenchmark()

    # LLM_BACKEND selects the coaching LLM's serving stack, not the model or
    # prompt -- defaults to the existing local Ollama path so nothing changes
    # for anyone not opting in. "fireworks" is UNTESTED as of this writing
    # (see rag/llm.py:OpenAICompatibleBackend, PROJECT_LOG.md 2026-08-07) --
    # do not treat as a validated path until a confirmatory re-run is logged.
    llm_backend_choice = os.environ.get("LLM_BACKEND", "ollama")
    if llm_backend_choice == "fireworks":
        print("[startup] configuring Fireworks-hosted backend (qwen3-8b, FP8, UNTESTED)...")
        fireworks_key = os.environ.get("FIREWORKS_API_KEY")
        if not fireworks_key:
            raise RuntimeError("LLM_BACKEND=fireworks requires FIREWORKS_API_KEY to be set.")
        _state["llm_backend"] = OpenAICompatibleBackend(
            model=FIREWORKS_MODEL, api_key=fireworks_key, base_url=FIREWORKS_BASE_URL)
    else:
        # qwen3:14b (2026-08-26, promoted from qwen3:8b): the Qwen3
        # LLM-quality-lever trial (PROJECT_LOG.md, "LLM-quality-lever test")
        # found 14B fits comfortably alongside everything else resident on
        # the 3060 at Q4_K_M and showed a real grounding/quality improvement
        # over 8B on identical prompts -- see that log entry for the actual
        # before/after numbers this is based on.
        # think=False (2026-08-26): real 8-paired-trial A/B test (PROJECT_LOG.md
        # "qwen3:14b thinking-mode A/B test") found thinking OFF both 2.22x
        # faster AND more reliably grounded (8/8 vs 6/8 fully grounded) on
        # diverse real records -- not a speed-for-quality tradeoff, a genuine
        # win on grounding too. Real caveat: think=False showed more verbatim
        # cross-session boilerplate reuse in that test; tracked as a follow-up,
        # not blocking. Shared instance -- also affects stroke chat (line ~617),
        # which wasn't separately tested with chat-shaped prompts but shares
        # the same citation-discipline prompt structure.
        print("[startup] configuring Ollama backend (qwen3:14b, thinking=off)...")
        _state["llm_backend"] = OllamaBackend(
            model="qwen3:14b", host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            think=False)

    print("[startup] ready.")


def decode_and_resample_video(video_path: str, seq_len: int = SEQ_LEN,
                               max_raw_frames: int = MAX_RAW_FRAMES, min_raw_frames: int = MIN_RAW_FRAMES):
    """Reads every readable frame from the uploaded video, then uniformly
    resamples (not naive truncation) down or up to exactly seq_len frames
    -- StrokeGRU's ONNX graph hard-codes seq_len=60 (confirmed via
    ir/verify_onnx_exports.py), so this must land on exactly that count,
    not "close enough." This is the pre-trimming decision from
    mvp/DESIGN.md: the recording UI asks for one short swing; this
    function's only job is temporal resampling, not finding a stroke
    inside a longer video (explicit non-goal for this MVP).

    Bails out mid-read (not after) if the raw frame count exceeds
    max_raw_frames -- see MAX_RAW_FRAMES for why this must happen during
    the read loop, not as a check on the fully-materialized frame list."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("This video file couldn't be opened -- it may be corrupted or in an "
                          "unsupported format. Try recording a new clip.")

    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(frames) > max_raw_frames:
            cap.release()
            raise ValueError(
                f"This video is too long (over {max_raw_frames} frames). Record a short clip of "
                f"just one stroke, a few seconds long -- not a full rally or match.")
    cap.release()

    if len(frames) == 0:
        raise ValueError("This video file contained no readable video frames. Try recording a new clip.")
    if len(frames) < min_raw_frames:
        raise ValueError(
            f"This video is too short ({len(frames)} frame(s), need at least {min_raw_frames}). "
            f"Record at least about a second so there's a real swing to analyze.")

    if len(frames) == seq_len:
        sampled = frames
    else:
        idx = np.linspace(0, len(frames) - 1, seq_len).round().astype(int)
        sampled = [frames[i] for i in idx]

    return sampled, fps, len(frames)


def run_pose_extraction(frames, landmarker):
    """RAW frames, no crop -- validated on real non-dataset footage in
    mvp/stress_test_pose.py (100% detection rate, no red-shirt-hack
    needed, on 3 real amateur/practice clips)."""
    n_landmarks = 33
    feats = np.zeros((len(frames), n_landmarks * 3), dtype=np.float32)
    n_detected = 0
    for i, img_rgb in enumerate(frames):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = landmarker.detect(mp_img)
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            feats[i] = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).reshape(-1)
            n_detected += 1
    return feats, n_detected


def _append_log(record: dict, path: str = SESSION_LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_session_metrics(stroke_event_dict, class_probabilities, predicted_class, pose_detection_rate):
    """Phase 1 instrumentation (2026-08-01): logs the confidence-signalling inputs
    (mobile/components/ResultView.js) for EVERY session that reaches classification,
    regardless of whether a caveat displayed client-side -- logging only flagged
    sessions would bias the calibration data away from exactly the cases most worth
    finding: confident-but-wrong predictions (see the documented volley/forehand
    confusability, training/stroke_classification/README.md). Written immediately
    after classification, deliberately BEFORE coaching generation runs, so this data
    survives even if coaching hangs, fails, or the client disconnects before the full
    response completes -- see log_session_outcome() below for the coaching-outcome
    half, written separately and later, correlated by the same event_id.

    Deliberately non-identifying by construction, per explicit instruction: no
    player_id, no source_video filename (a tester could name a file after
    themselves), no IP/device info -- only the model's own output and derived
    numbers, keyed by the already-random event_id."""
    sorted_probs = sorted(class_probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top1_class, top1_prob = sorted_probs[0]
    runner_up_class, top2_prob = sorted_probs[1]
    _append_log({
        "event": "classification",
        "event_id": stroke_event_dict["event_id"],
        "created_at": stroke_event_dict["created_at"],
        "predicted_class": predicted_class,
        "confidence": top1_prob,
        "runner_up_class": runner_up_class,
        "top1_top2_gap": top1_prob - top2_prob,
        "pose_detection_rate": pose_detection_rate,
    })


def log_session_outcome(event_id, t0, coaching):
    """Phase 1 instrumentation (2026-08-01): logs the coaching-generation outcome
    (whether the retry-gate succeeded, how many attempts it took) and total
    processing time for a session that reached classification -- the roadmap's
    "whether the coaching gate retried, and how many times" and "processing time"
    instrumentation items. Correlated with log_session_metrics()'s entry via the
    same event_id rather than combined into one write, so classification data isn't
    lost if this step never completes (see that function's docstring)."""
    _append_log({
        "event": "outcome",
        "event_id": event_id,
        "processing_time_s": time.monotonic() - t0,
        "coaching_success": coaching["success"],
        "coaching_attempts": coaching["n_attempts"],
        "coaching_failure_reason": coaching.get("failure_reason"),
    })


def log_rejection(failure_mode: str, t0: float):
    """Phase 1 instrumentation (2026-08-01): logs a session rejected before reaching
    classification -- Phase C's upload/content failure modes (bad_video_file,
    no_player_detected, low_pose_detection, and not_tennis added 2026-08-12). The
    roadmap's "any failure mode
    triggered" instrumentation item; without this, rejected sessions were invisible
    to instrumentation entirely, silently undercounting how often testers hit these.
    No StrokeEvent exists yet at rejection time, so there's no real event_id to reuse
    -- generates a fresh one purely to give each log row a unique key, not to
    identify anyone (same non-identifying-by-construction rule as the other two
    logging functions: no player_id, no filename, no IP/device info)."""
    _append_log({
        "event": "rejection",
        "event_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "failure_mode": failure_mode,
        "processing_time_s": time.monotonic() - t0,
    })


def run_stroke_classification(pose_seq, session, temperature=1.0):
    """temperature=1.0 (default) reproduces the original raw-softmax behavior --
    every diagnostic script under ir/ calls this without a temperature and should
    keep seeing uncalibrated numbers, since they're evaluating the raw model, not
    the live pipeline's reported confidence. The live pipeline (below) passes the
    fitted _state["stroke_temperature"] explicitly."""
    ins = session.get_inputs()[0]
    x = pose_seq.astype(np.float32)[None]
    logits = session.run(None, {ins.name: x})[0][0]
    scaled_logits = logits / temperature
    probs = np.exp(scaled_logits - scaled_logits.max())
    probs = probs / probs.sum()
    class_probabilities = {c: float(p) for c, p in zip(STROKE_CLASSES, probs)}
    predicted_class = max(class_probabilities, key=class_probabilities.get)
    return predicted_class, class_probabilities


@app.get("/api/v1/health")
def health():
    # Reachability check differs by backend: Ollama exposes an unauthenticated
    # local /api/tags; OpenAICompatibleBackend (Fireworks etc.) requires the
    # real API key on every call, so "reachable" there just confirms the host
    # resolves/responds, not that the key is valid -- a real generate() call
    # is the only true test of that (see the confirmatory re-run requirement).
    llm_backend = _state["llm_backend"]
    llm_reachable = False
    try:
        import urllib.request
        if isinstance(llm_backend, OllamaBackend):
            with urllib.request.urlopen(f"{llm_backend.host}/api/tags", timeout=3) as resp:
                llm_reachable = resp.status == 200
        elif isinstance(llm_backend, OpenAICompatibleBackend):
            req = urllib.request.Request(llm_backend.base_url, method="HEAD")
            try:
                urllib.request.urlopen(req, timeout=3)
                llm_reachable = True
            except urllib.error.HTTPError:
                llm_reachable = True  # any HTTP response (even 4xx) means the host is up
    except Exception:
        llm_reachable = False

    return {
        "stroke_onnx_loaded": "stroke_session" in _state,
        "pose_landmarker_loaded": "landmarker" in _state,
        "percentile_benchmark_loaded": "benchmark" in _state,
        "llm_backend": llm_backend.name,
        "llm_reachable": llm_reachable,
    }


# ---- Accounts (Phase 2, 2026-08-07) --------------------------------------
# Email + password, opaque session tokens -- see backend/auth.py's docstring
# and PROJECT_LOG.md for the full reasoning. Still behind the global
# require_trial_passcode middleware above like every other route; this is a
# second, independent layer on top of the trial gate, not a replacement.

@app.post("/api/v1/auth/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db_session)):
    user = User(
        id=str(uuid.uuid4()), email=body.email, password_hash=hash_password(body.password),
        name=body.name, playing_hand=None, created_at=utcnow(),
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    db.refresh(user)
    token_row = issue_token(db, user)
    return AuthResponse(token=token_row.token, user=user_to_out(user))


@app.post("/api/v1/auth/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db_session)):
    user = db.query(User).filter(User.email == body.email).first()
    if user is None or not verify_password(body.password, user.password_hash):
        # Deliberately the same message for "no such email" and "wrong password" --
        # distinguishing them would let a caller enumerate registered emails.
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    token_row = issue_token(db, user)
    return AuthResponse(token=token_row.token, user=user_to_out(user))


@app.post("/api/v1/auth/logout")
def logout(token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session)):
    db.delete(token_row)
    db.commit()
    return {"status": "logged_out"}


@app.get("/api/v1/auth/me", response_model=UserOut)
def me(token_row: AuthToken = Depends(get_current_token)):
    return user_to_out(token_row.user)


@app.patch("/api/v1/auth/profile", response_model=UserOut)
def update_profile(
    body: ProfileUpdateRequest,
    token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session),
):
    """Sets the coaching-tone profile (mobile/app/profile-setup.js, used both
    right after signup and later from Settings). ProfileUpdateRequest's own
    validator already enforces role-appropriate fields (backend/auth.py) --
    this just writes whatever validated shape arrives. Always overwrites the
    full profile (role + whichever pair of follow-up fields), not a partial
    patch of individual fields, matching the UI's own all-at-once submit."""
    user = token_row.user
    user.role = body.role
    user.skill_level = body.skill_level
    user.years_coaching = body.years_coaching
    user.certification_level = body.certification_level
    db.commit()
    db.refresh(user)
    return user_to_out(user)


@app.patch("/api/v1/auth/coach-name", response_model=UserOut)
def update_coach_name(
    body: CoachNameUpdateRequest,
    token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session),
):
    """Sets the player's chosen name for their AI coach persona
    (mobile/app/(tabs)/settings.js, StrokeChat's CoachBubble). Deliberately
    separate from PATCH /api/v1/auth/profile -- unlike that endpoint,
    coach_name does not require role to already be set, and has no
    role-dependent validation. CoachNameUpdateRequest's own validator
    already normalizes blank/whitespace-only input to None (backend/auth.py)
    -- this just writes whatever validated value arrives; the "Coach"
    fallback for a None value happens client-side wherever it's displayed,
    same as every other nullable profile field in this app."""
    user = token_row.user
    user.coach_name = body.coach_name
    db.commit()
    db.refresh(user)
    return user_to_out(user)


# ---- History (Phase 2, 2026-08-07) ----------------------------------------
# Backend-persisted swap-in for what mobile/lib/historyRepository.js used to
# read straight from on-device expo-sqlite -- rows are written automatically
# by analyze_stroke() above, tied to the authenticated user, not pushed here
# by the client. These two GETs return the same (id, created_at, result)
# shape the on-device version did, so historyRepository.js's own return
# shapes -- and every screen that calls it -- don't need to change, only
# where the data comes from.

@app.get("/api/v1/history")
def list_history(token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session)):
    rows = (
        db.query(StrokeHistory)
        .filter(StrokeHistory.user_id == token_row.user_id)
        .order_by(desc(StrokeHistory.created_at))
        .all()
    )
    return [
        # + "Z": created_at is stored as naive-but-UTC (see backend/db.py's
        # utcnow() docstring) -- without an explicit UTC marker, JS's
        # `new Date(...)` on the client would parse this as LOCAL time,
        # silently shifting every timestamp by the device's UTC offset.
        {"id": r.id, "created_at": r.created_at.isoformat() + "Z", "result_json": r.result_json}
        for r in rows
    ]


@app.get("/api/v1/history/{history_id}")
def get_history_item(
    history_id: str, token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session),
):
    row = db.get(StrokeHistory, history_id)
    # Checked separately from "not found" (both 404) rather than a 403 on
    # ownership mismatch -- a 403 would confirm to a caller that a given id
    # exists and belongs to someone else, a minor enumeration leak a 404
    # avoids for free.
    if row is None or row.user_id != token_row.user_id:
        raise HTTPException(status_code=404, detail="No session found with this id.")
    return {"id": row.id, "created_at": row.created_at.isoformat() + "Z", "result": json.loads(row.result_json)}


# ---- Stroke chat (2026-08-08) ----------------------------------------------
# See PROJECT_LOG.md's chat grounding architecture entry and rag/chat_router.py
# / rag/chat_pipeline.py's module docstrings for the full design. Deliberately
# STATELESS server-side (design decision: chat memory is ephemeral, lives only
# in mobile client state for the lifetime of the open Results/History-detail
# screen, not persisted) -- the client resends the conversation so far on
# every turn rather than the server tracking a chat session.

class ChatTurnMessage(BaseModel):
    role: Literal["player", "coach"]
    text: str


class ChatTurnRequestBody(BaseModel):
    message: str
    conversation_history: list[ChatTurnMessage] = []


def _compute_digest_for_user(db: Session, user_id: str, benchmark: BiomechanicsPercentileBenchmark):
    """Fetches this user's stored sessions (newest first, same query
    list_history() already uses) and computes a FRESH SessionDigest from
    them -- see ir/session_digest.py's module docstring: no caching, no
    persisted digest, full recomputation on every call, by design. Returns
    None if there's no usable session history yet (compute_session_digest's
    own definition of "usable" -- has a real biomechanics block)."""
    # Deliberately no row limit here (unlike a naive "just fetch 2*k") --
    # compute_session_digest() filters to USABLE sessions (real biomechanics
    # present) before windowing, so a hard row cap could silently undercount
    # if some of a user's most recent rows failed pose detection. At this
    # app's real session volumes per player, fetching full history (same
    # unlimited query list_history() already runs for the History tab) costs
    # nothing meaningful.
    rows = (
        db.query(StrokeHistory)
        .filter(StrokeHistory.user_id == user_id)
        .order_by(desc(StrokeHistory.created_at))
        .all()
    )
    sessions = [json.loads(row.result_json) for row in rows]
    digest = compute_session_digest(sessions, player_id=user_id, benchmark=benchmark)
    return digest.model_dump(mode="json") if digest is not None else None


@app.post("/api/v1/history/{history_id}/chat")
def chat_turn(
    history_id: str, body: ChatTurnRequestBody,
    token_row: AuthToken = Depends(get_current_token), db: Session = Depends(get_db_session),
):
    row = db.get(StrokeHistory, history_id)
    if row is None or row.user_id != token_row.user_id:
        raise HTTPException(status_code=404, detail="No session found with this id.")

    result = json.loads(row.result_json)
    coaching = result.get("coaching", {})
    if not coaching.get("success"):
        # No grounded coaching exists for this session (e.g. the retry-gate exhausted its
        # budget at analysis time, see analyze_stroke()) -- nothing exists yet for tier 1 to
        # replay and tier 2 has no prior citations to build "existing topics" context from.
        # Chat is still technically possible (tier 2/3 could run cold), but that's more scope
        # than this design covers -- refuse clearly rather than silently degrade.
        raise HTTPException(status_code=409, detail={
            "error_code": "no_coaching_available",
            "message": "This session doesn't have coaching feedback available to chat about.",
        })

    stroke_event_dict = result["stroke_event"]
    existing_response = CoachingResponse.model_validate({"items": coaching["items"]})
    existing_items = list(existing_response.items)
    existing_citations = [
        ExistingCitation(
            field_path=it.citation.ir_field_path,
            label=it.citation.ir_field_path.split(".")[-1],
            text=it.text,
        )
        for it in existing_items if it.citation is not None
    ]
    conversation_history = [(m.role, m.text) for m in body.conversation_history]
    digest = _compute_digest_for_user(db, token_row.user_id, _state["benchmark"])

    turn = handle_chat_turn(
        _state["llm_backend"], stroke_event_dict, _state["benchmark"],
        existing_items, existing_citations, conversation_history, body.message,
        tone_profile=user_tone_profile(token_row.user), digest=digest,
    )
    return {
        "route": turn.route,
        "tier": turn.tier,
        "text": turn.text,
        "citation": (turn.items[0].citation.model_dump()
                     if turn.items and turn.items[0].citation is not None else None),
        "success": turn.success,
        "failure_reason": turn.failure_reason,
        "n_attempts": turn.n_attempts,
        "total_latency_s": turn.total_latency_s,
        "escalated_from_tier3": turn.escalated_from_tier3,
    }


async def _read_upload_capped(video: UploadFile, max_bytes: int, t0: float) -> bytes:
    """Reads the upload in chunks, aborting as soon as max_bytes is exceeded
    -- rejects an oversized upload without first buffering the whole thing
    into memory (which `await video.read()` alone would do)."""
    chunks = []
    total = 0
    while True:
        chunk = await video.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            log_rejection("bad_video_file", t0)
            raise HTTPException(
                status_code=413,
                detail={
                    "error_code": "bad_video_file",
                    "message": f"This video is too large (over {max_bytes // (1024 * 1024)}MB). "
                               f"Record a short clip of just one stroke, not a long recording.",
                },
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.post("/api/v1/analyze-stroke")
async def analyze_stroke(
    video: UploadFile = File(...),
    # Session-type context (2026-08-12, mobile/app/(tabs)/index.js's practice/match
    # selection screen) -- a JSON string the client sends alongside the video, e.g.
    # {"mode":"practice","practiceType":"stroke","strokeType":"serve"} or
    # {"mode":"match"}. Purely descriptive metadata: does NOT feed classification,
    # biomechanics, or coaching generation -- those are unchanged. Stored as-is in
    # the persisted result_json (see StrokeHistory below) for future use (e.g. a
    # cross-session digest grouping by session type). Optional and best-effort:
    # a malformed or missing value never fails the request, it just means this
    # session has no recorded context, same as before this feature existed.
    session_context: str | None = Form(None),
    token_row: AuthToken = Depends(get_current_token),
    db: Session = Depends(get_db_session),
):
    t0 = time.monotonic()
    parsed_session_context = None
    if session_context:
        try:
            parsed_session_context = json.loads(session_context)
        except (json.JSONDecodeError, TypeError):
            parsed_session_context = None
    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    body = await _read_upload_capped(video, MAX_UPLOAD_BYTES, t0)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(body)
        tmp_path = tmp.name

    stage_timings_s = {}
    try:
        _stage_t0 = time.monotonic()
        try:
            frames, fps, n_real_frames = decode_and_resample_video(tmp_path)
        except ValueError as e:
            log_rejection("bad_video_file", t0)
            raise HTTPException(status_code=422, detail={"error_code": "bad_video_file", "message": str(e)})
        stage_timings_s["video_decode"] = time.monotonic() - _stage_t0

        # Zero-shot tennis-content gate (2026-08-12, ir/tennis_classifier.py) --
        # cheap (a handful of CLIP frame encodes), so it runs before the
        # expensive pose/classification/coaching stages, rejecting clearly
        # wrong content early rather than after paying for the rest of the
        # pipeline. See that module's docstring for what the locked 0.5
        # threshold was and wasn't validated against.
        _stage_t0 = time.monotonic()
        tennis_check = classify_frames(frames)
        stage_timings_s["tennis_classification"] = time.monotonic() - _stage_t0
        if not tennis_check["is_tennis"]:
            log_rejection("not_tennis", t0)
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "not_tennis",
                    "message": "We couldn't confirm this video shows tennis being played. Make sure the "
                               "clip clearly shows a player and a stroke, filmed on a tennis court.",
                },
            )

        _stage_t0 = time.monotonic()
        pose_seq, n_detected = run_pose_extraction(frames, _state["landmarker"])
        stage_timings_s["pose_extraction"] = time.monotonic() - _stage_t0
        detection_rate = n_detected / len(pose_seq)
        if n_detected == 0:
            log_rejection("no_player_detected", t0)
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "no_player_detected",
                    "message": "No player could be detected anywhere in this video. Make sure the "
                               "player is clearly visible in frame, and that only one person is in view.",
                },
            )
        if detection_rate < MIN_POSE_DETECTION_RATE:
            log_rejection("low_pose_detection", t0)
            raise HTTPException(
                status_code=422,
                detail={
                    "error_code": "low_pose_detection",
                    "message": f"Couldn't get a clear, reliable read on the player's movement -- a "
                               f"clear pose was only detected in {n_detected}/{len(pose_seq)} frames "
                               f"({detection_rate:.0%}, need at least {MIN_POSE_DETECTION_RATE:.0%}). "
                               f"Try better lighting, filming closer to the player, and make sure only "
                               f"one person is in frame.",
                },
            )

        _stage_t0 = time.monotonic()
        predicted_class, class_probabilities = run_stroke_classification(
            pose_seq, _state["stroke_session"], temperature=_state["stroke_temperature"])
        stage_timings_s["stroke_classification"] = time.monotonic() - _stage_t0

        _stage_t0 = time.monotonic()
        try:
            biomech_features = extract_biomechanics_features(pose_seq)
        except ValueError as e:
            log_rejection("bad_video_file", t0)
            raise HTTPException(
                status_code=422,
                detail={"error_code": "bad_video_file",
                        "message": f"This video couldn't be processed: {e} Try recording a new clip."},
            )
        stage_timings_s["biomechanics_extraction"] = time.monotonic() - _stage_t0

        biomechanics = BiomechanicsInfo(
            **biomech_features,
            source_model=ModelProvenance(
                component="biomechanics",
                checkpoint="pose_landmarker_full.task (upstream MediaPipe Pose; geometric derivation has no learned weights)",
                metric_name="n/a", metric_value=None,
                source_file=video.filename,
            ),
        )
        stroke = StrokeInfo(
            predicted_class=predicted_class,
            confidence=class_probabilities[predicted_class],
            class_probabilities=class_probabilities,
            source_model=ModelProvenance(
                component="stroke", checkpoint="stroke_gru.onnx",
                metric_name="val_macro_f1", metric_value=0.834,
                source_file=video.filename,
            ),
        )
        stroke_event = StrokeEvent(
            player_id=token_row.user_id,
            source_video=video.filename,
            frame_range=FrameRange(start_frame=0, end_frame=len(pose_seq) - 1, fps=fps),
            ball=None, court=None,
            stroke=stroke,
            biomechanics=biomechanics,
        )

        benchmark = _state["benchmark"]
        stroke_event_dict = stroke_event.model_dump(mode="json")
        log_session_metrics(stroke_event_dict, class_probabilities, predicted_class, detection_rate)
        _stage_t0 = time.monotonic()
        system_prompt, user_prompt = build_stroke_prompt(
            stroke_event_dict, benchmark, tone_profile=user_tone_profile(token_row.user))
        stage_timings_s["prompt_build_and_retrieval"] = time.monotonic() - _stage_t0
        source_records = {("StrokeEvent", stroke_event.event_id): stroke_event_dict}

        # Stroke classification + biomechanics above have already succeeded at this point --
        # a coaching-generation failure (LLM unreachable, or the retry-gate exhausting its
        # budget, see rag/gated_pipeline.py) must not throw that real result away. Confirmed
        # real bug (PROJECT_LOG.md, 2026-07-20): with Ollama unreachable, generate_grounded_response
        # -> OllamaBackend.generate raises urllib.error.URLError uncaught, which previously
        # propagated all the way out as a bare, non-JSON 500 -- discarding the already-computed
        # stroke_event and giving the client nothing to show. Caught here and turned into the
        # same explicit "coaching unavailable" shape the retry-gate's own exhaustion path
        # already returns, so the mobile client only has to handle one failure shape either way.
        try:
            # max_attempts=2 (2026-08-26, was 3): real latency investigation
            # (PROJECT_LOG.md) measured 7 real end-to-end qwen3:14b generations
            # through this exact endpoint -- 0/7 ever needed a 3rd attempt (5
            # succeeded on attempt 1, 2 needed exactly attempt 2). At 14B's
            # per-attempt cost (~35-110s), a 3rd attempt is pure worst-case
            # latency with no observed reliability payoff on this sample --
            # capping at 2 keeps the same 7/7 real success rate at a lower
            # worst-case bound. Small sample (n=7); if a genuine need for a
            # 3rd attempt shows up later, revisit rather than assume this holds.
            gated = generate_grounded_response(
                _state["llm_backend"], system_prompt, user_prompt, source_records,
                max_attempts=2, percentile_benchmark=benchmark,
            )
            coaching = {
                "success": gated.success,
                "n_attempts": gated.n_attempts,
                "total_latency_s": gated.total_latency_s,
                "attempt_latencies_s": [a.latency_s for a in gated.attempts],
            }
            if gated.success:
                coaching["items"] = [
                    {
                        "text": item.text, "type": item.type,
                        "citation": item.citation.model_dump() if item.citation else None,
                    }
                    for item in gated.response.items
                ]
            else:
                coaching["failure_reason"] = gated.failure_reason
        except Exception as e:
            coaching = {
                "success": False,
                "n_attempts": 0,
                "total_latency_s": 0.0,
                "failure_reason": f"coaching feedback service unavailable ({type(e).__name__}: {e})",
            }

        stage_timings_s["llm_generation"] = coaching.get("total_latency_s", 0.0)
        log_session_outcome(stroke_event.event_id, t0, coaching)

        response_body = {
            "stroke_event": stroke_event_dict,
            "coaching": coaching,
            "session_context": parsed_session_context,
            "diagnostics": {
                "real_frames_in_upload": n_real_frames,
                "pose_detection_rate": n_detected / len(pose_seq),
                "total_processing_time_s": time.monotonic() - t0,
                "stage_timings_s": stage_timings_s,
            },
        }

        # Persisted server-side, atomically with the analysis itself, rather than
        # requiring a second client-initiated "save to history" call -- a client
        # that crashes/backgrounds right after receiving the response would
        # otherwise silently lose the session from history. Reuses the
        # StrokeEvent's own event_id as the history row's id (see backend/db.py's
        # StrokeHistory docstring) rather than minting a second id for the same
        # thing.
        db.add(StrokeHistory(
            id=stroke_event.event_id, user_id=token_row.user_id,
            created_at=utcnow(), result_json=json.dumps(response_body),
        ))
        db.commit()

        return response_body
    finally:
        os.unlink(tmp_path)


class FeedbackSubmission(BaseModel):
    """Phase 1 structured feedback (2026-08-01) -- two tracks kept separate per the
    roadmap: `reliability` (was the shot identified correctly?) and
    `usefulness_rating`/`usefulness_text` (was the coaching useful?), plus
    `actual_class` -- real ground truth on real phone footage, collected only when
    the prediction was wrong or partly wrong, which the project has none of
    otherwise. `role`/`coach_would_say_to_student` are optional and only meaningful
    when the submitter identified as a coach (mobile/app/(tabs)/settings.js) -- a
    bare role CATEGORY, not an identifier, never tied to a name or persistent
    per-tester id."""
    event_id: str
    reliability: Literal["yes", "no", "partly"]
    actual_class: str | None = None
    usefulness_rating: int
    usefulness_text: str | None = None
    coach_would_say_to_student: bool | None = None
    role: Literal["player", "coach"] | None = None

    @model_validator(mode="after")
    def _check_consistency(self):
        if not (1 <= self.usefulness_rating <= 5):
            raise ValueError("usefulness_rating must be between 1 and 5")
        if self.reliability in ("no", "partly") and self.actual_class is None:
            raise ValueError("actual_class is required when reliability is 'no' or 'partly'")
        if self.actual_class is not None and self.actual_class not in STROKE_CLASSES:
            raise ValueError(f"actual_class must be one of {STROKE_CLASSES}")
        return self


@app.post("/api/v1/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    _append_log({
        "event_id": feedback.event_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reliability": feedback.reliability,
        "actual_class": feedback.actual_class,
        "usefulness_rating": feedback.usefulness_rating,
        "usefulness_text": feedback.usefulness_text,
        "coach_would_say_to_student": feedback.coach_would_say_to_student,
        "role": feedback.role,
    }, path=FEEDBACK_LOG_PATH)
    return {"status": "recorded"}
