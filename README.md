# VisioTennis

**AI tennis coaching that shows its work.**

VisioTennis turns a short smartphone video of a tennis stroke into coaching feedback that is mechanically verified against real measured data — not just fluent-sounding text. Every specific claim in a generated coaching response is checked against a structured, schema-validated evidence record before it's ever shown to a user. If a claim can't be verified, it doesn't ship.

---

## Why this exists

Language models can generate plausible, confident-sounding technical advice that was never actually supported by what a video shows. For a coaching tool, that's not a minor flaw — it's the core risk. VisioTennis addresses this with a **schema-validated Intermediate Representation (IR) layer** that sits between computer vision and language generation, so the model is never trusted to accurately report its own evidence.

---

## Architecture

```
Video → Pose extraction (MediaPipe) → Stroke classification → Biomechanics
      → IR layer (Pydantic-validated: StrokeEvent / SessionRecord / SessionDigest)
      → Retrieval-augmented coaching generation (local Qwen3, via Ollama)
      → Mechanical citation verification → Retry-gate (≤3 attempts) → Response
```

Every generated coaching claim carries a **citation** — a record type, record ID, field path, and cited value — which is checked by exact lookup against the real stored data, not inferred or trusted. Responses that fail verification are regenerated; if verification still fails after 3 attempts, the app returns an explicit failure state rather than an unverified answer.

A separate **context plane** conditions *how* feedback is explained — based on the user's registered role (Player or Coach) and skill level (Beginner / Intermediate / Advanced) — without ever changing the underlying evidence, citations, or measured values.

---

## Key results — reported honestly, including what didn't work

This project treats negative results as real findings, not failures to hide.

| Result | Value |
|---|---|
| Grounded response delivery, ungated → gated (retry) | 66.7% → **100%** |
| Stroke classification (corrected, player-disjoint split) | 68.8% accuracy / 0.647 macro-F1 |
| Superseded metric — do not cite | ~~84.5% / 0.834~~ (leaky split — 100% player overlap across train/val/test, found and corrected) |
| Deployed model's real-footage accuracy | 57.1% (8/14) |
| Serve classification on real footage | **0/3 — unresolved after 6 independent interventions** (documented in full) |
| Court homography, in-distribution vs. real footage | 2.2–10.5 cm → 1.22–6.27 m reprojection error |

**Two independent data-leakage bugs were found and corrected** in this project — one in court detection's train/validation split, one in the stroke classifier's headline metric. Both are documented with before/after numbers, not quietly fixed and forgotten.

**Ball tracking and court homography are trained but deliberately excluded from the live coaching path** — real-footage validation showed severe generalization failure, and the exclusion (made cautiously beforehand) was confirmed correct by that evidence.

Full findings, including the closed six-point serve-classification investigation and every ruled-out hypothesis, are in `PROJECT_LOG.md`.

---

## Features

- Stroke classification (forehand / backhand / serve / volley / smash) from a single short clip
- Grounded, citation-verified coaching feedback with automatic quality/confidence indicators
- Skill- and role-conditioned language — the same evidence, explained differently for a beginner player vs. a coach
- Cross-session progress digest ("How am I trending?"), computed fresh from real stored data every time — never chained from a prior AI-generated summary
- Multi-turn coaching chatbot with tiered grounding and adversarially-tested safety backstops
- Full light/dark adaptive theming
- An illustrated, per-user-nameable Coach Avatar

---

## Tech stack

- **Mobile:** React Native (Expo)
- **Backend:** FastAPI (Python)
- **Computer vision:** MediaPipe Pose, PyTorch (fine-tuned GRU stroke classifier), ONNX Runtime for deployed inference
- **Language generation:** Qwen3, served locally via Ollama
- **Validation:** Pydantic v2 for all IR schema enforcement

---

## Project structure

```
backend/     FastAPI server and the full coaching pipeline entrypoint
mobile/      React Native / Expo client
rag/         Grounding, citation verification, retry-gate, knowledge base
training/    Model training scripts and small deployed checkpoints
ir/          IR schema definitions, validation, and diagnostic tooling
mvp/real_footage_test/   Real, hand-labeled validation clips used throughout
PROJECT_LOG.md           Full, dated development log — every finding, fix, and negative result
start_visiotennis.ps1    One-command local startup script
```

---

## Getting started

**Prerequisites:** Python 3.x, Node.js + Expo CLI, [Ollama](https://ollama.com) with a Qwen3 model pulled, an Android emulator or device.

```bash
# Backend
cd backend
pip install -r requirements.txt
export TRIAL_PASSCODE=<choose-a-value>   # required — no hardcoded default
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Mobile
cd mobile
npm install
npx expo start
```

Configure `mobile/lib/config.js` with your backend's reachable URL and matching `TRIAL_PASSCODE`.

> **Note:** third-party datasets (THETIS, TenniSet) used for model development are not included in this repository due to unresolved licensing terms on the original distributions. Only this project's own real test fixtures (`mvp/real_footage_test/`) are included.

---

## Known limitations

- Real-footage serve classification remains an open, well-characterized problem — 0/3 across every tested checkpoint and architecture, investigated across six independent interventions with the ruled-out explanations documented.
- The real-footage validation set is small (14 hand-labeled clips, no volley/smash coverage) — sufficient to detect the serve failure, not sufficient to estimate overall deployment accuracy.
- Ball tracking and court homography are not reliable on unconstrained real footage and are excluded from the active coaching path.
- The coaching knowledge base is hand-authored and has not yet undergone a review by a qualified tennis coach.
- Momentum tracking currently demonstrates reconstruction of a known formula from complete information, not independently validated forecasting.

---

## Research

This project is accompanied by a research paper detailing the IR architecture, grounding methodology, and full deployment-aware failure analysis.

---

## Contributors

- Syed Hassan Raza Bukhari
- Khwaja Ali bin Usman
- Malaika Saleem

**Supervisor:** Daniyal Adeeb
Department of Artificial Intelligence, University of Management and Technology, Lahore

---

## Acknowledgments

- [THETIS](https://github.com/THETIS-dataset/dataset) — Three-Dimensional Tennis Shots dataset (Gourgari et al., CVPR 2013)
- [TenniSet](https://github.com/HaydenFaulkner/Tennis) — dense event annotation dataset for tennis broadcasts
- The Qwen team, for the open-weight language models this project runs locally
- MediaPipe, for pose estimation

---

## License

Not yet finalized. Given upstream dataset licensing constraints affecting parts of model development, a license for this repository will be added separately — do not assume permissive reuse until one is specified.

<video src="assets/demo.mp4" controls width="100%"></video>
