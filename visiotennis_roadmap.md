# VisioTennis — Consolidated Plan

From where the project stands today to the full product vision. Phases are ordered by dependency, not by the order a user experiences them.

---

## Phase 0 — Get the trial running (days, not weeks)

The goal is a bounded trial with university tennis players and a coach. Everything here blocks that.

**Reachability**
- Free public URL for the backend (Tailscale Funnel, or Cloudflare rotating quick-tunnel as fallback)
- No domain purchase, no paid tier
- Verified with a real round trip from a device not on the local network

**App foundation**
- Real device/emulator verification of the Phase B navigation rebuild — tap-through, not compile-check
- Specifically: SQLite persistence across app restart, and History refresh on tab focus
- Phase C: map all 5 backend failure modes to human-readable screen states, each saying what happened and what to do

**Consent and data handling**
- What happens to a tester's video, how long it's kept, when it's deleted
- Explicit opt-in before any upload
- Includes a separate, clearly-worded opt-in for training-data retention (see Phase 1)

**Coach review — runs in parallel, needs no code**
- Knowledge base exported as a standalone review document
- Per entry: is it technically correct? would you say it to a player? what's missing?
- Flag the biomechanical percentile framing as the area needing most scrutiny
- Ideally two coaches, not one — coaching philosophy varies

---

## Phase 1 — Build the trial to actually produce data

These are additions to the trial, not separate work. Without them the trial yields anecdotes instead of evidence.

**Automatic session instrumentation**
Log per session, server-side:
- Pose detection rate
- Predicted class + confidence
- Whether the coaching gate retried, and how many times
- Processing time
- Any failure mode triggered

**Structured feedback, two tracks kept separate**
- *Reliability*: was the shot identified correctly? (yes / no / partly)
- *Usefulness*: was the coaching useful? 1–5, plus optional free text
- For the coach specifically: would you say this to a student?

The value is in the join — e.g. does low pose detection predict low usefulness ratings?

**Training-data capture, with explicit consent**
- Store the video alongside the tester's correction of the predicted shot
- This is the most valuable data the project will ever get cheaply: real phone footage, real conditions, real labels
- Must be built in from day one — retrofitting means discarding it
- Requires its own consent checkbox, separate from "process my video"

**Confidence signalling in the UI**
- When classification confidence or pose detection is low, say so rather than presenting a confident answer
- Pair it with actionable guidance: framing, lighting, distance, clear the frame of other people
- Cheap to build, uses signals that already exist, and does more for perceived reliability than a few points of accuracy

---

## Phase 2 — Run the trial

Not a build phase. Real players and a real coach using the app on their own courts and phones.

This is the first genuine test of generalisation: different bodies, courts, lighting, camera angles, and phones than anything in training or testing so far.

**Exit criteria — decide these before starting, not after:**
- What shot-recognition accuracy on tester footage is acceptable?
- What average usefulness rating counts as "the coaching works"?
- What does the coach have to say for the knowledge base to be considered sound?

---

## Phase 3 — The fork

The trial determines the next move. Two honest branches:

**If the coaching lands well** → proceed to Phase 4 and build toward the product.

**If the coaching is thin, generic, or wrong** → stop building product. Go back to the knowledge base and the models. This is the correct outcome to act on, not a failure — finding it now is exactly what the trial is for.

Do not skip this decision point. Everything after it is expensive.

---

## Phase 4 — Foundation for a real product

**Cloud migration** (the biggest single piece) — **IN PROGRESS as of 2026-08-07**, see `PROJECT_LOG.md`
- The backend cannot stay on one PC once there are accounts and paying users
- Includes the LLM decision: cloud GPU hosting vs. swapping to a hosted API
- The `ClaudeBackend` stub exists but is untested — that's a real validation effort, not a config change

**Accounts and server-side persistence**
- Everything downstream depends on this
- Unblocks moving session history off-device
- Design the age question now: a tennis app will have under-18 users, and retrofitting age handling into a live social system is painful

**Retrain on trial data**
- Use the labelled footage captured in Phase 1
- This is the first real chance to broaden the training distribution beyond the original narrow set

---

## Phase 5 — The product experience

In dependency order, not user-journey order:

1. **Profile, progression, charts** — the retention loop; what makes someone open the app twice
2. **Onboarding and intro tour** — polish the front door once there's something behind it
3. **Coaching chatbot** — needs real design work first: multi-turn grounding is a different problem from one-shot. What happens when a follow-up question can't be answered from the data? Refusing is annoying; answering ungrounded defeats the entire architecture
4. **Social layer** — leaderboard, friends, lobbies. Requires moderation, reporting, and blocking designed in, not bolted on. Meaningless without a user base anyway
5. **Paywall** — deliberately last. The trial and early usage tell you what people would actually miss if you took it away. Guessing now means guessing wrong

**Platform note:** Apple and Google have specific rules on subscription paywalls, including that a hard paywall shown before any demonstrated value is a common rejection reason. Reorder to: intro → sign up → experience something → paywall.

---

## Positioning — decide this deliberately

SwingVision wins on line-calling accuracy and camera robustness. Competing there means fighting on their turf with fewer resources.

They barely do explainable coaching. The grounded-citation architecture — every claim provably tied to a real measurement — is genuinely novel and is the strongest thing this project has built.

**Own "the app that tells you why, and proves it."** Make that an explicit product decision rather than something that happens by default.

---

## The main risk

Not technical. The risk is that a product vision this large absorbs unlimited effort while the core question — *is the coaching actually good?* — stays unanswered.

There is an FYP deadline and a real product ambition, and they pull in different directions. The trial serves both. Protect it from being deprioritised by more interesting engineering work.

Everything past Phase 3 is a guess until Phase 2 is done.
