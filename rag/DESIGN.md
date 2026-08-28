# RAG Coaching Layer — design document (objective 3)

**Status (updated 2026-07-18): grounding harness, retry-gate, KB (24-entry
placeholder, session-level), embedding-based semantic retrieval, and LLM
choice (Qwen3 8B, local via Ollama) are all built and evaluated.** Step 1's
KB-content-quality question (placeholder vs. licensed/open-sourced content)
remains open — see "Recommendation" below, still needs sign-off on options
1/2 if higher-authority content is ever pursued. Step 2 is RESOLVED for
session-level coaching (statistical, self-referential deviation via
`explain_anomaly()`, no external benchmark needed) and stroke-level
biomechanical benchmarking is formally DEFERRED (blocked on schema/pose-
feature work that doesn't exist yet), not an open question being actively
worked.

## Purpose

Objective 3 calls for a RAG pipeline that turns IR records (`StrokeEvent`,
`SessionRecord` — see `ir/schema.py`) into natural-language coaching
feedback, citing "the measured deviation that triggered each recommendation"
for each piece of advice, with a stated target of **>95% factual grounding
rate**. This document covers: what reference material actually exists to
retrieve from (Step 1), where "deviation" benchmarks come from (Step 2), the
retrieval/prompt/LLM architecture (Step 3), and how grounding rate is
actually measured, not just asserted (Step 4).

---

## Step 1: Knowledge base inventory — nothing exists locally, open question on how to proceed

**Checked, not assumed:** searched the entire repo (`find . -iname
"*coach*" -o -iname "*itf*" -o -iname "*usta*" -o -iname "*.pdf"`, etc.) for
any coaching-reference material. Result: **nothing found.** The only PDF
hits were unrelated `matplotlib` package assets; no ITF Coaching Manual, USTA
guidelines, articles, or any other coaching-content file exists anywhere
under this repo. `visiotennis_training_prompt.md` (the closest thing to a
written spec present locally) covers Models 1-4 only and says nothing about
RAG content sourcing. This matches the expectation stated in the request —
confirmed, not assumed.

**The constraint:** ITF/USTA coaching manuals are commercial, copyrighted
publications. Scraping or bulk-reproducing them is not legitimate use, FYP
context or not — this needs to be ruled out explicitly, not treated as "just
go find a PDF online."

**Three realistic options, not mutually exclusive:**

1. **Licensed access.** If the university library, a coaching-certification
   course, or a personal purchase gives legitimate access to ITF/USTA
   materials, that's the highest-quality source — but it's an acquisition
   step outside this codebase's control, and I have no way to verify
   licensing status from here. **Needs the user to confirm whether this is
   realistic/available**, not something to assume or fake.
2. **Publicly available, openly-licensed coaching content.** E.g.
   Wikipedia/Wikibooks tennis-technique articles (CC-BY-SA, redistribution
   with attribution is fine), open-access sports-biomechanics papers that
   publish coaching-relevant findings, or federation resources explicitly
   marked public-domain/open-licensed (not all federation content is — needs
   checking per-source, not assumed wholesale). This is legally clean but
   requires actual research effort to find and vet each source's license
   before use — not something to fabricate a list of right now.
3. **A small hand-authored placeholder set.** A handful of short coaching
   guideline snippets (e.g. 5-10 per stroke type: contact point, weight
   transfer, racket face angle, follow-through, common errors), written by
   us, informed by general/common tennis coaching knowledge, and **explicitly
   marked "authored, not sourced"** in the content itself and in any
   citation the RAG pipeline produces from it. This is not a substitute for
   a real knowledge base in the final system, but it is sufficient and
   honest for prototyping the retrieval + citation-grounding *mechanics*
   (the harness in Step 4 doesn't care whether the underlying content is
   licensed material or hand-authored — it only checks whether generated
   claims trace back to real IR field values).

**Recommendation: start with option 3 for pipeline prototyping, pursue
option 1/2 in parallel as a content-quality upgrade, not a blocker.** Option
3 is buildable immediately and lets Steps 3-4's plumbing (retrieval, prompt
template, grounding harness) get built and tested now, honestly labeled the
whole way. Options 1/2 are about *content quality/authority*, which is a
separate, slower-moving question — swapping in real retrieved passages later
shouldn't require re-architecting the pipeline if the retrieval layer is
built as a clean interface (see Step 3). **This needs your sign-off before I
write a single placeholder snippet** — in particular, confirm: (a) is
licensed ITF/USTA access realistically available to you, making option 1
worth pursuing now rather than later; (b) are you comfortable with option 3's
"authored, not sourced" placeholder framing for the prototype, understanding
it's explicitly not a source of coaching authority in the FYP writeup.

---

## Step 2: Deviation-detection benchmarks — RESOLVED (session-level); stroke-level explicitly DEFERRED

To cite "the measured deviation that triggered" a recommendation, the
pipeline needs a **reference value** to compare a measured quantity against.
This is unresolved, and — importantly — **it's tangled up with a schema gap
I found while working through this**, not just a benchmark-sourcing
question.

### The schema gap

`StrokeEvent` (`ir/schema.py`) currently captures: stroke classification +
confidence, ball trajectory (pixel + court-relative position), court
homography, and a `momentum` placeholder. **It does not capture any
per-stroke biomechanical measurement** — no contact-point height, no
racket-face angle, no knee bend, no shoulder-hip separation. These are
exactly the kind of things the example in the request ("what's a normal
contact point height for a forehand?") is asking about, and none of it
exists in the IR schema today.

We do have an available upstream signal for this: `ir/populate_e2e.py`
already proved real MediaPipe Pose extraction works (33 landmarks x, y, z
per frame) on our own footage. Deriving biomechanical features (e.g. contact
height = wrist-landmark y-position at the contact frame, relative to
hip-landmark y-position) from those landmarks is plausible future work, but
**it doesn't exist yet** — no derivation code, no schema fields for it. This
is a second open item, tightly coupled to Step 2, not a separate one: a
benchmark is only useful once there's a matching measured field to compare
it against.

**By contrast, `SessionRecord` already has a working, grounded deviation
mechanism today**, no new work needed: Model 4's `explain_anomaly()`
(`training/anomaly_detection/train.py`) computes real z-scores against the
fitted population mean/std of `ANOMALY_FEATURE_COLUMNS`, and that's already
what populates `SessionRecord.explanation` (see
`ir/example_output/session_record_0.json`: `"double_fault_rate=0.16 (4.6 std
above average)"` — a real, checkable deviation). Session-level coaching
feedback (fatigue, error-rate trends, serve consistency) is **buildable now**
against real grounded deviations. Stroke-level biomechanical feedback (the
"contact point height" example) is **not** buildable yet without resolving
both the benchmark question below and the schema gap above.

### Three candidate benchmark approaches (for stroke-level biomechanics, once the schema gap is addressed)

1. **Skill-tier-relative benchmarks.** Bucket players by skill level
   (beginner/intermediate/advanced) and define expected ranges per tier from
   coaching literature norms. Highest coaching validity, but needs
   literature-sourced ranges — the same sourcing/licensing question as Step
   1, since numeric technique norms mostly live inside the same commercial
   coaching manuals. Paraphrased/derived numeric ranges are more legally
   defensible than verbatim text reproduction, but still needs real sourcing
   research, not invention.
2. **Pro-player reference ranges from public data.** Some open-access
   sports-biomechanics papers publish numeric findings (contact height,
   swing angles) as citable statistics, which is standard academic use, not
   the copyright problem bulk manual reproduction would be. Requires
   literature research to find legitimately-reusable published numbers —
   a scoped research task, not something to fabricate placeholder numbers
   for.
3. **Population percentiles from our own dataset.** Once stroke-level
   biomechanical features are derived (the schema-gap work above) across
   our own `StrokeClassification`/THETIS-derived pose data, compute
   percentile distributions and flag a given stroke as a statistical outlier
   *within our own sample*. Fully self-contained, zero copyright exposure,
   immediately buildable once the schema gap is closed — but it measures
   "unusual relative to our own dataset" (mostly amateur players), not
   "technically correct," and must be labeled that way, not conflated with
   coaching authority.

**Recommendation:** option 3 is the only one buildable without new external
sourcing, so it's the natural first step for prototyping stroke-level
grounded feedback — but it answers a different question ("is this unusual
for our sample?") than options 1/2 ("is this technically sound?"), and the
FYP writeup needs to be precise about which claim is actually being made.

### Decision — RESOLVED (2026-07-18)

**Session-level coaching is the adopted grounding mechanism for this
project — no external benchmark sourcing needed, and none is being pursued.**
The deviation being cited for session-level feedback is **statistical**
(this session vs. that same player's own historical distribution, via
Model 4's `explain_anomaly()` z-scores), not a comparison against an
external "ideal" or "correct" technique value. That means Step 2's
sourcing/copyright problem (which affects options 1/2 above, and Step 1's
KB-content question) simply doesn't apply here: `explain_anomaly()` already
computes a real, checkable number (z-score against the fitted population
mean/std) entirely from data already produced inside this pipeline, with no
external reference material required. This has been true since Model 4 was
built; what changed here is formally confirming and documenting it as the
adopted mechanism, not an interim workaround pending something better.

**Stroke-level biomechanical benchmarking (contact height, racket angle,
etc.) is formally logged as DEFERRED, not pursued in this phase.** This is
a decision to stop actively working the question, not a restatement of it
being merely "open." It remains blocked on two things, both still true and
unaddressed: (1) the schema gap above (`StrokeEvent` has no biomechanical
fields, and no derivation code exists to compute them from the proven
MediaPipe Pose landmarks); (2) a benchmark source for whatever such fields
would be measured against (the three candidate approaches above remain
recorded for when this is picked back up, not discarded). Nothing about the
KB expansion or embedding retrieval built in this phase touches
stroke-level content — `rag/kb_placeholder.py` stays scoped to the 12
`ANOMALY_FEATURE_COLUMNS` session-level topics, consistent with this
decision.

---

## Step 3: Pipeline architecture

### Retrieval approach

Given Step 1 is unresolved, retrieval is designed against an abstract
interface, not a specific corpus, so the KB decision doesn't require
re-architecting anything:

- A small local vector index (e.g. `sentence-transformers` embeddings +
  cosine similarity, or even simple keyword/BM25 given how small an initial
  KB will be — dozens to low hundreds of snippets, not thousands) over
  short, atomic coaching-guideline passages, each tagged with metadata:
  `{stroke_class, topic (e.g. "contact_point", "footwork"), source
  ("authored" | "<citation>")}`.
- Retrieval query is constructed from the IR record being explained: for a
  `StrokeEvent`, something like `f"{stroke.predicted_class} coaching
  guidance relevant to {deviation_description}"`; for a `SessionRecord`,
  the top-k deviated features from `explanation` (already computed by
  `explain_anomaly()`).
- Top-k (k=2-3) retrieved passages are passed into the prompt, each with its
  `source` tag intact — so if/when option 3's authored placeholders are
  swapped for real licensed/open content later, the prompt template and
  grounding harness don't change, only the index contents and their `source`
  tags do.
- This is a design, not yet implemented — no index exists, because there's
  no content to index yet (Step 1).

### Prompt template design — structural grounding, not a style instruction

The naive approach ("please cite your sources") produces prose that's hard
to verify mechanically — checking whether a sentence is "grounded" would
require NLP parsing of free text, which is exactly the kind of fuzzy check
Step 4 needs to avoid. Instead, **the prompt requires the LLM to return
structured JSON matching `rag/schema.py:CoachingResponse`**, not free-form
text:

```python
class Citation(BaseModel):
    record_type: Literal["StrokeEvent", "SessionRecord"]
    record_id: str
    ir_field_path: str        # e.g. "features.unforced_error_rate"
    cited_value: float | int | str | bool

class CoachingFeedbackItem(BaseModel):
    text: str
    type: Literal["grounded_observation", "general_note"]
    citation: Citation | None  # required iff type == "grounded_observation"
```

Every substantive coaching claim must be a `grounded_observation` with an
explicit `Citation` naming the exact record and field it's based on, and the
exact value found there. Only non-factual filler (encouragement, framing)
may be a `citation`-free `general_note`. This is enforced two ways: (1)
Pydantic's `model_validator` on `CoachingFeedbackItem` rejects a
`grounded_observation` with no citation or a `general_note` with one at
parse time — a malformed LLM response fails construction, same "100%
compliance by construction" principle as the IR schema; (2) the actual
prompt instructs the model explicitly to use `general_note` sparingly and
never for anything containing a number or comparison. The system prompt
includes the actual serialized IR record(s) (`model_dump_json()`) as the
only source of truth the model is allowed to cite from — it isn't asked to
recall facts, only to reference values already in front of it.

### LLM choice — RESOLVED: Qwen3 8B (Q4_K_M, local via Ollama), not LLaMA 3.1 8B

**Original plan (this section, previous revision) was to prototype with the
Claude API first and treat local inference as a later cost-reduction step.**
That changed: no `ANTHROPIC_API_KEY` is available in this environment, and
rather than block on acquiring one, the user chose to stand up local
inference first and evaluate it on its own merits.

**Model substitution: Qwen3 8B instead of the proposal's LLaMA 3.1 8B.**
Both are open, ~8B-parameter, locally-quantizable models serving the same
role (free, on-device coaching-feedback generation) — the substitution is a
same-goal, better-current-fit swap, not a scope change: Qwen3 is a newer
model with generally stronger instruction-following on constrained,
context-only-answering tasks (exactly what the grounding requirement needs)
per current public benchmarks, discovered by checking what's actually
available now rather than assuming the proposal's original pick was still
the best option a year+ later. Set up via Ollama (recommended for ease of
setup): `ollama pull qwen3:8b` — Ollama's default tag resolved to **Q4_K_M**
quantization, confirmed via `ollama show qwen3:8b` (8.2B params, Q4_K_M).

**VRAM: confirmed comfortably within the 3060's 12GB.** Baseline usage
before loading: ~1.1GB. After loading + one generation: **6.5-6.6GB** — well
under budget, leaving headroom for a larger context window or a second
concurrent model if ever needed.

**Reliability check: 6 trials of the identical session-level coaching
prompt (same `SessionRecord`, same KB context, temperature 0.2), not a
single run.** The first two ad-hoc runs already showed why a single run
would have been misleading: run 1 was perfectly grounded; run 2, the very
next call with an *identical* prompt, fabricated a fake numeric `record_id`
and two feature names that don't exist anywhere in the schema
(`net_approach_pct`, `backhand_slice_pct`). A 4-trial follow-up batch
(`rag/run_qwen_multitrial.py`) was run to get an actual reliability read
rather than anchoring on either extreme:

| Trial | Result | Latency |
|---|---|---|
| 1 (initial) | 3/3 grounded (100%) | 64.2s |
| 2 (initial) | **0/6 grounded (0%)** — fabricated `record_id`, 2 fabricated feature names | 72.8s |
| 3 (batch) | Pydantic `ValidationError`: trailing item missing required `"type"` field | 29.9s |
| 4 (batch) | 3/3 grounded (100%) | 24.6s |
| 5 (batch) | 3/3 grounded (100%) | 30.5s |
| 6 (batch) | 3/3 grounded (100%) | 22.4s |

**4 of 6 trials fully clean (100% grounding, well-formed JSON); 1 of 6
completely hallucinated a record and two nonexistent fields (0%); 1 of 6
was well-grounded on its first 3 items but violated the output schema on a
trailing summary sentence (omitted `"type"`, causing a hard parse failure,
not a grounding failure).** Pooled across the 5 trials that parsed
successfully: 12/18 grounded_observation items passed (66.7%) — but that
single pooled number undersells what's actually happening, because the
failure isn't evenly distributed noise, it's binary per-trial (a trial is
either clean or it isn't): **3 distinct, honestly-different failure modes
observed, not one**:
  1. Clean and correct (4/6 trials).
  2. Full hallucination — wrong record, invented fields (1/6 trials).
  3. Schema malformation on a non-critical trailing item, first 3 items
     still correctly grounded (1/6 trials).

Latency was also bimodal: the two ad-hoc runs (64-73s) ran noticeably
slower than the 4-trial batch (22-31s, mean 26.8s) despite being warm in
both cases — plausibly explained by variable-length hidden "thinking" (Qwen3
is a reasoning model; Ollama separates `<think>` content from final output,
so longer/more uncertain reasoning traces don't show up in the visible JSON
but do cost wall-clock time), though 6 samples isn't enough to confirm that
correlation, only note it.

**Decision: adopt Qwen3 8B (Q4_K_M, local via Ollama) as the pipeline's
default LLM**, with a mandatory mitigation baked into the pipeline design,
not left as a known-but-unaddressed risk: **`rag/grounding.py`'s
`check_grounding()` must run as a runtime gate, not only an offline eval
tool.** Any generated response — Qwen3 or otherwise — that (a) fails to
parse against `CoachingResponse` or (b) scores below a grounding-rate
threshold (proposed: reject below 100% initially, since every observed
clean trial hit exactly 100% and the failures were binary, not marginal)
should trigger an automatic retry before ever being shown to a user. This
is cheap to implement (the harness and schema validator already exist,
built in the previous phase) and turns "4/6 raw reliability" into a much
higher *delivered* reliability, at the cost of occasional extra latency on
a retry — an acceptable trade given generation is already only ~25-70s and
this is not a real-time interaction. **This retry-gate is a required
follow-up before any user-facing generation, not yet implemented** — flagged
here rather than silently assumed.

Claude API remains unevaluated (no key configured) and is not part of this
decision, only deferred — `rag/llm.py:ClaudeBackend` implements the same
interface and can be evaluated on the identical prompt the moment a key is
available, with zero changes to `rag/prompt.py`, `rag/schema.py`, or
`rag/grounding.py`.

---

### Hosted-provider choice for cloud migration — RESOLVED: Fireworks AI

Phase 1 of the cloud-migration plan (`PROJECT_LOG.md` 2026-08-07) required
moving Qwen3 8B off the local 3060 onto a hosted inference provider, without
losing the reliability profile established above. Four candidates were
named up front (Groq, Together.ai, Fireworks.ai, DeepInfra) and each was
checked directly for real, current availability/pricing/quantization/latency
— not assumed from name recognition, since this decision has real monthly
cost attached and the four providers' catalogs change quickly.

**Availability check — this eliminated 3 of 4 candidates before pricing or
latency even mattered:**

| Provider | Hosts Qwen3 8B (dense)? | What they host instead |
|---|---|---|
| **Groq** | No | Qwen3 32B (deprecated 2026-06-17) and Qwen3.6 27B — never added the 8B size to their LPU-hosted catalog |
| **Together.ai** | No | Catalog has moved on to Qwen3.5-9B (a different, newer model generation) and larger variants — the 8B dense model isn't listed |
| **DeepInfra** | No | Qwen3-Max and Qwen3.5/3.6/3.8 MoE variants only — no 8B dense |
| **Fireworks AI** | **Yes** | Exact model match, confirmed on Fireworks' own model page (`accounts/fireworks/models/qwen3-8b`) |

Only Fireworks was actually evaluable on the other two axes as a result —
this is worth stating plainly rather than presenting it as a close 4-way
contest that Fireworks happened to win.

**Quantization — an honest caveat, not a match.** No hosted inference
provider offers literal **Q4_K_M** (the GGUF/llama.cpp k-quant format
Ollama uses locally) — that format is specific to local/edge serving
stacks, not something cloud providers like Fireworks/Together/DeepInfra
serve from (they use vLLM/TensorRT-LLM-style stacks with FP16/FP8/AWQ/GPTQ
instead). Fireworks serves Qwen3 8B at **FP8** — 8-bit floating point,
which is *higher* precision than 4-bit Q4_K_M, so this is a precision
upgrade relative to local, not a downgrade or an exact match. It's still a
different serving stack with different sampling defaults, which is exactly
why the confirmatory re-run (`rag/run_fireworks_confirmatory.py`) is
required before trusting this in production, not treated as a formality.

**Pricing (confirmed on Fireworks' own listing, flat rate, current as of
2026-08-07 — verify at signup, this space moves fast):** $0.20 / 1M input
tokens, $0.20 / 1M output tokens. At trial-scale volume (dozens–low
hundreds of coaching-generation requests/month, each call using a
SessionRecord + KB-snippet prompt of roughly 1.5-3K input tokens and a
structured-JSON response of a few hundred tokens), this comes out to
single-digit dollars per month, not the $150-400/month a self-managed
always-on cloud GPU would cost for the same workload.

**Latency (from Artificial Analysis' cross-provider Qwen3-8B benchmark,
independently sourced, not Fireworks' own marketing numbers):** **0.64s
time-to-first-token, 144.8 output tokens/sec** — the lowest TTFT of any
provider benchmarked for this model (next-best was Alibaba Cloud's own
first-party hosting at 2.15s TTFT / 38 t/s). Note this measures per-request
API latency, not the full ~22-73s wall-clock time the local Ollama setup
takes for one complete gated-pipeline generation (which includes Qwen3's
hidden reasoning-token generation) — the two numbers aren't directly
comparable, but a >3x faster TTFT and >3x higher token throughput than the
next-best benchmarked alternative is a reasonable basis to expect
meaningfully lower end-to-end latency, to be confirmed by the actual
confirmatory re-run's measured latency, not assumed from the TTFT number
alone.

**Decision: Fireworks AI, `accounts/fireworks/models/qwen3-8b`, FP8.** Not
because it won a close comparison against Groq/Together/DeepInfra on
latency or price — it was the only one of the four that hosts this exact
model at all. Full source list below.

Sources checked live during this decision (2026-08-07): Groq's own model
docs (console.groq.com) and deprecation notice; Together AI's serverless
model catalog docs (docs.together.ai); DeepInfra's models and pricing
pages (deepinfra.com); Fireworks' Qwen3 8B model page (fireworks.ai) and
Artificial Analysis' cross-provider Qwen3-8B-instruct provider comparison
(artificialanalysis.ai) for the independently-sourced latency/throughput
numbers.

---

### Retry-gate design and evaluation — IMPLEMENTED

The mandatory follow-up flagged above is now built: `rag/gated_pipeline.py`.

**Design.** `generate_grounded_response(backend, system_prompt, user_prompt,
source_records, max_attempts=3)` wraps a single `backend.generate()` call in
a check-and-retry loop:

1. Generate a response.
2. Attempt to parse it as a `CoachingResponse` (Pydantic). Parse failure →
   reject, retry.
3. Run `check_grounding()`. Accept only if `grounding_rate == 100%` for
   every `grounded_observation` item present — the same bar the reliability
   data above justified (every clean trial hit exactly 100%; failures were
   binary, never marginal, so a softer threshold like ">80%" would not have
   changed which trials passed).
4. Repeat up to `max_attempts` (default 3) full regenerations — same prompt,
   same temperature, no prompt modification between attempts, since the
   retries are purely exploiting Ollama's run-to-run sampling variance, not
   a corrected prompt.
5. If no attempt within the budget qualifies, return an explicit failure
   (`GatedResult(success=False, response=None, ...)`) with a
   `failure_reason` string. Callers MUST treat this as "could not generate a
   grounded response" — there is no fallback path that surfaces the last
   (rejected) attempt to a user under any circumstance.

This is a mechanical backstop, not a fix to the model's occasional
hallucination/malformation — it trades latency for a hard delivered-output
guarantee.

**Evaluation: the same 6-trial reliability test, re-run through the gate**
(`rag/run_gated_multitrial.py`, `max_attempts=3`), same prompt/KB
context/SessionRecord as the ungated 6-trial run above:

| Trial | Attempts | Outcome | Total latency |
|---|---|---|---|
| 1 | 3 | ACCEPTED (attempt 3) — attempts 1-2 both failed to parse (Pydantic `ValidationError`: missing `ir_field_path`/`cited_value`/`type` fields) | 192.6s |
| 2 | 1 | ACCEPTED (attempt 1), 3/3 grounded | 41.0s |
| 3 | 1 | ACCEPTED (attempt 1), 3/3 grounded | 31.3s |
| 4 | 1 | ACCEPTED (attempt 1), 3/3 grounded | 22.1s |
| 5 | 1 | ACCEPTED (attempt 1), 3/3 grounded | 23.6s |
| 6 | 1 | ACCEPTED (attempt 1), 3/3 grounded | 33.7s |

**Before/after result:**

- **Ungated: 4/6 trials clean (66.7%)** delivered directly to a user, with
  1/6 fully hallucinated and 1/6 schema-malformed both would have reached the
  user as-is.
- **Gated: 6/6 trials (100%) delivered a fully grounded, well-formed
  response** — zero ungrounded or malformed output reached the user-facing
  boundary. 1/6 trials needed retries (2 extra generations, both parse
  failures, not hallucinations this time — a reminder that "malformed" and
  "hallucinated" are separate failure modes and either can show up on a
  retry).
- **Cost: mean attempts per trial 1.33 (successes only), mean latency per
  trial including retries 57.4s** (vs. a raw single-shot mean of ~26.8s on
  the earlier clean batch) — total latency across all 6 gated trials was
  344.3s vs. what would have been roughly 6× a single-shot latency ungated.
  The one retried trial (192.6s) dominates this mean; the other 5 trials
  cost no more than an ungated call would have.
- This is only 6 trials — not enough to pin down a stable "retries needed
  X% of the time" figure, but it is enough to demonstrate the gate does
  exactly what it was built for: convert a 66.7% raw clean rate into a 100%
  delivered rate, at a bounded, occasional latency cost, with zero silent
  failures (every rejected attempt is visible in the per-attempt log, not
  swallowed).

Per-trial detail (attempts, latencies, accept/reject reasoning) saved under
`rag/example_output/qwen3_gated_trials/`.

---

### Retrieval accuracy across diverse anomaly types — validated 2026-07-18

The reliability numbers above (both ungated and gated) were all measured on
**the same repeated SessionRecord** — a real result, but it can't say
anything about whether retrieval generalizes, since the same 3 KB snippets
were retrieved every single time. Two things were needed to actually test
that: (1) the KB had to be big enough that retrieval is a real ranking
decision (Step 1 expansion, 6→24 entries, all 12 `ANOMALY_FEATURE_COLUMNS`
covered — see `rag/kb_placeholder.py`), and (2) retrieval had to be
semantic, not exact-keyword lookup, since a keyword match against a 1:1-keyed
KB isn't a meaningful test of ranking quality (see `rag/embeddings.py` /
`rag/prompt.py:retrieve_snippets()` above).

**Diverse SessionRecords.** `ir/populate.py:populate_session_records()`'s
first 3 flagged rows all happen to share the same dominant deviated
features (see `session_record_0/1/2.json` — `unforced_error_rate` and
`break_points_saved_pct` dominate all three), because the synthetic
"anomalous" archetype shifts all 12 features together. To get genuinely
different anomaly types, `ir/populate_diverse_sessions.py` crafts 6 rows
each targeting a distinct feature theme (serve breakdown, second-serve/ace
decline, movement/fatigue, rally-style shift, clutch/break-point
breakdown, general unforced-error spike), run through the same real fitted
`IsolationForest` + `explain_anomaly()` — not fabricated flags/explanations,
only the input rows are crafted. (Honest finding kept in that script's
docstring: a pure single-theme perturbation, however extreme, often isn't
enough to trip this particular fitted forest — 4 of the 6 profiles needed
an additional mild blend toward the real anomalous population's mean across
all features to get flagged `-1`, layered under a much larger targeted
override so the theme features still dominate the top-3 explanation.)
**All 6 crafted sessions were confirmed genuinely flagged anomalous
(`anomaly_flag=-1`) by the real model before being used.**

**Retrieval recall per session, first pass** (does `retrieve_snippets()`'s
top-3 result include a KB entry for every feature actually named in that
session's `explain_anomaly()` output?):

| Session | Explanation's top-3 features | Retrieved KB topics | Recall |
|---|---|---|---|
| clutch_breakdown | break_points_saved_pct, unforced_error_rate, avg_reaction_time_ms | same 3 | 3/3 (100%) |
| movement_fatigue | avg_reaction_time_ms, court_coverage_speed_mps, total_distance_covered_m | same 3 | 3/3 (100%) |
| serve_breakdown | first_serve_pct, first_serve_points_won_pct, double_fault_rate | same 3 | 3/3 (100%) |
| unforced_error_spike | unforced_error_rate, avg_reaction_time_ms, total_distance_covered_m | same 3 | 3/3 (100%) |
| rally_style_shift | winners_rate, avg_rally_length, unforced_error_rate | avg_rally_length, unforced_error_rate | 2/3 (67%) |
| second_serve_ace_decline | second_serve_points_won_pct, ace_rate, unforced_error_rate | first_serve_points_won_pct, unforced_error_rate | 1/3 (33%) |

First-pass mean: 15/18 (83.3%). Investigating the two misses found and fixed
two separate bugs, then left one genuine limitation undisturbed:

**Bug 1 (fixed) — dedup fallthrough.** `retrieve_snippets()` picked each
phrase's single highest-similarity match and, if that index was already
claimed by an earlier phrase, dropped the phrase entirely instead of
falling through to its next-best unclaimed candidate. This is why
`rally_style_shift` returned only 2 snippets instead of 3. Fixed by
ranking all 24 candidates per phrase and walking down until an unclaimed
one is found (`rag/prompt.py:retrieve_snippets()`).

**Bug 2 (fixed) — literal cross-topic mentions in authored KB text.**
Fixing bug 1 alone didn't fix `rally_style_shift`: with fallthrough
working, the `winners_rate` phrase's single best match across *all* 24
entries turned out to be an `unforced_error_rate` snippet, not either of
the two `winners_rate` snippets. Root cause: that snippet's authored text
literally contained the string `winners_rate` ("...compare against
winners_rate in the same session..."), and a small (45MB) embedding model
is lexically sensitive enough that the literal snake_case token match
outweighed genuine topical relevance. Audited the full 24-entry KB for the
same pattern (`any other topic's exact snake_case name appearing in a
snippet's text`, checked mechanically, not by eye) and found exactly one
other instance (`total_distance_covered_m`'s entry mentioning
`avg_rally_length`). Both were isolated, mechanical content fixes:
reworded the two cross-references in natural language instead of the raw
identifier (e.g. "winners_rate" → "how often those shots landed as clean
winners"). Re-verified directly (no LLM calls needed, retrieval is local
and fast): `rally_style_shift` now retrieves all 3 correct topics, and
`second_serve_ace_decline` incidentally improved too (1/3 → 2/3, now
correctly retrieving `ace_rate`).

**Final mean recall after both fixes: 17/18 (94.4%).** One miss remains,
deliberately left as a documented limitation rather than chased further:
`second_serve_ace_decline`'s `second_serve_points_won_pct` phrase retrieves
the `first_serve_points_won_pct` snippet instead of either
`second_serve_points_won_pct` snippet. This is neither of the two bugs
above — no literal cross-mention, no dedup collision — it's the embedding
model genuinely ranking "first serve points won" and "second serve points
won" as more similar to each other than either is to its own two
authored variants, on two short passages that share most of their
vocabulary and differ mainly in "first"/"second". Fixing this properly
would mean either a larger/less lexically-shallow embedding model, adding
a lightweight rerank step, or restructuring how first/second-serve KB
content is worded/tagged to be more distinguishable to a small model —
real design work, not a quick edit, so it's left as a known, understood
limitation rather than being iterated on further here.

**What the semantic-miss case actually produces.** Before fixing bug 2,
`second_serve_ace_decline`'s generated response (grounding 3/3, 100%) was:

```
"The player's second-serve points won percentage is 9.07..., which is 6.0
standard deviations below average." (cites features.second_serve_points_won_pct)
"The player's ace rate is 0.0, which is 3.2 standard deviations below
average." (cites features.ace_rate)
"The player's unforced error rate is 0.25..., which is 1.6 standard
deviations above average." (cites features.unforced_error_rate)
```

This answers a real question worth naming explicitly: despite retrieving a
less-relevant snippet (`first_serve_points_won_pct` framing instead of
`second_serve_points_won_pct` or `ace_rate` framing), the model did **not**
misapply the wrong-topic advice to the right numbers — it fell back to
bare, grounded fact restatement with essentially no coaching interpretation
at all. Every sentence here is a correct citation with no added
interpretive claim, so nothing in it is *false*, but the coaching value is
close to zero compared to the richer, interpretive sentences seen on
well-retrieved sessions (e.g. `clutch_breakdown`'s "...suggests
pressure-related issues, particularly on break points or late in service
games."). **Named limitation for the FYP writeup: the grounding harness
validates citation accuracy, not retrieval relevance — a response can score
100% grounded while being retrieval-degraded, and in the one case observed
here that degradation showed up as generic/thin output rather than
confidently-wrong advice, but the harness has no way to detect or penalize
that difference. Grounding rate and retrieval recall are two independent
reliability axes and must be reported separately, not conflated into one
"pipeline accuracy" number.**

**Gated pipeline result on these 6 diverse sessions, each using its own
freshly-retrieved (not hand-picked) KB context (first full pass, before the
two retrieval fixes above — grounding/gate behavior is retrieval-independent,
so these numbers stand):**

| Session | Attempts | Outcome | Latency |
|---|---|---|---|
| clutch_breakdown | 1 | ACCEPTED, 4/4 grounded | 68.3s |
| movement_fatigue | 1 | ACCEPTED, 4/4 grounded | 33.1s |
| rally_style_shift | 1 | ACCEPTED, 3/3 grounded | 64.0s |
| second_serve_ace_decline | 1 | ACCEPTED, 3/3 grounded | 72.8s |
| serve_breakdown | 1 | ACCEPTED, 3/3 grounded | 46.4s |
| unforced_error_spike | 1 | ACCEPTED, 4/4 grounded | 79.7s |

**6/6 (100%) gated success, 0/6 needed a retry**, mean 1.00 attempts/trial,
total latency 364.3s across all 6.

**One more honest data point, not smoothed over:** re-running
`rally_style_shift` alone afterward (to confirm the retrieval fix) hit the
gate's failure path for the first time in this project — all 3 attempts
failed to parse, each for the same recurring reason seen earlier in the
Qwen3 reliability evaluation (a trailing coaching-summary item omitting the
required `"type"` field), producing `GatedResult(success=False, ...)`. This
is not a regression from the retrieval fixes (retrieval and generation are
independent, and the retrieval fix was verified separately with no LLM
call) — it's the retry-gate's designed failure state actually firing in
practice on real generation variance, confirming that path is real and
reachable, not just a defensive branch that never triggers. A production
caller hitting this would need to surface "could not generate a grounded
response" rather than any output, exactly as designed.

Per-session results, retrieval decisions, and grounding reports saved under
`rag/example_output/qwen3_gated_diverse_trials/`.

---

## Step 4: Grounding validation harness — built and demonstrated now

**This exists as real, tested code already** (`rag/schema.py`,
`rag/grounding.py`, `rag/example_grounding_check.py`) — no LLM call, no KB,
no retrieval code needed to build or prove it, because it's designed to be
independent of how a `CoachingResponse` was produced.

### Design

`grounding_rate` is defined mechanically, not via NLP judgment:

```
grounding_rate = (# grounded_observation items that pass all 3 checks)
                / (# grounded_observation items total)
```

For each `grounded_observation` item, `check_grounding()`
(`rag/grounding.py`) runs three checks in order, any failure stopping at
that check:

1. **Record exists** — the citation's `(record_type, record_id)` must be
   among the real IR records this feedback was actually generated from
   (passed in explicitly by the caller, never inferred/trusted from the
   LLM's own claim) → else `WRONG_RECORD`.
2. **Field resolves** — `resolve_field_path()` walks the citation's dotted
   `ir_field_path` (supports nested keys and list indices, e.g.
   `"ball.trajectory[3].pixel.x_px"`) against that record's real
   `model_dump()` dict → else `FIELD_NOT_FOUND`.
3. **Value matches** — the resolved real value must match `cited_value`
   (numeric within relative/absolute tolerance, exact for strings/bools) →
   else `VALUE_MISMATCH`.

All three pass → `GROUNDED`. `general_note` items are excluded from the
rate entirely (they're not claims) but are scanned for a stray digit as a
soft `SUSPECT_GENERAL_NOTE` warning — catches a factual claim that dodged
the citation requirement by mislabeling itself as non-factual, which the
hard rate alone wouldn't catch.

### Demonstration (`rag/example_grounding_check.py`)

Run against one real `SessionRecord` (`ir/example_output/session_record_0.json`
— real `IsolationForest` + `explain_anomaly()` output, see the IR layer
closeout) and a **hand-written, deliberately adversarial**
`CoachingResponse` (not LLM-generated — the point here is proving the
checker works, not showing off a generation result) exercising all six
verdict paths: one correctly grounded claim, one with a fabricated value
(cites `0.01` where the real `double_fault_rate` is `0.159`), one citing a
field that doesn't exist (`first_serve_spin_rpm`), one citing a record never
provided to the harness, one clean general note, and one general note that
smuggles in a number. Result, exactly as designed:

```
grounding_rate = 1/4 grounded_observation items pass (25.0%)
WARNING: 1 general_note item(s) contain a digit with no citation

  [GROUNDED] ... features.unforced_error_rate=0.3397... confirmed
  [VALUE_MISMATCH] ... cited 0.01, actual value at 'features.double_fault_rate' is 0.159...
  [FIELD_NOT_FOUND] ... key 'first_serve_spin_rpm' not found ...
  [WRONG_RECORD] ... no known SessionRecord with id=sess_last_tuesday_not_provided ...
  [OK_GENERAL_NOTE] 'Keep up the consistent effort ...'
  [SUSPECT_GENERAL_NOTE] "You've now had 4 sessions in a row with elevated error rates."
```

A deliberately bad 25% score on an adversarial example is the right result
here — it demonstrates the harness *catches* fabrication, mismatch,
misattribution, and mislabeling, which is the property that makes a future
">95%" claim on real generated feedback actually mean something, rather than
being asserted on the honor system.

---

## Not yet built

- Higher-authority KB content (Step 1 options 1/2 — licensed ITF/USTA
  access, or vetted openly-licensed sources). `rag/kb_placeholder.py` is now
  a 24-entry hand-authored, explicitly "authored not sourced" set covering
  all 12 session-level `ANOMALY_FEATURE_COLUMNS` topics (2 entries each) —
  a real, exercised placeholder, still not a resolution of the content-
  authority question if that's ever pursued.
- Stroke-level biomechanical benchmarking — formally DEFERRED (see Step 2
  "Decision — RESOLVED" above), blocked on the schema gap (no biomechanical
  fields on `StrokeEvent`, no derivation code from the proven MediaPipe
  Pose landmarks) and a benchmark source, neither pursued in this phase.
- Claude API evaluation on the same prompt (`rag/llm.py:ClaudeBackend`
  exists and is interface-complete, but untested — no API key configured).
- Any pipeline entry point that takes an arbitrary `SessionRecord`/
  `StrokeEvent` end to end and returns feedback to a caller — everything so
  far is evaluation/comparison scripts (`rag/run_qwen_eval.py`,
  `rag/run_qwen_multitrial.py`), not a callable pipeline function.

## What's now built and verified

- `rag/schema.py` — structural citation-grounding schema
  (`CoachingResponse` / `CoachingFeedbackItem` / `Citation`), enforced at
  construction time.
- `rag/grounding.py` — the grounding-rate checker, field-path resolver,
  and per-item verdict reporting.
- `rag/example_grounding_check.py` — adversarial demonstration proving the
  checker correctly identifies all 6 verdict categories against one real IR
  record.
- `rag/kb_placeholder.py` — hand-authored placeholder KB, expanded
  2026-07-18 to 24 entries (2 per topic) covering all 12
  `ANOMALY_FEATURE_COLUMNS`, session-level topics only, clearly labeled as
  non-authoritative.
- `rag/embeddings.py` — `OllamaEmbedder` (local `all-minilm`, 45MB, via
  Ollama's `/api/embed`) + cosine similarity, added 2026-07-18 to replace
  keyword-matching retrieval with real semantic ranking.
- `rag/prompt.py` — shared prompt construction (retrieval + schema
  instructions), used identically regardless of backend so a model
  comparison actually compares models, not prompts. `retrieve_snippets()`
  now embeds each SessionRecord deviation phrase and ranks all 24 KB
  entries by cosine similarity (previously: exact keyword/topic-string
  match) — see "Retrieval accuracy across diverse anomaly types" below.
- `rag/llm.py` — `OllamaBackend` (used for the Qwen3 evaluation) and
  `ClaudeBackend` (interface-complete, untested), behind one shared
  `generate(system_prompt, user_prompt) -> GenerationResult` interface.
- **Qwen3 8B (Q4_K_M) installed via Ollama, confirmed loading and generating
  within the RTX 3060's 12GB VRAM (6.5-6.6GB used), and evaluated across 6
  real generation trials of the actual session-level coaching prompt against
  the actual grounding harness** — see "LLM choice" above for the full
  reliability breakdown (4/6 clean, 1/6 hallucinated, 1/6 schema-malformed)
  and the resulting decision to adopt it with a mandatory retry-gate.
  Raw outputs and per-trial grounding reports saved under
  `rag/example_output/qwen3_trials/`.
- `rag/gated_pipeline.py` — the retry-gate itself
  (`generate_grounded_response()`), and `rag/run_gated_multitrial.py` — the
  same 6-trial reliability test re-run through the gate. Result: 6/6
  (100%) trials delivered a fully grounded, well-formed response (vs. 4/6
  ungated), 1/6 needed retries, mean latency per trial including retries
  57.4s. See "Retry-gate design and evaluation" above for the full
  before/after table. Per-trial results saved under
  `rag/example_output/qwen3_gated_trials/`.
- `rag/kb_placeholder.py` expanded to 24 entries (all 12
  `ANOMALY_FEATURE_COLUMNS`), `rag/embeddings.py` (local `all-minilm`
  semantic retrieval), `ir/populate_diverse_sessions.py` (6 SessionRecords
  with genuinely distinct anomaly profiles, confirmed anomaly-flagged by
  the real `IsolationForest`), and `rag/run_gated_diverse_trials.py` — the
  gated pipeline re-run across all 6 diverse sessions with real per-session
  retrieval. Result: 6/6 (100%) gated success (one later isolated re-run
  did hit the gate's failure path — see below), mean retrieval recall
  17/18 (94.4%) after fixing two retrieval bugs found during this
  evaluation (a dedup fallthrough bug, and literal cross-topic mentions in
  2/24 KB entries confusing the embedding model). One residual retrieval
  limitation (first-serve/second-serve confusion) documented, not chased
  further. See "Retrieval accuracy across diverse anomaly types" above for
  the full breakdown, including what a grounded-but-retrieval-degraded
  response actually looks like. Per-session results saved under
  `rag/example_output/qwen3_gated_diverse_trials/`.

---

## RAG integration: stroke-level biomechanics coaching (2026-07-19)

Extends the RAG coaching layer to consume `ir/biomechanics_percentiles.py`'s
percentile data, now that the biomechanics feature-extraction phase is at
its own scoped completion point (see `PROJECT_LOG.md`). Same discipline as
every phase so far: design and flag open questions before building
generation code.

### Step 1: Scope check — does stroke-level coaching reuse the existing machinery cleanly? Partially, not fully

**What reuses completely unchanged, verified not just assumed:**
- `CoachingResponse` / `CoachingFeedbackItem` / `Citation` (`rag/schema.py`)
  — `Citation.record_type` already included `"StrokeEvent"` from the
  original design, so no schema change was needed there.
- `check_grounding()`'s `WRONG_RECORD` / `FIELD_NOT_FOUND` / `VALUE_MISMATCH`
  checks — a citation of a RAW biomechanics field (e.g.
  `"biomechanics.wrist_height_ratio"`) is just another field inside
  `StrokeEvent.model_dump()`, resolved by the same generic
  `resolve_field_path()` used for every other IR field. No change needed.
- The retrieval ranking algorithm itself (embed phrases, rank all KB
  entries by cosine similarity, dedup-with-fallthrough) — refactored out
  of `rag/prompt.py:retrieve_snippets()` into a KB-agnostic
  `retrieve_snippets_for_phrases(phrases, kb, query_prefix, ...)`, with
  `retrieve_snippets()` kept as a thin session-level wrapper. Verified
  byte-for-byte that this refactor changed nothing about session-level
  behavior: `SCHEMA_INSTRUCTIONS`'s exact string is checked equal to the
  pre-refactor original, and `retrieve_snippets()` was re-run against
  `session_record_diverse_rally_style_shift.json`, reproducing the exact
  same 3/3 topics as before the refactor.

**What does NOT reuse cleanly — found before writing any generation code, per the requested discipline:**

1. **Percentile citations require a genuinely different verification
   operation, not a field-path lookup.** A raw value citation
   (`"biomechanics.wrist_height_ratio" = 5.65`) is checked by resolving a
   static path — but a percentile claim (`"93rd percentile for serve"`)
   isn't a field stored anywhere in `StrokeEvent`; per the schema
   decision, percentiles are deliberately NOT baked into
   `BiomechanicsInfo`. Verifying one means RECOMPUTING it via
   `BiomechanicsPercentileBenchmark.percentile(value, stroke_class,
   feature)` and comparing against what was cited — a function call
   against a live reference population, not a dict lookup. Resolved by
   adding `Citation.percentile_of: str | None = None` (default `None`
   preserves all existing behavior exactly) and a new branch in
   `check_grounding()` that recomputes and compares when it's set,
   keyed off the same record's own `stroke.predicted_class`.
2. **The record's ID field has a different name.** `SessionRecord` uses
   `record_id`; `StrokeEvent` uses `event_id`. The original
   `SCHEMA_INSTRUCTIONS` hardcoded the literal string `"record_id"`
   throughout — reusing it verbatim for `StrokeEvent` would have
   instructed the LLM to copy a field name (`record_id`) that doesn't
   exist in the JSON it's given. Resolved by parameterizing
   `SCHEMA_INSTRUCTIONS` into `build_schema_instructions(record_type,
   record_display_name, id_field_name, feature_scope_rule,
   include_percentile_rule)`, with `SCHEMA_INSTRUCTIONS =
   build_schema_instructions()` (SessionRecord defaults) verified
   byte-identical to the original fixed string.
3. **There is no upstream "which feature(s) are worth mentioning" signal
   for a stroke, unlike a session.** `SessionRecord.explanation` is
   precomputed by Model 4's `explain_anomaly()` — the anomaly-detection
   step itself decides and stores the top-k deviated features. A
   `StrokeEvent`'s `biomechanics` has no equivalent upstream flagging
   step (every stroke has all 8 ranked percentiles; nothing says which
   are notable). Built `select_notable_biomechanics_features()`
   (`rag/stroke_prompt.py`) as the missing piece: ranks the
   `RANKED_FEATURES` present on the event by `|percentile - 50|`
   (furthest-from-the-median first) and takes the top-k — the
   percentile-domain analog of a z-score-magnitude ranking, built fresh
   because there was nothing to reuse here, not because reuse was
   rejected.

**Bottom line: the citation/grounding/retrieval INFRASTRUCTURE reuses
almost completely; the citation-of-a-DERIVED-STATISTIC case (percentile)
and the feature-selection step are the two genuinely new pieces, both
scoped and built below, not assumed away.**

### Step 2: KB expansion — `rag/kb_biomechanics.py`, scoped to exactly the 7 validated fields

14 entries (2 per topic) covering only the fields `ir/DESIGN.md`'s final
usability table marked validated/percentile-ranked: `wrist_height_ratio`,
`weight_transfer_ratio`, `knee_flex` (one combined topic for left+right),
`stance_width_ratio`, `follow_through_extension_ratio`, `elbow_angle_deg`,
`shoulder_hip_separation_deg`. **Deliberately no entries for the 3 excluded
rotation fields** — there's nothing groundable to say about a field that
can never be cited with a real percentile.

**`elbow_angle_deg` and `shoulder_hip_separation_deg` entries are written
generically, not around a stroke-class pattern** — per the full-dataset
finding that both fields' one striking small-sample pattern (forehand-vs-
backhand elbow contrast; "serve has low X-factor") was contradicted by the
real population. Their KB text discusses what the feature measures and
how to talk about a percentile-relative observation, and explicitly does
NOT say anything like "servers typically show X" for either field, so a
retrieved snippet can't smuggle a now-known-false claim into generated
coaching text.

### Step 3: Grounding harness extension — tested adversarially before trusting it

`rag/example_percentile_grounding_check.py` (same pattern as the original
`rag/example_grounding_check.py`) constructs a hand-written
`CoachingResponse` against a real `StrokeEvent`
(`stroke_event_biomech_serve.json`) with 5 cases and asserts the exact
expected verdict sequence:

| # | Case | Expected verdict | Actual |
|---|---|---|---|
| 1 | Correct percentile citation (real 92.7th recomputed) | `GROUNDED` | `GROUNDED` |
| 2 | Fabricated percentile (`50.0` cited, real is `92.7`) | `PERCENTILE_MISMATCH` | `PERCENTILE_MISMATCH` |
| 3 | Percentile claimed for an EXCLUDED field (`shoulder_rotation_deg`) | `PERCENTILE_FIELD_NOT_RANKED` | `PERCENTILE_FIELD_NOT_RANKED` |
| 4 | Percentile citation naming a record never provided | `WRONG_RECORD` | `WRONG_RECORD` |
| 5 | `percentile_of` omitted, cited value is the percentile not the raw value | `VALUE_MISMATCH` | `VALUE_MISMATCH` |

All 5 passed exactly as expected (`assert`-verified, not eyeballed). Also
confirmed calling `check_grounding()` WITHOUT a `percentile_benchmark`
correctly rejects a percentile citation as `PERCENTILE_BENCHMARK_MISSING`
rather than silently passing or crashing — a caller can't accidentally
skip verification by forgetting to pass the benchmark.

### Step 4: Validation on real data

Built 4 real `StrokeEvent`s with real `BiomechanicsInfo` attached
(`ir/populate_biomechanics_stroke_events.py`) — one per stroke class
(forehand, backhand, serve, volley), same COMPOSITE pattern already
established in `ir/populate.py` (real-but-unrelated ball/court source
clips), with the one addition that `stroke` and `biomechanics` here come
from the SAME real THETIS pose sequence, not two unrelated sources. Only
`clean`-bucket clips used (per the full-dataset run).

**Real finding surfaced while picking examples, not hidden:** the first
clean test-split example for `forehand` and `backhand` were each
misclassified by StrokeGRU (predicted `volley` and `forehand`
respectively — a real consequence of the classifier's 83.4% test
macro-F1, not a bug in this validation). Swapped to alternate clean
examples where `stroke.predicted_class` matches the true label, so this
validation exercises 4 genuinely distinct predicted classes. **Flagging
this as a real, load-bearing limitation for stroke-level coaching**:
percentile ranking keys off `stroke.predicted_class` (the only thing
available at inference time — there is no ground truth to fall back on
in production), so a misclassified stroke gets percentile-ranked against
the WRONG class's reference population, and nothing in the grounding
harness can catch this (it's a semantically wrong-but-internally-
consistent citation, not a fabricated one) — the same shape of problem as
the RAG layer's grounding/retrieval-relevance decoupling finding, one
level up the stack: **grounding checks correctness against the record as
given; it cannot check whether the record's own upstream classification
was correct.**

**Gated pipeline result: 0/4 (0%) — but driven by a different, more
fundamental failure mode than anything seen in session-level testing, not
citation fabrication.**

Two real, quickly-diagnosed-and-fixed prompt bugs were found and fixed
along the way (both confirmed via re-running, same discipline as every
other bug in this project):

1. **Citation JSON-key mismatch (first run, 4/4 trials, 100% parse
   failure, every single attempt).** The prompt's `id_field_name`
   parameter was used both to say WHERE to copy the ID value from
   (`STROKE_EVENT`'s own `"event_id"` field — correct) and, by the same
   token substitution, to name the CITATION OBJECT'S OWN JSON key
   (`"event_id"` instead of the fixed `"record_id"` `Citation` actually
   requires) — so the LLM correctly followed the (buggy) instructions and
   omitted `record_id` every time. Fixed by decoupling the two concepts in
   `build_schema_instructions()`: the citation's JSON key is always
   literally `"record_id"`; only the source-field name it copies FROM
   varies. Verified the SessionRecord path (where the two names already
   coincided) stays byte-identical.
2. **Missing comma in the percentile-field JSON example** (a second,
   smaller bug caught in the same pass, before re-running) — fixed.

**After both fixes, re-ran: still 0/4, but the dominant failure mode
changed completely.** Every attempt across every stroke class now
abandons the required `{"items": [...]}` schema entirely and invents its
own ad-hoc structure instead — e.g. (real raw output, backhand, attempt
1):

```json
{
  "stroke_class": "backhand",
  "biomechanical_metrics": {
    "knee_flex_right_deg": {"value": 123.4, "percentile": 98.8},
    ...
  },
  "coaching_insights": [
    {
      "metric": "knee_flex_right_deg", "percentile": 98.8,
      "interpretation": "Your right knee flexion is exceptionally high (98.8th percentile), indicating a highly compact leg drive. This could enhance power but may increase strain on the knee joint if not balanced with proper core engagement.",
      "source": "coaching_snippet_1"
    }, ...
  ]
}
```

Other attempts invented yet other shapes (`"feedback": [...]`, `"stroke":
{"class": ...}`) — not one consistent wrong schema, a different invented
one nearly every attempt, across every stroke class and every retry. This
is a materially different failure mode than the session-level evaluation
ever showed (which mostly followed the correct schema, with occasional
hallucinated content within it) — worth flagging as structurally
surprising rather than pushed past.

**Content quality observation, worth keeping separate from the schema
failure:** the free-form text the model produces when it abandons the
schema is NOT the "thin/generic" pattern seen in the session-level
semantic-miss case — it's specific, percentile-grounded, and
substantively useful coaching language (see the `coaching_insights`
example above). **The model is not struggling to produce good stroke
coaching content; it is struggling to package that content into the
required schema for this prompt specifically.** This matters for how the
finding should be read: this is a schema-compliance problem, not a
content-quality or grounding-accuracy problem, and the two must not be
conflated when reporting this result.

**Working hypothesis for the root cause, checked against real numbers,
not asserted:** the stroke-level user prompt is **~3.8x larger** than the
session-level one (10,364 vs. 2,755 chars) because the full
`StrokeEvent` JSON shown to the model includes a complete ball trajectory
(12 frames, each with pixel/court coordinates, visibility, confidence) and
14 court keypoints plus a homography matrix — none of which biomechanics
coaching needs or is allowed to cite (rule 5 restricts citations to the
`biomechanics` object) — versus `SessionRecord`'s compact ~12-scalar
`features` dict. A large, mostly-irrelevant JSON blob, combined with an
already-well-structured "NOTABLE BIOMECHANICAL PERCENTILES" summary block
placed right before the schema instructions are meant to govern, is a
plausible reason a smaller local model would drift toward inventing its
own natural-feeling response shape instead of following the meta-schema
strictly. Not proven by these 4 trials alone (n=4, no controlled
ablation run), but consistent with what's actually in the prompt.

**Stopping here to flag this rather than pushing further with more prompt
iterations unilaterally, per the explicit instruction for this phase.**
A plausible next step (trimming the STROKE_EVENT JSON shown in the prompt
to just the fields biomechanics coaching can actually cite, while keeping
the full record available to the grounding harness for WRONG_RECORD/
FIELD_NOT_FOUND checks) is a real, scoped fix worth trying — but it is a
big enough prompt-content change, with its own design questions (does
trimming what's SHOWN diverge from what's CITABLE in a way that breaks
the "the prompt is the only source of truth" principle this project has
held since Step 3's original design?), that it's flagged for a decision
rather than silently implemented.

### Trimming the prompt (chosen next step, implemented and validated)

Chose to trim: `rag/stroke_prompt.py:_build_citable_view()` builds a
reduced dict (`event_id`, `player_side`, `stroke.predicted_class`/
`confidence`, `biomechanics.*`) for what's SHOWN in the user prompt —
removing the ball trajectory, court keypoints/homography, and
`class_probabilities`, none of which biomechanics coaching can cite. **The
"prompt is the only source of truth" principle is preserved, not
compromised:** the grounding harness's `source_records` still receives the
FULL, untrimmed `StrokeEvent` (unchanged in `rag/run_gated_stroke_trials.py`)
— trimming affects only what the model is shown, never what's actually
checkable, so a citation to a field outside the trimmed view would still
correctly resolve (or correctly fail) against the real record. User prompt
size dropped from 10,364 to 3,405 chars (~3x smaller).

**Re-ran: schema compliance recovered almost completely** (parse failures
dropped from 12/12 attempts to a small minority), confirming the
prompt-size/irrelevant-context hypothesis. **But a new, different issue
surfaced immediately: every parsed response cited the RAW value while
still correctly setting the (then-existing) `percentile_of` flag** — e.g.
citing `0.147` (the raw `wrist_height_ratio`) while `percentile_of:
"wrist_height_ratio"` was set and the *text* correctly said "14.5th
percentile." Diagnosed via a live raw-output inspection, not assumed: the
original design asked the model to override `cited_value`'s meaning based
on a *separate* field elsewhere in the same object — an exception to the
"`ir_field_path` resolves to `cited_value`" invariant this schema relies
on everywhere else — and the model reliably defaulted back to that
invariant's normal meaning (cite what's really there) regardless of the
flag. **Fixed by folding the percentile signal into `ir_field_path`
itself** (`"biomechanics_percentile.<feature>"` vs.
`"biomechanics.<feature>"`), removing the separate `percentile_of` field
from `Citation` entirely (`rag/schema.py`) and updating
`check_grounding()` to route on the path prefix (`rag/grounding.py`). This
keeps a single invariant with no exception for the model to break: the
value always means what the path says it means, just resolved two
different ways depending on prefix. **Re-ran the adversarial test
(`rag/example_percentile_grounding_check.py`) under the new design — all 5
expected verdicts still confirmed exactly**, including a new case
(percentile number cited against a raw-value path, correctly caught as
`VALUE_MISMATCH`).

**Final gated pipeline result after both real fixes: 4/4 (100%) success,
0/4 needed a retry**, mean 1.00 attempts/trial, total latency 132.7s
across all 4 (21.4s–43.6s per trial) — a clean result, not a marginal one.

| Stroke | Attempts | Grounded | Latency |
|---|---|---|---|
| backhand | 1 | 3/3 | 43.6s |
| forehand | 1 | 3/3 | 41.5s |
| serve | 1 | 3/3 | 21.4s |
| volley | 1 | 3/3 | 26.2s |

**Spot-check of the actual generated text — explicitly checking for the
thin/generic-vs-useful distinction found in the session-level
semantic-miss case, not just trusting the 100% grounding rate:**

All 12 grounded items across the 4 strokes are substantively specific and
interpretive, not bare number restatement — e.g. serve's "Your
shoulder-hip separation angle (1.98°) is at the 4.8th percentile for
serves, indicating a highly compact upper body rotation **relative to
peers**" correctly frames the observation as percentile-relative, never
claiming an absolute "servers typically show X" pattern — exactly the
generic, non-claim framing the Step 2 KB entries for this exact field
were deliberately written to encourage, given that pattern was falsified
at full-dataset scale. Forehand's elbow-angle item similarly hedges
("may indicate a compact swing... could be intentional or worth
exploring") rather than asserting a fault, again matching the
deliberately generic KB text.

**One real overclaiming case found, worth flagging on its own — the
grounding pass rate alone would not have caught it:** volley's
`weight_transfer_ratio` item states "suggesting a highly proactive weight
shift **toward the ball direction** during contact." The percentile
citation itself (88.4th) is correctly grounded — but "toward the ball
direction" is a specific directional claim the underlying measurement
cannot actually support: `rag/kb_biomechanics.py`'s own
`weight_transfer_ratio` entries explicitly caveat that this feature's
sign depends on camera framing per clip and isn't independently verified
as consistent, so no directional claim beyond the raw magnitude is
justified. **This is the stroke-level echo of the session-level
grounding/retrieval-relevance finding, in a new form: the grounding
harness verifies the CITED NUMBER, not the interpretive gloss the model
layers on top of a correctly-cited number** — a response can be 100%
grounded while still containing an unsupported directional/causal claim
the number alone doesn't establish. Not a reason to distrust the 100%
result (the citation itself is genuinely correct), but a concrete
reminder that grounding rate and full-claim accuracy are still two
different things, one level further into this layer than the original
finding.
