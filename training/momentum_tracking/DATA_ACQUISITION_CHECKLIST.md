# Model 3 (Momentum Tracking) — data acquisition checklist

`TennisMomentumLSTM` in `model.py` is a definitions-only scaffold. Nothing in
this folder trains, and nothing should until the items below are satisfied —
per the project rule of inspecting real files before writing a `Dataset` class,
extended here to: don't build a `Dataset` class at all until real files exist
to inspect.

## Feasibility check against existing Tennalysis MCP ingestion (2026-07-28)

Before scoping new acquisition, checked whether the Match Charting Project (MCP)
data already ingested for the sibling Tennalysis-web project (`D:\Tennalysis-web`,
a separate repo — not part of this VisioTennis-app repo, hence the earlier
2026-07-17 "no point-by-point data anywhere in the local repo" note in
`PROJECT_LOG.md` was correct for *this* repo but incomplete for the user's
wider project set) already covers what Model 3 needs. Verified directly against
the live data (raw CSVs in `D:\Tennalysis-web\data\raw\mcp\`, the
`charted_match_stats` table via a live DB query, and the stage-5 loader
`data_pipeline/stages/load_charted_stats.py`) rather than assumed from the row
count alone.

**What the 10,022 rows actually are:** one row per (charted match, player) in
the Postgres `charted_match_stats` table — confirmed via `SELECT COUNT(*)`
(10,022 exactly) and `SELECT COUNT(DISTINCT charted_match_id)` (5,011 distinct
matches × ~2 players each). Provenance, traced through the pipeline:
- `data/raw/mcp/charting-m-matches.csv`: 7,566 charted matches (raw MCP index).
- 5,027 of those reconciled to a known ATP `match_id` (`data/processed/mcp_reconciliation.parquet`,
  built by stage 4 `reconcile_mcp.py`).
- 5,011 of those had a usable `Overview.csv` "Total" row (stage 5 skips
  matched-but-statless slots) → 10,022 player-match rows.

**What's in the 5 raw MCP files actually downloaded** (`charting-m-matches.csv`,
`charting-m-stats-Overview.csv`, `-NetPoints.csv`, `-ServeBasics.csv`,
`-KeyPointsServe.csv`, `-KeyPointsReturn.csv`): every one of these is a
**match/set/situational-bucket aggregate**, not a point-by-point log —
confirmed by inspecting real rows, not just column names:
- `Overview.csv`: totals **per set** (`set` ∈ {1..5, "Total"}) — aces, DFs,
  serve/return points won, winners, unforced errors, etc. as *counts*.
- `NetPoints.csv` / `ServeBasics.csv`: totals per serve number (1st/2nd) or
  overall — same shape, still counts.
- `KeyPointsServe.csv` / `KeyPointsReturn.csv`: totals bucketed by situational
  category (`row` ∈ {`BP`, `GP`, `Deuce`, `STotal`} = break point / game point
  / deuce / set total) — e.g. "8 of 11 break points won" for a whole match.
  This is the closest thing to a "pressure" signal in the data, but it's a
  match-level aggregate *within* a bucket, not a record of any individual
  point, its position in the match, or its order relative to other points.

None of these files contain a `Pt`/`Gm#`/`Svr`/`PtWinner`-style per-point row
(the actual MCP point-by-point log format, distributed by the same source
project as separate `charting-m-points*.csv` files) — that file type was never
downloaded into `data/raw/mcp/`, and there is no rally/point table anywhere in
`db/migrations/001_init.sql` to hold it. Tennalysis's own Milestone 4
point-probability work (`ml/point_probabilities.py`) confirms this
independently: it derives per-player serve/return point-win *probabilities*
from these same match-level aggregates and a Bayesian-shrinkage/opponent-
adjustment model, then feeds a Markov-chain match *simulator*
(`ml/match_simulator.py`) — i.e., Tennalysis itself had to model around the
absence of real point sequences rather than read them off the data, which is
the same wall Model 3 hits.

**Answers to the 5 scoping questions:**

1. **Can point-level sequence/outcome be reconstructed?** No. Nothing in the
   ingested data records which player won a given point, in order, linked to a
   running score/set/server state. The `BP`/`GP`/`Deuce` buckets in
   KeyPointsServe/Return give *aggregate* pressure-point performance per match,
   not a sequence.
2. **How many matches, and is coverage complete or partial?** 10,022 rows /
   5,011 distinct matches. Coverage is "complete" only in the sense that each
   matched-and-statted charted match has a full Overview total — but that's
   complete *aggregate* coverage, not partial point-by-point coverage, because
   no point-by-point data was ingested at all (zero matches have it, not "some
   matches have partial coverage").
3. **Which of the 3 candidate label approaches are computable from this data?**
   None, as scoped (all three require a per-point sequence). Approach #2's
   prerequisite win-probability model could in principle reuse the
   *distributional* per-player point-win probabilities Tennalysis already
   built (`ml/point_probabilities.py`) as a prior, but that only estimates
   *how often* a player wins a point of a given type in general — it still
   provides no way to know the order or context of specific points within a
   specific match, so it doesn't unblock building a real per-point momentum
   *label* from observed match data. Approaches #1 and #3 need actual per-point
   sequences and have no substitute here.
4. **If sufficient (even partially), how many usable match-sequences?** Zero.
   Not a small pilot — a true zero, at the point-sequence granularity Model 3's
   `TennisMomentumLSTM` is built for (per-point timesteps). This is a hard gap,
   not a small one.
5. **What's precisely missing:** the actual MCP point-by-point log files
   (`charting-m-points.csv` and its decade-split variants, in the same source
   project — `JeffSackmann/tennis_MatchChartingProject` on GitHub — that
   `charting-m-matches.csv` and the stats files already came from). This is a
   concrete, scoped acquisition target, not a vague "need more data": pull
   those files into `D:\Tennalysis-web\data\raw\mcp\`, inspect their real
   schema (2-3 files) the same way every other model's data was inspected in
   this project, and add a `points` table + loader stage that joins to the
   existing `mcp_reconciliation.parquet` mapping (so points can resolve to
   real ATP `player_id`s the same way `load_charted_stats.py` already does).
   Since coverage of the *points* file is historically known to be lower than
   the stats files for some matches (not verified here — flag for whoever
   does the acquisition to check once the file is in hand), expect somewhat
   fewer than 5,011 usable matches once pulled, not automatically the same
   count as the stats-based training set.

**Recommendation: acquisition is still required**, with a specific,
scoped target — not a "go find more data" search. This is a same-source
follow-up download (same MCP project, additional file type), not a new data
source to identify, and it plugs into pipeline infrastructure (reconciliation
mapping, player-id resolution) that already exists in Tennalysis-web. Proceed
with checklist items 1-5 below using `charting-m-points*.csv` as the concrete
candidate source, starting with item 2 (schema inspection of 2-3 real files)
once those files are pulled in.

## Schema confirmation: real charting-m-points files (2026-07-28)

Pulled 3 real files directly from the source repo
(`raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master/`)
into the scratchpad for inspection — not committed anywhere, no ingestion code
written yet, per instruction:
- `charting-m-points-2020s.csv` (56.6MB, 547,478 point rows, 3,337 matches)
- `charting-m-points-2010s.csv` (36.5MB, 357,229 point rows, 2,232 matches)
- `charting-m-points-to-2009.csv` (38.0MB, 379,569 point rows, 1,963 matches)

**Confirmed schema** (header, verified against real rows, not just the column
names): `match_id, Pt, Set1, Set2, Gm1, Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd,
Notes, PtWinner`. This is genuinely a per-point log, in order:
- `Pt`: point number within the match — increases monotonically within each
  match in the large majority of cases (see caveat below).
- `Svr` / `PtWinner`: server and point winner for that specific point, coded
  1/2 consistently (checked full value_counts — no stray values, no nulls in
  either column across all 547,478 rows of the 2020s file).
- `Set1`/`Set2`, `Gm1`/`Gm2`, `Pts` (e.g. `"40-30"`, `"AD-40"`), `Gm#`: full
  running score state at the time of the point — enough to derive break
  point / game point / set point / match point status with standard tennis
  scoring rules (not present as an explicit boolean column, but computable
  from these fields; not attempted here since this pass is schema
  confirmation only, not label-building).
- `1st`/`2nd`: shot-by-shot rally notation for the point (charting shorthand).
  Mostly present for `1st` (charted whenever there was a serve), `2nd` null
  63% of the time (points won on the first serve — expected, not a data gap).
- Core columns (`match_id`, `Pt`, `Set1/2`, `Gm1/2`, `Pts`, `Gm#`, `TbSet`,
  `Svr`, `PtWinner`) have **zero nulls** across all three files.
- Point-count-per-match distribution is realistic for real tennis matches:
  mean 164, median 149, min 15, max 408 (2020s file) — consistent with real
  charted matches, not corrupted/truncated data.

**Caveat found, worth flagging honestly:** 597 of 3,337 matches (~18%) in the
2020s file have at least one non-`+1` jump in `Pt` (e.g. jumping from point 34
back to 91 then 150, with `Set1`/`Gm1` resetting mid-sequence) — inspected a
sample and this looks like either a correction/re-chart appended under the
same `match_id`, or two charters' logs concatenated under one ID, not a
sequencing bug in the *individual* point rows themselves (each row's own
score fields are internally consistent). This needs a real fix (e.g. sort +
dedupe by `Set1/Set2/Gm1/Gm2/Pts` fingerprint, or drop affected matches) when
the ingestion code is actually written — not a blocker for this schema-only
pass, but don't assume every `match_id`'s points are one clean, gapless
sequence without handling this.

**Does it support the 3 candidate momentum-label approaches?** Yes, in terms
of raw data availability, now that this file is confirmed real:
1. **EWMA win-rate** — trivially computable directly from the `PtWinner`
   sequence, per match, in order. No modeling prerequisite.
2. **ΔWP** — the required input (score state at each point: `Set1/2`,
   `Gm1/2`, `Pts`, `Svr`, plus the eventual point/game/match winner to fit
   against) is present. Still requires building and validating the
   win-probability model itself (unchanged from the original checklist item
   3) — that's separate work, not a data gap.
3. **Point-importance-weighting** — same story: the score-state fields needed
   to classify break/game/set/match points are present; the weighting formula
   itself is a separate design decision (unchanged from the original
   checklist), not blocked by data availability anymore.

**Cross-check against already-ingested Tennalysis matches** (queried the live
`charted_match_stats` table directly, 5,011 distinct matches): of the 7,532
distinct matches across all 3 points files, **4,997 (99.7% of the 5,011
already-ingested matches) already have point-by-point data available in the
same source**. This means point-level ingestion for those matches would join
directly onto the existing `mcp_reconciliation.parquet` mapping (`charted_match_id`
→ `matched_atp_id` → real `player_id`) — no new reconciliation pass needed,
confirming the plan in the feasibility check above.

**Scale estimate if this moves from a 2-3 file pilot to the full set:**
pulling all 3 already-downloaded files in for real ingestion gives **~4,997
usable matches** immediately (matches already reconciled to ATP data whose
points are also present) — essentially the same population size as the
existing 5,011-match aggregate-stats set, just with per-point granularity
added rather than a smaller subset. That's a real training-set-sized corpus,
not a small pilot, once the `Pt`-sequencing caveat above is handled. (Not
pulled here, but worth noting for later: the source repo also has parallel
`charting-w-points-*.csv` women's files, which would roughly double the pool
if Tennalysis ever ingests women's matches — currently it's men's-only, so
this is a future option, not part of the current estimate.)

## What's needed before this can move to a training script

1. **A point-by-point (not just match-summary) source**, covering, at minimum,
   per-point: server identity, point winner, running game/set score, and point
   outcome (ace / winner / unforced error / forced error / double fault).
   Match-summary stats (aggregate first-serve %, total winners, etc.) are
   NOT sufficient — momentum is a sequence signal and needs point-level
   granularity. **Update 2026-07-28:** source identified and de-risked, not
   an open question anymore — see the feasibility check above. The
   Tennalysis-web project already ingests match-level aggregates from the
   Match Charting Project (`JeffSackmann/tennis_MatchChartingProject`); the
   point-by-point files (`charting-m-points-2020s/2010s/to-2009.csv`) come
   from that same source repo and have now been pulled and schema-confirmed
   (see below) — server identity, point winner, and running score are all
   present per point. Licensing terms still need re-verification for these
   specific files before real ingestion (same repo as the already-reviewed
   stats files, but terms can differ by file — not re-checked in this pass).
2. **Confirmed file format and schema — DONE (2026-07-28)**, see the schema
   confirmation section above: real columns, dtypes, null rates, point
   ordering, and per-match point-count distribution all inspected against 3
   real files (1.28M point rows across 7,532 matches) before any ingestion
   code was written, the same way every other model's data was inspected in
   this project. The feature list in `model.py`'s docstring is still a
   provisional starting point, not a contract — expect to revise it once
   real per-point feature engineering starts (this pass only computed the
   EWMA momentum label, not the model input features).
3. **A decision on how "momentum" is operationalized as a training target** —
   see the dedicated OPEN DESIGN QUESTION section below. This is the single
   biggest unresolved item for this model and is not just a data-format
   question, so it gets expanded treatment rather than a one-line bullet.
4. **A match-count / point-count sanity check** for whether an LSTM is even
   the right capacity for the data volume available — if the acquired dataset
   is small (a handful of matches), a simpler model (e.g. logistic regression
   on a rolling window, or a small 1D-CNN) may generalize better than an LSTM;
   revisit architecture choice once real data volume is known rather than
   assuming the LSTM in `model.py` is final.
5. **Variable-length handling strategy.** Matches have different point counts.
   `model.py`'s `forward()` accepts an optional `lengths` argument for
   `pack_padded_sequence`-style masking but doesn't implement it yet — decide
   between padding+masking vs. fixed-length windowing once real sequence-length
   distributions are known.

## First real pipeline: ingestion + EWMA label + sanity check (2026-07-28)

Scoped deliberately narrow, per instruction: ingestion + EWMA momentum label
(approach #1 only) + sanity check on real matches. **No ΔWP, no
point-importance-weighting, no `Dataset`/`DataLoader`, no training loop** —
all still explicitly out of scope until EWMA is validated. Code:
`training/momentum_tracking/ingest_points.py` (ingestion, dedupe/sort fix,
EWMA label) and `training/momentum_tracking/sanity_check_ewma.py` (example
selection + plotting). Raw CSVs (130MB, the same 3 files pulled during
schema confirmation) now live in `training/momentum_tracking/data/raw/` —
not yet added to git; worth a deliberate decision (commit vs. `.gitignore`)
before this goes further, not something silently decided here.

**Dedupe/sort fix — root cause and verification.** Investigated the ~18%
non-sequential-`Pt` figure from the schema-confirmation pass directly (not
just patched around it): the overwhelming majority (1,521 of 7,532 matches,
~20.2%) turned out to be a **pure row-ordering artifact** in the raw CSVs —
each match's rows contain a complete, correct 1..N point sequence, just not
written to the file in `Pt` order. Sorting by `Pt` alone cleanly resolves all
of these. A much smaller number (11 matches) have **genuine duplicate rows**
for the same `Pt` — inspected these directly and found the duplicated pair
always agrees on every score-relevant column (`Set1/2`, `Gm1/2`, `Pts`,
`Gm#`, `Svr`, `PtWinner`) and only differs in shot-notation detail (`1st`/
`2nd`/`Notes`), consistent with the same point charted twice with different
annotation styles rather than conflicting data. Deduplicating on the
score-relevant columns resolved 6 of these 11 cleanly. The remaining
**5 matches could not be cleanly resolved** (duplicate `Pt` rows that
disagree on score-relevant columns — genuinely conflicting, not a formatting
difference) and were dropped rather than guessed at:
`19850907-M-US_Open-SF-Ivan_Lendl-Jimmy_Connors`,
`19890609-M-Roland_Garros-SF-Michael_Chang-Andrei_Chesnokov`,
`19920705-M-Wimbledon-F-Andre_Agassi-Goran_Ivanisevic`,
`19930704-M-Wimbledon-F-Jim_Courier-Pete_Sampras`,
`20220812-M-Canada_Masters-QF-Hubert_Hurkacz-Nick_Kyrgios`.

Post-fix verification was explicit, not assumed: every remaining match's `Pt`
column was checked to be strictly `+1` increasing starting at 1, with an
assertion that fails loudly if any match doesn't satisfy this (see
`dedupe_and_sort()` in `ingest_points.py`) — this ran clean.

**Match count after the fix:** 7,532 raw matches → **7,527 clean matches**
(99.93% retention; only the 5 unresolvable matches above were dropped) →
**4,996 of those are already-ingested/reconciled ATP matches** in
Tennalysis-web's `charted_match_stats` table. This is effectively identical
to the ~4,997 figure estimated during schema confirmation (off by exactly 1,
consistent with one of the 5 dropped matches — Hurkacz-Kyrgios 2022 — having
been in that set). **The dedupe fix does not meaningfully shrink the usable
corpus.**

**A real bug the sanity check itself caught:** while picking example matches,
naively reading each match's LAST row's `Set1`/`Set2` and comparing them to
determine the winner was WRONG in **31.2% of matches** (3,437/4,996 agreed
with the real winner from Tennalysis's `matches.winner_id`). Root cause:
`Set1`/`Set2`/`Gm1`/`Gm2`/`Pts` record the score **entering** each point (point
1 of every match is always logged at "0-0"), not the score after it — so the
last row reflects the state before the final point was played, not the
actual completed result. Fixed by using Tennalysis's real `winner_id` instead
of re-deriving it from the point log. This matters beyond just picking good
example matches: **anything that needs to know when a game/set/match was
actually won (approach #3's break/set/match-point detection, in particular)
must NOT assume a row's raw score columns reflect a completed game/set** —
apply standard scoring rules point-by-point instead. Flagged here so
whoever builds approach #3 doesn't rediscover this the hard way.

**Scope of this bug, stated explicitly (2026-07-28 closeout) — does NOT
affect the EWMA label values themselves.** Re-checked `compute_ewma_momentum()`
in `ingest_points.py` line by line to confirm: it reads only `PtWinner` and
`match_id` (for grouping), relying on `Pt`-order sorting that `dedupe_and_sort()`
already verifies with a hard assertion — it never touches `Set1`/`Set2`/`Gm1`/
`Gm2`/`Pts` at all. The naive last-row-Set-comparison bug lived entirely in
`sanity_check_ewma.py` (`validate_winner_derivation()` and the "winner's
perspective" sign-flip used only for plot orientation/captioning), which runs
strictly *after* `momentum_ewma_p1` has already been computed and written to
`points_ewma_momentum.parquet`. So: **every `momentum_ewma_p1` value in the
processed output is correct and unaffected by this bug** — what was wrong was
only the human-facing verification/plotting/match-selection layer built on
top of it (which example matches got picked, and how their captions/plot
orientation described the "winner"), not the label data itself. The warning
above for whoever builds approach #3 stands regardless, since #3's
break/set/match-point detection will need to read the score columns directly
in a way approach #1 never had to.

**EWMA sanity check — looks like real signal, not noise.** Plotted
`momentum_ewma_p1` (halflife = 10 points, `EWMA_HALFLIFE_POINTS` in
`ingest_points.py` — provisional, not validated) for 3 real matches, verified
against real final scores from `matches.score`:
- **2004 Roland Garros Final, Gaudio d. Coria, 0-6 3-6 6-4 6-1 8-6** — one of
  the most famous comebacks in tennis history (down 2 sets to love). The
  curve tracks it: sustained deeply negative momentum (Coria's favor) for the
  first ~100 points, crossing positive around the match's midpoint, staying
  mostly positive (with realistic short-term noise) through to the finish.
  This is exactly the shape a real comeback should produce.
- **2015 Rome Masters SF, Federer d. Wawrinka, 6-4 6-2** — a tidy scoreline on
  paper. The point-level curve tells a more interesting story underneath it:
  Wawrinka built a strong early lead (momentum near +1.0 in the first ~20
  points) that Federer's momentum overturned by roughly the match's halfway
  point, finishing negative (Federer's favor) — real texture a set/game score
  alone doesn't show. (An earlier draft of this checklist entry incorrectly
  captioned this as "Wawrinka d. Federer, a real upset," written from memory
  without checking — the sanity-check plot's own winner label caught the
  error before it went further. Left in the writeup as a reminder that this
  project's own real-vs-assumed data discipline caught a real mistake here,
  not just a hypothetical one.)
- **2015 US Open SF, Djokovic d. Cilic, 6-0 6-1 6-2 (control)** — picked as a
  dominant, lopsided win to check the label doesn't manufacture false drama.
  Momentum stayed mostly positive (Djokovic's favor) throughout with no deep
  sustained dip, consistent with a real blowout.

Also worth flagging: the EWMA curve is visibly volatile/extreme (near ±1) in
roughly the first `~1.5x halflife` points of every match — expected behavior
for an EWMA with little history to average yet, not a defect, but something
to account for (e.g. a warm-up mask, or a longer halflife) before trusting
early-match labels at face value in training.

**Conclusion: EWMA (approach #1) is validated well enough to move forward
with.** It is not noise — it produces distinguishable, plausible curves for a
real comeback, a real "tidy scoreline hides real drama" case, and a real
blowout. Processed output: `training/momentum_tracking/data/processed/
points_ewma_momentum.parquet` (1,281,126 point rows across 7,527 matches,
including the `already_ingested` flag and `momentum_ewma_p1` label).
**Approaches #2 (ΔWP) and #3 (point-importance-weighted) remain explicitly
deferred**, per instruction, until this label is put in front of an actual
model and shown to produce something sensible — not before.

## OPEN DESIGN QUESTION: momentum has no ground truth

This is unresolved regardless of which point-by-point data source gets
acquired, so it's called out on its own rather than buried as a checklist
line item.

**The problem:** `TennisMomentumLSTM` is defined as a regression model
outputting a continuous `[-1, 1]` per-point momentum score
(`model.py:TennisMomentumLSTM`). Point-by-point data gives *outcomes*
(who won each point, under what score context) — it does not give
*momentum*. "Momentum" is a construct, not a measurement: there is no sensor
or scoreboard field anywhere that records it directly. Whoever builds the
training pipeline for this model has to first commit to a specific
mathematical definition of momentum, derive labels from raw point outcomes
using that definition, and only then train against it. Picking a definition
silently (e.g. defaulting to "whatever's easiest to code") would bake an
arbitrary, unstated assumption into the model's entire output — this has to
be a deliberate decision, made with the user, and documented, not an
implementation detail.

**Candidate label-construction approaches** (starting points to evaluate
against real data once acquired, not a final choice — and citations below
should be verified independently before use in the FYP writeup, since exact
titles/years aren't being re-derived here from a primary source):

1. **Smoothed/exponentially-weighted recent win-rate.** Simplest option:
   label at point *t* = an EWMA (or fixed-window rolling average) of
   `+1`/`-1` point outcomes over the last *K* points, from the
   point-of-interest player's perspective. Cheap to compute, easy to explain,
   but the choice of *K* and decay rate is arbitrary and directly determines
   what the model learns to predict — needs justification, not just a default.
   Treats every point as equally important, which is a known weakness (see #3).

2. **Win-probability-derivative ("win probability added"-style) momentum.**
   Fit a separate point-level win-probability model first — e.g. logistic
   regression or a simple lookup table over score state (game/set score,
   server, break/set/match point flags) predicting P(win match | current
   state) — then define momentum as the *change* in that win probability
   from one point to the next (ΔWP). This mirrors "win probability added"
   metrics used in other sports analytics (baseball, basketball, American
   football), and there is tennis-specific literature on in-match win
   probability modeling (e.g. work building on Klaassen & Magnus's research
   on point dependence/serial correlation in tennis) that could inform the
   win-probability model itself. More principled than #1 since it's
   score-context-aware, but requires building and validating a second model
   (the win-probability model) before this one can even be trained.

3. **Point-importance-weighted momentum.** Not every point is equally
   significant to a match's outcome — a point at break/set/match point
   swings the win-probability far more than a point at 40-0. Tennis analytics
   has a long-standing "importance of a point" concept (originating from
   Morris's work on point importance in tennis, extended by later authors)
   that quantifies exactly this. Momentum could be defined as an
   importance-weighted running sum/average of recent point outcomes rather
   than treating all points equally as in #1 — conceptually the most
   tennis-aware option of the three, but the most work to implement
   correctly (requires either the win-probability model from #2 as a
   building block, or a separate point-importance formula).

**Recommendation for whoever picks this up:** don't default to #1 just
because it's simplest — bring #1/#2/#3 (or others) back to the user as an
explicit decision once real data is in hand, the same way the 12→5 class
collapse decision was made for Model 2 rather than assumed.

## Explicitly out of scope until the above is resolved

**Update 2026-07-28:** ingestion (`ingest_points.py`) and the EWMA momentum
label (approach #1) are now real and validated — see the pipeline section
above. Everything below is still out of scope:

- Writing `dataset.py` / a `DataLoader` for this model.
- Writing `train.py` for this model.
- Approaches #2 (ΔWP) and #3 (point-importance-weighted momentum) —
  deliberately deferred until EWMA has actually been used to train something
  and shown to produce a sensible result, per instruction.
- Generating synthetic/placeholder point-by-point data to "test" the training
  loop — synthetic data would validate tensor shapes at best and risks being
  mistaken for a real result later. `model.py`'s `__main__` shape check uses
  `torch.randn` explicitly labeled as non-representative, for exactly this
  reason — extend that pattern if any further scaffolding is added here.
