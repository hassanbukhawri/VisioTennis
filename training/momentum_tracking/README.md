# Momentum Tracking (Model 3) — status and results

## Status: genuinely predictive (leak-free features, horizon-shifted target), evaluated against a real baseline. Does NOT yet beat naive persistence. NOT deployed.

This supersedes the original 2026-08-23 status below (kept further down,
not deleted, as the historical record of what "reconstruction" meant and
why it wasn't evidence of predictive power). That gap is now closed, and
this section states the new result with the same directness the old one
used for its own limitation — success is not stated more quietly than the
caveat was.

**What changed (2026-08-26):** the input feature set now contains ONLY
match state ENTERING the point in question — `server`, `game_score_diff`,
`set_score_diff`, `is_break_point`, `is_set_point`, `is_match_point`
(`input_dim` 12 → 6). `point_winner` and every other point-outcome-derived
feature (rally length, ace/winner/unforced-error flags, server streak) are
gone from the input entirely, at every lag — not just the current point.
The target is no longer the same-point EWMA value; `MomentumDataset` now
takes a `horizon` (default 5), and position *i*'s target is the EWMA value
5 points after the state its input describes. See `dataset.py`'s module
docstring for the full detail.

**Held-out TEST set (500 matches, 78,271 points — 80,271 minus 500×4
horizon-truncated trailing points): MSE=0.03745, MAE=0.13520,
RMSE=0.19352.** ~34× the old reconstruction task's MAE and ~936× its MSE —
exactly the order-of-magnitude jump expected once every outcome-derived
feature was removed and the target was pushed 5 points into the future.
This number is measuring real forecasting difficulty now, not a
near-zero reconstruction floor.

**The number alone doesn't say whether 0.135 MAE is actually good — two
baselines were computed on the identical test set before calling this
useful, not after:**

| approach | MAE | MSE | RMSE |
|---|---|---|---|
| **trained model** | 0.13520 | 0.03745 | 0.19352 |
| persistence ("momentum hasn't changed in 5 points") | **0.13048** | **0.02837** | **0.16844** |
| neutral (always predict 0) | 0.18669 | 0.06238 | 0.24975 |

**Honest finding, stated with the same weight as the headline number: the
trained model is slightly WORSE than the naive persistence baseline** on
every metric (MAE +3.6%, MSE +32% relative to persistence), while clearly
beating the neutral/predict-0 baseline (MAE −28%, MSE −40%). The model has
genuinely learned to use score-context to make informed forecasts rather
than defaulting to a neutral guess — that part is real, and is itself new
relative to the reconstruction-task version, which demonstrated nothing
about score-context specifically. But it has not learned to forecast
momentum better than assuming momentum hasn't moved much in 5 points. The
train/val curve shows no overfitting gap (epoch 28: train_mae=0.1366,
val_mae=0.1357), so this reads as a real ceiling for what these 6
score-context features predict about momentum 5 points out, not an
undertrained model.

**Qualitative check** (`evaluate.py`, `figures/trained_model_vs_ewma_examples.png`,
plotting each forecast against the point it actually targets):
- **Gaudio comeback:** right shape, forecast 5 points ahead of the real
  swing — negative through roughly the first 110-120 points (label's own
  sustained-negative span runs to ~140), crosses positive from there,
  stays mostly positive afterward. Consistently damped in amplitude
  relative to the label (model peaks ~0.4-0.65 vs the label's swings to
  ±0.85-1.0) — expected, not a defect: the true EWMA can swing to the
  extremes off a handful of point outcomes a context-only model never
  sees.
- **Negative control (Djokovic/Cilic, genuinely held out — VAL split):**
  stays positive (~0.15-0.45) for nearly the whole match, no invented
  reversal — the specific failure mode this check exists to catch did not
  occur. One honest exception: a brief, shallow dip to ~-0.05 around
  points 105-112 before recovering — not a sustained reversal, but not
  perfectly flat either.
- **Federer/Wawrinka:** correct direction of the turnaround, same damped-
  amplitude pattern as Gaudio.

**Split disclosure, re-verified for this run, not assumed carried over
from 2026-08-23:** Gaudio and Federer/Wawrinka are both in TRAIN; only
Djokovic/Cilic is genuinely unseen (VAL). Matters more here than it did
for the reconstruction task, since this result is no longer near-ceiling —
only the negative control and the quantitative TEST-set numbers above are
on genuinely unseen data.

**Bottom line: partially useful, not fully.** This is no longer a
validated reconstruction of a formula the model could see the exact input
to — the held-out error moved by 2-3 orders of magnitude exactly as
removing that input would predict, and the model demonstrably uses
score-context for real (beats neutral; qualitatively correct direction and
rough timing on both positive examples; no fabricated drama on the
negative control). But by the most relevant bar — does it forecast
momentum better than assuming it hasn't changed — it does not, on this
test set, with this feature set. **Not a case for wiring this into the
product as a "prediction" surface as-is.** A real next step, if revisited:
a richer leak-free feature set (only game/set-transition score-state is
here now; fatigue/rest proxies or tournament/round context could plausibly
help), or accepting persistence as the baseline every future attempt here
should be reported against explicitly.

**Correctly scoped as model-building and evaluation, not a deployed
capability.** The trained checkpoint is not wired into `backend/app.py` or
any other live surface.

---

## What was built

- **`dataset.py`** — feature engineering against the real, confirmed
  `charting-m-points-*.csv` schema. `FEATURE_NAMES` (6, leak-free: state
  entering the point only) vs `LEAKY_FEATURE_NAMES` (6, still computed for
  diagnostics, never passed to the model: `point_winner` and everything
  derived from how the point was actually played). `MomentumDataset`'s
  `horizon` parameter shifts the target that many points ahead, truncating
  each match's trailing `horizon-1` positions (no valid future target).
  Match-level train/val/test split (3,995/499/500 matches), zero-overlap
  verified by assertion.
- **`model.py`** — `TennisMomentumLSTM`, 2-layer LSTM (hidden_dim=64) +
  small FC head, Tanh-bounded `[-1,1]` regression output, `input_dim=6`.
- **`train.py`** — masked-MSE training loop (padded timesteps excluded
  from the loss), `--horizon` CLI arg, smoke-test mode, best-checkpoint-
  by-val-MAE selection (not last epoch), horizon saved into the checkpoint
  so `evaluate.py` never has to assume what a checkpoint was trained with.
- **`evaluate.py`** — held-out test MSE/MAE/RMSE, plus the qualitative
  check against the same 3 hand-picked real matches `sanity_check_ewma.py`
  validated the EWMA LABEL against originally: a real comeback (2004
  Roland Garros Final, Gaudio d. Coria), a match with real within-match
  swings despite a tidy scoreline (2015 Rome Masters SF, Federer d.
  Wawrinka), and a dominant-win negative control (2015 US Open SF,
  Djokovic d. Cilic).

## Real bugs found and fixed while building this, not glossed over

1. **`TbSet` is a SET-level flag, not point-level.** Reads `True` for
   every point in a set from the first 0-0 point onward whenever that set
   eventually reaches 6-6 — trusting it to decide how to parse the `Pts`
   score string per-point silently misparsed ordinary early-game scores
   (e.g. "15-0") as raw tiebreak point counts. Caught by hand-checking a
   real match's point sequence against the computed flag (an initial 55.6%
   "is_set_point" rate was the tell). Fixed by detecting tiebreak-style
   scoring directly from whether `Pts`' values fall inside the standard
   `0/15/30/40/AD` vocabulary, independent of `TbSet`'s real meaning.
2. **1 NaN value in 823,874 real points** (`Gm2`, one specific match, a
   genuine gap in the raw MCP source file) reached the first full training
   run and produced `nan` loss from epoch 1 — caught immediately (killed
   at epoch ~9), root-caused to the exact row, fixed with a verified
   within-match forward/back-fill. Did not recur in the leak-free
   reframe's full run (fix already in place).
3. **`server_streak_norm` originally recorded the streak AFTER folding in
   the current point's own outcome** — a leak in that specific diagnostic
   column, moot for training now that it's excluded from the model's
   input, but fixed anyway (2026-08-26) so the column itself is honest.

## History: the original reconstruction-task result (2026-08-23), superseded above

**Held-out test set (500 matches, 80,271 points): MSE=0.00004, MAE=0.00401,
RMSE=0.00624.** `point_winner` — whether the point-of-interest player won
the point just played — was both a per-timestep INPUT feature and the raw
signal the EWMA training target was itself computed from. Given that
input, reconstructing the target was a known, deterministic smoothing
formula the model could see the raw ingredients of — a near-zero MSE
confirmed the LSTM learned to implement that EWMA recurrence correctly,
not that it had learned anything predictive. The qualitative check (same 3
matches) reproduced the Gaudio comeback and Djokovic/Cilic negative-control
shapes on that version too, for the same reason (a near-perfect
reconstruction of the label reproduces the label's own shape by
construction).
