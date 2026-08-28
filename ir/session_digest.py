"""
Cross-session progress digest computation ("Option C" -- see PROJECT_LOG.md's
2026-08-12 entry for the full design writeup).

Core design commitment: a digest is ALWAYS computed fresh, directly from real
stored session rows, every single time one is requested. There is no
persisted digest, no running/incremental aggregate, and no "update the last
digest" code path anywhere in this file or its callers -- the only way a
SessionDigest is ever produced is by re-reading the raw session history and
recomputing from scratch. This is a deliberate trade against a cheaper
"maintain a running summary" design: recomputation means every digest can be
independently re-derived and checked against source_session_ids (the exact
rows it claims to summarize), the same "verify against real data, don't trust
a cached claim" philosophy rag/grounding.py already applies to every other
citable record in this project. The cost is real (a digest costs one query +
one percentile lookup per feature per session, not O(1)), but at the session
volumes this app will see per player for the foreseeable future, that cost is
trivial next to the auditability it buys.
"""
import statistics
from collections import Counter

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark, RANKED_FEATURES
from ir.schema import SessionDigest

# DEFAULT_LOOKBACK_K -- a judgment call, not empirically tuned (no real usage
# data exists yet to tune it against; flagged here as something to revisit
# once Phase 1's session instrumentation has accumulated real per-player
# session-frequency data during the actual trial). Reasoning for k=5:
#
# 1. Noise floor: a single session's biomechanics numbers are noisy by
#    construction -- this project's own real-footage validation work
#    documented pose-detection rates swinging from 65% to 100% clip-to-clip,
#    and percentile ranks are correspondingly jumpy session to session.
#    Averaging over too few sessions (k=2-3) lets one unusual session (bad
#    lighting, an off day, a genuinely different swing) dominate a "trend"
#    claim the player would reasonably read as durable. k=5 gives real
#    smoothing without needing a large history.
# 2. Reachability: the Phase 2 trial (see visiotennis_roadmap.md) runs with
#    real university players over a bounded window -- requiring a large k
#    (e.g. 10, needing 20 total sessions before ANY trend claim exists, since
#    a trend needs 2*k) would likely mean most trial participants never see
#    this feature fire at all. k=5 means a trailing average appears after
#    just 1 session (degrades gracefully, see compute_session_digest below)
#    and a real trend delta appears after 10 sessions -- realistically
#    reachable within the trial's timeframe for an engaged player.
# 3. It's a common, unremarkable choice for a trailing window (5- and
#    10-session/day moving windows are standard in fitness/analytics UIs) --
#    not picked for novelty, picked to be unsurprising.
#
# This is explicitly NOT validated against real session-frequency data --
# said plainly rather than dressed up as more rigorous than it is.
DEFAULT_LOOKBACK_K = 5


def compute_session_digest(
    sessions: list[dict], player_id: str, benchmark: BiomechanicsPercentileBenchmark,
    lookback_k: int = DEFAULT_LOOKBACK_K,
) -> SessionDigest | None:
    """sessions: this player's stored analyze-stroke result dicts (same shape
    as backend/app.py's response_body), ordered NEWEST FIRST. Returns None if
    there are zero usable sessions (no biomechanics -- e.g. every one of this
    player's sessions failed pose detection) -- callers must handle "no
    digest available" explicitly (see rag/chat_pipeline.py's
    progress_digest route) rather than receive a degenerate all-zero digest
    that looks like real data.

    "Usable" = has a real biomechanics block (stroke.predicted_class +
    biomechanics.<feature> for every ir/biomechanics_percentiles.py feature).
    Sessions that failed pose detection, or predate biomechanics being
    computed, are skipped, not zero-filled."""
    usable = [s for s in sessions if s.get("stroke_event", {}).get("biomechanics")]
    if not usable:
        return None

    recent_window = usable[:lookback_k]
    prior_window = usable[lookback_k:lookback_k * 2]

    def _percentile(session: dict, feature: str) -> float:
        stroke_class = session["stroke_event"]["stroke"]["predicted_class"]
        raw_value = session["stroke_event"]["biomechanics"][feature]
        return benchmark.percentile(raw_value, stroke_class, feature)["percentile"]

    trailing_averages = {}
    trend_deltas = {}
    for feat in RANKED_FEATURES:
        recent_pcts = [_percentile(s, feat) for s in recent_window]
        trailing_averages[feat] = round(statistics.fmean(recent_pcts), 1)
        if prior_window:
            prior_pcts = [_percentile(s, feat) for s in prior_window]
            trend_deltas[feat] = round(statistics.fmean(recent_pcts) - statistics.fmean(prior_pcts), 1)

    stroke_classes = [s["stroke_event"]["stroke"]["predicted_class"] for s in recent_window]
    stroke_type_mode = Counter(stroke_classes).most_common(1)[0][0]

    return SessionDigest(
        player_id=player_id,
        lookback_k=lookback_k,
        session_count=len(recent_window),
        total_sessions_considered=len(usable[:lookback_k * 2]),
        stroke_type_mode=stroke_type_mode,
        trailing_averages=trailing_averages,
        trend_deltas=trend_deltas,
        trend_prior_window_size=len(prior_window),
        source_session_ids=[s["stroke_event"]["event_id"] for s in recent_window + prior_window],
    )
