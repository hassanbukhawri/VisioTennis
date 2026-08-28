"""
PLACEHOLDER knowledge base -- see rag/DESIGN.md Step 1. These snippets are
HAND-AUTHORED BY US, informed by general/common tennis-coaching knowledge,
NOT sourced from the ITF Coaching Manual, USTA guidelines, or any other
copyrighted publication (none exists locally -- confirmed by a full-repo
search, see DESIGN.md). They exist ONLY to exercise the retrieval + prompt +
grounding pipeline mechanics end to end. Any citation the RAG pipeline
produces from a snippet below must be labeled "authored, not sourced" -- this
is explicitly not a claim of coaching authority, and should not be presented
as one in the FYP writeup or anywhere else.

Scoped to session-level topics only (SessionRecord's ANOMALY_FEATURE_COLUMNS
-- see training/anomaly_detection/train.py), matching DESIGN.md's Step 2
finding that session-level feedback has a real grounded-deviation mechanism
(Model 4's explain_anomaly() z-scores) today, whereas stroke-level
biomechanical feedback is explicitly DEFERRED (see DESIGN.md Step 2 --
blocked on pose-feature extraction work that doesn't exist yet). Do NOT
extend this file with stroke-level (contact height, racket angle, etc.)
snippets until that is unblocked.

Expanded (2026-07-18) from an initial 6-topic set to all 12
FEATURE_COLUMNS, with 2 differently-framed entries per topic (24 entries
total) -- deliberately more than one obviously-matching snippet per test
case, so retrieval (rag/embeddings.py) has to actually rank candidates by
semantic similarity rather than do a trivial 1:1 lookup.

REWRITTEN (2026-08-01) for player-facing tone, per real-user testing
feedback: the previous wording read as descriptive measurement reporting
("X is at the Yth percentile"), not coaching -- overwhelming for a casual
player, states-and-interprets but never says what to actually try. Rewrite
rules: lead with plain-language meaning (no "ratio"/"percentile" jargon in
the main sentence), every entry ends with something the player can try
next time, state findings directly rather than hedging on a metric that's
already been measured as unusual, and keep it to ~2 sentences so multiple
stacked results stay skimmable. Does NOT change retrieval or grounding
mechanics -- confirmed via code read (rag/grounding.py never inspects KB
text; rag/prompt.py embeds the full rewritten text same as before) and via
re-running rag/run_gated_diverse_trials.py post-rewrite (see PROJECT_LOG.md).
Entries that introduce a concrete new remedy not present in the prior
wording are flagged in rag/KB_COACH_REVIEW.md for the coach's attention --
reworded-only entries are not flagged.
"""
from dataclasses import dataclass


@dataclass
class KBSnippet:
    topic: str  # matches a training/anomaly_detection/train.py:FEATURE_COLUMNS name
    text: str
    source: str = "authored (placeholder, not sourced -- see rag/DESIGN.md Step 1)"


PLACEHOLDER_KB = [
    # --- first_serve_pct ---
    KBSnippet(
        "first_serve_pct",
        "Your first serve isn't landing as often as usual for you. Before changing "
        "anything in your motion, try playing it safer for a bit -- a little less "
        "pace, a little more margin -- and see if it comes back on its own.",
    ),
    KBSnippet(
        "first_serve_pct",
        "You're missing more first serves than usual, even though your toss looks "
        "steady on video -- that points to over-swinging for extra pace, not a "
        "motion breakdown. Try taking 10-15% off your target serve speed and see "
        "if your percentage comes back.",
    ),
    # --- first_serve_points_won_pct ---
    KBSnippet(
        "first_serve_points_won_pct",
        "Your first serve is landing fine, but you're not winning as many points "
        "behind it as usual -- that points to what you do with the point right "
        "after the serve, not the serve itself. Look at your shot choice on that "
        "first ball, not your service motion.",
    ),
    KBSnippet(
        "first_serve_points_won_pct",
        "You're winning fewer points behind your first serve than usual this "
        "session. Before assuming something's wrong with your game, consider that "
        "the returner may just have read your serve well today -- this kind of "
        "dip is often about them, not you.",
    ),
    # --- second_serve_points_won_pct ---
    KBSnippet(
        "second_serve_points_won_pct",
        "Your second serve isn't winning points the way it usually does, and your "
        "double faults are creeping up too -- that combination usually points to "
        "overhitting or pressing on the second serve, not playing it too safe. "
        "Try taking a little pace off and aiming for more margin over the net, "
        "rather than going for more on that ball.",
    ),
    KBSnippet(
        "second_serve_points_won_pct",
        "Your second serve is losing points while your first serve is winning "
        "them normally -- that points to the second serve specifically, not your "
        "overall service motion or footwork. Any changes are better spent there "
        "than on reworking your whole serve.",
    ),
    # --- ace_rate ---
    KBSnippet(
        "ace_rate",
        "Your ace count is down today, but if your first-serve points won is "
        "holding steady, this probably isn't about your serve -- the returner "
        "likely just read you well. Only worry about it if the pattern keeps "
        "showing up over several sessions.",
    ),
    KBSnippet(
        "ace_rate",
        "Your aces dropped along with a slower reaction time today -- that "
        "combination usually means fatigue is taking a bit off your racket speed "
        "on serve, not a targeting problem. If you're feeling tired, this is one "
        "to let go rather than trying to fix your placement.",
    ),
    # --- double_fault_rate ---
    KBSnippet(
        "double_fault_rate",
        "You double-faulted more than usual today, alongside a lower first-serve "
        "percentage -- that combination usually means you pushed too hard on the "
        "second serve under pressure, not a toss problem. Check whether they "
        "clustered on break points or late in games; if so, that's pressure, not "
        "technique.",
    ),
    KBSnippet(
        "double_fault_rate",
        "Your double faults bunched up in the second half of the session instead "
        "of spreading out evenly -- that's usually fatigue catching up with your "
        "toss, not a technical flaw. Consider a shorter warm-up or more rest "
        "between sets before changing your serve motion.",
    ),
    # --- break_points_saved_pct ---
    KBSnippet(
        "break_points_saved_pct",
        "Your break-point save rate was low this session, but a handful of break "
        "points isn't enough to draw real conclusions from. Don't change "
        "anything based on this alone -- just watch this number over your next "
        "few sessions before treating it as a pattern.",
    ),
    KBSnippet(
        "break_points_saved_pct",
        "Your break-point save rate is low, but your overall unforced-error rate "
        "looks normal -- that points to how you're playing the biggest points "
        "specifically, not your game as a whole. On break points, try dialing "
        "back your risk a notch rather than going bigger.",
    ),
    # --- unforced_error_rate ---
    KBSnippet(
        "unforced_error_rate",
        "You made more unforced errors than usual today. Check when they "
        "happened -- a cluster late in the session points to fatigue, so rest a "
        "bit more between reps; errors on one specific shot point to that shot's "
        "technique instead.",
    ),
    KBSnippet(
        "unforced_error_rate",
        "Your errors went up while your points got shorter -- you were probably "
        "going for bigger shots than usual. Check how many of those bigger "
        "swings actually landed as winners; if not many did, try taking a little "
        "pace off until your errors settle back down.",
    ),
    # --- winners_rate ---
    KBSnippet(
        "winners_rate",
        "Your winners were down today, but your unforced errors didn't go up -- "
        "that usually means you played more conservatively, not that your shots "
        "got worse. If that was a deliberate choice against a tough opponent, no "
        "need to change anything; if not, it's worth noticing.",
    ),
    KBSnippet(
        "winners_rate",
        "Your winners and unforced errors moved in the same direction today -- "
        "both up, or both down. Read that as one shift in how aggressively you "
        "were playing, not two separate problems to fix.",
    ),
    # --- avg_rally_length ---
    KBSnippet(
        "avg_rally_length",
        "Your points got shorter than usual today. If your winners were also up, "
        "that's you taking control earlier in points -- a good sign; if your "
        "unforced errors were up instead, points are ending before you can build "
        "them, which is worth addressing.",
    ),
    KBSnippet(
        "avg_rally_length",
        "Your points ran longer than usual while your winners were down -- "
        "that's a more passive, defensive session. That's fine against a tough "
        "opponent on the day, but worth noticing if it keeps happening rather "
        "than being a one-off.",
    ),
    # --- court_coverage_speed_mps ---
    KBSnippet(
        "court_coverage_speed_mps",
        "You covered the court more slowly than usual today, and your reaction "
        "time was down too -- together that's a fatigue signal, not a footwork "
        "problem. Consider more recovery time or a lighter session before your "
        "next one.",
    ),
    KBSnippet(
        "court_coverage_speed_mps",
        "You moved slower than usual today, but your total distance covered was "
        "normal -- that points to less urgency on individual points, not less "
        "overall effort. Worth checking your split-step timing on video to see "
        "if you're getting a late start on points.",
    ),
    # --- avg_reaction_time_ms ---
    KBSnippet(
        "avg_reaction_time_ms",
        "Your reaction time was slower than usual today, especially later in the "
        "session -- that's a classic fatigue marker, not a technical issue. "
        "Check how long you'd been playing and what time of day it was before "
        "reading too much into it.",
    ),
    KBSnippet(
        "avg_reaction_time_ms",
        "Slower reaction time along with slower court coverage points to overall "
        "fatigue. If it's reaction time alone though, with normal movement "
        "speed, try watching the ball earlier out of your opponent's racket "
        "rather than working on fitness.",
    ),
    # --- total_distance_covered_m ---
    KBSnippet(
        "total_distance_covered_m",
        "You covered less ground than usual today. Before treating that as a "
        "fitness or effort issue, check whether the session was shorter or "
        "points were ending quicker than usual -- both would explain it without "
        "anything being wrong.",
    ),
    KBSnippet(
        "total_distance_covered_m",
        "Distance covered, court-speed, and reaction time all dropping together "
        "is a fairly complete fatigue picture. If only one of the three is down, "
        "treat it as something narrower and specific, not overall fatigue.",
    ),
]
