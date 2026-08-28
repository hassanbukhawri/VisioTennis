"""
Turn-router for the multi-turn stroke chatbot.

Design (see PROJECT_LOG.md's chat grounding architecture entry): every chat
turn is classified into one of 4 routes BEFORE any expensive generation
happens, so the full retry-gate (rag/gated_pipeline.py, 27-200s+) only runs
when a turn genuinely needs it:

  - "existing_citation": replay an already-grounded item from this session's
    real CoachingResponse (rag/schema.py). NEVER generates new text -- so a
    wrong match here can only surface a real, already-verified (if slightly
    off-topic) answer, never a fabricated one. That bounded failure mode is
    what makes it acceptable to ship an imperfect topic-matcher.
  - "new_grounded_claim": needs the full retry-gate. Also the fail-safe
    default whenever classification is uncertain or unparseable.
  - "general_note" / "unavailable_comparison": cheap single-shot generation,
    no citation possible -- but see check_tier3_response_for_smuggled_claims
    below for why this path isn't blindly trusted either.

Two mechanical backstops, neither trusting the LLM's self-report, both
following rag/grounding.py's existing philosophy (verify against real data,
don't parse/guess):

  1. classify_turn()'s "existing_citation" route is checked against the
     REAL set of this session's citation field paths before being trusted --
     a claimed match to a field that was never actually cited is downgraded
     to "new_grounded_claim" (same pattern as grounding.py's WRONG_RECORD
     check never trusting a citation's claimed record).
  2. check_tier3_response_for_smuggled_claims() catches the case where the
     router itself misclassifies a should-be-grounded question as
     "general_note" -- if the resulting free-text response smuggles in a
     citation-shaped claim or a general_note with a stray digit (already a
     tripwire in check_grounding's SUSPECT_GENERAL_NOTE), it's escalated to
     the full gate instead of shown. Reuses check_grounding() unchanged,
     called with an EMPTY source_records -- a tier-3 response is never
     allowed to cite any real record, so any grounded_observation item that
     appears is automatically WRONG_RECORD.
  3. check_for_skill_tier_mismatch() (added 2026-08-12) catches a THIRD,
     different failure mode from the two above: a "new_grounded_claim"
     generation that is itself fully, mechanically grounded (passes
     check_grounding() with a real, correctly-cited percentile) but still
     shouldn't be shown, because the question was framed as a skill-tier/
     ability comparison ("what's normal for a player AT MY LEVEL") and
     ir/biomechanics_percentiles.py's population is flat -- confirmed to
     have no skill/tier stratification anywhere in that module. A citation
     into that benchmark can be numerically perfect and still mislabel a
     flat population percentile as a skill-tier comparison it cannot
     actually support. This is a real, OBSERVED failure (see
     PROJECT_LOG.md's 2026-08-12 entry): the router prompt above already
     tells the classifier "at my level" belongs in "unavailable_comparison"
     (rule 4), but a live run on the exact phrase "what's a good weight
     transfer ratio for a player at my level" was routed to
     "new_grounded_claim" instead and produced a real, 100%-grounded
     percentile citation -- exactly the case this backstop exists for.
"""
import json
import re
from dataclasses import dataclass

from rag.grounding import PERCENTILE_PREFIX, check_grounding
from rag.run_qwen_eval import clean_json_text
from rag.schema import ChatRouteDecision, CoachingFeedbackItem, CoachingResponse

# Fail-safe default whenever the classifier's output can't be trusted for any
# reason (unparseable, or an existing_citation claim that doesn't check out).
# Always SAFE, only ever costs latency -- never skips grounding.
_FAILSAFE_DECISION = ChatRouteDecision(route="new_grounded_claim", matched_field_path=None)


@dataclass
class ExistingCitation:
    """One real grounded_observation citation already generated for this
    stroke's session (i.e. one CoachingFeedbackItem.citation from the
    original analyze-stroke response). field_path is what matching happens
    against; label/text are shown to the router LLM for topic matching."""
    field_path: str  # e.g. "biomechanics_percentile.wrist_height_ratio"
    label: str  # short human topic name, e.g. "wrist height / contact point"
    text: str  # the original coaching sentence, for topic-matching context


ROUTER_SYSTEM_PROMPT = """You are a routing classifier for a tennis coaching chatbot. You do NOT answer the \
player's question -- you decide how it should be handled. Respond with ONLY a single JSON object (no markdown \
fences, no prose before or after) matching exactly this shape:

{
  "route": "existing_citation" | "new_grounded_claim" | "general_note" | "unavailable_comparison" | "progress_digest",
  "matched_field_path": "<copy the EXACT field_path string of the matched topic below, or null>"
}

You are given a list of topics ALREADY discussed for this stroke (each with the exact citation field_path it maps \
to) and the player's new message. Choose exactly one route:

1. "existing_citation" -- ONLY if the new message is clearly asking about (a rephrase of, or direct follow-up on) \
one of the EXACT topics listed below, same field, same kind of claim. Set "matched_field_path" to that topic's \
field_path, copied exactly. If you are not confident it's the SAME topic -- including if it could plausibly be a \
different field, a different kind of question (e.g. asking to compare across time/sessions when the topic was only \
ever discussed for this one stroke), or if which topic it refers to is ambiguous (e.g. an unclear "that" after \
multiple topics were just discussed) -- do NOT use this route. When genuinely unsure, prefer "new_grounded_claim" \
instead of guessing.
2. "new_grounded_claim" -- the message asks something specific about the player's OWN measured data for THIS \
stroke that is NOT one of the listed topics (a different biomechanical field, or the same field but a materially \
different kind of question). Also use this route whenever you are uncertain between this and "existing_citation".
3. "general_note" -- generic tennis advice/knowledge with NO implied comparison to the player's own data or to any \
benchmark/average/other player (e.g. "what's a good drill for this", "how do I fix a short follow-through in \
general").
4. "unavailable_comparison" -- the message asks for a personalized comparison, benchmark, or "is X good/normal" \
judgment that would require data this system doesn't have (e.g. comparing to other players "at my level", to a \
pro, or asking whether a specific number is "good" in the abstract without it being one of the listed topics' own \
stated percentile). Does NOT cover progress-over-time questions -- see rule 5.
5. "progress_digest" -- the message asks about the player's progress or trend ACROSS SESSIONS/OVER TIME (e.g. \
"how am I trending", "am I improving", "how has my [stroke] been going lately", "how does this session compare to \
my recent ones"). This is DIFFERENT from rule 4's "unavailable_comparison": a real cross-session digest exists for \
this exact question, so it is NOT an unavailable comparison -- route it here, not to "unavailable_comparison".

If a message has multiple parts and ANY part is not covered by an existing topic, route the WHOLE message as \
"new_grounded_claim" (do not split a message across routes).

"matched_field_path" MUST be null for every route except "existing_citation", and MUST exactly match one of the \
field_path values listed below when the route IS "existing_citation" -- never invent or paraphrase a field_path.

WORKED EXAMPLES (topics here are illustrative placeholders, not the real topics for this stroke -- apply the same \
reasoning pattern to the real topics given below):

Example 1 -- topics discussed: [field_path="X.metric_a"]. Message: "what's a good exercise to work on this?" \
-> route "general_note". No reference to the player's own number or a comparison -- pure generic advice, even \
though it follows a data-grounded topic.

Example 2 -- topics discussed: [field_path="X.metric_a", field_path="X.metric_b"]. Message: "was that good?" \
(right after both were just discussed) -> route "new_grounded_claim", NOT "unavailable_comparison". It's not \
asking for a benchmark this app lacks -- it's ambiguous WHICH of the two topics "that" refers to, so it needs a \
fresh generation that can address the ambiguity directly, not the fixed "no benchmark data" disclaimer.

Example 3 -- topics discussed: [field_path="X.metric_a"]. Message: "how does this compare to last week's \
session?" -> route "unavailable_comparison". Comparing to a DIFFERENT, SPECIFIC session's raw data is never \
something this app can ground (there is no per-session "diff against one other session" capability) even though \
the field itself was discussed here -- this is the disclaimer case, not "progress_digest", because the player is \
asking about one other specific session, not an aggregate trend.

Example 4 -- topics discussed: [field_path="X.metric_a"]. Message: "how have I been progressing lately?" \
-> route "progress_digest", NOT "unavailable_comparison". This asks about a TREND across recent sessions in \
aggregate, not a comparison to one specific other session or player -- exactly what the digest exists to answer.

Example 5 -- topics discussed: [field_path="X.metric_a"]. Message: "is my backhand better than last time?" \
-> route "progress_digest". "Better than last time" is a trend/recency comparison against the player's own recent \
history in aggregate, not a request for one specific past session's raw numbers -- still the digest's job, even \
though it's phrased as a single comparison rather than "trend" explicitly."""


def _format_existing_citations(existing_citations: list[ExistingCitation]) -> str:
    if not existing_citations:
        return "(no topics have been discussed yet for this stroke)"
    return "\n".join(
        f'- field_path="{c.field_path}" ({c.label}): "{c.text}"' for c in existing_citations
    )


def _format_conversation_history(conversation_history: list[tuple[str, str]]) -> str:
    if not conversation_history:
        return "(no prior turns in this chat)"
    return "\n".join(f"{role}: {text}" for role, text in conversation_history)


def build_router_prompt(
    existing_citations: list[ExistingCitation],
    conversation_history: list[tuple[str, str]],
    user_message: str,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the routing classifier.
    conversation_history: list of (role, text) tuples, e.g.
    [("player", "..."), ("coach", "...")], oldest first -- lets the router
    resolve pronouns/follow-ups ("was that normal?") against recent turns,
    or correctly refuse to when it can't."""
    user_prompt = (
        f"TOPICS ALREADY DISCUSSED FOR THIS STROKE:\n{_format_existing_citations(existing_citations)}\n\n"
        f"RECENT CONVERSATION:\n{_format_conversation_history(conversation_history)}\n\n"
        f'PLAYER\'S NEW MESSAGE: "{user_message}"\n\n'
        f"Classify this message now."
    )
    return ROUTER_SYSTEM_PROMPT, user_prompt


def classify_turn(
    backend,
    existing_citations: list[ExistingCitation],
    conversation_history: list[tuple[str, str]],
    user_message: str,
) -> ChatRouteDecision:
    """Classifies one chat turn. Never raises on a bad/unparseable LLM
    output or an unverifiable citation match -- always falls back to the
    safe default (_FAILSAFE_DECISION) instead, since a routing mistake here
    must only ever cost latency, never correctness."""
    system_prompt, user_prompt = build_router_prompt(existing_citations, conversation_history, user_message)
    result = backend.generate(system_prompt, user_prompt)

    try:
        cleaned = clean_json_text(result.raw_text)
        decision = ChatRouteDecision.model_validate(json.loads(cleaned))
    except Exception:
        return _FAILSAFE_DECISION

    if decision.route == "existing_citation":
        real_paths = {c.field_path for c in existing_citations}
        if decision.matched_field_path not in real_paths:
            # Mechanical backstop: never trust the classifier's claimed match --
            # verify it against the real citation set, same pattern as
            # grounding.py's WRONG_RECORD check never trusting a citation's
            # claimed record. A plausible-sounding but nonexistent/wrong
            # field_path downgrades to the safe, if more expensive, route.
            return _FAILSAFE_DECISION

    return decision


def check_tier3_response_for_smuggled_claims(response: CoachingResponse) -> bool:
    """Returns True if a general_note/unavailable_comparison-tier response is
    safe to show as-is; False if it must be escalated to the full retry-gate
    instead. This is the misclassification safety net: if classify_turn()
    wrongly routed a should-be-grounded question to "general_note", the LLM
    asked to write generic text will often still smuggle in the player's
    real number as an uncited claim -- caught here via check_grounding()
    called with an EMPTY source_records, so:
      - any "grounded_observation" item is automatically WRONG_RECORD (no
        record is legitimately citable on this tier), and
      - any "general_note" item containing a stray digit is flagged
        SUSPECT_GENERAL_NOTE (check_grounding's existing tripwire).
    Both fail this check and force escalation."""
    report = check_grounding(response, source_records={})
    return all(v.verdict == "OK_GENERAL_NOTE" for v in report.verdicts)


# Skill-tier / ability-comparison framing. Not exhaustive -- a fixed phrase
# list can always be phrased around -- but covers the documented real failure
# plus its natural rephrasings. Deliberately narrow (no bare "beginner"/
# "advanced" as standalone triggers) to avoid false-positiving on legitimate
# technical text that happens to use those words (e.g. a coach-tone response
# describing an "advanced technique").
_SKILL_TIER_PATTERNS = [
    r"\bat (my|your) level\b",
    r"\bfor (my|your) (skill )?level\b",
    r"\b(my|your) skill level\b",
    r"\bplayers? (like|at) (my|your) (level|stage)\b",
    r"\bsomeone at (my|your) (stage|level)\b",
    r"\b(compared?|compares?) (to|with) (players|other players|someone) (like me|like you|at (my|your) level)\b",
    r"\bfor (a |an )?(beginner|intermediate|advanced) (player|like me)\b",
    r"\b(my|your) (experience|ability) (level|group)\b",
]
_SKILL_TIER_RE = re.compile("|".join(_SKILL_TIER_PATTERNS), re.IGNORECASE)


def _contains_skill_tier_language(text: str) -> bool:
    return bool(_SKILL_TIER_RE.search(text))


def _has_percentile_citation(items: list[CoachingFeedbackItem]) -> bool:
    """True if any item cites a population-percentile-derived value -- either
    a direct "biomechanics_percentile.<feature>" StrokeEvent citation, or ANY
    SessionDigest citation (added 2026-08-12): trailing_averages/trend_deltas
    are themselves computed by averaging the same flat, non-skill-stratified
    BiomechanicsPercentileBenchmark percentiles across sessions (see
    ir/session_digest.py) -- the skill-tier mismatch risk this backstop
    exists for applies just as much to a digest's averaged percentile as to
    a single stroke's raw one, so both citation shapes trip it the same way."""
    return any(
        item.citation is not None and (
            item.citation.ir_field_path.startswith(PERCENTILE_PREFIX)
            or item.citation.record_type == "SessionDigest"
        )
        for item in items
    )


def check_for_skill_tier_mismatch(user_message: str, items: list[CoachingFeedbackItem]) -> bool:
    """Returns True if a "new_grounded_claim" response is safe to show;
    False if it must be rejected and the turn forced to
    "unavailable_comparison" instead, REGARDLESS of what classify_turn()
    originally decided or how well the response itself grounds (see module
    docstring, backstop #3). Trips only when BOTH conditions hold, checked
    mechanically, not via the model's self-report:
      1. skill-tier/ability-comparison language appears in the player's OWN
         message (the reliable, literal site of the framing -- "at my
         level" originates in the question, not necessarily in the
         generated answer) OR in the response text itself (checked too,
         belt-and-suspenders, in case the model echoes the framing back);
         AND
      2. the response actually cites a "biomechanics_percentile.*" field --
         ir/biomechanics_percentiles.py's population, confirmed to have no
         skill/tier stratification, so this citation can never legitimately
         answer a skill-tier-framed question no matter how numerically
         correct it is.
    A response with skill-tier language but NO percentile citation (e.g. a
    plain general_note) is left alone -- this backstop is specifically about
    a flat-population number being mislabeled as a skill-tier comparison,
    not about the phrase appearing at all."""
    response_text = " ".join(item.text for item in items)
    skill_tier_framing = _contains_skill_tier_language(user_message) or _contains_skill_tier_language(response_text)
    return not (skill_tier_framing and _has_percentile_citation(items))
