"""
Stroke-level coaching prompt construction, built on top of the
session-level machinery in rag/prompt.py (retrieve_snippets_for_phrases,
build_schema_instructions) rather than duplicating it -- see
rag/DESIGN.md "RAG integration: Step 1 scope check" for exactly what does
and doesn't reuse cleanly:

- CoachingResponse/CoachingFeedbackItem/Citation (rag/schema.py), the
  retrieval ranking algorithm, and the grounding-check machinery for a RAW
  field citation all reuse unchanged.
- What does NOT reuse unchanged, and is built fresh here: (1) which
  features are "notable" for a stroke -- SessionRecord already has this
  precomputed by explain_anomaly(); a StrokeEvent's biomechanics has no
  upstream anomaly-flagging step, so a selection rule is needed
  (top-k by |percentile - 50|, the percentile-domain analog of a z-score
  magnitude ranking); (2) the prompt's id-field name (StrokeEvent uses
  event_id, not record_id); (3) the percentile-citation JSON rule
  (rag/prompt.py:build_schema_instructions(include_percentile_rule=True)).
"""
import json

from rag.prompt import build_schema_instructions, build_tone_instruction, retrieve_snippets_for_phrases
from rag.kb_biomechanics import BIOMECHANICS_KB
from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark, RANKED_FEATURES

STROKE_SCHEMA_INSTRUCTIONS = build_schema_instructions(
    record_type="StrokeEvent",
    record_display_name="STROKE_EVENT",
    id_field_name="event_id",
    feature_scope_rule=(
        'Only reference biomechanical values that are actually present in {RECORD}\'s "biomechanics" '
        'object. Do not invent a feature name that is not listed there, and do not state a percentile '
        'for a field not present in the reference snippets/context below.'
    ),
    include_percentile_rule=True,
)


def select_notable_biomechanics_features(stroke_event: dict, benchmark: BiomechanicsPercentileBenchmark, top_k=3):
    """Analogous to explain_anomaly()'s top-k z-score deviation ranking, but
    for percentiles: a StrokeEvent's biomechanics has no upstream
    anomaly-flagging step (unlike SessionRecord, whose `explanation` is
    precomputed by Model 4), so "which feature(s) are worth commenting on"
    has no existing answer to reuse -- ranks every RANKED_FEATURES field
    present on this event by |percentile - 50| (furthest from the middle
    of this stroke class's own distribution first) and returns the top_k.

    Returns a list of dicts: {feature, value, percentile, n, low_confidence}."""
    biomechanics = stroke_event.get("biomechanics")
    stroke_class = stroke_event.get("stroke", {}).get("predicted_class")
    if not biomechanics or not stroke_class:
        return []

    scored = []
    for feature in RANKED_FEATURES:
        if feature not in biomechanics:
            continue
        value = biomechanics[feature]
        result = benchmark.percentile(value, stroke_class, feature)
        scored.append(dict(feature=feature, value=value, stroke_class=stroke_class, **result))

    scored.sort(key=lambda r: abs(r["percentile"] - 50), reverse=True)
    return scored[:top_k]


def _build_citable_view(stroke_event: dict) -> dict:
    """Returns a TRIMMED view of stroke_event containing only the fields
    biomechanics coaching is actually allowed to cite (event_id,
    player_side, stroke.predicted_class/confidence, biomechanics.*) --
    NOT the full StrokeEvent (ball trajectory, court keypoints/homography,
    class_probabilities), which biomechanics coaching can never cite (rule
    5) but which made the first Step-4 validation's prompt ~3.8x larger
    than the session-level one and correlated with the model abandoning
    the required JSON schema on every attempt (see rag/DESIGN.md).

    IMPORTANT: this trimmed dict is used ONLY for what's shown in the
    prompt text. The grounding harness (rag/gated_pipeline.py's
    source_records) is still given the FULL, untrimmed stroke_event --
    trimming what's SHOWN never trims what's actually verifiable, so a
    citation to a field outside this view would still correctly resolve
    (or fail) against the real record, it just isn't something the model
    is being shown or encouraged to reference."""
    return {
        "event_id": stroke_event["event_id"],
        "player_side": stroke_event.get("player_side"),
        "stroke": {
            "predicted_class": stroke_event["stroke"]["predicted_class"],
            "confidence": stroke_event["stroke"]["confidence"],
        },
        "biomechanics": stroke_event["biomechanics"],
    }


def build_stroke_prompt(
    stroke_event: dict, benchmark: BiomechanicsPercentileBenchmark, embedder=None, tone_profile: dict | None = None,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for stroke-level biomechanics
    coaching -- the StrokeEvent analog of rag/prompt.py:build_prompt().

    Shows the model a TRIMMED citable view (_build_citable_view), not the
    full StrokeEvent -- see that function's docstring. The grounding
    harness still checks against the full record via source_records,
    unaffected by this trimming.

    tone_profile (2026-08-10, backend/db.py's User.role/skill_level/etc.,
    see rag/prompt.py:build_tone_instruction) -- optional, defaults to None
    so every existing caller that doesn't pass one (eval/trial scripts)
    keeps generating the exact same prompt text as before this feature
    existed."""
    notable = select_notable_biomechanics_features(stroke_event, benchmark)
    phrases = [
        f"{n['feature']}={n['value']:.3f} ({n['percentile']}th percentile for {n['stroke_class']})"
        for n in notable
    ]
    snippets = retrieve_snippets_for_phrases(
        phrases, BIOMECHANICS_KB, "Player's stroke shows this biomechanical percentile: ", top_k=3, embedder=embedder)

    tone_instruction = build_tone_instruction(tone_profile)
    tone_block = f"\n\n{tone_instruction}" if tone_instruction else ""

    system_prompt = (
        "You are a tennis coaching assistant generating stroke-technique feedback for a player, "
        "based on body-pose-derived biomechanical measurements from ONE identified stroke. "
        "You must base every factual or comparative claim STRICTLY on the STROKE_EVENT JSON given "
        "to you in the user message -- never on general tennis knowledge, never on numbers you "
        "compute yourself, including percentiles: only state a percentile that is explicitly given "
        "to you in this prompt's biomechanics context, never estimate or guess one. This output "
        "will be checked automatically against the real STROKE_EVENT data and its real percentile "
        "ranking; any claim that doesn't trace back exactly will be flagged as ungrounded."
        + tone_block + "\n\n"
        + STROKE_SCHEMA_INSTRUCTIONS
    )

    notable_block = "\n".join(
        f"- {n['feature']} = {n['value']:.3f}, which is the {n['percentile']}th percentile among "
        f"{n['stroke_class']} clips in our dataset (n={n['n']}){' [LOW-CONFIDENCE TAIL ESTIMATE -- small reference sample]' if n['low_confidence'] else ''}"
        for n in notable
    ) or "(no notable biomechanical percentiles available for this stroke)"

    kb_block = "\n\n".join(
        f"[{s.topic}] (source: {s.source})\n{s.text}" for s in snippets
    ) or "(no matching reference snippets retrieved for this stroke's biomechanics)"

    citable_view = _build_citable_view(stroke_event)
    user_prompt = (
        f"STROKE_EVENT:\n{json.dumps(citable_view, indent=2)}\n\n"
        f"NOTABLE BIOMECHANICAL PERCENTILES for this stroke (precomputed against our own dataset's "
        f"per-stroke-class population -- these are the ONLY percentiles you may state; do not invent "
        f"or recompute any other percentile):\n{notable_block}\n\n"
        f"COACHING REFERENCE SNIPPETS (for tone/framing only -- these are hand-authored placeholder "
        f"content, NOT an authoritative coaching source, see labels above):\n{kb_block}\n\n"
        f"Generate the stroke-technique coaching feedback JSON now."
    )
    return system_prompt, user_prompt


def _format_chat_history(conversation_history: list[tuple[str, str]]) -> str:
    if not conversation_history:
        return "(no earlier turns in this conversation)"
    return "\n".join(f"{role}: {text}" for role, text in conversation_history)


CHAT_SCHEMA_INSTRUCTIONS = build_schema_instructions(
    record_type="StrokeEvent", record_display_name="STROKE_EVENT", id_field_name="event_id",
    feature_scope_rule=(
        'Only reference biomechanical values that are actually present in {RECORD}\'s "biomechanics" '
        'object. Do not invent a feature name that is not listed there, and do not state a percentile '
        'for a field not present in the reference snippets/context below.'
    ),
    include_percentile_rule=True,
    item_count_rule=(
        'Produce 1 to 3 items that directly answer the player\'s NEW question below -- do NOT repeat '
        'feedback already given earlier in this conversation, and do not answer questions the player '
        'did not ask.'
    ),
)


def build_stroke_chat_prompt(
    stroke_event: dict, benchmark: BiomechanicsPercentileBenchmark,
    conversation_history: list[tuple[str, str]], user_message: str, embedder=None, tone_profile: dict | None = None,
) -> tuple[str, str]:
    """Tier-2 ("new_grounded_claim") chat prompt -- the multi-turn analog of
    build_stroke_prompt(), reusing the exact same citable view, notable-
    percentile selection, and KB retrieval, but scoped to answering ONE new
    follow-up question (with conversation history for context) instead of
    generating the initial 3-6 item coaching summary. Goes through the same
    generate_grounded_response() retry-gate unchanged -- this only changes
    what's asked for, not how the answer is verified.

    See rag/chat_router.py's module docstring for where this fits in the
    overall 4-route chat design (PROJECT_LOG.md's chat grounding architecture
    entry) -- this is called only after classify_turn() has already decided
    the turn needs a fresh grounded claim.

    tone_profile: see build_stroke_prompt()'s docstring -- same optional,
    additive-only convention."""
    notable = select_notable_biomechanics_features(stroke_event, benchmark)
    phrases = [
        f"{n['feature']}={n['value']:.3f} ({n['percentile']}th percentile for {n['stroke_class']})"
        for n in notable
    ]
    snippets = retrieve_snippets_for_phrases(
        phrases, BIOMECHANICS_KB, "Player's stroke shows this biomechanical percentile: ", top_k=3, embedder=embedder)

    tone_instruction = build_tone_instruction(tone_profile)
    tone_block = f"\n\n{tone_instruction}" if tone_instruction else ""

    system_prompt = (
        "You are a tennis coaching assistant continuing a chat conversation with a player about ONE "
        "identified stroke. The player has already received initial coaching feedback (shown below as "
        "conversation history) and is now asking a NEW follow-up question. You must base every factual "
        "or comparative claim in your answer STRICTLY on the STROKE_EVENT JSON given to you in the user "
        "message -- never on general tennis knowledge, never on numbers you compute yourself, including "
        "percentiles: only state a percentile that is explicitly given to you in this prompt's "
        "biomechanics context, never estimate or guess one. This output will be checked automatically "
        "against the real STROKE_EVENT data and its real percentile ranking; any claim that doesn't "
        "trace back exactly will be flagged as ungrounded."
        + tone_block + "\n\n"
        + CHAT_SCHEMA_INSTRUCTIONS
    )

    notable_block = "\n".join(
        f"- {n['feature']} = {n['value']:.3f}, which is the {n['percentile']}th percentile among "
        f"{n['stroke_class']} clips in our dataset (n={n['n']}){' [LOW-CONFIDENCE TAIL ESTIMATE -- small reference sample]' if n['low_confidence'] else ''}"
        for n in notable
    ) or "(no notable biomechanical percentiles available for this stroke)"

    kb_block = "\n\n".join(
        f"[{s.topic}] (source: {s.source})\n{s.text}" for s in snippets
    ) or "(no matching reference snippets retrieved for this stroke's biomechanics)"

    citable_view = _build_citable_view(stroke_event)
    user_prompt = (
        f"STROKE_EVENT:\n{json.dumps(citable_view, indent=2)}\n\n"
        f"NOTABLE BIOMECHANICAL PERCENTILES for this stroke (precomputed against our own dataset's "
        f"per-stroke-class population -- these are the ONLY percentiles you may state; do not invent "
        f"or recompute any other percentile):\n{notable_block}\n\n"
        f"COACHING REFERENCE SNIPPETS (for tone/framing only -- these are hand-authored placeholder "
        f"content, NOT an authoritative coaching source, see labels above):\n{kb_block}\n\n"
        f"CONVERSATION SO FAR:\n{_format_chat_history(conversation_history)}\n\n"
        f'PLAYER\'S NEW QUESTION: "{user_message}"\n\n'
        f"Generate the JSON reply now."
    )
    return system_prompt, user_prompt
