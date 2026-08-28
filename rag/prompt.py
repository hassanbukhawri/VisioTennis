"""
Shared prompt construction for session-level coaching feedback -- used
identically regardless of which LLM backend (rag/llm.py) executes it, so a
model comparison is actually comparing the models, not different prompts.

Retrieval (per rag/DESIGN.md Step 1, updated 2026-07-18): a SessionRecord's
`explanation` field (Model 4's real explain_anomaly() z-score output, see
training/anomaly_detection/train.py) names the top-k deviated feature(s) in
plain text, e.g. "double_fault_rate=0.16 (4.6 std above average)". Each
deviation phrase is embedded (rag/embeddings.py, local `all-minilm` via
Ollama) and matched by cosine similarity against embeddings of the full
24-entry KB (rag/kb_placeholder.py) -- real semantic retrieval over the
whole set, not a keyword/topic-string lookup. (An earlier version of this
function did exact keyword matching; replaced once the KB grew past a
handful of 1:1-keyed snippets, per DESIGN.md Step 1.)
"""
import json
import re

from rag.kb_placeholder import PLACEHOLDER_KB
from rag.embeddings import OllamaEmbedder, cosine_similarity

_FEATURE_NAME_RE = re.compile(r"([a-z_]+)=")

_kb_embedding_caches = {}  # keyed by id(kb_list) -- supports multiple independent KBs (session-level, biomechanics)


def build_tone_instruction(tone_profile: dict | None) -> str | None:
    """Returns a short paragraph tuning language complexity to the reader's
    account profile (backend/db.py's User.role/skill_level/years_coaching/
    certification_level), or None if there's nothing to add.

    `tone_profile` is `{"role": ..., "skill_level": ..., "years_coaching": ...,
    "certification_level": ...}` (all optional keys) -- or None entirely, for
    an account that hasn't set a role yet (nullable by design, see
    backend/db.py) or a caller that doesn't have a user in scope at all (eval
    scripts). Returning None in that case means every existing caller that
    doesn't pass a profile keeps generating the exact same prompt text as
    before this feature existed -- this is purely additive.

    Deliberately does NOT scale coach tone by years_coaching/certification_level
    -- coach output is always technical regardless of experience level (a
    flat flag on role, not a second scaling axis to build/maintain). Those
    two fields are collected and stored (backend/db.py) but not read here;
    see PROJECT_LOG.md for why a coach-tone scale was explicitly ruled out."""
    role = (tone_profile or {}).get("role")
    if role == "coach":
        return (
            "TONE: the person reading this is a tennis COACH, not the player themselves -- use "
            "precise, technical biomechanical language throughout (exact joint angles, kinetic-chain "
            "terminology, percentile framing where relevant) rather than simplified player-facing "
            "phrasing. Do not soften or simplify terminology for a coach audience. State each cited "
            "value and its direction/magnitude plainly and trust the reader to interpret it -- do not "
            "add interpretive hedges like 'this alone doesn't indicate a problem' or 'if this doesn't "
            "feel problematic' to a routine finding; a coach doesn't need reminding that one metric "
            "isn't the whole picture. The reference snippets below are written for a PLAYER audience "
            "and often defer to 'get a coach's eye on this' -- for a coach reader, translate that into "
            "the underlying technical point directly instead of repeating the deferral, since the coach "
            "IS that eye. Reserve hedging for a genuine data-quality caveat that holds regardless of "
            "audience (a value measured at a single instant, a measurement that depends on camera "
            "framing, a thin sample size) -- state that plainly as a caveat about the measurement, not "
            "as doubt about whether the finding is worth attention."
        )
    if role == "player":
        skill_level = (tone_profile or {}).get("skill_level")
        if skill_level == "beginner":
            return (
                "TONE: the person reading this is a BEGINNER player -- use simple, plain language, "
                "avoid technical jargon entirely (no joint-angle numbers, no biomechanics terms like "
                "'kinetic chain' or 'shoulder-hip separation'), and if a technical idea is genuinely "
                "necessary, explain it in everyday words. Keep sentences short and concrete."
            )
        if skill_level == "intermediate":
            return (
                "TONE: the person reading this is an INTERMEDIATE player -- plain, direct language is "
                "the default, and light technical terms (naming a joint, a general concept like "
                "'follow-through') are fine if the meaning is clear in context, but avoid dense "
                "biomechanics jargon or precise angle/percentile figures in the prose."
            )
        if skill_level == "advanced":
            return (
                "TONE: the person reading this is an ADVANCED player -- more technical language is "
                "fine (specific joint angles, precise biomechanical terms), though it should still read "
                "as coaching feedback addressed to the player, not a lab report."
            )
    return None


def _get_kb_embeddings(embedder, kb=PLACEHOLDER_KB):
    cache_key = id(kb)
    if cache_key not in _kb_embedding_caches:
        texts = [f"{s.topic}: {s.text}" for s in kb]
        _kb_embedding_caches[cache_key] = embedder.embed(texts)
    return _kb_embedding_caches[cache_key]


def extract_deviation_phrases(session_record: dict) -> list[str]:
    """Splits a SessionRecord's `explanation` field into its individual
    per-feature deviation phrases, e.g. ["double_fault_rate=0.16 (4.6 std
    above average)", "unforced_error_rate=0.34 (3.4 std above average)"].
    Exposed separately (not just inlined in retrieve_snippets) so an
    evaluation script can independently compute "which feature(s) were
    actually flagged" to check retrieval accuracy against."""
    return [p.strip() for p in session_record.get("explanation", "").split(";") if p.strip()]


def extract_deviated_feature_names(session_record: dict) -> list[str]:
    """The feature names named in the explanation, in order, e.g.
    ["double_fault_rate", "unforced_error_rate", "first_serve_points_won_pct"]."""
    return _FEATURE_NAME_RE.findall(session_record.get("explanation", ""))

def build_schema_instructions(
    record_type: str = "SessionRecord",
    record_display_name: str = "SESSION_RECORD",
    id_field_name: str = "record_id",
    feature_scope_rule: str = (
        'Only reference feature values that are actually present in {RECORD}\'s "features" object '
        'or its "anomaly_score"/"anomaly_flag" fields. Do not invent a feature name that is not listed.'
    ),
    include_percentile_rule: bool = False,
    item_count_rule: str = "Produce 3 to 6 items total.",
    require_implication_field: bool = False,
) -> str:
    """Generates the SCHEMA_INSTRUCTIONS text, parameterized by record type
    -- generalized 2026-07-19 for stroke-level biomechanics coaching
    (rag/stroke_prompt.py), which cites StrokeEvent (id field "event_id"),
    not SessionRecord (id field "record_id"). See rag/DESIGN.md "RAG
    integration: Step 1 scope check" -- the id-field-name mismatch and the
    percentile-citation rule are two concrete, non-cosmetic reasons this
    couldn't just be reused verbatim as one fixed string.

    include_percentile_rule adds the citation.percentile_of field to the
    JSON shape and a rule explaining when to use it (rag/schema.py's
    Citation.percentile_of) -- off by default so the exact byte-for-byte
    session-level prompt text this project already validated does not
    change.

    item_count_rule overrides rule 7's fixed text (2026-08-08, for the
    stroke chat's tier-2 path, rag/stroke_prompt.py:build_stroke_chat_prompt
    -- answering one follow-up question needs 1-3 items, not a full 3-6 item
    coaching summary). Defaults to the original hardcoded string, so every
    existing caller's prompt text is byte-identical to before this param
    existed."""
    R = record_display_name  # e.g. "SESSION_RECORD" / "STROKE_EVENT", the placeholder name shown to the LLM
    percentile_field_line = ""
    percentile_rule = ""
    cited_value_caveat = ""
    implication_field_line = ""
    rule_8_text = (
        'A "grounded_observation"\'s "text" must never stop at restating the cited number and its '
        'statistical deviation alone (e.g. "X is 9.07, which is 6.0 standard deviations below average" '
        'and nothing else) -- that is a citation, not coaching. Always connect the value to a concrete '
        "implication for the player: what it suggests, why it matters, or something to try. Use the "
        "reference snippets' framing when one closely matches this specific feature. If no closely-relevant "
        "snippet was retrieved, still give a plain-language, appropriately cautious implication grounded in "
        "the direction/magnitude of the real cited value -- do not invent a cause or a fix the data doesn't "
        "support, but do not fall back to a bare number either."
    )
    if require_implication_field:
        # Schema-level version of the same rule (2026-08-26, lever #3 --
        # see rag/implication_schema.py): "implication" is now its OWN
        # required JSON key for every grounded_observation item, not just
        # something the "text" sentence is instructed to fold in. Kept
        # SEPARATE from "text" deliberately -- "text" can stay a clean,
        # citation-grounded description; "implication" is where the
        # concrete takeaway is required to live, and StrictCoachingResponse
        # will reject the whole response if it's missing, empty, or a
        # copy of "text".
        implication_field_line = (
            ',\n      "implication": "<REQUIRED for grounded_observation: a concrete takeaway distinct '
            'from \\"text\\" -- what this suggests, why it matters, or something to try. Must not be '
            'empty or a copy of \\"text\\". Omit this key entirely for general_note items.>"'
        )
        rule_8_text = (
            'Every "grounded_observation" item MUST include a non-empty "implication" key (see the shape '
            'above), DISTINCT from "text" -- this is enforced by the response schema itself, not just this '
            'instruction: a grounded_observation with a missing, empty, or "text"-duplicate "implication" '
            'will fail to parse. "text" states the finding (the cited value and its direction/magnitude, in '
            'plain player-facing language); "implication" states the concrete takeaway: what it suggests, '
            "why it matters, or something to try. Use the reference snippets' framing when one closely "
            "matches this specific feature. If no closely-relevant snippet was retrieved, still give a "
            "plain-language, appropriately cautious implication grounded in the direction/magnitude of the "
            "real cited value -- do not invent a cause or a fix the data doesn't support, but do not leave "
            '"implication" as a restatement of "text" either. "implication" must be omitted entirely (not '
            'present as a key) for "general_note" items.'
        )
    if include_percentile_rule:
        # Design note (see rag/schema.py's Citation docstring): percentile-ness is
        # signaled by ir_field_path's PREFIX, not a separate flag field -- an
        # earlier design used a separate "percentile_of" field and every real
        # generation set it correctly while still citing the raw value in
        # "cited_value" (breaking the "ir_field_path resolves to cited_value"
        # invariant this schema relies on everywhere else). Folding the signal
        # into the path itself means that invariant never has an exception.
        percentile_rule = (
            f'\n9. To CITE a PERCENTILE, use "ir_field_path": "biomechanics_percentile.<feature>" (note the '
            f'DIFFERENT prefix -- e.g. "biomechanics_percentile.wrist_height_ratio", NOT '
            f'"biomechanics.wrist_height_ratio") and set "cited_value" to the percentile number (0-100). To '
            f'state the RAW value instead, use "ir_field_path": "biomechanics.<feature>" (no "_percentile") and '
            f'"cited_value" is the raw number. Never mix these: cited_value must always equal exactly what '
            f'ir_field_path\'s prefix says it is. Do not state a percentile for any field not explicitly listed '
            f'as available for percentile ranking in the reference snippets/context below -- some fields have '
            f'no valid percentile and must not be described in percentile terms. IMPORTANT -- this rule governs '
            f'the CITATION JSON only, not the prose: the "text" sentence itself must NEVER spell out the '
            f'percentile number or the word "percentile"/"ratio" (e.g. do NOT write "at the 80th percentile" or '
            f'"your wrist height ratio is X"). Instead describe what it means in the same plain, direct, '
            f'player-facing language as the reference snippets below (e.g. "unusually high", "on the higher '
            f'end", "typical for this shot") -- the exact number lives only in "cited_value", never spelled out '
            f'in "text".'
        )
        cited_value_caveat = ' (unless "ir_field_path" uses the "biomechanics_percentile." prefix -- see rule 9)'
    feature_scope_rule = feature_scope_rule.format(RECORD=R)

    # The citation object's own JSON key is ALWAYS literally "record_id"
    # (that's the fixed Pydantic field name on rag/schema.py:Citation,
    # regardless of record_type) -- but the SOURCE record's own id field is
    # named differently per record type (SessionRecord.record_id vs.
    # StrokeEvent.event_id). Found by running the actual LLM (2026-07-19,
    # see rag/DESIGN.md): the original single-string template conflated
    # these two into `id_field_name`, so for StrokeEvent it correctly told
    # the model to copy from "event_id" but ALSO told it to name the
    # citation's own JSON key "event_id" -- producing a citation object
    # missing the required "record_id" key on every single generation,
    # a 100% failure rate, not a rare hallucination. When id_field_name
    # already equals "record_id" (the SessionRecord case), the two
    # concepts coincide and the ORIGINAL simpler wording is kept verbatim
    # (verified byte-identical) rather than switched to the more verbose
    # disambiguating phrasing, since there's nothing to disambiguate there.
    if id_field_name == "record_id":
        record_id_key_line = f'"record_id": "<copy the record_id field from the {R} JSON below, exactly>"'
        record_id_rule = f'"record_id" must be copied exactly from {R}\'s own "record_id" field. Do not invent or reference any other record.'
    else:
        record_id_key_line = (
            f'"record_id": "<the citation JSON key is ALWAYS literally \\"record_id\\" -- copy the VALUE '
            f'from {R}\'s own \\"{id_field_name}\\" field into it, exactly>"'
        )
        record_id_rule = (
            f'The citation\'s JSON key is always "record_id" (never "{id_field_name}", even though that is '
            f'the name of the corresponding field on {R} itself) -- its VALUE must be copied exactly from '
            f'{R}\'s own "{id_field_name}" field. Do not invent or reference any other record.'
        )

    return f"""You must respond with ONLY a single JSON object (no markdown fences, no prose before or after) matching exactly this shape:

{{
  "items": [
    {{
      "text": "<one coaching sentence>",
      "type": "grounded_observation" | "general_note",
      "citation": {{
        "record_type": "{record_type}",
        {record_id_key_line},
        "ir_field_path": "<a dotted path into {R}, e.g. features.unforced_error_rate>",
        "cited_value": <copy the exact numeric value found at that path in {R} -- do not round, do not recompute>{percentile_field_line}
      }}{implication_field_line}
    }}
  ]
}}

Hard rules:
1. Every item with type "grounded_observation" MUST include a "citation" object. Every item with type "general_note" must NOT include a "citation" key at all.
2. A "citation"'s "cited_value" must be copied EXACTLY (same number of decimal places, no rounding) from the {R} JSON provided below at the path named in "ir_field_path"{cited_value_caveat}. Do not state any number that is not copied verbatim from {R} in this way.
3. {record_id_rule}
4. Do not use "general_note" for any sentence that contains a number, a comparison ("higher than", "below average", etc.), or any claim that could be verified/falsified -- those must be "grounded_observation" with a citation.
5. {feature_scope_rule}
6. Match the plain, direct, player-facing tone of the coaching reference snippets below in every "text" sentence -- not jargon-heavy measurement reporting -- and use them for framing and general advice content; any advice text must still be paired with a citation to a REAL {R} field if it makes a specific/comparative claim.
7. {item_count_rule}
8. {rule_8_text}{percentile_rule}"""


SCHEMA_INSTRUCTIONS = build_schema_instructions()  # SessionRecord defaults -- byte-identical to the original fixed string


def retrieve_snippets_for_phrases(phrases: list[str], kb: list, query_prefix: str, top_k=3, embedder=None):
    """KB-agnostic semantic retrieval, generalized 2026-07-19 out of the
    original session-level-only retrieve_snippets() so stroke-level
    biomechanics coaching (rag/stroke_prompt.py) reuses this exact ranking
    logic rather than duplicating it -- the retrieval algorithm itself
    doesn't care whether phrases came from a SessionRecord's z-score
    explanation or a StrokeEvent's biomechanics percentiles, only the
    caller-supplied `kb` list and `query_prefix` text differ.

    Embeds each phrase, embeds every KB snippet once (cached per kb list
    identity), and for each phrase picks its highest-cosine-similarity
    snippet not already claimed by an earlier phrase -- deduplicated across
    phrases, capped at top_k. Nothing here guarantees the retrieved
    snippet's `topic` matches the phrase's feature name, which is exactly
    what makes it worth checking empirically (see
    rag/run_gated_diverse_trials.py).

    Fixed 2026-07-18 (on the original session-level-only version): an
    earlier version only checked each phrase's single argmax and skipped
    the phrase entirely if that top match was already claimed by an
    earlier phrase, rather than falling through to its next-best unclaimed
    candidate -- this silently dropped a legitimately relevant snippet
    whenever two phrases' top picks collided. Now falls through the full
    similarity ranking per phrase."""
    if not phrases:
        return []

    embedder = embedder or OllamaEmbedder()
    kb_embeddings = _get_kb_embeddings(embedder, kb)
    query_texts = [f"{query_prefix}{p}" for p in phrases]
    query_embeddings = embedder.embed(query_texts)

    picked = []
    seen_idx = set()
    for q_emb in query_embeddings:
        sims = [cosine_similarity(q_emb, kb_emb) for kb_emb in kb_embeddings]
        ranked_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        for idx in ranked_idx:
            if idx not in seen_idx:
                picked.append(kb[idx])
                seen_idx.add(idx)
                break
        if len(picked) >= top_k:
            break
    return picked


def retrieve_snippets(session_record: dict, top_k=3, embedder=None):
    """Session-level retrieval: embeds each deviation phrase from the
    SessionRecord's `explanation` (Model 4's real explain_anomaly() z-score
    output) and ranks against the full session-level KB
    (rag/kb_placeholder.py). Thin wrapper around
    retrieve_snippets_for_phrases() -- see that function for the retrieval
    algorithm itself and the dedup-fallthrough bug fix."""
    phrases = extract_deviation_phrases(session_record)
    return retrieve_snippets_for_phrases(
        phrases, PLACEHOLDER_KB, "Player session shows this statistical deviation: ", top_k, embedder)


def build_prompt(session_record: dict, embedder=None, use_few_shot: bool = False,
                  require_implication_field: bool = False) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt).

    use_few_shot (2026-08-26, default False -- purely additive, every
    existing caller's prompt text is byte-identical to before this param
    existed): appends rag/few_shot_examples.py's 2 real, already-grounded
    example responses to the system prompt. Off by default until the A/B
    test (rag/run_few_shot_ab_trial.py) shows a real improvement -- see
    PROJECT_LOG.md, "LLM-quality-lever test," item 2.

    require_implication_field (2026-08-26, default False -- same
    byte-identical-by-default guarantee): swaps in
    build_schema_instructions(require_implication_field=True) instead of
    the cached SCHEMA_INSTRUCTIONS constant. Pair with
    rag/implication_schema.py:StrictCoachingResponse as the parsing schema
    (see rag/run_implication_field_ab_trial.py) -- passing this without
    also parsing against StrictCoachingResponse would tell the model to
    emit a field nothing then validates. PROJECT_LOG.md, "LLM-quality-lever
    test," item 3."""
    snippets = retrieve_snippets(session_record, embedder=embedder)

    schema_instructions = (
        build_schema_instructions(require_implication_field=True)
        if require_implication_field else SCHEMA_INSTRUCTIONS
    )
    system_prompt = (
        "You are a tennis coaching assistant generating session feedback for a player. "
        "You must base every factual or comparative claim STRICTLY on the SESSION_RECORD JSON "
        "given to you in the user message -- never on general tennis knowledge, never on numbers "
        "you compute yourself. This output will be checked automatically against the real "
        "SESSION_RECORD data; any claim that doesn't trace back exactly will be flagged as "
        "ungrounded.\n\n" + schema_instructions
    )
    if use_few_shot:
        from rag.few_shot_examples import build_few_shot_block
        system_prompt += "\n\n" + build_few_shot_block()

    kb_block = "\n\n".join(
        f"[{s.topic}] (source: {s.source})\n{s.text}" for s in snippets
    ) or "(no matching reference snippets retrieved for this session's deviations)"

    user_prompt = (
        f"SESSION_RECORD:\n{json.dumps(session_record, indent=2)}\n\n"
        f"COACHING REFERENCE SNIPPETS (for tone/framing only -- these are hand-authored "
        f"placeholder content, NOT an authoritative coaching source, see labels above):\n"
        f"{kb_block}\n\n"
        f"Generate the coaching feedback JSON now."
    )
    return system_prompt, user_prompt
