"""
Structured coaching-feedback schema for the RAG grounding-validation harness.

DESIGN ARTIFACT -- see rag/DESIGN.md, "Prompt template design," for the
rationale. Summary: rather than parsing free-form LLM prose after the fact to
guess which sentences make a factual claim, the prompt template REQUIRES the
LLM to emit feedback as a list of explicitly typed items -- each either a
"grounded_observation" (must cite an exact IR field path + the value found
there) or a "general_note" (no citation, and must not smuggle in a numeric
claim). This turns "is >95% of the feedback grounded" from a fuzzy NLP
judgment call into an exact, mechanical check: does this citation's field
path resolve against the real record, and does the resolved value match what
was cited? See rag/grounding.py for the checker.

No generation code depends on this yet -- this schema (and the harness built
against it in rag/grounding.py) exists and is verified BEFORE any LLM call is
made, per this project's design-before-code discipline.
"""
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    """Points at exactly one field in one real IR record (StrokeEvent or
    SessionRecord) that a coaching sentence claims to describe. record_id is
    required (not just a field path) so a citation can't point at "some
    plausible-looking field" in the abstract -- it must name the specific
    record instance the feedback is about, which the harness then looks up
    for real.

    Percentile citations (added for stroke-level biomechanics coaching,
    see rag/DESIGN.md "RAG integration: Step 1 scope check"): a
    "biomechanics_percentile.<feature>" ir_field_path (instead of the
    usual "biomechanics.<feature>") means cited_value is the POPULATION
    PERCENTILE of that field's real value, computed against the reference
    population for the record's own stroke.predicted_class -- see
    rag/grounding.py's percentile_benchmark parameter, which recomputes and
    compares rather than doing a static dict lookup.

    Design note: an earlier version used a separate `percentile_of` flag
    field alongside a "biomechanics.<feature>" path, asking the model to
    keep cited_value's MEANING in sync with an out-of-band flag. Every
    single real generation broke that sync the same way (correctly set
    the flag, text, and field name, but still cited the RAW value instead
    of the percentile) -- the model reliably follows "ir_field_path
    resolves to cited_value" everywhere else in this schema, so an
    exception to that invariant, signaled by a field elsewhere in the
    object, was the wrong design. Folding the signal INTO the path itself
    (a different prefix, not a different flag) keeps a single invariant
    for the model to hold: whatever ir_field_path says, that's what
    cited_value must equal -- it's just that "biomechanics_percentile.X"
    resolves via a live recomputation instead of a static lookup, on the
    verification side, transparently to the model.

    SessionDigest citations (added 2026-08-12, ir/schema.py:SessionDigest)
    need NO new verification logic at all, unlike percentile citations --
    ir_field_path is a plain dotted path (e.g.
    "trailing_averages.wrist_height_ratio", "trend_deltas.knee_flex_right_deg",
    "stroke_type_mode", "session_count") that resolves against the digest's
    own model_dump() via the SAME resolve_field_path()/_values_match() the
    "biomechanics.<feature>" branch already uses -- because the digest's
    fields already ARE the pre-computed values being cited, there's nothing
    left to recompute at verification time the way a percentile citation
    needs. This is the whole point of extending record_type rather than
    building a parallel checking path: the existing exact-match mechanism
    already covers it."""
    record_type: Literal["StrokeEvent", "SessionRecord", "SessionDigest"]
    record_id: str
    ir_field_path: str  # dotted path into the record's model_dump(), e.g. "features.unforced_error_rate";
    # or "biomechanics_percentile.<feature>" for a percentile citation (see class docstring)
    cited_value: float | int | str | bool


class CoachingFeedbackItem(BaseModel):
    text: str
    type: Literal["grounded_observation", "general_note"]
    citation: Citation | None = None
    # implication (2026-08-26, added for the LLM-quality-lever test's item 3,
    # PROJECT_LOG.md): OPTIONAL here and NOT enforced by this base class --
    # every existing caller (gated_pipeline.py, chat_pipeline.py,
    # backend/app.py, stroke_prompt.py) keeps parsing exactly as before,
    # whether or not a response includes this key. The HARD requirement
    # (non-empty, distinct from `text`) that actually tightens the schema
    # lives in rag/implication_schema.py:StrictCoachingResponse, a separate
    # subclass used only by that lever's own A/B trial
    # (rag/run_implication_field_ab_trial.py) -- kept out of this file so
    # testing a stricter schema can never accidentally break every other
    # already-working caller of the production CoachingResponse/
    # CoachingFeedbackItem classes.
    implication: str | None = None

    @model_validator(mode="after")
    def check_citation_presence(self):
        if self.type == "grounded_observation" and self.citation is None:
            raise ValueError("grounded_observation items must include a citation")
        if self.type == "general_note" and self.citation is not None:
            raise ValueError("general_note items must not include a citation -- "
                              "use grounded_observation instead")
        return self


class CoachingResponse(BaseModel):
    """Top-level structured output for one coaching-feedback generation call
    (whatever produced it -- an LLM, or, for harness testing, hand-written)."""
    items: list[CoachingFeedbackItem] = Field(min_length=1)


class ChatRouteDecision(BaseModel):
    """Output of the chat turn-router (rag/chat_router.py) -- classifies one
    user chat turn against a specific stroke's already-generated coaching
    BEFORE any new generation happens, so the expensive retry-gate only runs
    when a turn actually needs it. See PROJECT_LOG.md's chat grounding
    architecture entry for the full design (4-way routing, why tier 1 never
    generates new text, the tier-3 misclassification backstop).

    "existing_citation": this turn is asking about a topic already covered
      by one of this session's real grounded_observation citations --
      matched_field_path names WHICH one (its ir_field_path, e.g.
      "biomechanics_percentile.wrist_height_ratio"). Routes to replaying the
      stored item, never a fresh generation -- so a wrong match here can
      only surface a real, already-verified (if slightly off-topic) answer,
      never a fabricated one.
    "new_grounded_claim": this turn asks something about the stroke's real
      data that ISN'T already covered by an existing citation -- needs a
      fresh generation through the full retry-gate (rag/gated_pipeline.py).
      Also the fail-safe default when classification itself is uncertain,
      ambiguous, or fails to parse -- always safe, only costs latency.
    "general_note": generic advice/knowledge with no implied comparison to
      the player's own data (e.g. "what's a good drill for this").
    "unavailable_comparison": sounds like it wants a personalized/benchmark
      comparison this app doesn't have data for (e.g. "what's normal for my
      level") -- gets a fixed, code-authored disclaimer, never one the LLM
      is trusted to remember to write itself.
    "progress_digest" (added 2026-08-12): the message asks about the
      player's progress/trend ACROSS SESSIONS ("how am I trending", "am I
      improving", "how's my progress been") -- distinct from
      "unavailable_comparison" (which is for comparisons this app genuinely
      CANNOT ground, e.g. to another player or a past session's raw data)
      because a real, freshly-computed SessionDigest (ir/schema.py) now
      exists for exactly this question. Routes to a dedicated grounded
      generation using that digest, through the same retry-gate as
      "new_grounded_claim" -- see rag/chat_pipeline.py."""
    route: Literal[
        "existing_citation", "new_grounded_claim", "general_note",
        "unavailable_comparison", "progress_digest",
    ]
    matched_field_path: str | None = None

    @model_validator(mode="after")
    def check_matched_field_path_presence(self):
        if self.route == "existing_citation" and not self.matched_field_path:
            raise ValueError('route "existing_citation" requires a non-empty matched_field_path')
        if self.route != "existing_citation" and self.matched_field_path is not None:
            raise ValueError(f'matched_field_path must be null for route {self.route!r}')
        return self
