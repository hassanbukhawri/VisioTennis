"""
Strict schema variant for lever #3 of the 2026-08-26 LLM-quality-lever test
(PROJECT_LOG.md) -- "tighten the structured-output schema to reduce the
thin/generic failure mode at the source rather than patching it after
generation."

The 2026-08-21 fix for this same failure mode (PROJECT_LOG.md, "2.2 --
thin/generic bare-restatement pattern") was a PROMPT INSTRUCTION only
(rag/prompt.py:build_schema_instructions() rule 8: "must connect [the
cited value] to a concrete implication"). That's a rule the model can
still just not follow -- nothing about the JSON becomes malformed if it
doesn't. This module makes the same requirement a SCHEMA constraint
instead: a `grounded_observation` item must include a non-empty
`implication` field, DISTINCT from `text`, or the response fails to parse
at all -- so a bare restatement can no longer validate as well-formed
JSON in the first place, not just get flagged by an instruction.

Deliberately a SEPARATE subclass, not a change to rag/schema.py's
production `CoachingFeedbackItem`/`CoachingResponse` (see that file's
`implication` field comment) -- so testing whether this actually helps can
never accidentally break every other already-working caller of the
production schema while the lever is still being evaluated.
"""
from pydantic import Field, model_validator

from rag.schema import Citation, CoachingFeedbackItem, CoachingResponse

IMPLICATION_MIN_LENGTH = 15  # chars, after strip() -- see StrictCoachingFeedbackItem docstring


class StrictCoachingFeedbackItem(CoachingFeedbackItem):
    """Same shape as CoachingFeedbackItem, plus one additional hard
    constraint on `implication`.

    Two mechanical checks, no NLP judgment call -- consistent with this
    project's grounding-harness philosophy of exact, non-fuzzy validation:
    1. non-empty after strip(), at least IMPLICATION_MIN_LENGTH chars --
       rejects an empty string, whitespace, or a lazy placeholder like
       "N/A"/"-"/"none", without trying to judge prose QUALITY (that's the
       A/B trial's job, not this validator's).
    2. not identical (case-insensitive, stripped) to `text` -- blocks the
       most obvious way to game check #1: copying the same sentence into
       both fields instead of adding a real implication.
    Neither check can catch a technically-present-but-still-thin
    implication (e.g. a generic "monitor this over time" that doesn't
    engage with the specific finding) -- that's exactly what
    rag/run_implication_field_ab_trial.py evaluates qualitatively, on real
    generations, not assumed solved by adding the field."""

    @model_validator(mode="after")
    def check_implication_presence(self):
        if self.type != "grounded_observation":
            return self
        if self.implication is None or len(self.implication.strip()) < IMPLICATION_MIN_LENGTH:
            raise ValueError(
                f'grounded_observation items must include a non-empty "implication" '
                f"(at least {IMPLICATION_MIN_LENGTH} characters) -- a concrete takeaway, "
                f"not just the cited value restated"
            )
        if self.implication.strip().lower() == self.text.strip().lower():
            raise ValueError('"implication" must not be identical to "text" -- restating the '
                              "same sentence in both fields is not a real implication")
        return self


class StrictCoachingResponse(CoachingResponse):
    items: list[StrictCoachingFeedbackItem] = Field(min_length=1)
