"""
Runtime retry-gate for the RAG coaching layer.

Motivation: the 6-trial Qwen3 8B reliability eval (rag/run_qwen_multitrial.py,
see rag/DESIGN.md "LLM choice -- RESOLVED") found the model is NOT uniformly
reliable on an identical prompt -- 4/6 trials were fully grounded, 1/6
hallucinated a fake record_id + nonexistent fields (0% grounded), and 1/6
failed to parse at all (a required field omitted on a trailing item). That is
not acceptable to surface directly to a user. rag/grounding.py already gives a
mechanical, non-NLP way to detect both failure modes -- this module turns that
offline check into an online gate: generate, check, and only ever return a
response that is 100% grounded and well-formed. If every attempt in the
budget fails, return an explicit failure state instead of a bad response.

This is NOT a smarter prompt, a different model, or a fix to Qwen3's
occasional hallucination -- it is a mechanical backstop that trades latency
(extra generations on failure) for a hard guarantee: nothing ungrounded or
malformed reaches a user. See rag/DESIGN.md "Retry-gate design" for the
before/after reliability numbers this bought in practice.
"""
import json
import time
from dataclasses import dataclass, field

from rag.grounding import GroundingReport, check_grounding
from rag.llm import GenerationResult
from rag.run_qwen_eval import clean_json_text
from rag.schema import CoachingResponse

FULLY_GROUNDED_TOL = 1.0  # require grounding_rate == 100% (not just "high")


@dataclass
class AttemptRecord:
    attempt_num: int
    latency_s: float
    parse_ok: bool
    parse_error: str = ""
    grounded_observation_count: int = 0
    grounded_pass_count: int = 0
    suspect_general_note: bool = False
    accepted: bool = False


@dataclass
class GatedResult:
    success: bool
    response: CoachingResponse | None
    report: GroundingReport | None
    attempts: list = field(default_factory=list)  # list[AttemptRecord]
    total_latency_s: float = 0.0
    failure_reason: str = ""

    @property
    def n_attempts(self):
        return len(self.attempts)


def _try_one_generation(backend, system_prompt, user_prompt, source_records, attempt_num,
                         percentile_benchmark=None, response_schema=CoachingResponse):
    """Runs one generation + parse + grounding check. Returns
    (AttemptRecord, CoachingResponse | None, GroundingReport | None).

    percentile_benchmark: passed straight through to check_grounding() --
    None for session-level coaching (unchanged behavior), an
    ir.biomechanics_percentiles.BiomechanicsPercentileBenchmark for
    stroke-level biomechanics coaching (see rag/stroke_prompt.py).

    response_schema (2026-08-26, default CoachingResponse -- every existing
    caller unchanged): which Pydantic model to validate the raw JSON
    against. Lets rag/run_implication_field_ab_trial.py pass
    rag/implication_schema.py:StrictCoachingResponse for the "after"
    condition of that lever's A/B test without duplicating this function.

    SUSPECT_GENERAL_NOTE gate (2026-08-26, closing the gap found during the
    Qwen3 14B trial -- PROJECT_LOG.md, "LLM-quality-lever test"): an
    attempt is now rejected (not just flagged) if ANY general_note item
    contains an uncited digit, even when every grounded_observation item is
    100% grounded. Same principle as the 2026-08-12 skill-tier backstop
    (rag/chat_router.py:check_for_skill_tier_mismatch) -- never trust the
    model's own labeling of what counts as a "claim"; a general_note with a
    real number in it is a claim that dodged the citation requirement, and
    gets forced through it via a retry (regenerate from scratch), the same
    remedy as any other ungrounded content -- not silently accepted with a
    warning attached."""
    result: GenerationResult = backend.generate(system_prompt, user_prompt)

    try:
        cleaned = clean_json_text(result.raw_text)
        response = response_schema.model_validate(json.loads(cleaned))
    except Exception as e:
        rec = AttemptRecord(
            attempt_num=attempt_num, latency_s=result.latency_s,
            parse_ok=False, parse_error=str(e),
        )
        return rec, None, None

    report = check_grounding(response, source_records, percentile_benchmark=percentile_benchmark)
    n = report.grounded_observation_count
    passed = report.grounded_pass_count
    rate = report.grounding_rate if n else 0.0
    suspect = report.has_suspect_general_note
    fully_grounded = n > 0 and rate >= FULLY_GROUNDED_TOL and not suspect

    rec = AttemptRecord(
        attempt_num=attempt_num, latency_s=result.latency_s, parse_ok=True,
        grounded_observation_count=n, grounded_pass_count=passed,
        suspect_general_note=suspect, accepted=fully_grounded,
    )
    return rec, response, report


def generate_grounded_response(backend, system_prompt, user_prompt, source_records,
                                max_attempts=3, percentile_benchmark=None,
                                response_schema=CoachingResponse) -> GatedResult:
    """Generates via `backend`, validates + grounds the result, and retries
    (regenerating from scratch -- same prompt, same temperature, so retries
    rely on run-to-run sampling variance, not a changed prompt) up to
    `max_attempts` times. Returns as soon as an attempt is both well-formed
    AND 100% grounded. If no attempt qualifies within the budget, returns a
    GatedResult with success=False and response=None -- callers must treat
    that as an explicit "could not generate a grounded response" failure,
    never fall back to the last (ungrounded/malformed) attempt.

    response_schema: see _try_one_generation's docstring."""
    attempts = []
    total_latency = 0.0

    for i in range(1, max_attempts + 1):
        rec, response, report = _try_one_generation(
            backend, system_prompt, user_prompt, source_records, i,
            percentile_benchmark=percentile_benchmark, response_schema=response_schema)
        attempts.append(rec)
        total_latency += rec.latency_s

        if rec.accepted:
            return GatedResult(
                success=True, response=response, report=report,
                attempts=attempts, total_latency_s=total_latency,
            )

    last = attempts[-1]
    if not last.parse_ok:
        reason = (f"all {max_attempts} attempts failed: last attempt did not parse "
                  f"as a valid CoachingResponse ({last.parse_error})")
    elif last.grounded_pass_count < last.grounded_observation_count or last.grounded_observation_count == 0:
        reason = (f"all {max_attempts} attempts failed: last attempt grounded "
                  f"{last.grounded_pass_count}/{last.grounded_observation_count} "
                  f"items, below the required 100%")
    else:
        # All grounded_observation items passed (100%) but a general_note
        # item contained an uncited digit -- a different failure mode from
        # the one above, worth a distinct message rather than the
        # misleading "below the required 100%" when it was actually 100%.
        reason = (f"all {max_attempts} attempts failed: last attempt's "
                  f"{last.grounded_pass_count}/{last.grounded_observation_count} grounded_observation "
                  f"items were fully grounded, but at least one general_note item contained an "
                  f"uncited digit (SUSPECT_GENERAL_NOTE) -- rejected per the same-citation-requirement "
                  f"gate, not accepted with a warning")

    return GatedResult(
        success=False, response=None, report=None,
        attempts=attempts, total_latency_s=total_latency,
        failure_reason=reason,
    )
