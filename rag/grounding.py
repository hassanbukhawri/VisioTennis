"""
Grounding validation harness for RAG coaching feedback.

The proposal's target is >95% factual grounding rate. That number is only
meaningful if there's an automated, mechanical way to check it -- this module
is that check. It does NOT use NLP/heuristic parsing of free-text sentences
to guess whether a claim is "factual"; instead it relies on rag/schema.py's
structural constraint (every substantive claim must be an explicit
`grounded_observation` item with a `Citation`) and does exact verification:

    1. Does the cited record (record_type, record_id) actually exist in the
       set of real IR records this feedback was generated from? (catches an
       LLM citing a plausible-sounding but nonexistent/wrong record)
    2. Does the cited field path resolve against that record's real data?
       (catches a fabricated field name)
    3. Does the resolved value match what was cited? (catches a fabricated
       or mis-transcribed number)

grounding_rate = (# grounded_observation items that pass all three checks)
                / (# grounded_observation items total)

`general_note` items are excluded from the rate (they're not claims), but are
still scanned for a stray digit as a soft warning -- a general_note
containing a number is a likely sign a factual claim slipped past the
citation requirement rather than being flagged as one.

This harness is deliberately independent of any generation code -- it
consumes a `CoachingResponse` (rag/schema.py), however it was produced, plus
the real IR records it's allowed to reference. Built and demonstrated (see
rag/example_grounding_check.py) BEFORE any LLM call exists in this project,
per the design-before-code discipline used for the IR layer.
"""
import re
from dataclasses import dataclass, field
from typing import Any

from rag.schema import CoachingResponse

NUMERIC_PATTERN = re.compile(r"\d")
_PATH_TOKEN_RE = re.compile(r"[^.\[\]]+|\[\d+\]")

# Module-level so other verification code (rag/chat_router.py's skill-tier
# backstop) can identify a percentile citation without duplicating this
# string -- see Citation's docstring in rag/schema.py for what this prefix means.
PERCENTILE_PREFIX = "biomechanics_percentile."


class FieldPathError(Exception):
    pass


def resolve_field_path(record: dict, path: str) -> Any:
    """Resolves a dotted path, e.g. 'features.unforced_error_rate' or
    'ball.trajectory[3].pixel.x_px', against `record` (a plain dict, e.g.
    from StrokeEvent.model_dump() / SessionRecord.model_dump()). Raises
    FieldPathError -- never silently returns None -- if any segment of the
    path doesn't exist, so a fabricated field name is always caught rather
    than resolving to a coincidental None/default."""
    tokens = _PATH_TOKEN_RE.findall(path)
    if not tokens:
        raise FieldPathError(f"empty or unparseable field path {path!r}")
    cur = record
    consumed = []
    for tok in tokens:
        if tok.startswith("["):
            idx = int(tok[1:-1])
            if not isinstance(cur, list) or idx >= len(cur):
                raise FieldPathError(
                    f"index {tok} invalid after {'.'.join(consumed) or '<root>'} in path {path!r}")
            cur = cur[idx]
        else:
            if not isinstance(cur, dict) or tok not in cur:
                raise FieldPathError(
                    f"key {tok!r} not found after {'.'.join(consumed) or '<root>'} in path {path!r}")
            cur = cur[tok]
        consumed.append(tok)
    return cur


def _values_match(cited, actual, rel_tol=1e-3, abs_tol=1e-6):
    if isinstance(cited, bool) or isinstance(actual, bool):
        return cited == actual
    if isinstance(cited, (int, float)) and isinstance(actual, (int, float)):
        return abs(cited - actual) <= max(abs_tol, rel_tol * abs(actual))
    return cited == actual


@dataclass
class ItemVerdict:
    text: str
    type: str
    verdict: str  # GROUNDED | WRONG_RECORD | FIELD_NOT_FOUND | VALUE_MISMATCH | OK_GENERAL_NOTE | SUSPECT_GENERAL_NOTE
    detail: str = ""


@dataclass
class GroundingReport:
    verdicts: list = field(default_factory=list)

    @property
    def grounded_observation_count(self):
        return sum(1 for v in self.verdicts if v.type == "grounded_observation")

    @property
    def grounded_pass_count(self):
        return sum(1 for v in self.verdicts if v.type == "grounded_observation" and v.verdict == "GROUNDED")

    @property
    def grounding_rate(self):
        n = self.grounded_observation_count
        return self.grounded_pass_count / n if n else float("nan")

    @property
    def has_suspect_general_note(self):
        """True if any general_note item contains a digit with no citation
        -- a likely factual claim that dodged the citation requirement by
        being mislabeled as non-factual. Deliberately kept OUT of
        grounding_rate's own denominator (that metric's definition -- "of
        the items that WERE submitted as claims, how many check out" --
        stays unchanged, still meaningful on its own for existing callers
        that only read grounding_rate/summary()). This property exists so a
        CALLER that wants "no ungrounded claim anywhere, regardless of how
        it was labeled" can check that separately -- see
        rag/gated_pipeline.py's retry-gate, which does exactly that
        (2026-08-26, closing the gap found during the Qwen3 14B trial:
        PROJECT_LOG.md, 'LLM-quality-lever test')."""
        return any(v.verdict == "SUSPECT_GENERAL_NOTE" for v in self.verdicts)

    def summary(self) -> str:
        n = self.grounded_observation_count
        if n:
            head = (f"grounding_rate = {self.grounded_pass_count}/{n} grounded_observation "
                     f"items pass ({self.grounding_rate * 100:.1f}%)")
        else:
            head = "grounding_rate = N/A (no grounded_observation items in this response)"
        lines = [head]
        n_suspect = sum(1 for v in self.verdicts if v.verdict == "SUSPECT_GENERAL_NOTE")
        if n_suspect:
            lines.append(f"WARNING: {n_suspect} general_note item(s) contain a digit with no "
                         f"citation -- possible ungrounded claim mislabeled as non-factual")
        lines.append("")
        for v in self.verdicts:
            suffix = f" -- {v.detail}" if v.detail else ""
            lines.append(f"  [{v.verdict}] ({v.type}) {v.text!r}{suffix}")
        return "\n".join(lines)


def check_grounding(response: CoachingResponse, source_records: dict, percentile_benchmark=None) -> GroundingReport:
    """source_records: {(record_type, record_id): record_dict} -- the exact
    set of real IR records this feedback is allowed to cite (typically the
    StrokeEvent/SessionRecord(s) that were actually retrieved/passed into the
    generation prompt for this response). A citation naming any other
    (record_type, record_id) is rejected as WRONG_RECORD even if that ID
    looks plausible -- the harness never trusts the LLM's claim about which
    record it used, only what's independently known to have been provided.

    percentile_benchmark: optional ir.biomechanics_percentiles.BiomechanicsPercentileBenchmark
    (or None). Required only for citations whose ir_field_path starts with
    "biomechanics_percentile." (stroke-level biomechanics percentile
    claims -- see rag/schema.py's Citation docstring). A percentile claim
    can't be checked by static field-path resolution the way a raw value
    can: it must be RECOMPUTED
    (percentile_benchmark.percentile(raw_value, stroke_class, feature)) and
    compared against what was cited, keyed off the record's own
    stroke.predicted_class -- a genuinely different verification operation
    from the dict-path lookup used for every other citation."""
    report = GroundingReport()
    for item in response.items:
        if item.type == "general_note":
            has_digit = bool(NUMERIC_PATTERN.search(item.text))
            verdict = "SUSPECT_GENERAL_NOTE" if has_digit else "OK_GENERAL_NOTE"
            detail = "contains a digit but was not submitted as a citable claim" if has_digit else ""
            report.verdicts.append(ItemVerdict(item.text, item.type, verdict, detail))
            continue

        c = item.citation
        key = (c.record_type, c.record_id)
        if key not in source_records:
            report.verdicts.append(ItemVerdict(
                item.text, item.type, "WRONG_RECORD",
                f"no known {c.record_type} with id={c.record_id} among the records provided"))
            continue
        record = source_records[key]

        if not c.ir_field_path.startswith(PERCENTILE_PREFIX):
            try:
                actual = resolve_field_path(record, c.ir_field_path)
            except FieldPathError as e:
                report.verdicts.append(ItemVerdict(item.text, item.type, "FIELD_NOT_FOUND", str(e)))
                continue
            if not _values_match(c.cited_value, actual):
                report.verdicts.append(ItemVerdict(
                    item.text, item.type, "VALUE_MISMATCH",
                    f"cited {c.cited_value!r}, actual value at {c.ir_field_path!r} is {actual!r}"))
                continue
            report.verdicts.append(ItemVerdict(
                item.text, item.type, "GROUNDED", f"{c.ir_field_path}={actual!r} confirmed"))
            continue

        # "biomechanics_percentile.<feature>": recompute the percentile of the
        # underlying RAW field, don't trust the LLM's stated number.
        feature = c.ir_field_path[len(PERCENTILE_PREFIX):]
        raw_path = f"biomechanics.{feature}"
        try:
            actual = resolve_field_path(record, raw_path)
        except FieldPathError as e:
            report.verdicts.append(ItemVerdict(item.text, item.type, "FIELD_NOT_FOUND", str(e)))
            continue
        if percentile_benchmark is None:
            report.verdicts.append(ItemVerdict(
                item.text, item.type, "PERCENTILE_BENCHMARK_MISSING",
                f"citation claims a percentile ({c.ir_field_path}) but no percentile_benchmark "
                f"was provided to check_grounding() -- cannot verify"))
            continue
        try:
            stroke_class = resolve_field_path(record, "stroke.predicted_class")
        except FieldPathError as e:
            report.verdicts.append(ItemVerdict(
                item.text, item.type, "FIELD_NOT_FOUND",
                f"percentile citation requires record.stroke.predicted_class: {e}"))
            continue
        try:
            result = percentile_benchmark.percentile(float(actual), stroke_class, feature)
        except ValueError as e:
            report.verdicts.append(ItemVerdict(item.text, item.type, "PERCENTILE_FIELD_NOT_RANKED", str(e)))
            continue
        computed_pct = result["percentile"]
        if not _values_match(c.cited_value, computed_pct, abs_tol=0.6):
            report.verdicts.append(ItemVerdict(
                item.text, item.type, "PERCENTILE_MISMATCH",
                f"cited {c.cited_value!r}th percentile, recomputed {computed_pct!r}th percentile "
                f"for {raw_path}={actual!r} among {stroke_class} clips (n={result['n']})"))
            continue
        report.verdicts.append(ItemVerdict(
            item.text, item.type, "GROUNDED",
            f"{raw_path}={actual!r} confirmed at {computed_pct}th percentile "
            f"for {stroke_class} (n={result['n']})"))
    return report
