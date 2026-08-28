"""
Adversarial demonstration that rag/grounding.py's percentile-verification
extension actually catches a fabricated percentile claim, rather than
trusting the LLM's stated number. Same pattern as
rag/example_grounding_check.py's original adversarial test: built and run
BEFORE trusting this in the gated pipeline, not assumed to work because
the code looks right.

Percentile-ness is signaled by an ir_field_path PREFIX
("biomechanics_percentile.<feature>"), not a separate flag field -- see
rag/schema.py's Citation docstring for why the earlier separate-flag
design was replaced (real generations kept the flag correct while still
citing the raw value, breaking the "path resolves to value" invariant).

Usage:
    python rag/example_percentile_grounding_check.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.schema import CoachingResponse  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402

with open(os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_serve.json")) as f:
    serve_event = json.load(f)

benchmark = BiomechanicsPercentileBenchmark()
real_pct = benchmark.percentile(
    serve_event["biomechanics"]["wrist_height_ratio"], serve_event["stroke"]["predicted_class"], "wrist_height_ratio"
)
print(f"Real recomputed percentile for this event's wrist_height_ratio: {real_pct}")

response = CoachingResponse.model_validate({
    "items": [
        {
            "text": "Your contact height on this serve is at the 93.0th percentile among serves in our dataset -- well above typical.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "StrokeEvent",
                "record_id": serve_event["event_id"],
                "ir_field_path": "biomechanics_percentile.wrist_height_ratio",
                "cited_value": real_pct["percentile"],  # correct -- should pass
            },
        },
        {
            "text": "Your contact height on this serve is at the 50.0th percentile (perfectly average).",
            "type": "grounded_observation",
            "citation": {
                "record_type": "StrokeEvent",
                "record_id": serve_event["event_id"],
                "ir_field_path": "biomechanics_percentile.wrist_height_ratio",
                "cited_value": 50.0,  # FABRICATED -- real percentile is ~93rd, not 50th
            },
        },
        {
            "text": "Your shoulder rotation is at the 40th percentile for serve.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "StrokeEvent",
                "record_id": serve_event["event_id"],
                "ir_field_path": "biomechanics_percentile.shoulder_rotation_deg",  # EXCLUDED field -- should be rejected
                "cited_value": 40.0,
            },
        },
        {
            "text": "Your second-serve points won percentage is at the 20th percentile.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "StrokeEvent",
                "record_id": "sess_fake_id_not_provided",  # WRONG record entirely
                "ir_field_path": "biomechanics_percentile.wrist_height_ratio",
                "cited_value": 20.0,
            },
        },
        {
            "text": "Your wrist height ratio value is 5.65.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "StrokeEvent",
                "record_id": serve_event["event_id"],
                "ir_field_path": "biomechanics.wrist_height_ratio",  # RAW-value path, not percentile
                "cited_value": real_pct["percentile"],
                # cited_value here is the PERCENTILE number, but the path says raw-value mode --
                # this must FAIL as a value mismatch (real raw value is ~5.65, not 92.7),
                # confirming the raw-value path isn't silently treated as percentile mode.
            },
        },
    ]
})

source_records = {("StrokeEvent", serve_event["event_id"]): serve_event}

print("\n--- WITHOUT percentile_benchmark (should reject item 1 as unverifiable, not silently pass) ---")
report_no_benchmark = check_grounding(response.model_copy(update={"items": response.items[:1]}), source_records)
print(report_no_benchmark.summary())

print("\n--- WITH percentile_benchmark (real check) ---")
report = check_grounding(response, source_records, percentile_benchmark=benchmark)
print(report.summary())

expected_verdicts = ["GROUNDED", "PERCENTILE_MISMATCH", "PERCENTILE_FIELD_NOT_RANKED", "WRONG_RECORD", "VALUE_MISMATCH"]
actual_verdicts = [v.verdict for v in report.verdicts]
print(f"\nExpected verdict sequence: {expected_verdicts}")
print(f"Actual verdict sequence:   {actual_verdicts}")
assert actual_verdicts == expected_verdicts, "adversarial test did NOT produce the expected verdicts -- do not trust this harness yet"
print("\n[PASS] percentile-verification extension correctly rejects fabricated percentiles, "
      "excluded-field percentile claims, wrong-record citations, and correctly rejects a "
      "percentile number cited against a raw-value path.")
