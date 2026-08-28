"""
Few-shot examples for the session-level coaching-generation prompt
(rag/prompt.py:build_prompt) -- lever #2 of the 2026-08-26 LLM-quality-lever
test (see PROJECT_LOG.md).

Both examples below are REAL, already-generated, already-100%-grounded
model outputs, not hand-written -- reconstructed exactly from
`rag/example_output/qwen3_gated_diverse_trials/{name}_report.txt` (the
verified [GROUNDED] text + field path + cited value for each item) paired
with the real `record_id` from the source
`ir/example_output/session_record_diverse_{name}.json` SessionRecord.
Re-validated against `rag/schema.py:CoachingResponse` and
`rag/grounding.py:check_grounding()` before being used here (both score
3/3, 100%) -- not just copied and trusted. `clutch_breakdown` and
`movement_fatigue` were picked because both succeeded in ONE attempt at
FULL (1.0) retrieval recall in the original diverse-session trial
(`rag/example_output/qwen3_gated_diverse_trials/summary.json`) -- the
cleanest, least-retry-assisted examples available, not cherry-picked past
that objective bar.

Deliberately excludes `rally_style_shift`, `second_serve_ace_decline`,
`serve_breakdown`, and `unforced_error_spike` -- those stay held out as
this lever's own A/B test set (`rag/run_few_shot_ab_trial.py`), so the
few-shot prompt is never evaluated on the exact sessions it was built
from.
"""
FEW_SHOT_EXAMPLES = [
    {
        "session_summary": "break_points_saved_pct=0.00 (4.6 std below average); "
                            "unforced_error_rate=0.25 (1.6 std above average); "
                            "avg_reaction_time_ms=264.77 (1.5 std above average)",
        "response": {
            "items": [
                {
                    "text": "Your break-point save rate was zero today, but a handful of break points "
                             "isn't enough to draw real conclusions. Monitor this number across your next "
                             "few sessions before treating it as a pattern.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b932148d-edeb-4ca5-820a-e4fbd6d3c38e",
                        "ir_field_path": "features.break_points_saved_pct",
                        "cited_value": 0.0,
                    },
                },
                {
                    "text": "You made more unforced errors than usual today. Check when they happened "
                             "— a cluster late in the session points to fatigue, so rest more between "
                             "reps; errors on one specific shot point to that shot's technique.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b932148d-edeb-4ca5-820a-e4fbd6d3c38e",
                        "ir_field_path": "features.unforced_error_rate",
                        "cited_value": 0.2489394921634392,
                    },
                },
                {
                    "text": "Your reaction time was slower than usual today, especially later in the "
                             "session — that's a classic fatigue marker, not a technical issue. Check "
                             "how long you'd been playing and what time of day it was before reading too "
                             "much into it.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b932148d-edeb-4ca5-820a-e4fbd6d3c38e",
                        "ir_field_path": "features.avg_reaction_time_ms",
                        "cited_value": 264.7665917403021,
                    },
                },
            ]
        },
    },
    {
        "session_summary": "avg_reaction_time_ms=403.56 (6.0 std above average); "
                            "court_coverage_speed_mps=0.76 (6.0 std below average); "
                            "total_distance_covered_m=689.72 (6.0 std below average)",
        "response": {
            "items": [
                {
                    "text": "Your reaction time was slower than usual today, which is a classic fatigue "
                             "marker. Check how long you'd been playing and the time of day before "
                             "overinterpreting.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b9d345d2-d56a-4f19-9f96-f7127c5f0fc1",
                        "ir_field_path": "features.avg_reaction_time_ms",
                        "cited_value": 403.5580028607616,
                    },
                },
                {
                    "text": "You moved slower than usual today, but your total distance covered was "
                             "normal. This suggests less urgency on individual points, not less overall "
                             "effort. Check your split-step timing on video.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b9d345d2-d56a-4f19-9f96-f7127c5f0fc1",
                        "ir_field_path": "features.court_coverage_speed_mps",
                        "cited_value": 0.7578988393420136,
                    },
                },
                {
                    "text": "You covered less ground than usual today. Before treating that as a "
                             "fitness or effort issue, check whether the session was shorter or points "
                             "ended quicker than usual.",
                    "type": "grounded_observation",
                    "citation": {
                        "record_type": "SessionRecord",
                        "record_id": "b9d345d2-d56a-4f19-9f96-f7127c5f0fc1",
                        "ir_field_path": "features.total_distance_covered_m",
                        "cited_value": 689.7231725258166,
                    },
                },
            ]
        },
    },
]


def build_few_shot_block() -> str:
    """Renders FEW_SHOT_EXAMPLES into the exact prompt text block, with an
    explicit framing that these are FORMAT/QUALITY calibration only, not
    content to reuse -- a session-level generation call always has its own
    real SESSION_RECORD, and copying these citations onto a different
    record_id would itself be a grounding failure the harness would catch,
    but the instruction is stated up front rather than relying on that
    backstop alone."""
    import json

    blocks = []
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, start=1):
        blocks.append(
            f"Example {i} -- a DIFFERENT real session (deviations: {ex['session_summary']}):\n"
            f"{json.dumps(ex['response'], indent=2)}"
        )
    examples_text = "\n\n".join(blocks)
    return (
        "EXAMPLE OUTPUTS (real, previously generated, independently verified 100% grounded -- "
        "for FORMAT and QUALITY calibration only. These are from DIFFERENT sessions with DIFFERENT "
        "record_ids -- do not reuse their record_id, field paths, or cited values for the current "
        "session below; every citation must point at THIS session's own SESSION_RECORD. Notice each "
        "item pairs the cited number with a concrete implication or check, in plain player-facing "
        "language -- match that quality bar, not just the JSON shape.)\n\n"
        f"{examples_text}"
    )
