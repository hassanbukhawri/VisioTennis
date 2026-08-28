"""
Proves the grounding-validation harness (rag/grounding.py) actually catches
what it claims to catch, BEFORE any generation pipeline exists to produce
real LLM feedback. Uses one real SessionRecord
(ir/example_output/session_record_0.json, from Model 4's real
IsolationForest + explain_anomaly() output -- see ir/populate.py) and a
HAND-WRITTEN CoachingResponse (not LLM-generated) constructed to exercise
every verdict path the harness supports:

  - a correctly grounded observation (cites a real field, real value)       -> GROUNDED
  - an observation citing a real field but a WRONG value                    -> VALUE_MISMATCH
  - an observation citing a field path that doesn't exist on the record     -> FIELD_NOT_FOUND
  - an observation citing a record_id that wasn't provided                  -> WRONG_RECORD
  - a clean general note (no citation, no digits)                          -> OK_GENERAL_NOTE
  - a general note that sneaks in a number with no citation                -> SUSPECT_GENERAL_NOTE

If the harness is working, exactly one of the five citable items should pass,
giving a deliberately-bad grounding_rate of 1/4 = 25% on this adversarial
example -- the point is to prove detection works, not to show a good score.

Usage:
    python rag/example_grounding_check.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.schema import CoachingResponse  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")

if __name__ == "__main__":
    with open(SESSION_RECORD_PATH) as f:
        session_record = json.load(f)
    real_id = session_record["record_id"]
    real_unforced_error_rate = session_record["features"]["unforced_error_rate"]
    real_double_fault_rate = session_record["features"]["double_fault_rate"]
    print(f"[setup] loaded real SessionRecord id={real_id}")
    print(f"        real features.unforced_error_rate = {real_unforced_error_rate}")
    print(f"        real features.double_fault_rate    = {real_double_fault_rate}")

    response = CoachingResponse(items=[
        {
            "text": f"Your unforced error rate this session was {real_unforced_error_rate:.2f} per point, "
                    f"noticeably above your usual range -- worth reviewing shot selection on medium-pace balls.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "SessionRecord", "record_id": real_id,
                "ir_field_path": "features.unforced_error_rate",
                "cited_value": real_unforced_error_rate,  # matches -> GROUNDED
            },
        },
        {
            "text": "Your double fault rate was unusually low this session at 0.01 per service point.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "SessionRecord", "record_id": real_id,
                "ir_field_path": "features.double_fault_rate",
                "cited_value": 0.01,  # WRONG -- real value is ~0.159 -> VALUE_MISMATCH
            },
        },
        {
            "text": "Your average first-serve spin rate was well below tour average this session.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "SessionRecord", "record_id": real_id,
                "ir_field_path": "features.first_serve_spin_rpm",  # doesn't exist -> FIELD_NOT_FOUND
                "cited_value": 1800,
            },
        },
        {
            "text": "Compared to your session last Tuesday, your movement speed has dropped noticeably.",
            "type": "grounded_observation",
            "citation": {
                "record_type": "SessionRecord", "record_id": "sess_last_tuesday_not_provided",
                "ir_field_path": "features.court_coverage_speed_mps",  # never given to the harness -> WRONG_RECORD
                "cited_value": 2.1,
            },
        },
        {
            "text": "Keep up the consistent effort -- small, steady improvements add up over a season.",
            "type": "general_note",  # no citation, no digits -> OK_GENERAL_NOTE
        },
        {
            "text": "You've now had 4 sessions in a row with elevated error rates.",
            "type": "general_note",  # digit with no citation -> SUSPECT_GENERAL_NOTE
        },
    ])

    source_records = {("SessionRecord", real_id): session_record}
    report = check_grounding(response, source_records)

    print()
    print(report.summary())
    print()
    print(f"[expected] 1/4 grounded_observation items should pass (25.0%) -- "
          f"this is a deliberately adversarial example proving detection works, "
          f"not a demonstration of a good score.")

    assert report.grounded_observation_count == 4
    assert report.grounded_pass_count == 1
    verdict_by_text = {v.text: v.verdict for v in report.verdicts}
    assert verdict_by_text["Your double fault rate was unusually low this session at 0.01 per service point."] \
        == "VALUE_MISMATCH"
    assert verdict_by_text["Your average first-serve spin rate was well below tour average this session."] \
        == "FIELD_NOT_FOUND"
    assert verdict_by_text["Compared to your session last Tuesday, your movement speed has dropped noticeably."] \
        == "WRONG_RECORD"
    assert verdict_by_text["Keep up the consistent effort -- small, steady improvements add up over a season."] \
        == "OK_GENERAL_NOTE"
    assert verdict_by_text["You've now had 4 sessions in a row with elevated error rates."] \
        == "SUSPECT_GENERAL_NOTE"
    print("\n[PASS] all 6 verdicts matched expectations -- harness correctly distinguishes "
          "grounded claims from fabricated/mismatched/misattributed/mislabeled ones.")

    out_path = os.path.join(REPO_ROOT, "rag/example_output/grounding_check_demo.txt")
    with open(out_path, "w") as f:
        f.write(report.summary())
    print(f"[done] wrote {out_path}")
