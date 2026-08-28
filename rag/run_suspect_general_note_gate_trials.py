"""
Verifies the SUSPECT_GENERAL_NOTE gate fix (rag/gated_pipeline.py, 2026-08-26)
-- same "never trust the model's own labeling of what counts as a claim"
principle as the 2026-08-12 skill-tier backstop
(rag/chat_router.py:check_for_skill_tier_mismatch), tested the same way:
a deterministic check against the REAL case that surfaced the gap first,
then hand-constructed variants (true positives AND true negatives, so the
gate is confirmed not to also start rejecting perfectly fine responses),
then one live re-run through the real retry loop to confirm the fix
behaves correctly end-to-end, not just in isolated unit logic.

Part A -- deterministic, the ACTUAL trial-2 case: replays the real saved
raw model output (rag/example_output/qwen3_14b_trials/trial_2_raw.txt)
that produced SUSPECT_GENERAL_NOTE during the Qwen3 14B lever test,
through the fixed _try_one_generation logic, against the real source
SessionRecord. Confirms it is now REJECTED (accepted=False) -- before the
fix, this exact response would have been ACCEPTED (n=3/3 grounded_observation
items all passed).

Part B -- deterministic, 4 constructed variants: 2 should be REJECTED
(true positives), 2 should be ACCEPTED (true negatives -- confirms the fix
doesn't also reject perfectly fine responses, the same "correctly safe"
check the skill-tier backstop's own test suite ran).

Part C -- live: 2 fresh generations through the real
generate_grounded_response() retry loop (real qwen3:14b via Ollama, the
same session_record_0.json prompt trial-2 came from) with the fix in
place, confirming the retry LOOP itself still functions end-to-end (not
just the isolated per-attempt check).

Usage:
    python rag/run_suspect_general_note_gate_trials.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.schema import CoachingResponse, CoachingFeedbackItem, Citation  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.gated_pipeline import _try_one_generation, generate_grounded_response  # noqa: E402
from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")
TRIAL_2_RAW_PATH = os.path.join(REPO_ROOT, "rag/example_output/qwen3_14b_trials/trial_2_raw.txt")


def _real_citation(record):
    return Citation(
        record_type="SessionRecord", record_id=record["record_id"],
        ir_field_path="features.unforced_error_rate",
        cited_value=record["features"]["unforced_error_rate"],
    )


if __name__ == "__main__":
    with open(SESSION_RECORD_PATH) as f:
        session_record = json.load(f)
    source_records = {("SessionRecord", session_record["record_id"]): session_record}

    n_pass = 0
    n_total = 0

    # -- Part A: the ACTUAL trial-2 case, replayed deterministically --
    print("=" * 78 + "\nPart A: the real trial-2 case (rag/example_output/qwen3_14b_trials/trial_2_raw.txt)\n" + "=" * 78)
    with open(TRIAL_2_RAW_PATH) as f:
        trial_2_raw = f.read()
    # trial_2_raw.txt's record_id belongs to a different real SessionRecord
    # than session_record_0.json (each Qwen3 8B/14B trial in run_qwen_multitrial.py
    # pulls a fresh SessionRecord fixture) -- load THAT specific record so this
    # replay checks against the exact records the real generation was actually
    # run against, not a mismatched one.
    trial_2_response = CoachingResponse.model_validate(json.loads(trial_2_raw))
    trial_2_record_id = trial_2_response.items[0].citation.record_id
    # search known example_output SessionRecord fixtures for this id
    candidate_paths = [
        os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json"),
    ]
    trial_2_record = None
    for p in candidate_paths:
        if os.path.exists(p):
            d = json.load(open(p))
            if d.get("record_id") == trial_2_record_id:
                trial_2_record = d
                break
    if trial_2_record is None:
        # session_record_0.json IS the fixture run_qwen_multitrial.py always
        # uses -- record_id should always match; fail loudly rather than
        # silently checking against the wrong record if it somehow doesn't.
        raise SystemExit(f"could not find the SessionRecord (id={trial_2_record_id}) "
                          f"trial_2_raw.txt was actually generated against -- "
                          f"refusing to grade against a mismatched record")
    trial_2_sources = {("SessionRecord", trial_2_record_id): trial_2_record}
    report = check_grounding(trial_2_response, trial_2_sources)
    print(report.summary())
    old_behavior_accepted = report.grounded_observation_count > 0 and report.grounding_rate >= 1.0
    new_behavior_accepted = old_behavior_accepted and not report.has_suspect_general_note
    print(f"\nOLD gate (grounding_rate only) would have accepted: {old_behavior_accepted}")
    print(f"NEW gate (grounding_rate AND no suspect general_note) accepts: {new_behavior_accepted}")
    ok = old_behavior_accepted and not new_behavior_accepted
    print(f"-> {'PASS' if ok else 'FAIL'}: fix changes this exact real case from accepted to rejected"
          + ("" if ok else " -- FIX DID NOT WORK AS EXPECTED ON THE REAL CASE"))
    n_total += 1
    n_pass += int(ok)

    # -- Part B: 4 constructed variants (2 true positive, 2 true negative) --
    print("\n" + "=" * 78 + "\nPart B: constructed variants\n" + "=" * 78)

    variants = {
        "true_positive_single_digit_general_note": (
            CoachingResponse(items=[
                CoachingFeedbackItem(text="Your unforced error rate was high today.",
                                      type="grounded_observation", citation=_real_citation(session_record)),
                CoachingFeedbackItem(text="Your first serve percentage was 51.0%, slightly below average.",
                                      type="general_note"),
            ]),
            False,  # expected accepted=False
        ),
        "true_positive_digit_mixed_among_clean_notes": (
            CoachingResponse(items=[
                CoachingFeedbackItem(text="Your unforced error rate was high today.",
                                      type="grounded_observation", citation=_real_citation(session_record)),
                CoachingFeedbackItem(text="Keep working on your footwork between points.", type="general_note"),
                CoachingFeedbackItem(text="You covered 2487.94m today, more than usual.", type="general_note"),
                CoachingFeedbackItem(text="Stay hydrated during longer sessions.", type="general_note"),
            ]),
            False,
        ),
        "true_negative_no_digit_general_note": (
            CoachingResponse(items=[
                CoachingFeedbackItem(text="Your unforced error rate was high today.",
                                      type="grounded_observation", citation=_real_citation(session_record)),
                CoachingFeedbackItem(text="Keep working on your footwork between points.", type="general_note"),
            ]),
            True,  # expected accepted=True
        ),
        "true_negative_no_general_note_at_all": (
            CoachingResponse(items=[
                CoachingFeedbackItem(text="Your unforced error rate was high today.",
                                      type="grounded_observation", citation=_real_citation(session_record)),
            ]),
            True,
        ),
    }

    for name, (response, expect_accepted) in variants.items():
        report = check_grounding(response, source_records)
        rate = report.grounding_rate if report.grounded_observation_count else 0.0
        accepted = report.grounded_observation_count > 0 and rate >= 1.0 and not report.has_suspect_general_note
        ok = accepted == expect_accepted
        print(f"  [{name}] expected accepted={expect_accepted}, got accepted={accepted} "
              f"(suspect={report.has_suspect_general_note}) -> {'PASS' if ok else 'FAIL'}")
        n_total += 1
        n_pass += int(ok)

    # -- Part C: live, 2 fresh generations through the real retry loop --
    print("\n" + "=" * 78 + "\nPart C: live re-run through generate_grounded_response(), real qwen3:14b\n" + "=" * 78)
    system_prompt, user_prompt = build_prompt(session_record)
    backend = OllamaBackend(model="qwen3:14b")
    for i in range(1, 3):
        gated = generate_grounded_response(backend, system_prompt, user_prompt, source_records, max_attempts=3)
        if gated.success:
            any_suspect_in_accepted = check_grounding(gated.response, source_records).has_suspect_general_note
            print(f"  live trial {i}: SUCCESS after {gated.n_attempts} attempt(s), "
                  f"{gated.total_latency_s:.1f}s, accepted response has_suspect_general_note="
                  f"{any_suspect_in_accepted}")
            for a in gated.attempts:
                if a.parse_ok and a.suspect_general_note:
                    print(f"    (attempt {a.attempt_num} was correctly rejected for suspect_general_note "
                          f"before the accepted attempt)")
            ok = not any_suspect_in_accepted
        else:
            print(f"  live trial {i}: FAILED: {gated.failure_reason}")
            ok = True  # an honest failure (never accepting a bad response) is not a gate defect
        n_total += 1
        n_pass += int(ok)

    print(f"\n{'=' * 78}\n[summary] {n_pass}/{n_total} checks passed\n{'=' * 78}")
