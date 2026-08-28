"""
Adversarial grounding test for SessionDigest citations (2026-08-12) --
mirrors the exact standard rag/example_grounding_check.py set for the
original StrokeEvent/SessionRecord grounding harness, and the deterministic
unit test rag/chat_router.py's skill-tier backstop was verified against:
hand-constructed adversarial CoachingResponse objects (no LLM involved),
checked against check_grounding()/check_for_skill_tier_mismatch() directly,
proving the mechanism catches each specific failure mode -- not just
trusting the design because it was sound on paper.

Per explicit instruction: "construct a case designed to produce a false or
unverifiable trend claim, confirm it's caught and rejected."

Usage:
    python rag/run_session_digest_grounding_check.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from ir.schema import SessionDigest  # noqa: E402
from rag.chat_router import check_for_skill_tier_mismatch  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.schema import Citation, CoachingFeedbackItem, CoachingResponse  # noqa: E402

# A real digest (not fabricated) -- same shape compute_session_digest()
# produces, hand-copied from a real run against this project's own dev DB
# (test1@example.com, 7 real stored sessions) so the "real record" side of
# every adversarial case is genuine, not synthetic.
REAL_DIGEST = SessionDigest(
    record_id="82bbf801-9043-4df2-b4ce-9f22d7e9f792",
    player_id="f79976ff-4537-44a6-aee7-0e2d59412148",
    lookback_k=5,
    session_count=5,
    total_sessions_considered=7,
    stroke_type_mode="backhand",
    trailing_averages={
        "wrist_height_ratio": 81.2, "shoulder_hip_separation_deg": 39.3,
        "knee_flex_left_deg": 63.6, "knee_flex_right_deg": 97.1,
        "stance_width_ratio": 65.9, "weight_transfer_ratio": 99.4,
        "follow_through_extension_ratio": 73.7, "elbow_angle_deg": 27.8,
    },
    trend_deltas={
        "wrist_height_ratio": -17.5, "shoulder_hip_separation_deg": -51.7,
        "knee_flex_left_deg": -2.2, "knee_flex_right_deg": 0.1,
        "stance_width_ratio": 10.8, "weight_transfer_ratio": 2.4,
        "follow_through_extension_ratio": -19.7, "elbow_angle_deg": 24.8,
    },
    trend_prior_window_size=2,
    source_session_ids=["a", "b", "c", "d", "e", "f", "g"],
)
DIGEST_DICT = REAL_DIGEST.model_dump(mode="json")
SOURCE_RECORDS = {("SessionDigest", REAL_DIGEST.record_id): DIGEST_DICT}
BENCHMARK = BiomechanicsPercentileBenchmark()


def _case(name, response, expected_verdicts, user_message=None, expect_skill_tier_catch=None):
    report = check_grounding(response, SOURCE_RECORDS, percentile_benchmark=BENCHMARK)
    actual_verdicts = [v.verdict for v in report.verdicts]
    ok = actual_verdicts == expected_verdicts
    print(f"[{name}]")
    print(f"  expected: {expected_verdicts}")
    print(f"  actual:   {actual_verdicts}")
    for v in report.verdicts:
        if v.detail:
            print(f"    -- {v.detail}")
    print(f"  {'PASS' if ok else 'FAIL -- GROUNDING CHECK DID NOT CATCH THIS'}")

    skill_tier_ok = True
    if user_message is not None:
        safe = check_for_skill_tier_mismatch(user_message, response.items)
        skill_tier_caught = not safe  # "caught" means the backstop correctly flagged it unsafe
        skill_tier_ok = skill_tier_caught == expect_skill_tier_catch
        print(f"  skill-tier backstop: safe={safe} (expected catch={expect_skill_tier_catch}) "
              f"{'PASS' if skill_tier_ok else 'FAIL'}")
    print()
    return ok and skill_tier_ok


if __name__ == "__main__":
    results = []

    # 1. Wrong record_id -- claims a plausible-looking but nonexistent digest id.
    results.append(_case(
        "wrong_record_id",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your backhand has been improving.", type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id="fabricated-digest-id-0000",
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=-17.5),
        )]),
        expected_verdicts=["WRONG_RECORD"],
    ))

    # 2. Fabricated field path -- a feature name that doesn't exist in trailing_averages/trend_deltas.
    results.append(_case(
        "fabricated_field_path",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your racket head speed has increased.", type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=REAL_DIGEST.record_id,
                               ir_field_path="trend_deltas.racket_head_speed", cited_value=12.0),
        )]),
        expected_verdicts=["FIELD_NOT_FOUND"],
    ))

    # 3. Value mismatch -- real field, real record, WRONG number (the actual
    # false-trend-claim case: claiming a much bigger improvement than really happened).
    results.append(_case(
        "value_mismatch_inflated_trend",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your wrist height ratio has climbed dramatically in recent sessions.",
            type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=REAL_DIGEST.record_id,
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=45.0),
            # real value is -17.5 (a DECLINE), not +45.0 -- both wrong magnitude AND wrong direction
        )]),
        expected_verdicts=["VALUE_MISMATCH"],
    ))

    # 4. Citing a trend field when trend_deltas is EMPTY on the real record --
    # simulates a new player with only 1 usable session (no trend window exists
    # at all) where the model still tries to claim a trend. No special-casing
    # needed: the key simply doesn't exist in the real (empty) dict.
    thin_digest = REAL_DIGEST.model_copy(update={
        "session_count": 1, "total_sessions_considered": 1, "trend_deltas": {}, "trend_prior_window_size": 0,
        "source_session_ids": ["a"],
    })
    thin_dict = thin_digest.model_dump(mode="json")
    thin_source_records = {("SessionDigest", thin_digest.record_id): thin_dict}
    thin_report = check_grounding(
        CoachingResponse(items=[CoachingFeedbackItem(
            text="You've been improving steadily.", type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=thin_digest.record_id,
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=-17.5),
        )]),
        thin_source_records, percentile_benchmark=BENCHMARK,
    )
    ok4 = [v.verdict for v in thin_report.verdicts] == ["FIELD_NOT_FOUND"]
    print("[fabricated_trend_with_no_trend_window]")
    print(f"  expected: ['FIELD_NOT_FOUND']")
    print(f"  actual:   {[v.verdict for v in thin_report.verdicts]}")
    print(f"  {'PASS' if ok4 else 'FAIL'}\n")
    results.append(ok4)

    # 5. Legitimate, correctly-cited claim -- confirms the mechanism doesn't
    # just reject everything (a checker that fails everything would trivially
    # "catch" all the above for the wrong reason).
    results.append(_case(
        "legitimate_correct_citation",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your wrist height has been trending lower in recent sessions.",
            type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=REAL_DIGEST.record_id,
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=-17.5),
        )]),
        expected_verdicts=["GROUNDED"],
    ))

    # 6. Skill-tier mismatch backstop, extended to SessionDigest citations --
    # a real, correctly-grounded digest citation answering a skill-tier-framed
    # progress question ("has my progress been good for my level?").
    results.append(_case(
        "skill_tier_mismatch_on_digest_citation",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your wrist height has been trending lower in recent sessions.",
            type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=REAL_DIGEST.record_id,
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=-17.5),
        )]),
        expected_verdicts=["GROUNDED"],  # mechanically grounded -- the NUMBER is real
        user_message="has my progress been good for a player at my level?",
        expect_skill_tier_catch=True,  # but the backstop must still reject it for the framing mismatch
    ))

    # 7. Control: the SAME digest citation, WITHOUT skill-tier framing -- must
    # NOT be caught by the backstop (proves it isn't over-triggering on every
    # digest citation regardless of message content).
    results.append(_case(
        "digest_citation_no_skill_tier_framing_control",
        CoachingResponse(items=[CoachingFeedbackItem(
            text="Your wrist height has been trending lower in recent sessions.",
            type="grounded_observation",
            citation=Citation(record_type="SessionDigest", record_id=REAL_DIGEST.record_id,
                               ir_field_path="trend_deltas.wrist_height_ratio", cited_value=-17.5),
        )]),
        expected_verdicts=["GROUNDED"],
        user_message="how has my progress been lately?",
        expect_skill_tier_catch=False,
    ))

    n_pass = sum(results)
    print(f"{'=' * 70}\n[summary] {n_pass}/{len(results)} adversarial cases behaved as expected.\n{'=' * 70}")
    if n_pass != len(results):
        sys.exit(1)
