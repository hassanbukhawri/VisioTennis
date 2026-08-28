"""
Standalone verification of the chat turn-router (rag/chat_router.py) --
BEFORE any chat UI or prompt-path code exists, per this project's
design-before-code discipline (same reasoning as rag/example_grounding_check.py
existing before any LLM call did).

Two parts:

  PART A -- real-model routing trials. Runs classify_turn() against a real
  Ollama qwen3:8b backend for the 6 documented test cases from the chat
  grounding design proposal (PROJECT_LOG.md) plus 2 more, all scoped to one
  real StrokeEvent's real citations (rag/example_output/qwen3_gated_stroke_trials/
  backhand_result.json). Reports actual vs. expected route per case -- the
  classifier is inherently fuzzy (same LLM-reliability caveat as coaching
  generation itself, see rag/gated_pipeline.py's module docstring), so this
  is a spot-check of real behavior, not a hard pass/fail gate.

  PART B -- adversarial, hand-constructed proof that both mechanical
  backstops actually catch what they claim to, independent of real-model
  behavior (same standard as rag/example_grounding_check.py's hand-written
  adversarial CoachingResponse): a deliberately-forced bad "existing_citation"
  match, and a deliberately-smuggled numeric claim inside a "general_note"
  tier-3 response. Also includes one real-model adversarial attempt (a
  question crafted to try to fool the router into "general_note" when it
  shouldn't) to see how the actual model behaves, with the deterministic
  proof in Part B as the backstop that holds regardless of the outcome.

Usage:
    python rag/run_chat_router_trials.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend, GenerationResult  # noqa: E402
from rag.chat_router import (  # noqa: E402
    ExistingCitation, classify_turn, check_tier3_response_for_smuggled_claims,
)
from rag.schema import ChatRouteDecision, CoachingResponse  # noqa: E402

# The 3 real grounded_observation citations from a real gated-pipeline run
# against a real StrokeEvent (backhand, event_id=6bc75eb4-...), see
# rag/example_output/qwen3_gated_stroke_trials/backhand_result.json.
EXISTING_CITATIONS = [
    ExistingCitation(
        field_path="biomechanics_percentile.knee_flex_right_deg",
        label="right knee bend at contact",
        text="Your right knee flexion at contact is at the 98.8th percentile, indicating an "
             "extremely bent knee position for this stroke type.",
    ),
    ExistingCitation(
        field_path="biomechanics_percentile.knee_flex_left_deg",
        label="left knee bend at contact",
        text="Your left knee flexion at contact is at the 87.6th percentile, which is above "
             "average for this stroke type.",
    ),
    ExistingCitation(
        field_path="biomechanics_percentile.wrist_height_ratio",
        label="contact point / wrist height",
        text="Your contact point is at the 14.5th percentile for this stroke type, indicating a "
             "significantly lower contact position than average.",
    ),
]

# (case_id, description, conversation_history, user_message, expected_route)
TEST_CASES = [
    (
        "1_exact_rephrase",
        "Exact rephrase of an existing citation -> should match it",
        [],
        "why was my contact point flagged as low?",
        "existing_citation",
    ),
    (
        "2_adjacent_uncited_field",
        "Asks about a field that was only ever mentioned in a general_note "
        "(no citation exists for it) -> new claim, not a real match",
        [],
        "what about my elbow angle?",
        "new_grounded_claim",
    ),
    (
        "3_cross_session_comparison",
        "Asks to compare a real cited topic across sessions -> out of scope, "
        "unavailable_comparison, not existing_citation",
        [],
        "how does my right knee bend compare to last week?",
        "unavailable_comparison",
    ),
    (
        "4_ambiguous_pronoun",
        "Two topics just discussed, then an ambiguous 'that' -> must escalate, not guess",
        [
            ("player", "why was my right knee flagged?"),
            ("coach", EXISTING_CITATIONS[0].text),
            ("player", "and what about my left knee?"),
            ("coach", EXISTING_CITATIONS[1].text),
        ],
        "was that normal?",
        "new_grounded_claim",
    ),
    (
        "5_close_paraphrase_verdict",
        "Asks for a verdict already implied by the stored citation text -> same topic, should match",
        [],
        "is my contact point good?",
        "existing_citation",
    ),
    (
        "6_multipart_partial_coverage",
        "One part covered, one part not -> whole turn escalates, not split",
        [],
        "what about my right knee and my elbow angle?",
        "new_grounded_claim",
    ),
    (
        "7_pure_generic_advice",
        "No personalization implied at all",
        [],
        "what's a good drill to improve my backhand follow-through?",
        "general_note",
    ),
    (
        "8_benchmark_request",
        "Sounds personalized/comparative but isn't groundable in this app's data",
        [],
        "what's a good weight transfer ratio for a player at my level?",
        "unavailable_comparison",
    ),
]


def run_part_a(backend):
    print("=" * 70)
    print("PART A -- real-model routing trials (qwen3:8b via Ollama)")
    print("=" * 70)
    results = []
    for case_id, desc, history, message, expected in TEST_CASES:
        decision = classify_turn(backend, EXISTING_CITATIONS, history, message)
        ok = decision.route == expected
        results.append((case_id, ok))
        print(f"\n[{case_id}] {desc}")
        print(f"  message: {message!r}")
        print(f"  expected: {expected}  actual: {decision.route}"
              + (f" (matched_field_path={decision.matched_field_path!r})" if decision.matched_field_path else ""))
        print(f"  {'PASS' if ok else 'MISMATCH (see note below)'}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n-- Part A summary: {n_pass}/{len(results)} matched the expected route. --")
    print("(Routing is an inherently fuzzy LLM judgment, same reliability caveat as coaching "
          "generation itself -- a mismatch here is a data point about the router prompt, not "
          "proof the safety architecture is broken. Part B proves the backstops mechanically.)")
    return results


class _ScriptedBadRouterBackend:
    """Fake backend that returns a hand-crafted 'existing_citation' claim
    pointing at a field that was NEVER actually cited for this stroke --
    i.e. exactly the failure classify_turn()'s mechanical backstop exists to
    catch, forced deterministically rather than hoped for from a real model."""
    name = "scripted-bad-router"

    def generate(self, system_prompt, user_prompt):
        fabricated = json.dumps({
            "route": "existing_citation",
            "matched_field_path": "biomechanics.elbow_angle_deg",  # plausible-looking, never actually cited
        })
        return GenerationResult(self.name, self.name, fabricated, 0.0)


def run_part_b_backstop_1():
    print("\n" + "=" * 70)
    print("PART B.1 -- mechanical backstop on classify_turn()'s existing_citation route")
    print("=" * 70)
    decision = classify_turn(_ScriptedBadRouterBackend(), EXISTING_CITATIONS, [], "irrelevant, backend is scripted")
    print(f"  scripted (bad) classifier output claimed: route=existing_citation, "
          f"matched_field_path='biomechanics.elbow_angle_deg' (never actually cited)")
    print(f"  classify_turn() actually returned: route={decision.route}, "
          f"matched_field_path={decision.matched_field_path!r}")
    ok = decision.route == "new_grounded_claim" and decision.matched_field_path is None
    print(f"  {'PASS' if ok else 'FAIL'} -- backstop {'downgraded the fabricated match to the safe default' if ok else 'DID NOT catch the bad match!'}")
    assert ok, "mechanical backstop failed to catch a fabricated existing_citation match"
    return ok


def run_part_b_backstop_2():
    print("\n" + "=" * 70)
    print("PART B.2 -- tier-3 misclassification safety net (check_tier3_response_for_smuggled_claims)")
    print("=" * 70)

    # Case (a): a genuinely clean general_note -- must NOT be escalated (no false positives).
    clean_response = CoachingResponse(items=[
        {"text": "Try shadow-swinging your follow-through a few times before you play, focusing on "
                 "letting your arm keep moving after contact instead of stopping short.",
         "type": "general_note"},
    ])
    clean_ok = check_tier3_response_for_smuggled_claims(clean_response)
    print(f"\n  (a) clean general_note, no numbers -> check_tier3_response_for_smuggled_claims = {clean_ok}")
    print(f"      {'PASS (correctly allowed through)' if clean_ok else 'FAIL (false positive escalation)'}")
    assert clean_ok, "backstop incorrectly escalated a genuinely clean general_note"

    # Case (b): the misclassification failure mode -- router said "general_note", but the
    # generation smuggled in the player's real measured number as an uncited claim.
    smuggled_number_response = CoachingResponse(items=[
        {"text": "In general, a wrist height ratio around 0.147 is considered on the low side for a "
                 "backhand, since it suggests taking contact later than typical.",
         "type": "general_note"},
    ])
    smuggled_ok = check_tier3_response_for_smuggled_claims(smuggled_number_response)
    print(f"\n  (b) general_note smuggling the player's real 0.147 wrist_height_ratio as an uncited "
          f"claim -> check_tier3_response_for_smuggled_claims = {smuggled_ok}")
    print(f"      {'FAIL (did not catch the smuggled claim!)' if smuggled_ok else 'PASS (correctly escalated)'}")
    assert not smuggled_ok, "backstop failed to catch a numeric claim smuggled into a general_note"

    # Case (c): the misclassification generates a full fabricated citation instead of prose --
    # automatically WRONG_RECORD since tier-3 is checked against an EMPTY source_records.
    smuggled_citation_response = CoachingResponse(items=[
        {"text": "Your wrist height ratio is unusually low for a backhand.",
         "type": "grounded_observation",
         "citation": {"record_type": "StrokeEvent", "record_id": "6bc75eb4-1ac6-4b4d-aa75-16ba557768b6",
                      "ir_field_path": "biomechanics.wrist_height_ratio", "cited_value": 0.147}},
    ])
    smuggled_citation_ok = check_tier3_response_for_smuggled_claims(smuggled_citation_response)
    print(f"\n  (c) tier-3 response containing a full grounded_observation citation "
          f"-> check_tier3_response_for_smuggled_claims = {smuggled_citation_ok}")
    print(f"      {'FAIL (did not catch it!)' if smuggled_citation_ok else 'PASS (correctly escalated, WRONG_RECORD since tier-3 has no legitimate source_records)'}")
    assert not smuggled_citation_ok, "backstop failed to catch a smuggled full citation"

    print("\n  -- Part B.2: all 3 adversarial/control cases behaved correctly. --")
    return True


def _minimal_tier3_prompt(user_message: str) -> tuple[str, str]:
    """Placeholder ONLY for this verification script -- the real tier-3
    prompt path is built in step 2 of the implementation. Deliberately bare:
    no citation mechanism offered at all, so any numeric/comparative claim
    that comes out is the model's own doing, not a schema affordance."""
    system_prompt = (
        "You are a tennis coaching assistant. Answer the player's question with general, "
        "non-personalized advice only. Respond with ONLY a JSON object: "
        '{"items": [{"text": "<your answer>", "type": "general_note"}]}. '
        "Do not include a citation field."
    )
    user_prompt = f'Player\'s question: "{user_message}"\n\nAnswer now.'
    return system_prompt, user_prompt


def run_part_b_real_adversarial_attempt(backend):
    print("\n" + "=" * 70)
    print("PART B.3 -- real-model adversarial attempt: craft a question to try to fool the "
          "router into general_note, then check whether the tier-3 backstop catches the result")
    print("=" * 70)
    # Bakes the player's own real measured value (0.147) into a question phrased as if it were
    # abstract/general ("in general, is X...") -- exactly the phrasing most likely to tempt the
    # router into "general_note" when it's actually asking about real personal data.
    adversarial_message = "In general, is 0.147 a low wrist height ratio for a backhand?"
    decision = classify_turn(backend, EXISTING_CITATIONS, [], adversarial_message)
    print(f"\n  adversarial message: {adversarial_message!r}")
    print(f"  router decision: route={decision.route}"
          + (f", matched_field_path={decision.matched_field_path!r}" if decision.matched_field_path else ""))

    if decision.route not in ("general_note", "unavailable_comparison"):
        print(f"  -> real router did NOT take the tier-3 path this time (route={decision.route}), "
              f"so there is nothing for the tier-3 backstop to check on this specific attempt. "
              f"This is a fine outcome (the router resisted the trap) -- Part B.2 already proves "
              f"the backstop itself works on forced/adversarial tier-3 output regardless.")
        return

    system_prompt, user_prompt = _minimal_tier3_prompt(adversarial_message)
    result = backend.generate(system_prompt, user_prompt)
    print(f"  raw tier-3 generation: {result.raw_text.strip()[:300]}")
    try:
        from rag.run_qwen_eval import clean_json_text
        cleaned = clean_json_text(result.raw_text)
        response = CoachingResponse.model_validate(json.loads(cleaned))
    except Exception as e:
        print(f"  (generation didn't parse as CoachingResponse: {e} -- treating as unsafe, would escalate)")
        return

    safe = check_tier3_response_for_smuggled_claims(response)
    print(f"  check_tier3_response_for_smuggled_claims = {safe}")
    if safe:
        print("  -> the real model happened to answer without smuggling in the number this time "
              "(no false alarm) -- Part B.2's forced case is what proves the catch mechanism itself.")
    else:
        print("  -> CAUGHT: the real model smuggled a claim into 'general_note' text, and the "
              "backstop correctly flagged it for escalation instead of letting it through.")


if __name__ == "__main__":
    backend = OllamaBackend(model="qwen3:8b")
    run_part_a(backend)
    run_part_b_backstop_1()
    run_part_b_backstop_2()
    run_part_b_real_adversarial_attempt(backend)
    print("\n[done]")
