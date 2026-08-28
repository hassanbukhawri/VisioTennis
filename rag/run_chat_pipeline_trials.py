"""
Standalone verification of the full chat pipeline (rag/chat_pipeline.py:
handle_chat_turn) -- classifier + all 3 prompt paths wired together, run
against a real StrokeEvent and a real Ollama qwen3:8b backend, BEFORE any
UI/endpoint code exists. Complements rag/run_chat_router_trials.py (which
verified the classifier + backstops in isolation) by proving the actual
end-to-end dispatch: real turns hitting tier 1 (template replay, no
generation), tier 2 (full retry-gate), and tier 3 (both general_note and
unavailable_comparison, including the fixed disclaimer).

Usage:
    python rag/run_chat_pipeline_trials.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from rag.chat_pipeline import UNAVAILABLE_COMPARISON_DISCLAIMER, handle_chat_turn  # noqa: E402
from rag.chat_router import ROUTER_SYSTEM_PROMPT, ExistingCitation  # noqa: E402
from rag.llm import GenerationResult, OllamaBackend  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402


class _ForceRouteBackend:
    """Wraps a real backend, but scripts the ROUTER's decision (recognized by
    its distinctive system prompt) to a fixed route -- every OTHER call
    (tier-2/tier-3 generation) goes to the real backend untouched. Used to
    deterministically exercise the "unavailable_comparison" dispatch path
    (disclaimer prepending) without depending on the classifier happening to
    choose that route on a given live run -- classify_turn()'s own real-model
    reliability is already covered separately in run_chat_router_trials.py."""
    name = "force-route-wrapper"

    def __init__(self, real_backend, forced_route, forced_field_path=None):
        self.real_backend = real_backend
        self.forced_route = forced_route
        self.forced_field_path = forced_field_path

    def generate(self, system_prompt, user_prompt):
        if system_prompt == ROUTER_SYSTEM_PROMPT:
            payload = json.dumps({"route": self.forced_route, "matched_field_path": self.forced_field_path})
            return GenerationResult(self.name, self.name, payload, 0.0)
        return self.real_backend.generate(system_prompt, user_prompt)

STROKE_EVENT_PATH = os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_backhand.json")
COACHING_RESULT_PATH = os.path.join(REPO_ROOT, "rag/example_output/qwen3_gated_stroke_trials/backhand_result.json")

# (case_id, user_message, expected_tier, expected_route)
TEST_TURNS = [
    ("tier1_existing_citation", "why was my contact point flagged as low?", 1, "existing_citation"),
    ("tier2_new_grounded_claim", "what about my elbow angle?", 2, "new_grounded_claim"),
    ("tier3_general_note", "what's a good drill to improve my backhand follow-through?", 3, "general_note"),
    ("tier3_unavailable_comparison", "what's a good weight transfer ratio for a player at my level?", 3, "unavailable_comparison"),
]

if __name__ == "__main__":
    with open(STROKE_EVENT_PATH) as f:
        stroke_event = json.load(f)
    with open(COACHING_RESULT_PATH) as f:
        coaching_result = json.load(f)

    existing_response = CoachingResponse.model_validate({"items": coaching_result["generated_items"]})
    existing_items = list(existing_response.items)
    existing_citations = [
        ExistingCitation(field_path=it.citation.ir_field_path, label=it.citation.ir_field_path.split(".")[-1], text=it.text)
        for it in existing_items if it.citation is not None
    ]
    print(f"[setup] loaded StrokeEvent event_id={stroke_event['event_id']} (backhand)")
    print(f"[setup] {len(existing_citations)} existing real citations from a prior gated-pipeline run:")
    for c in existing_citations:
        print(f"    - {c.field_path}")

    backend = OllamaBackend(model="qwen3:8b")
    benchmark = BiomechanicsPercentileBenchmark()

    results = []
    for case_id, message, expected_tier, expected_route in TEST_TURNS:
        print(f"\n{'=' * 70}\n[{case_id}] message: {message!r}\n{'=' * 70}")
        turn = handle_chat_turn(
            backend, stroke_event, benchmark, existing_items, existing_citations,
            conversation_history=[], user_message=message,
        )
        ok_tier = turn.tier == expected_tier
        ok_route = turn.route == expected_route
        print(f"  expected: tier={expected_tier} route={expected_route}")
        print(f"  actual:   tier={turn.tier} route={turn.route} success={turn.success} "
              f"n_attempts={turn.n_attempts} total_latency_s={turn.total_latency_s:.1f} "
              f"escalated_from_tier3={turn.escalated_from_tier3}")
        print(f"  text: {turn.text}")
        if turn.route == "unavailable_comparison":
            has_disclaimer = turn.text.startswith(UNAVAILABLE_COMPARISON_DISCLAIMER)
            print(f"  disclaimer present and fixed: {has_disclaimer}")
            ok_route = ok_route and has_disclaimer
        ok = ok_tier and ok_route
        results.append((case_id, ok))
        print(f"  {'PASS' if ok else 'MISMATCH'}")

    n_pass = sum(1 for _, ok in results if ok)
    print(f"\n{'=' * 70}\n[summary] {n_pass}/{len(results)} pipeline trials matched the expected tier+route.\n{'=' * 70}")

    print(f"\n{'=' * 70}\n[forced] deterministically forcing route=unavailable_comparison to prove the "
          f"disclaimer-prepending dispatch path itself, independent of the classifier's live route choice\n{'=' * 70}")
    forced_backend = _ForceRouteBackend(backend, forced_route="unavailable_comparison")
    forced_turn = handle_chat_turn(
        forced_backend, stroke_event, benchmark, existing_items, existing_citations,
        conversation_history=[], user_message="what's a good weight transfer ratio for a player at my level?",
    )
    forced_ok = (
        forced_turn.route == "unavailable_comparison"
        and forced_turn.tier == 3
        and forced_turn.text.startswith(UNAVAILABLE_COMPARISON_DISCLAIMER)
    )
    print(f"  route={forced_turn.route} tier={forced_turn.tier}")
    print(f"  text: {forced_turn.text}")
    print(f"  {'PASS' if forced_ok else 'FAIL'} -- disclaimer dispatch path {'confirmed working' if forced_ok else 'BROKEN'}")
    assert forced_ok, "unavailable_comparison dispatch did not prepend the fixed disclaimer as designed"
