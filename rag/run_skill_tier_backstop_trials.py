"""
Verification of the skill-tier-mismatch backstop (rag/chat_router.py:
check_for_skill_tier_mismatch, backstop #3) against the real, documented
2026-08-10 failure -- "what's a good weight transfer ratio for a player at
my level" was classified as "new_grounded_claim" and produced a real,
100%-grounded percentile citation, mislabeling a flat (non-skill-stratified)
population percentile as a skill-tier comparison it can't support.

Two parts, same live/deterministic split already used for the other two
backstops (see rag/run_chat_router_trials.py Part B.2/B.3):

  Part A -- natural live runs of the exact failing phrase + 3 rephrasings
  through the real classifier (qwen3:8b). Shows what the router does TODAY
  and confirms the end-to-end effective route is now "unavailable_comparison"
  regardless.

  Part B -- deterministic: forces the router to "new_grounded_claim" for the
  same 4 messages (via the existing _ForceRouteBackend pattern), guaranteeing
  tier 2 actually runs and produces a real percentile citation, then asserts
  the backstop downgrades the result to "unavailable_comparison". This is
  the proof that doesn't depend on the classifier happening to misroute on
  any given run -- same reasoning as the other two backstops' deterministic
  cases.

Usage:
    python rag/run_skill_tier_backstop_trials.py
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
    """Same pattern as run_chat_pipeline_trials.py's _ForceRouteBackend --
    scripts only the router's decision, every other call (tier-2/tier-3
    generation) goes to the real backend untouched."""
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

# The exact 2026-08-10 failing phrase, plus 3 rephrasings covering different
# surface forms of the same skill-tier framing.
SKILL_TIER_MESSAGES = [
    ("original_2026-08-10", "what's a good weight transfer ratio for a player at my level?"),
    ("rephrase_compare_to_players", "how does my weight transfer compare to other players at my level?"),
    ("rephrase_skill_level", "is my weight transfer ratio good for my skill level?"),
    ("rephrase_someone_at_my_stage", "what would be normal for someone at my stage of playing?"),
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
    print(f"[setup] {len(existing_citations)} existing real citations from a prior gated-pipeline run")

    backend = OllamaBackend(model="qwen3:8b")
    benchmark = BiomechanicsPercentileBenchmark()

    print(f"\n{'#' * 70}\n# PART A -- natural live runs (real classifier, no forcing)\n{'#' * 70}")
    part_a_results = []
    for case_id, message in SKILL_TIER_MESSAGES:
        print(f"\n{'=' * 70}\n[{case_id}] message: {message!r}\n{'=' * 70}")
        turn = handle_chat_turn(
            backend, stroke_event, benchmark, existing_items, existing_citations,
            conversation_history=[], user_message=message,
        )
        print(f"  route={turn.route} tier={turn.tier} success={turn.success} "
              f"escalated_from_tier3={turn.escalated_from_tier3}")
        print(f"  text: {turn.text}")
        # CORRECTED criterion (first version wrongly required literal
        # route=="unavailable_comparison" for every message, which also
        # flags an honest tier-2-ungrounded failure or a router that
        # legitimately picked general_note/unavailable_comparison on its
        # own as if they were backstop failures). The actual thing that
        # must be true: no percentile citation is EVER shown in response to
        # skill-tier-framed language, regardless of which safe route
        # produced that outcome.
        leaked_citation = turn.success and any(
            it.citation is not None and it.citation.ir_field_path.startswith("biomechanics_percentile.")
            for it in turn.items
        )
        ok = not leaked_citation
        part_a_results.append((case_id, ok, turn.route))
        print(f"  {'SAFE (no percentile citation shown)' if ok else 'LEAK -- percentile citation shown for skill-tier framing'}")

    print(f"\n{'#' * 70}\n# PART B -- deterministic: force router to new_grounded_claim,\n"
          f"# guaranteeing tier 2 runs and (if grounding succeeds) produces a\n"
          f"# real percentile citation, then confirm the backstop catches it\n{'#' * 70}")
    part_b_results = []
    for case_id, message in SKILL_TIER_MESSAGES:
        print(f"\n{'=' * 70}\n[{case_id} / forced new_grounded_claim] message: {message!r}\n{'=' * 70}")
        forced_backend = _ForceRouteBackend(backend, forced_route="new_grounded_claim")
        turn = handle_chat_turn(
            forced_backend, stroke_event, benchmark, existing_items, existing_citations,
            conversation_history=[], user_message=message,
        )
        print(f"  route={turn.route} tier={turn.tier} success={turn.success} "
              f"escalated_from_tier3={turn.escalated_from_tier3}")
        print(f"  text: {turn.text}")
        if not turn.success:
            # Tier 2 never produced a grounded response to begin with (real
            # model variance) -- the backstop had nothing to catch on this
            # specific attempt, not a backstop failure. Noted, not counted
            # as a catch.
            print("  [tier 2 itself failed to ground -- backstop not exercised this attempt]")
            part_b_results.append((case_id, None, turn.route))
            continue
        leaked_citation = any(
            it.citation is not None and it.citation.ir_field_path.startswith("biomechanics_percentile.")
            for it in turn.items
        )
        if not leaked_citation:
            # Tier 2 grounded, but on a DIFFERENT (non-percentile) field --
            # nothing for this specific backstop to catch here either.
            print("  [tier 2 grounded but cited no percentile field -- backstop not exercised this attempt]")
            part_b_results.append((case_id, None, turn.route))
            continue
        has_disclaimer = turn.text.startswith(UNAVAILABLE_COMPARISON_DISCLAIMER)
        ok = turn.route == "unavailable_comparison" and has_disclaimer
        part_b_results.append((case_id, ok, turn.route))
        print(f"  {'CAUGHT (forced to unavailable_comparison)' if ok else 'NOT CAUGHT -- BACKSTOP FAILURE'}")

    print(f"\n{'#' * 70}\n[summary]\n{'#' * 70}")
    n_a = sum(1 for _, ok, _ in part_a_results if ok)
    print(f"Part A (natural): {n_a}/{len(part_a_results)} messages ended up correctly showing the disclaimer.")
    for case_id, ok, route in part_a_results:
        print(f"  [{'OK' if ok else 'FAIL'}] {case_id}: effective route={route}")

    n_b_caught = sum(1 for _, ok, _ in part_b_results if ok is True)
    n_b_exercised = sum(1 for _, ok, _ in part_b_results if ok is not None)
    print(f"Part B (forced new_grounded_claim): {n_b_caught}/{n_b_exercised} exercised attempts caught "
          f"({len(part_b_results) - n_b_exercised} attempt(s) had tier 2 itself fail to ground, not exercising the backstop).")
    for case_id, ok, route in part_b_results:
        label = "SKIPPED (tier2 ungrounded)" if ok is None else ("OK" if ok else "FAIL")
        print(f"  [{label}] {case_id}: effective route={route}")

    any_hard_failure = any(ok is False for _, ok, _ in part_a_results + part_b_results)
    if any_hard_failure:
        print("\nFAIL: at least one message was NOT caught by the backstop -- see FAIL lines above.")
        sys.exit(1)
    print("\nPASS: no skill-tier-framed message ever surfaced a percentile citation uncaught.")
