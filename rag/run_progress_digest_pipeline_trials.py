"""
Live end-to-end verification of the "progress_digest" chat route -- real
qwen3:8b via Ollama, real stroke_event context (same fixtures the other chat
pipeline trials use), and a REAL SessionDigest computed from this project's
own dev DB (test1@example.com's 7 real stored sessions), not synthetic data.

Complements rag/run_session_digest_grounding_check.py (which proves the
CATCH mechanism deterministically, no LLM) by proving the actual routing +
generation + grounding integration works together against a real model.

Usage:
    python rag/run_progress_digest_pipeline_trials.py
"""
import json
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from ir.session_digest import compute_session_digest  # noqa: E402
from rag.chat_pipeline import NO_DIGEST_AVAILABLE_MESSAGE, handle_chat_turn  # noqa: E402
from rag.chat_router import ExistingCitation  # noqa: E402
from rag.llm import OllamaBackend  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402

STROKE_EVENT_PATH = os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_backhand.json")
COACHING_RESULT_PATH = os.path.join(REPO_ROOT, "rag/example_output/qwen3_gated_stroke_trials/backhand_result.json")
DB_PATH = os.path.join(REPO_ROOT, "backend/app.db")
REAL_USER_ID = "f79976ff-4537-44a6-aee7-0e2d59412148"  # test1@example.com, 7 real sessions

TEST_MESSAGES = [
    ("progress_trend", "how have I been progressing lately?"),
    ("progress_comparative_phrasing", "is my backhand better than it was before?"),
    ("progress_skill_tier_combined", "has my progress been good for a player at my level?"),
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

    benchmark = BiomechanicsPercentileBenchmark()
    backend = OllamaBackend(model="qwen3:8b")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT result_json FROM stroke_history WHERE user_id=? ORDER BY created_at DESC", (REAL_USER_ID,))
    sessions = [json.loads(r[0]) for r in cur.fetchall()]
    print(f"[setup] {len(sessions)} real stored sessions fetched for {REAL_USER_ID}")

    digest_obj = compute_session_digest(sessions, player_id=REAL_USER_ID, benchmark=benchmark)
    assert digest_obj is not None, "expected a real digest from real stored sessions -- got None"
    digest = digest_obj.model_dump(mode="json")
    print(f"[setup] real digest computed: session_count={digest['session_count']} "
          f"trend_prior_window_size={digest['trend_prior_window_size']} "
          f"stroke_type_mode={digest['stroke_type_mode']}\n")

    results = []
    for case_id, message in TEST_MESSAGES:
        print(f"{'=' * 70}\n[{case_id}] message: {message!r}\n{'=' * 70}")
        turn = handle_chat_turn(
            backend, stroke_event, benchmark, existing_items, existing_citations,
            conversation_history=[], user_message=message, digest=digest,
        )
        print(f"  route={turn.route} tier={turn.tier} success={turn.success} "
              f"n_attempts={turn.n_attempts} total_latency_s={turn.total_latency_s:.1f}")
        print(f"  text: {turn.text}")
        for item in turn.items:
            if item.citation is not None:
                print(f"  citation: record_type={item.citation.record_type} "
                      f"field={item.citation.ir_field_path} value={item.citation.cited_value}")
        # Safety property that actually matters: no unsafe/ungrounded claim
        # ever reaches this point silently -- either route=="progress_digest"
        # with a real digest citation, or a safe fallback route
        # (unavailable_comparison) was used instead. Never route=="progress_digest"
        # with a citation that ISN'T a real SessionDigest field.
        leaked_bad_digest_citation = any(
            it.citation is not None and it.citation.record_type == "SessionDigest"
            and it.citation.ir_field_path not in
            [f"trailing_averages.{k}" for k in digest["trailing_averages"]]
            + [f"trend_deltas.{k}" for k in digest["trend_deltas"]]
            + ["stroke_type_mode", "session_count"]
            for it in turn.items
        )
        ok = turn.success and not leaked_bad_digest_citation
        results.append((case_id, ok, turn.route))
        print(f"  {'OK' if ok else 'FAIL -- unsafe digest citation leaked'}\n")

    print(f"{'=' * 70}\n[real digest-not-available case]\n{'=' * 70}")
    turn_none = handle_chat_turn(
        backend, stroke_event, benchmark, existing_items, existing_citations,
        conversation_history=[], user_message="how have I been progressing lately?", digest=None,
    )
    ok_none = turn_none.text == NO_DIGEST_AVAILABLE_MESSAGE and turn_none.route == "progress_digest" and turn_none.tier == 3
    print(f"  route={turn_none.route} tier={turn_none.tier} text={turn_none.text!r}")
    print(f"  {'OK -- fixed message shown, no generation attempted' if ok_none else 'FAIL'}\n")
    results.append(("digest_none_fixed_message", ok_none, turn_none.route))

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"{'=' * 70}\n[summary] {n_ok}/{len(results)} live cases behaved safely.\n{'=' * 70}")
    for case_id, ok, route in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {case_id}: route={route}")
