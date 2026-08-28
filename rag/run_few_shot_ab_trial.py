"""
Lever #2 (few-shot examples) A/B trial -- PROJECT_LOG.md 2026-08-26,
"LLM-quality-lever test." Runs the SAME held-out diverse SessionRecords
through the gated pipeline twice, side by side: once with
rag/prompt.py:build_prompt's default (use_few_shot=False, current live
behavior) and once with use_few_shot=True (rag/few_shot_examples.py's 2
real, already-grounded examples appended to the system prompt). Same
before/after comparison method as rag/rerun_thin_response_check.py and
rag/rerun_coach_tone_check.py.

TARGET_SESSIONS deliberately excludes clutch_breakdown/movement_fatigue
(those ARE the few-shot examples -- testing on them would be near-trivial).
second_serve_ace_decline is the ORIGINAL documented thin/generic-response
case (PROJECT_LOG.md 2026-07-18), already improved once by the 2026-08-21
rule-8 prompt-instruction fix -- reused again here as the sharpest real
test of whether few-shot examples move it further, not a fresh unrelated
pick.

Usage:
    python rag/run_few_shot_ab_trial.py
"""
import json
import os
import sys
import time
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt, retrieve_snippets, extract_deviated_feature_names  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402

# Same documented transport-level flakiness as rag/run_gated_diverse_trials.py
# (PROJECT_LOG.md 2026-08-02: intermittent urllib TimeoutError with no
# response at all after a handful of sequential /api/chat calls -- not a
# prompt/content issue, every attempt that DID complete was still gradeable).
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY_S = 5

TARGET_SESSIONS = ["second_serve_ace_decline", "rally_style_shift"]
OUT_DIR = os.path.join(REPO_ROOT, "rag", "example_output", "few_shot_ab_trial")


def run_one(backend, session_record, source_records, use_few_shot, max_attempts=3):
    system_prompt, user_prompt = build_prompt(session_record, use_few_shot=use_few_shot)
    for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
        try:
            return generate_grounded_response(backend, system_prompt, user_prompt, source_records,
                                               max_attempts=max_attempts)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            print(f"  [network retry {attempt}/{NETWORK_RETRY_ATTEMPTS}] {type(e).__name__}: {e}", flush=True)
            if attempt == NETWORK_RETRY_ATTEMPTS:
                raise
            time.sleep(NETWORK_RETRY_DELAY_S)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    backend = OllamaBackend(model="qwen3:8b")

    all_results = {}
    for target in TARGET_SESSIONS:
        path = os.path.join(REPO_ROOT, f"ir/example_output/session_record_diverse_{target}.json")
        with open(path) as f:
            session_record = json.load(f)
        source_records = {("SessionRecord", session_record["record_id"]): session_record}

        print(f"\n{'=' * 78}\n[{target}]\n{'=' * 78}")
        print(f"  explanation: {session_record['explanation']}")

        print(f"\n  --- BEFORE (use_few_shot=False) ---")
        before = run_one(backend, session_record, source_records, use_few_shot=False)
        if before.success:
            print(f"  SUCCESS after {before.n_attempts} attempt(s), {before.total_latency_s:.1f}s")
            for item in before.response.items:
                print(f"    [{item.type}] {item.text!r}")
        else:
            print(f"  FAILED: {before.failure_reason}")

        print(f"\n  --- AFTER (use_few_shot=True) ---")
        after = run_one(backend, session_record, source_records, use_few_shot=True)
        if after.success:
            print(f"  SUCCESS after {after.n_attempts} attempt(s), {after.total_latency_s:.1f}s")
            for item in after.response.items:
                print(f"    [{item.type}] {item.text!r}")
        else:
            print(f"  FAILED: {after.failure_reason}")

        all_results[target] = dict(
            before=dict(
                success=before.success, n_attempts=before.n_attempts,
                total_latency_s=before.total_latency_s,
                items=[dict(type=it.type, text=it.text) for it in before.response.items] if before.success else None,
                failure_reason=before.failure_reason,
            ),
            after=dict(
                success=after.success, n_attempts=after.n_attempts,
                total_latency_s=after.total_latency_s,
                items=[dict(type=it.type, text=it.text) for it in after.response.items] if after.success else None,
                failure_reason=after.failure_reason,
            ),
        )

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {out_path}")
