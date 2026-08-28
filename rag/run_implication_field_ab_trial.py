"""
Lever #3 (schema-tightened "implication" field) A/B trial -- PROJECT_LOG.md
2026-08-26, "LLM-quality-lever test." Same before/after comparison method
as rag/run_few_shot_ab_trial.py and rag/rerun_thin_response_check.py, same
2 held-out sessions (second_serve_ace_decline -- the ORIGINAL documented
thin/generic case, PROJECT_LOG.md 2026-07-18 -- and rally_style_shift).

BEFORE: current production behavior exactly -- rag/prompt.py:build_prompt()
default (require_implication_field=False, prompt rule 8 only) parsed
against rag/schema.py:CoachingResponse.
AFTER: build_prompt(require_implication_field=True) (schema-described
REQUIRED "implication" key) parsed against
rag/implication_schema.py:StrictCoachingResponse (the actual hard
validator -- see that file). A response missing implication, or where
implication duplicates text, fails to PARSE here (not just "ungrounded"),
which rag/gated_pipeline.py's retry-gate already treats as a real failure
needing a retry -- so this trial also reports how often that firing costs
an extra attempt, not just the eventual text quality.

Usage:
    python rag/run_implication_field_ab_trial.py
"""
import json
import os
import sys
import time
import urllib.error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.implication_schema import StrictCoachingResponse  # noqa: E402

# Same documented transport-level flakiness as rag/run_gated_diverse_trials.py
# (PROJECT_LOG.md 2026-08-02) -- see rag/run_few_shot_ab_trial.py's own copy
# of this same note.
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY_S = 5

TARGET_SESSIONS = ["second_serve_ace_decline", "rally_style_shift"]
OUT_DIR = os.path.join(REPO_ROOT, "rag", "example_output", "implication_field_ab_trial")


def run_one(backend, session_record, source_records, strict, max_attempts=3):
    system_prompt, user_prompt = build_prompt(session_record, require_implication_field=strict)
    schema = StrictCoachingResponse if strict else CoachingResponse
    for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
        try:
            return generate_grounded_response(backend, system_prompt, user_prompt, source_records,
                                               max_attempts=max_attempts, response_schema=schema)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            print(f"  [network retry {attempt}/{NETWORK_RETRY_ATTEMPTS}] {type(e).__name__}: {e}", flush=True)
            if attempt == NETWORK_RETRY_ATTEMPTS:
                raise
            time.sleep(NETWORK_RETRY_DELAY_S)


def print_result(label, gated):
    if gated.success:
        print(f"  {label}: SUCCESS after {gated.n_attempts} attempt(s), {gated.total_latency_s:.1f}s")
        for item in gated.response.items:
            impl = getattr(item, "implication", None)
            print(f"    [{item.type}] text={item.text!r}")
            if impl is not None:
                print(f"      implication={impl!r}")
        # how many attempts failed to PARSE (schema rejection), vs failed
        # grounding, vs succeeded -- distinguishes "the stricter schema
        # itself is costing retries" from "grounding is the bottleneck"
        parse_failures = sum(1 for a in gated.attempts if not a.parse_ok)
        if parse_failures:
            print(f"    ({parse_failures}/{gated.n_attempts} attempt(s) failed to parse before the accepted one)")
    else:
        print(f"  {label}: FAILED: {gated.failure_reason}")


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

        print(f"\n  --- BEFORE (require_implication_field=False, CoachingResponse) ---")
        before = run_one(backend, session_record, source_records, strict=False)
        print_result("before", before)

        print(f"\n  --- AFTER (require_implication_field=True, StrictCoachingResponse) ---")
        after = run_one(backend, session_record, source_records, strict=True)
        print_result("after", after)

        def _serialize(gated, strict):
            if not gated.success:
                return dict(success=False, n_attempts=gated.n_attempts,
                            total_latency_s=gated.total_latency_s, failure_reason=gated.failure_reason)
            items = []
            for it in gated.response.items:
                d = dict(type=it.type, text=it.text)
                if strict:
                    d["implication"] = it.implication
                items.append(d)
            parse_failures = sum(1 for a in gated.attempts if not a.parse_ok)
            return dict(success=True, n_attempts=gated.n_attempts, total_latency_s=gated.total_latency_s,
                        parse_failures_before_accepted=parse_failures, items=items)

        all_results[target] = dict(
            before=_serialize(before, strict=False),
            after=_serialize(after, strict=True),
        )

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {out_path}")
