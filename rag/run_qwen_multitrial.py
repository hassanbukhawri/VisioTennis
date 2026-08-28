"""
Runs the same session-level coaching prompt through local Qwen3 8B N times
(same prompt, same temperature) to check RELIABILITY, not just a single best-
or worst-case result. Motivated by an observed run-to-run failure: one trial
produced a perfectly grounded response (3/3, 100%), the very next identical
call fabricated a record_id and two feature names that don't exist in the
schema (0/6, 0%) -- a single run in either direction would have been
misleading. See rag/DESIGN.md for the full write-up.

Usage:
    python rag/run_qwen_multitrial.py [n_trials] [model]

model (2026-08-26, default "qwen3:8b" -- byte-identical default behavior
to before this param existed): lets the exact same trial code run against
a different Ollama model tag (e.g. "qwen3:14b" for the Qwen3-14B-feasibility
lever, PROJECT_LOG.md "LLM-quality-lever test," item 1) -- a true
side-by-side comparison uses the SAME script, not two subtly different
ones. Output goes to a model-specific subdirectory so neither run
overwrites the other's artifacts.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.run_qwen_eval import clean_json_text  # noqa: E402

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")

if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen3:8b"
    with open(SESSION_RECORD_PATH) as f:
        session_record = json.load(f)
    source_records = {("SessionRecord", session_record["record_id"]): session_record}
    system_prompt, user_prompt = build_prompt(session_record)
    backend = OllamaBackend(model=model)

    out_dir = os.path.join(REPO_ROOT, "rag", "example_output", f"{model.replace(':', '_')}_trials")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    for i in range(1, n_trials + 1):
        print(f"\n=== trial {i}/{n_trials} ===", flush=True)
        result = backend.generate(system_prompt, user_prompt)
        print(f"latency={result.latency_s:.2f}s", flush=True)
        with open(os.path.join(out_dir, f"trial_{i}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(result.raw_text)

        cleaned = clean_json_text(result.raw_text)
        try:
            response = CoachingResponse.model_validate(json.loads(cleaned))
            report = check_grounding(response, source_records)
            n = report.grounded_observation_count
            passed = report.grounded_pass_count
            rate = report.grounding_rate * 100 if n else float("nan")
            print(f"parsed OK: {passed}/{n} grounded ({rate:.0f}%)", flush=True)
            fail_kinds = sorted({v.verdict for v in report.verdicts if v.verdict != "GROUNDED"
                                  and v.verdict != "OK_GENERAL_NOTE"})
            results.append(dict(trial=i, latency_s=result.latency_s, parse_ok=True,
                                 n_items=n, n_pass=passed, rate_pct=rate, fail_kinds=fail_kinds))
            with open(os.path.join(out_dir, f"trial_{i}_report.txt"), "w") as f:
                f.write(report.summary())
        except Exception as e:
            print(f"PARSE FAILURE: {e}", flush=True)
            results.append(dict(trial=i, latency_s=result.latency_s, parse_ok=False,
                                 n_items=0, n_pass=0, rate_pct=0.0, fail_kinds=["PARSE_FAILURE"]))

    print("\n=== summary across all trials ===")
    for r in results:
        print(f"  trial {r['trial']}: latency={r['latency_s']:.1f}s parse_ok={r['parse_ok']} "
              f"grounded={r['n_pass']}/{r['n_items']} ({r['rate_pct']:.0f}%) fail_kinds={r['fail_kinds']}")
    total_items = sum(r["n_items"] for r in results)
    total_pass = sum(r["n_pass"] for r in results)
    print(f"\npooled grounding rate across all trials: {total_pass}/{total_items} "
          f"({100*total_pass/total_items:.1f}%)" if total_items else "no gradeable items")
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    print(f"mean latency: {avg_latency:.1f}s")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] wrote trial outputs + summary.json to {out_dir}")
