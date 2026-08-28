"""
Confirmatory re-run for the Fireworks-hosted Qwen3 8B (FP8) backend --
smaller than the original 6-trial + 6-diverse-session Ollama evaluation
(rag/run_gated_multitrial.py, rag/run_gated_diverse_trials.py), by design.
This is NOT re-establishing reliability from scratch: it's checking that the
hosted FP8 serving stack behaves consistently with the already-validated
local Q4_K_M results (66.7% raw / 100% gated on the repeated-session test,
94.4% retrieval recall across 6 diverse sessions) before trusting it for real
traffic. See PROJECT_LOG.md 2026-08-07 for why Fireworks was picked, and
rag/DESIGN.md for the original full evaluation this compares against.

Reuses the exact same gated pipeline (rag/gated_pipeline.py) and the same
repeated-session prompt as run_gated_multitrial.py's default, PLUS one of
the 6 diverse-session prompts (not all 6) as a second, independent data
point -- catching a retrieval/prompt-shape regression, not just a
repeated-prompt fluke, without paying for the full diverse-session suite
twice.

Usage:
    FIREWORKS_API_KEY=... python rag/run_fireworks_confirmatory.py [n_trials]
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend, OpenAICompatibleBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
FIREWORKS_MODEL = "accounts/fireworks/models/qwen3-8b"

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")
DIVERSE_SESSION_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_diverse_serve_breakdown.json")


def run_trials(backend, backend_label, session_record_path, n_trials, out_dir):
    with open(session_record_path) as f:
        session_record = json.load(f)
    source_records = {("SessionRecord", session_record["record_id"]): session_record}
    system_prompt, user_prompt = build_prompt(session_record)

    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for i in range(1, n_trials + 1):
        print(f"\n=== [{backend_label}] trial {i}/{n_trials} "
              f"(session={os.path.basename(session_record_path)}) ===", flush=True)
        gated = generate_grounded_response(backend, system_prompt, user_prompt, source_records, max_attempts=3)

        for a in gated.attempts:
            status = ("ACCEPTED" if a.accepted else
                       ("parse failure" if not a.parse_ok else "rejected (not fully grounded)"))
            print(f"  attempt {a.attempt_num}/3: latency={a.latency_s:.1f}s {status}"
                  + (f" ({a.grounded_pass_count}/{a.grounded_observation_count} grounded)"
                     if a.parse_ok else f" -- {a.parse_error}"), flush=True)

        if gated.success:
            print(f"  -> SUCCESS after {gated.n_attempts} attempt(s), "
                  f"total_latency={gated.total_latency_s:.1f}s", flush=True)
        else:
            print(f"  -> FAILED (exhausted 3 attempts): {gated.failure_reason}", flush=True)

        summaries.append(dict(trial=i, success=gated.success, n_attempts=gated.n_attempts,
                               total_latency_s=gated.total_latency_s))
        with open(os.path.join(out_dir, f"{backend_label}_trial_{i}.json"), "w") as f:
            json.dump(dict(
                backend=backend_label, trial=i, success=gated.success,
                n_attempts=gated.n_attempts, total_latency_s=gated.total_latency_s,
                failure_reason=gated.failure_reason,
                attempts=[dict(attempt_num=a.attempt_num, latency_s=a.latency_s, parse_ok=a.parse_ok,
                                parse_error=a.parse_error,
                                grounded_observation_count=a.grounded_observation_count,
                                grounded_pass_count=a.grounded_pass_count, accepted=a.accepted)
                          for a in gated.attempts],
            ), f, indent=2)
    return summaries


def print_summary(label, summaries):
    n = len(summaries)
    n_success = sum(1 for t in summaries if t["success"])
    mean_latency = sum(t["total_latency_s"] for t in summaries) / n
    print(f"\n[{label}] gated success: {n_success}/{n} ({100*n_success/n:.1f}%), "
          f"mean latency incl. retries: {mean_latency:.1f}s")


if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY not set -- nothing to test yet. "
              "Set it and re-run once the Fireworks account/key exist.")
        sys.exit(1)

    out_dir = os.path.join(REPO_ROOT, "rag", "example_output", "fireworks_confirmatory")
    fireworks_backend = OpenAICompatibleBackend(
        model=FIREWORKS_MODEL, api_key=api_key, base_url=FIREWORKS_BASE_URL)

    print("### Part 1: repeated-session prompt (same prompt run_gated_multitrial.py used) ###")
    repeated_summaries = run_trials(
        fireworks_backend, "fireworks_repeated", SESSION_RECORD_PATH, n_trials, out_dir)
    print_summary("fireworks / repeated-session", repeated_summaries)

    if os.path.exists(DIVERSE_SESSION_PATH):
        print("\n### Part 2: one diverse-session prompt (independent prompt shape, not the same session repeated) ###")
        diverse_summaries = run_trials(
            fireworks_backend, "fireworks_diverse_serve_breakdown", DIVERSE_SESSION_PATH, n_trials, out_dir)
        print_summary("fireworks / diverse serve_breakdown session", diverse_summaries)
    else:
        print(f"\n[skip] diverse-session file not found at {DIVERSE_SESSION_PATH} -- "
              f"Part 2 skipped, only the repeated-session result is available.")

    print(f"\n[done] wrote per-trial results to {out_dir}")
    print("\nCompare this against the original Ollama Q4_K_M baseline in "
          "rag/example_output/qwen3_gated_trials/summary.json (repeated-session) and "
          "rag/example_output/qwen3_gated_diverse_trials/ (diverse sessions) before "
          "deciding LLM_BACKEND=fireworks is trustworthy for real traffic.")
