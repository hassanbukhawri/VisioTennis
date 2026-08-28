"""
Re-runs the SAME 6-trial reliability test as rag/run_qwen_multitrial.py, but
through the gated pipeline (rag/gated_pipeline.py) instead of a single raw
generation. Each "trial" here means: one user-facing request, which may
internally cost 1-3 raw LLM calls before the gate accepts or gives up.

This measures the real user-facing reliability and latency cost of the gate,
not just whether it works in principle -- reports, per trial: attempts needed,
final accept/fail, and per-attempt latency; then pooled: gated success rate,
mean attempts-to-success, and total latency including retries.

Usage:
    python rag/run_gated_multitrial.py [n_trials] [max_attempts]
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")

if __name__ == "__main__":
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    max_attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    with open(SESSION_RECORD_PATH) as f:
        session_record = json.load(f)
    source_records = {("SessionRecord", session_record["record_id"]): session_record}
    system_prompt, user_prompt = build_prompt(session_record)
    backend = OllamaBackend(model="qwen3:8b")

    out_dir = os.path.join(REPO_ROOT, "rag", "example_output", "qwen3_gated_trials")
    os.makedirs(out_dir, exist_ok=True)

    trial_summaries = []
    for i in range(1, n_trials + 1):
        print(f"\n=== gated trial {i}/{n_trials} ===", flush=True)
        gated = generate_grounded_response(
            backend, system_prompt, user_prompt, source_records, max_attempts=max_attempts)

        for a in gated.attempts:
            status = ("ACCEPTED" if a.accepted else
                       ("parse failure" if not a.parse_ok else "rejected (not fully grounded)"))
            print(f"  attempt {a.attempt_num}/{max_attempts}: latency={a.latency_s:.1f}s "
                  f"{status}"
                  + (f" ({a.grounded_pass_count}/{a.grounded_observation_count} grounded)"
                     if a.parse_ok else f" -- {a.parse_error}"), flush=True)

        if gated.success:
            print(f"  -> trial {i} SUCCESS after {gated.n_attempts} attempt(s), "
                  f"total_latency={gated.total_latency_s:.1f}s", flush=True)
        else:
            print(f"  -> trial {i} FAILED (exhausted {max_attempts} attempts): "
                  f"{gated.failure_reason}", flush=True)

        with open(os.path.join(out_dir, f"trial_{i}_result.json"), "w") as f:
            json.dump(dict(
                trial=i,
                success=gated.success,
                n_attempts=gated.n_attempts,
                total_latency_s=gated.total_latency_s,
                failure_reason=gated.failure_reason,
                attempts=[dict(attempt_num=a.attempt_num, latency_s=a.latency_s,
                                parse_ok=a.parse_ok, parse_error=a.parse_error,
                                grounded_observation_count=a.grounded_observation_count,
                                grounded_pass_count=a.grounded_pass_count,
                                accepted=a.accepted)
                          for a in gated.attempts],
            ), f, indent=2)
        if gated.success:
            with open(os.path.join(out_dir, f"trial_{i}_response.json"), "w") as f:
                f.write(gated.report.summary())

        trial_summaries.append(dict(
            trial=i, success=gated.success, n_attempts=gated.n_attempts,
            total_latency_s=gated.total_latency_s,
        ))

    print("\n=== gated pipeline summary across all trials ===")
    n_success = sum(1 for t in trial_summaries if t["success"])
    n_needed_retry = sum(1 for t in trial_summaries if t["n_attempts"] > 1)
    total_attempts = sum(t["n_attempts"] for t in trial_summaries)
    total_latency = sum(t["total_latency_s"] for t in trial_summaries)
    avg_attempts_overall = total_attempts / len(trial_summaries)
    avg_attempts_on_success = (
        sum(t["n_attempts"] for t in trial_summaries if t["success"]) / n_success
        if n_success else float("nan"))

    for t in trial_summaries:
        print(f"  trial {t['trial']}: success={t['success']} attempts={t['n_attempts']} "
              f"total_latency={t['total_latency_s']:.1f}s")

    print(f"\ngated success rate: {n_success}/{n_trials} "
          f"({100*n_success/n_trials:.1f}%) -- i.e. user-facing failure rate = "
          f"{n_trials - n_success}/{n_trials}")
    print(f"trials that needed at least 1 retry: {n_needed_retry}/{n_trials}")
    print(f"mean attempts per trial (all trials): {avg_attempts_overall:.2f}")
    print(f"mean attempts per trial (successes only): {avg_attempts_on_success:.2f}")
    print(f"total latency across all {n_trials} trials incl. retries: {total_latency:.1f}s")
    print(f"mean latency per trial incl. retries: {total_latency/n_trials:.1f}s")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(
            n_trials=n_trials, max_attempts=max_attempts, n_success=n_success,
            n_needed_retry=n_needed_retry, total_attempts=total_attempts,
            avg_attempts_overall=avg_attempts_overall,
            avg_attempts_on_success=avg_attempts_on_success,
            total_latency_s=total_latency,
            trials=trial_summaries,
        ), f, indent=2)
    print(f"\n[done] wrote per-trial results + summary.json to {out_dir}")
