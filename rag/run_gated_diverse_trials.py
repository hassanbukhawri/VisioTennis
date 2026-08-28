"""
Re-runs the gated reliability test (rag/gated_pipeline.py) across 6 DIFFERENT
SessionRecords with distinct anomaly profiles (rag/DESIGN.md Step 3
"validation") -- not the same SessionRecord repeated 6 times
(rag/run_gated_multitrial.py). Each trial also uses real semantic retrieval
(rag/prompt.py:retrieve_snippets(), embedding-based via rag/embeddings.py)
picked fresh per session, not hand-selected KB context -- and reports
whether retrieval actually pulled KB entries matching that session's real
top-deviated features (rag/prompt.py:extract_deviated_feature_names()).

Usage:
    python rag/run_gated_diverse_trials.py [max_attempts]
"""
import glob
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

# 2026-08-02: this local Ollama setup intermittently stalls on the raw HTTP
# request itself (urllib TimeoutError, no response at all) after a handful
# of sequential /api/chat calls in one process -- reproduced 3x across
# otherwise-identical runs, at a DIFFERENT trial index each time, and a
# standalone single-call reproduction of the exact same prompt that hung
# came back in 65s. That rules out any particular prompt/KB content as the
# cause -- this is transport-level flakiness in the local serving setup,
# not a grounding or retrieval problem (every attempt that DID complete
# was 100% grounded). Retrying the whole trial from scratch (not touching
# rag/gated_pipeline.py's retry-gate, which is about ungrounded content,
# not network failures) is a validation-script-only workaround for that.
NETWORK_RETRY_ATTEMPTS = 3
NETWORK_RETRY_DELAY_S = 5


def generate_with_network_retry(*args, **kwargs):
    for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
        try:
            return generate_grounded_response(*args, **kwargs)
        except (TimeoutError, urllib.error.URLError, ConnectionError) as e:
            print(f"  [network retry {attempt}/{NETWORK_RETRY_ATTEMPTS}] {type(e).__name__}: {e}", flush=True)
            if attempt == NETWORK_RETRY_ATTEMPTS:
                raise
            time.sleep(NETWORK_RETRY_DELAY_S)

SESSION_RECORD_GLOB = os.path.join(REPO_ROOT, "ir/example_output/session_record_diverse_*.json")

if __name__ == "__main__":
    max_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    paths = sorted(glob.glob(SESSION_RECORD_GLOB))
    if not paths:
        raise SystemExit(f"no diverse SessionRecords found matching {SESSION_RECORD_GLOB} "
                          "-- run ir/populate_diverse_sessions.py first")

    backend = OllamaBackend(model="qwen3:8b")
    out_dir = os.path.join(REPO_ROOT, "rag", "example_output", "qwen3_gated_diverse_trials")
    os.makedirs(out_dir, exist_ok=True)

    trial_summaries = []
    for i, path in enumerate(paths, start=1):
        session_name = os.path.basename(path).replace("session_record_diverse_", "").replace(".json", "")
        with open(path) as f:
            session_record = json.load(f)
        source_records = {("SessionRecord", session_record["record_id"]): session_record}

        expected_features = set(extract_deviated_feature_names(session_record))
        retrieved = retrieve_snippets(session_record)
        retrieved_topics = set(s.topic for s in retrieved)
        overlap = expected_features & retrieved_topics
        retrieval_recall = len(overlap) / len(expected_features) if expected_features else float("nan")

        system_prompt, user_prompt = build_prompt(session_record)

        print(f"\n=== trial {i}/{len(paths)}: {session_name} ===", flush=True)
        print(f"  anomaly explanation: {session_record['explanation']}", flush=True)
        print(f"  expected deviated features: {sorted(expected_features)}", flush=True)
        print(f"  retrieved KB topics: {sorted(retrieved_topics)}", flush=True)
        print(f"  retrieval recall: {len(overlap)}/{len(expected_features)} "
              f"({retrieval_recall*100:.0f}%)", flush=True)

        gated = generate_with_network_retry(
            backend, system_prompt, user_prompt, source_records, max_attempts=max_attempts)

        for a in gated.attempts:
            status = ("ACCEPTED" if a.accepted else
                       ("parse failure" if not a.parse_ok else "rejected (not fully grounded)"))
            print(f"  attempt {a.attempt_num}/{max_attempts}: latency={a.latency_s:.1f}s {status}"
                  + (f" ({a.grounded_pass_count}/{a.grounded_observation_count} grounded)"
                     if a.parse_ok else f" -- {a.parse_error}"), flush=True)

        if gated.success:
            print(f"  -> {session_name} SUCCESS after {gated.n_attempts} attempt(s), "
                  f"total_latency={gated.total_latency_s:.1f}s", flush=True)
        else:
            print(f"  -> {session_name} FAILED (exhausted {max_attempts} attempts): "
                  f"{gated.failure_reason}", flush=True)

        with open(os.path.join(out_dir, f"{session_name}_result.json"), "w") as f:
            json.dump(dict(
                session_name=session_name,
                explanation=session_record["explanation"],
                expected_features=sorted(expected_features),
                retrieved_topics=sorted(retrieved_topics),
                retrieval_recall=retrieval_recall,
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
            with open(os.path.join(out_dir, f"{session_name}_report.txt"), "w") as f:
                f.write(gated.report.summary())

        trial_summaries.append(dict(
            session_name=session_name, success=gated.success, n_attempts=gated.n_attempts,
            total_latency_s=gated.total_latency_s, retrieval_recall=retrieval_recall,
        ))

    print("\n=== gated pipeline summary across all diverse sessions ===")
    n_success = sum(1 for t in trial_summaries if t["success"])
    n_needed_retry = sum(1 for t in trial_summaries if t["n_attempts"] > 1)
    total_attempts = sum(t["n_attempts"] for t in trial_summaries)
    total_latency = sum(t["total_latency_s"] for t in trial_summaries)
    avg_attempts_overall = total_attempts / len(trial_summaries)
    recalls = [t["retrieval_recall"] for t in trial_summaries if t["retrieval_recall"] == t["retrieval_recall"]]
    mean_recall = sum(recalls) / len(recalls) if recalls else float("nan")

    for t in trial_summaries:
        print(f"  {t['session_name']}: success={t['success']} attempts={t['n_attempts']} "
              f"total_latency={t['total_latency_s']:.1f}s "
              f"retrieval_recall={t['retrieval_recall']*100:.0f}%")

    print(f"\ngated success rate: {n_success}/{len(paths)} ({100*n_success/len(paths):.1f}%)")
    print(f"trials that needed at least 1 retry: {n_needed_retry}/{len(paths)}")
    print(f"mean attempts per trial: {avg_attempts_overall:.2f}")
    print(f"total latency across all {len(paths)} diverse trials incl. retries: {total_latency:.1f}s")
    print(f"mean retrieval recall across sessions: {mean_recall*100:.1f}%")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(
            n_trials=len(paths), max_attempts=max_attempts, n_success=n_success,
            n_needed_retry=n_needed_retry, total_attempts=total_attempts,
            avg_attempts_overall=avg_attempts_overall, total_latency_s=total_latency,
            mean_retrieval_recall=mean_recall, trials=trial_summaries,
        ), f, indent=2)
    print(f"\n[done] wrote per-session results + summary.json to {out_dir}")
