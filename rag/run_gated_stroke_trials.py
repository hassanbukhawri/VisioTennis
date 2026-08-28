"""
Step 4 validation: runs the gated pipeline (rag/gated_pipeline.py) on real
StrokeEvents with real BiomechanicsInfo populated, across 4 different
stroke classes (ir/populate_biomechanics_stroke_events.py) -- the
stroke-level analog of rag/run_gated_diverse_trials.py. Same discipline:
real retrieval per stroke (not hand-picked KB context), grounding pass
rate + retry rate reported, AND actual generated text spot-checked for the
thin/generic-vs-useful distinction found on the session-level
semantic-miss case (rag/DESIGN.md) -- a percentile citation can be 100%
grounded and still be low-value coaching text.

Usage:
    python rag/run_gated_stroke_trials.py [max_attempts]
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
from rag.stroke_prompt import build_stroke_prompt, select_notable_biomechanics_features  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402
from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402

# See rag/run_gated_diverse_trials.py's identical helper for why this exists
# (2026-08-02 local-Ollama transport-level flakiness, not a grounding issue).
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

STROKE_EVENT_GLOB = os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_*.json")

if __name__ == "__main__":
    max_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    paths = sorted(glob.glob(STROKE_EVENT_GLOB))
    if not paths:
        raise SystemExit(f"no biomechanics StrokeEvents found matching {STROKE_EVENT_GLOB} -- "
                          "run ir/populate_biomechanics_stroke_events.py first")

    benchmark = BiomechanicsPercentileBenchmark()
    backend = OllamaBackend(model="qwen3:8b")
    out_dir = os.path.join(REPO_ROOT, "rag", "example_output", "qwen3_gated_stroke_trials")
    os.makedirs(out_dir, exist_ok=True)

    trial_summaries = []
    for i, path in enumerate(paths, start=1):
        stroke_class = os.path.basename(path).replace("stroke_event_biomech_", "").replace(".json", "")
        with open(path) as f:
            stroke_event = json.load(f)
        source_records = {("StrokeEvent", stroke_event["event_id"]): stroke_event}

        notable = select_notable_biomechanics_features(stroke_event, benchmark)
        system_prompt, user_prompt = build_stroke_prompt(stroke_event, benchmark)

        print(f"\n=== trial {i}/{len(paths)}: {stroke_class} ===", flush=True)
        print(f"  predicted_class: {stroke_event['stroke']['predicted_class']}", flush=True)
        for n in notable:
            print(f"  notable: {n['feature']}={n['value']:.3f} -> {n['percentile']}th pctile "
                  f"(n={n['n']}){' [low-confidence]' if n['low_confidence'] else ''}", flush=True)

        gated = generate_with_network_retry(
            backend, system_prompt, user_prompt, source_records,
            max_attempts=max_attempts, percentile_benchmark=benchmark)

        for a in gated.attempts:
            status = ("ACCEPTED" if a.accepted else
                       ("parse failure" if not a.parse_ok else "rejected (not fully grounded)"))
            print(f"  attempt {a.attempt_num}/{max_attempts}: latency={a.latency_s:.1f}s {status}"
                  + (f" ({a.grounded_pass_count}/{a.grounded_observation_count} grounded)"
                     if a.parse_ok else f" -- {a.parse_error}"), flush=True)

        if gated.success:
            print(f"  -> {stroke_class} SUCCESS after {gated.n_attempts} attempt(s), "
                  f"total_latency={gated.total_latency_s:.1f}s", flush=True)
            print(f"  generated text:", flush=True)
            for item in gated.response.items:
                print(f"    [{item.type}] {item.text}", flush=True)
        else:
            print(f"  -> {stroke_class} FAILED (exhausted {max_attempts} attempts): "
                  f"{gated.failure_reason}", flush=True)

        with open(os.path.join(out_dir, f"{stroke_class}_result.json"), "w") as f:
            json.dump(dict(
                stroke_class=stroke_class,
                predicted_class=stroke_event["stroke"]["predicted_class"],
                notable_features=notable,
                success=gated.success,
                n_attempts=gated.n_attempts,
                total_latency_s=gated.total_latency_s,
                failure_reason=gated.failure_reason,
                generated_items=([dict(type=it.type, text=it.text,
                                        citation=it.citation.model_dump() if it.citation else None)
                                   for it in gated.response.items] if gated.success else None),
                attempts=[dict(attempt_num=a.attempt_num, latency_s=a.latency_s,
                                parse_ok=a.parse_ok, parse_error=a.parse_error,
                                grounded_observation_count=a.grounded_observation_count,
                                grounded_pass_count=a.grounded_pass_count,
                                accepted=a.accepted)
                          for a in gated.attempts],
            ), f, indent=2)
        if gated.success:
            with open(os.path.join(out_dir, f"{stroke_class}_report.txt"), "w") as f:
                f.write(gated.report.summary())

        trial_summaries.append(dict(
            stroke_class=stroke_class, success=gated.success, n_attempts=gated.n_attempts,
            total_latency_s=gated.total_latency_s,
        ))

    print("\n=== gated pipeline summary across all stroke-level trials ===")
    n_success = sum(1 for t in trial_summaries if t["success"])
    n_needed_retry = sum(1 for t in trial_summaries if t["n_attempts"] > 1)
    total_attempts = sum(t["n_attempts"] for t in trial_summaries)
    total_latency = sum(t["total_latency_s"] for t in trial_summaries)
    avg_attempts_overall = total_attempts / len(trial_summaries)

    for t in trial_summaries:
        print(f"  {t['stroke_class']}: success={t['success']} attempts={t['n_attempts']} "
              f"total_latency={t['total_latency_s']:.1f}s")

    print(f"\ngated success rate: {n_success}/{len(paths)} ({100*n_success/len(paths):.1f}%)")
    print(f"trials that needed at least 1 retry: {n_needed_retry}/{len(paths)}")
    print(f"mean attempts per trial: {avg_attempts_overall:.2f}")
    print(f"total latency across all {len(paths)} stroke-level trials incl. retries: {total_latency:.1f}s")

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(dict(
            n_trials=len(paths), max_attempts=max_attempts, n_success=n_success,
            n_needed_retry=n_needed_retry, total_attempts=total_attempts,
            avg_attempts_overall=avg_attempts_overall, total_latency_s=total_latency,
            trials=trial_summaries,
        ), f, indent=2)
    print(f"\n[done] wrote per-stroke results + summary.json to {out_dir}")
