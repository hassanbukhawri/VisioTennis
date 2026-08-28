"""
Thinking-mode A/B test for qwen3:14b, tracked as its own item (2026-08-26) --
NOT the same as the latency investigation's single paired sample
(rag/run_latency_investigation.py, PROJECT_LOG.md "qwen3:14b latency
investigation"), which found a real-but-thin 1.21x speedup on n=1 and
explicitly deferred a real recommendation pending genuine quality evidence.
This script is that follow-up: 8 paired trials (thinking on vs. off, same
prompt each pair) across 8 real, diverse records already used elsewhere in
this project for lever evaluation (6 diverse SessionRecords + 2 stroke-level
StrokeEvents, for cross-domain coverage) -- not one prompt repeated 8 times.

Every output on both sides goes through the SAME real grounding check
(rag/grounding.py:check_grounding()) -- a thinking-mode change is only a real
option if it doesn't cost grounding, same standard as every other change
this session. Quality is NOT reduced to the grounding pass/fail alone: this
script also does a real qualitative read of the text pairs for the
thin/generic-vs-specific distinction already characterized earlier this
session (rag/DESIGN.md's semantic-miss case; the few-shot/schema-tightening
lever entries in PROJECT_LOG.md) -- printed side-by-side per pair so a human
(or a follow-up read of this output) can judge, not just an automated metric.

Usage:
    python rag/run_thinking_mode_ab_trial.py
"""
import glob
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.stroke_prompt import build_stroke_prompt  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.run_qwen_eval import clean_json_text  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.run_latency_investigation import raw_ollama_call  # noqa: E402

SESSION_RECORD_GLOB = os.path.join(REPO_ROOT, "ir/example_output/session_record_diverse_*.json")
STROKE_EVENT_PATHS = [
    os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_forehand.json"),
    os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_backhand.json"),
]


def run_pair(name, system_prompt, user_prompt, source_records, benchmark=None):
    """Runs the SAME prompt twice (thinking on, thinking off), grounds both,
    returns a dict with everything needed for the latency + quality report."""
    raw_on, lat_on, body_on = raw_ollama_call(system_prompt, user_prompt, think=None)
    raw_off, lat_off, body_off = raw_ollama_call(system_prompt, user_prompt, think=False)

    result = {"name": name, "lat_on": lat_on, "lat_off": lat_off}

    for side, raw, body in [("on", raw_on, body_on), ("off", raw_off, body_off)]:
        thinking_field = body.get("message", {}).get("thinking", "")
        result[f"thinking_chars_{side}"] = len(thinking_field)
        try:
            cleaned = clean_json_text(raw)
            response = CoachingResponse.model_validate(json.loads(cleaned))
            report = check_grounding(response, source_records, percentile_benchmark=benchmark)
            n = report.grounded_observation_count
            result[f"parse_ok_{side}"] = True
            result[f"grounding_rate_{side}"] = report.grounding_rate if n else None
            result[f"suspect_{side}"] = report.has_suspect_general_note
            result[f"items_{side}"] = [
                {"type": it.type, "text": it.text} for it in response.items
            ]
        except Exception as e:
            result[f"parse_ok_{side}"] = False
            result[f"parse_error_{side}"] = f"{type(e).__name__}: {e}"
            result[f"items_{side}"] = []
    return result


def build_pairs():
    benchmark = BiomechanicsPercentileBenchmark()
    pairs = []

    session_paths = sorted(glob.glob(SESSION_RECORD_GLOB))
    for p in session_paths:
        with open(p) as f:
            session_record = json.load(f)
        sys_p, user_p = build_prompt(session_record)
        source_records = {("SessionRecord", session_record["record_id"]): session_record}
        pairs.append((os.path.basename(p), sys_p, user_p, source_records, None))

    for p in STROKE_EVENT_PATHS:
        with open(p) as f:
            stroke_event = json.load(f)
        sys_p, user_p = build_stroke_prompt(stroke_event, benchmark)
        source_records = {("StrokeEvent", stroke_event["event_id"]): stroke_event}
        pairs.append((os.path.basename(p), sys_p, user_p, source_records, benchmark))

    return pairs


OUT_PATH = os.path.join(REPO_ROOT, "rag", "example_output", "thinking_mode_ab_results.json")


def run_single_pair_mode(index):
    """Runs ONE pair (1-based index) and appends its result to OUT_PATH --
    called once per invocation (rag/run_thinking_mode_ab_trial.py N) so each
    real generation pair runs as its own short foreground call instead of one
    long unattended background job, per the user's explicit request after the
    background-task version was killed 3 times."""
    pairs = build_pairs()
    if not (1 <= index <= len(pairs)):
        raise SystemExit(f"index {index} out of range (1..{len(pairs)})")
    name, sys_p, user_p, source_records, bm = pairs[index - 1]
    print(f"[{index}/{len(pairs)}] {name} ...", flush=True)
    t0 = time.monotonic()
    r = run_pair(name, sys_p, user_p, source_records, benchmark=bm)
    print(f"  on:  {r['lat_on']:.1f}s (thinking {r['thinking_chars_on']} chars), "
          f"grounding={r.get('grounding_rate_on')}, suspect={r.get('suspect_on')}, "
          f"parse_ok={r['parse_ok_on']}")
    print(f"  off: {r['lat_off']:.1f}s, "
          f"grounding={r.get('grounding_rate_off')}, suspect={r.get('suspect_off')}, "
          f"parse_ok={r['parse_ok_off']}")
    print(f"  (pair took {time.monotonic() - t0:.1f}s total)", flush=True)

    existing = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
    existing = [x for x in existing if x["name"] != name]  # replace if re-run
    existing.append(r)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  saved ({len(existing)}/{len(pairs)} pairs done) -> {OUT_PATH}")


def print_summary():
    with open(OUT_PATH) as f:
        results = json.load(f)
    print("SUMMARY")
    print("=" * 78)
    lat_ons = [r["lat_on"] for r in results]
    lat_offs = [r["lat_off"] for r in results]
    mean_on = sum(lat_ons) / len(lat_ons)
    mean_off = sum(lat_offs) / len(lat_offs)
    print(f"n={len(results)} paired trials")
    print(f"mean latency, thinking ON:  {mean_on:.1f}s (range {min(lat_ons):.1f}-{max(lat_ons):.1f}s)")
    print(f"mean latency, thinking OFF: {mean_off:.1f}s (range {min(lat_offs):.1f}-{max(lat_offs):.1f}s)")
    print(f"mean speedup: {mean_on / mean_off:.2f}x")
    n_faster_off = sum(1 for r in results if r["lat_off"] < r["lat_on"])
    print(f"OFF faster than ON in {n_faster_off}/{len(results)} pairs")

    n_grounded_on = sum(1 for r in results if r.get("parse_ok_on") and r.get("grounding_rate_on") == 1.0 and not r.get("suspect_on"))
    n_grounded_off = sum(1 for r in results if r.get("parse_ok_off") and r.get("grounding_rate_off") == 1.0 and not r.get("suspect_off"))
    print(f"fully grounded (would pass the retry-gate): ON {n_grounded_on}/{len(results)}, OFF {n_grounded_off}/{len(results)}")

    print("\n" + "=" * 78)
    print("SIDE-BY-SIDE TEXT (for the thin/generic-vs-specific quality read)")
    print("=" * 78)
    for r in results:
        print(f"\n--- {r['name']} ---")
        print("  ON:")
        for it in r.get("items_on", []):
            print(f"    [{it['type']}] {it['text']}")
        if not r.get("parse_ok_on"):
            print(f"    PARSE FAILED: {r.get('parse_error_on')}")
        print("  OFF:")
        for it in r.get("items_off", []):
            print(f"    [{it['type']}] {it['text']}")
        if not r.get("parse_ok_off"):
            print(f"    PARSE FAILED: {r.get('parse_error_off')}")

    print(f"\nFull results saved to {OUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        print_summary()
    elif len(sys.argv) > 1:
        run_single_pair_mode(int(sys.argv[1]))
    else:
        raise SystemExit(
            "Usage: python rag/run_thinking_mode_ab_trial.py <pair_index 1-8>\n"
            "       python rag/run_thinking_mode_ab_trial.py summary"
        )
