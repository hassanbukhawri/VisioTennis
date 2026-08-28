"""
Temperature A/B test (2026-08-27), step 2 of the templating-rate follow-up
in PROJECT_LOG.md -- run only after step 1 (rag/run_templating_rate_recheck.py)
confirmed the boilerplate-reuse rate is real at larger n (15/143 = 10.5%,
n=24 sessions, up from the original 7/47 = 15% at n=8). This step checks
whether a modest bump to Ollama's generation temperature (currently a fixed
0.2 in rag/llm.py, never tuned per model) reduces verbatim cross-session
repetition WITHOUT costing grounding -- same 100%-grounding standard as
every other change this session.

Paired design, not a fresh sample: the SAME 24 source records used in the
step-1 recheck (16 from run_templating_rate_recheck.build_trials() + 8 from
run_thinking_mode_ab_trial.build_pairs(), the original A/B test's fixtures)
are re-run at temperature=0.5, think=False, isolating temperature as the
only changed variable. Grounding and duplicate-text rate are compared
directly against the temp=0.2 baseline already collected in
rag/example_output/templating_rate_recheck_results.json and
rag/example_output/thinking_mode_ab_results.json (items_off).

Usage:
    python rag/run_temperature_ab_trial.py <1-24>
    python rag/run_temperature_ab_trial.py report
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.grounding import check_grounding  # noqa: E402
from rag.run_qwen_eval import clean_json_text  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.run_latency_investigation import raw_ollama_call  # noqa: E402
from rag.run_templating_rate_recheck import build_trials as build_recheck_trials  # noqa: E402
from rag.run_thinking_mode_ab_trial import build_pairs  # noqa: E402

TEMPERATURE = float(os.environ.get("TEMP_AB_TEMPERATURE", "0.5"))
OUT_PATH = os.path.join(
    REPO_ROOT, "rag", "example_output",
    f"temperature_ab_results_t{str(TEMPERATURE).replace('.', '')}.json")
RECHECK_PATH = os.path.join(REPO_ROOT, "rag", "example_output", "templating_rate_recheck_results.json")
AB_RESULTS_PATH = os.path.join(REPO_ROOT, "rag", "example_output", "thinking_mode_ab_results.json")


def build_all_24():
    recheck = build_recheck_trials()  # 16, each (name, sys_p, user_p, source_records, benchmark)
    original = [(name, sp, up, sr, bm) for (name, sp, up, sr, bm) in build_pairs()]  # 8
    combined = recheck + original
    assert len(combined) == 24, f"expected 24, got {len(combined)}"
    return combined


def run_single_trial(index):
    trials = build_all_24()
    name, sys_p, user_p, source_records, benchmark = trials[index - 1]
    print(f"[{index}/24] {name} (temperature={TEMPERATURE}) ...", flush=True)

    raw, lat, body = raw_ollama_call(sys_p, user_p, think=False, extra_options={"temperature": TEMPERATURE})
    result = {"name": name, "latency_s": lat}
    try:
        cleaned = clean_json_text(raw)
        response = CoachingResponse.model_validate(json.loads(cleaned))
        report = check_grounding(response, source_records, percentile_benchmark=benchmark)
        n = report.grounded_observation_count
        result["parse_ok"] = True
        result["grounding_rate"] = report.grounding_rate if n else None
        result["suspect"] = report.has_suspect_general_note
        result["items"] = [{"type": it.type, "text": it.text} for it in response.items]
        print(f"  latency={lat:.1f}s grounding={result['grounding_rate']} "
              f"suspect={result['suspect']} n_items={len(result['items'])}")
    except Exception as e:
        result["parse_ok"] = False
        result["parse_error"] = f"{type(e).__name__}: {e}"
        result["items"] = []
        print(f"  PARSE FAILED: {result['parse_error']}")

    existing = []
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
    existing = [x for x in existing if x["name"] != name]
    existing.append(result)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  saved ({len(existing)}/24 done) -> {OUT_PATH}")


def _dedup_rate(sessions):
    total_items = sum(len(items) for _, items in sessions)
    text_to_sessions = {}
    for sess_name, items in sessions:
        for text in items:
            text_to_sessions.setdefault(text, set()).add(sess_name)
    duplicate_items = 0
    groups = []
    for text, sess_set in text_to_sessions.items():
        if len(sess_set) > 1:
            count = sum(1 for _, items in sessions if text in items)
            duplicate_items += count
            groups.append((text, sorted(sess_set), count))
    return total_items, duplicate_items, groups


def report():
    with open(OUT_PATH) as f:
        new_results = json.load(f)

    n_grounded = sum(1 for r in new_results if r.get("parse_ok") and r.get("grounding_rate") == 1.0 and not r.get("suspect"))
    n_parse_ok = sum(1 for r in new_results if r.get("parse_ok"))
    print(f"temperature={TEMPERATURE}: n={len(new_results)}, parse_ok={n_parse_ok}/{len(new_results)}, "
          f"fully_grounded={n_grounded}/{len(new_results)}")

    new_sessions = [(r["name"], [it["text"] for it in r["items"]]) for r in new_results if r.get("parse_ok")]
    total_new, dup_new, groups_new = _dedup_rate(new_sessions)
    print(f"\ntemp={TEMPERATURE} duplicate rate: {dup_new}/{total_new} ({100*dup_new/total_new:.1f}%)")
    print(f"distinct duplicate groups: {len(groups_new)}")
    for text, sess_list, count in sorted(groups_new, key=lambda x: -x[2]):
        print(f"  [{count}x across {len(sess_list)} sessions] {text[:120]}")
        for s in sess_list:
            print(f"    - {s}")

    # baseline (temp=0.2) recomputed the same way, same 24 sources, for a clean paired comparison
    with open(RECHECK_PATH) as f:
        recheck_baseline = json.load(f)
    with open(AB_RESULTS_PATH) as f:
        ab_baseline = json.load(f)
    baseline_sessions = [(r["name"], [it["text"] for it in r["items"]]) for r in recheck_baseline if r.get("parse_ok")]
    baseline_sessions += [(r["name"] + "::think_off(original)", [it["text"] for it in r["items_off"]])
                           for r in ab_baseline if r.get("parse_ok_off")]
    total_base, dup_base, _ = _dedup_rate(baseline_sessions)
    print(f"\nbaseline (temp=0.2, same 24 sources) duplicate rate: {dup_base}/{total_base} "
          f"({100*dup_base/total_base:.1f}%)")
    print(f"\nCOMPARISON: temp=0.2 -> {100*dup_base/total_base:.1f}% dup, "
          f"temp={TEMPERATURE} -> {100*dup_new/total_new:.1f}% dup, "
          f"grounding temp=0.2 -> (see step-1 report), temp={TEMPERATURE} -> {n_grounded}/{len(new_results)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    elif len(sys.argv) > 1:
        run_single_trial(int(sys.argv[1]))
    else:
        raise SystemExit("Usage: python rag/run_temperature_ab_trial.py <1-24>|report")
