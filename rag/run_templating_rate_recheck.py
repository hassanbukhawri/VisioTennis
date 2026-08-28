"""
Templating-rate larger-sample re-run (2026-08-27), step 1 of the named
follow-up in PROJECT_LOG.md ("think=False boilerplate/template reuse across
sessions"). The original 8-pair thinking-mode A/B test found 7/47 (15%)
think=False items reusing IDENTICAL boilerplate text word-for-word across
DIFFERENT sessions sharing a similar anomaly shape -- a small base (n=8
sessions, 47 items) to trust. This script adds ~20 MORE real, distinct-
source think=False trials before deciding the rate is real and worth
addressing.

Two source pools, chosen to maximize genuine source diversity rather than
re-running the same 8 fixtures (which would trivially "duplicate" on
re-run for the boring reason that identical input -> similar output at
temp=0.2, telling us nothing about cross-session template reuse):

  1. The 9 real fixture files in ir/example_output/ NOT used in the
     original 8-pair test (session_record_0/1/2, stroke_event_0/1/2,
     stroke_event_biomech_serve, stroke_event_biomech_volley,
     stroke_event_end_to_end).
  2. 11 real, distinct StrokeEvents pulled from backend/app.db's
     stroke_history table -- genuine production data from real
     analyze-stroke calls made throughout this project (27 stored rows,
     deduped to 11 distinct source_video clips, most recent row per
     clip kept). This is REAL data, not synthetic, and adds diversity
     the fixture set alone can't.

Total: 20 new sessions, think=False only (no thinking-on side needed --
the on/off comparison is already established; this step only firms up
the think=False templating-rate estimate). Every response is grounding-
checked the same way as everywhere else in this project
(rag/grounding.py:check_grounding, same 100%-grounding standard).

Usage:
    python rag/run_templating_rate_recheck.py <1-20>     # run one trial
    python rag/run_templating_rate_recheck.py report      # dedup analysis
"""
import glob
import json
import os
import sqlite3
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.stroke_prompt import build_stroke_prompt  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.run_qwen_eval import clean_json_text  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.run_latency_investigation import raw_ollama_call  # noqa: E402

APP_DB = os.path.join(REPO_ROOT, "backend", "app.db")
OUT_PATH = os.path.join(REPO_ROOT, "rag", "example_output", "templating_rate_recheck_results.json")
AB_RESULTS_PATH = os.path.join(REPO_ROOT, "rag", "example_output", "thinking_mode_ab_results.json")

# stroke_event_0/1/2 and stroke_event_end_to_end are pre-biomechanics-field
# schema (predate this feature) -- build_stroke_prompt requires
# stroke_event["biomechanics"] and KeyErrors on them; excluded, not silently
# worked around. Verified 2026-08-27 (see PROJECT_LOG.md recheck entry).
NEW_FIXTURE_PATHS = [
    os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json"),
    os.path.join(REPO_ROOT, "ir/example_output/session_record_1.json"),
    os.path.join(REPO_ROOT, "ir/example_output/session_record_2.json"),
    os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_serve.json"),
    os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_volley.json"),
]


def _load_real_history_stroke_events():
    """11 distinct real StrokeEvents from backend/app.db's stroke_history,
    deduped by source_video (most recent row per clip kept)."""
    con = sqlite3.connect(APP_DB)
    cur = con.cursor()
    cur.execute("SELECT id, created_at, result_json FROM stroke_history ORDER BY created_at")
    rows = cur.fetchall()
    by_video = {}
    for rid, created, rj in rows:
        d = json.loads(rj)
        se = d.get("stroke_event")
        if not se:
            continue
        key = se.get("source_video") or se.get("event_id")
        by_video[key] = (created, se)  # later rows overwrite -> most recent kept
    con.close()
    return [(f"real_history::{video}", se) for video, (created, se) in sorted(by_video.items())]


def build_trials():
    """Returns list of (name, sys_prompt, user_prompt, source_records,
    benchmark) -- benchmark is None for SessionRecord-based trials (no
    percentile citations in that path) and the real BiomechanicsPercentileBenchmark
    for StrokeEvent-based trials, matching check_grounding's percentile_benchmark
    param (its omission was a real bug caught mid-run 2026-08-27, same class of
    bug as rag/run_latency_investigation.py's earlier in this project -- see
    PROJECT_LOG.md)."""
    benchmark = BiomechanicsPercentileBenchmark()
    trials = []

    for p in NEW_FIXTURE_PATHS:
        with open(p) as f:
            record = json.load(f)
        name = os.path.basename(p)
        if "session_record" in name:
            sys_p, user_p = build_prompt(record)
            source_records = {("SessionRecord", record["record_id"]): record}
            trials.append((name, sys_p, user_p, source_records, None))
        else:
            sys_p, user_p = build_stroke_prompt(record, benchmark)
            source_records = {("StrokeEvent", record["event_id"]): record}
            trials.append((name, sys_p, user_p, source_records, benchmark))

    for name, se in _load_real_history_stroke_events():
        sys_p, user_p = build_stroke_prompt(se, benchmark)
        source_records = {("StrokeEvent", se["event_id"]): se}
        trials.append((name, sys_p, user_p, source_records, benchmark))

    return trials


def run_single_trial(index):
    trials = build_trials()
    if not (1 <= index <= len(trials)):
        raise SystemExit(f"index {index} out of range (1..{len(trials)})")
    name, sys_p, user_p, source_records, benchmark = trials[index - 1]
    print(f"[{index}/{len(trials)}] {name} ...", flush=True)

    raw, lat, body = raw_ollama_call(sys_p, user_p, think=False)
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
    print(f"  saved ({len(existing)}/{len(trials)} done) -> {OUT_PATH}")


def report():
    """Combines the original 8-session think=False pool (47 items) with
    the new sessions from this recheck, then re-runs the SAME exact-text
    cross-session dedup check the original finding used, on the larger n."""
    with open(AB_RESULTS_PATH) as f:
        ab_results = json.load(f)
    with open(OUT_PATH) as f:
        new_results = json.load(f)

    sessions = []  # (session_name, [item_text, ...])
    for r in ab_results:
        if r.get("parse_ok_off"):
            sessions.append((r["name"] + "::think_off(original)", [it["text"] for it in r["items_off"]]))
    n_fully_grounded_new = 0
    for r in new_results:
        if r.get("parse_ok"):
            sessions.append((r["name"], [it["text"] for it in r["items"]]))
            if r.get("grounding_rate") == 1.0 and not r.get("suspect"):
                n_fully_grounded_new += 1

    total_items = sum(len(items) for _, items in sessions)
    print(f"n_sessions={len(sessions)} (8 original + {len(new_results)} new), total_items={total_items}")
    print(f"new trials: parse_ok={sum(1 for r in new_results if r.get('parse_ok'))}/{len(new_results)}, "
          f"fully_grounded={n_fully_grounded_new}/{len(new_results)}")

    # exact-text match across DIFFERENT sessions (same method as the original finding)
    text_to_sessions = {}
    for sess_name, items in sessions:
        for text in items:
            text_to_sessions.setdefault(text, set()).add(sess_name)

    duplicate_items = 0
    dup_groups = []
    for text, sess_set in text_to_sessions.items():
        if len(sess_set) > 1:
            # count every occurrence across those sessions as a "duplicate item"
            count = sum(1 for _, items in sessions if text in items)
            duplicate_items += count
            dup_groups.append((text, sorted(sess_set), count))

    print(f"\nexact-duplicate items across DIFFERENT sessions: {duplicate_items}/{total_items} "
          f"({100*duplicate_items/total_items:.1f}%)")
    print(f"distinct duplicate text groups: {len(dup_groups)}")
    for text, sess_list, count in sorted(dup_groups, key=lambda x: -x[2]):
        print(f"\n  [{count} occurrences across {len(sess_list)} sessions] {text[:150]}")
        for s in sess_list:
            print(f"    - {s}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    elif len(sys.argv) > 1:
        run_single_trial(int(sys.argv[1]))
    else:
        raise SystemExit(
            "Usage: python rag/run_templating_rate_recheck.py <trial_index 1-20>\n"
            "       python rag/run_templating_rate_recheck.py report"
        )
