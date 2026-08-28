"""
One-off investigation script (2026-08-26): where does qwen3:14b generation
latency actually go, and are there concrete, low-risk levers (no model/quality
change) worth flipping? Not a permanent trial harness like
run_suspect_general_note_gate_trials.py -- run once, numbers go in
PROJECT_LOG.md, this file stays as a record of exactly how they were produced.

Uses the SAME real prompt-building path (rag.stroke_prompt.build_stroke_prompt)
and the SAME real gated retry loop (rag.gated_pipeline.generate_grounded_response)
the production backend/app.py endpoint uses -- not a synthetic/shortened prompt.

Usage:
    python rag/run_latency_investigation.py
"""
import json
import os
import re
import sys
import time
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark  # noqa: E402
from rag.stroke_prompt import build_stroke_prompt  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402
from rag.gated_pipeline import generate_grounded_response  # noqa: E402
from rag.llm import OllamaBackend  # noqa: E402
from rag.run_qwen_eval import clean_json_text  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402

HISTORY_JSON_PATH = os.path.join(REPO_ROOT, "scratch_history.json")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def raw_ollama_call(system_prompt, user_prompt, model="qwen3:14b", extra_options=None, think=None):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2, **(extra_options or {})},
    }
    if think is not None:
        payload["think"] = think
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_s = time.monotonic() - t0
    raw_text = body["message"]["content"]
    return raw_text, latency_s, body


if __name__ == "__main__":
    with open(HISTORY_JSON_PATH) as f:
        history = json.load(f)
    stroke_event = json.loads(history[0]["result_json"])["stroke_event"]
    source_records = {("StrokeEvent", stroke_event["event_id"]): stroke_event}
    benchmark = BiomechanicsPercentileBenchmark()
    system_prompt, user_prompt = build_stroke_prompt(stroke_event, benchmark)

    print("=" * 78)
    print("Part 1: prompt size (real prompt, as sent to the model)")
    print("=" * 78)
    print(f"system_prompt: {len(system_prompt)} chars")
    print(f"user_prompt:   {len(user_prompt)} chars")
    print(f"total:         {len(system_prompt) + len(user_prompt)} chars "
          f"(~{(len(system_prompt) + len(user_prompt)) // 4} tokens at a ~4 chars/token estimate)")
    print(f"model context length (qwen3:14b): 40960 tokens -- prompt is nowhere near this ceiling")

    print("\n" + "=" * 78)
    print("Part 2: is 'thinking' mode on by default, and what does it cost?")
    print("=" * 78)
    print("Baseline call (current code path -- no 'think' key sent, Ollama's default for a")
    print("thinking-capable model applies)...")
    raw_default, lat_default, body_default = raw_ollama_call(system_prompt, user_prompt, think=None)
    think_match = _THINK_RE.search(raw_default)
    think_len = len(think_match.group(0)) if think_match else 0
    final_len = len(_THINK_RE.sub("", raw_default))
    eval_count = body_default.get("eval_count")
    prompt_eval_count = body_default.get("prompt_eval_count")
    print(f"  latency: {lat_default:.1f}s")
    print(f"  eval_count (output tokens): {eval_count}, prompt_eval_count (input tokens): {prompt_eval_count}")
    print(f"  <think> block present in content: {think_match is not None}, think block chars: {think_len}, "
          f"final-answer chars: {final_len}")
    print(f"  message keys returned: {sorted(body_default.get('message', {}).keys())}")
    thinking_field = body_default.get("message", {}).get("thinking")
    if thinking_field:
        print(f"  SEPARATE 'thinking' field present: {len(thinking_field)} chars")
        print(f"  thinking field preview: {thinking_field[:300]!r}")
    if think_match:
        print(f"  think block preview: {think_match.group(0)[:300]!r}")

    print("\nSame prompt with think=False (Ollama's explicit thinking-mode-off switch)...")
    raw_nothink, lat_nothink, body_nothink = raw_ollama_call(system_prompt, user_prompt, think=False)
    eval_count_nt = body_nothink.get("eval_count")
    prompt_eval_count_nt = body_nothink.get("prompt_eval_count")
    print(f"  latency: {lat_nothink:.1f}s")
    print(f"  eval_count (output tokens): {eval_count_nt}, prompt_eval_count (input tokens): {prompt_eval_count_nt}")
    print(f"  <think> block present: {_THINK_RE.search(raw_nothink) is not None}")
    print(f"  message keys returned: {sorted(body_nothink.get('message', {}).keys())}")
    print(f"  raw output preview: {raw_nothink[:300]!r}")

    speedup = lat_default / lat_nothink if lat_nothink else float("nan")
    print(f"\n  -> think=False speedup this trial: {speedup:.2f}x ({lat_default:.1f}s -> {lat_nothink:.1f}s)")

    print("\nGrounding check on the think=False output (must still be 100% to count as viable)...")
    try:
        cleaned = clean_json_text(raw_nothink)
        response = CoachingResponse.model_validate(json.loads(cleaned))
        report = check_grounding(response, source_records, percentile_benchmark=benchmark)
        print(report.summary())
        print(f"  has_suspect_general_note: {report.has_suspect_general_note}")
    except Exception as e:
        print(f"  FAILED to parse/ground think=False output: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print("Part 3: max_attempts -- real first-attempt success rate under the CURRENT gate")
    print("(includes the just-shipped SUSPECT_GENERAL_NOTE check, which the original")
    print("'6/6 clean' eval predates)")
    print("=" * 78)
    N_TRIALS = 5
    first_attempt_ok = 0
    all_attempts_latencies = []
    for i in range(1, N_TRIALS + 1):
        gated = generate_grounded_response(
            OllamaBackend(model="qwen3:14b"), system_prompt, user_prompt, source_records, max_attempts=3,
            percentile_benchmark=benchmark)
        all_attempts_latencies.append([a.latency_s for a in gated.attempts])
        ok = gated.success and gated.n_attempts == 1
        first_attempt_ok += int(ok)
        print(f"  trial {i}: success={gated.success}, n_attempts={gated.n_attempts}, "
              f"per-attempt latencies={[f'{a.latency_s:.1f}s' for a in gated.attempts]}, "
              f"attempt reasons={[('parse_fail' if not a.parse_ok else ('suspect_note' if a.suspect_general_note else ('accepted' if a.accepted else 'ungrounded'))) for a in gated.attempts]}")
    print(f"\n  first-attempt success: {first_attempt_ok}/{N_TRIALS}")
