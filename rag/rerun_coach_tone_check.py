"""
Before/after comparison for the coach-tone hedging fix (rag/prompt.py:
build_tone_instruction, coach branch). "Before" is reconstructed by
monkeypatching in the exact original coach-tone string (as it read prior to
this fix) rather than needing a git-tracked revert, since backend/ and rag/
aren't committed to git in this repo. "After" uses the actual current
(fixed) rag/prompt.py.

Runs against ir/example_output/stroke_event_biomech_volley.json, the exact
fixture whose KB-templated "elbow_angle_deg"/"knee_flex" hedge language
("this alone doesn't indicate...", "get a coach's eye on this") is what
motivated the fix, with tone_profile={"role": "coach"}.

Usage:
    python rag/rerun_coach_tone_check.py
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import rag.prompt as prompt_module
import rag.stroke_prompt as stroke_prompt_module
from rag.llm import OllamaBackend
from rag.gated_pipeline import generate_grounded_response
from ir.biomechanics_percentiles import BiomechanicsPercentileBenchmark

STROKE_EVENT_PATH = os.path.join(REPO_ROOT, "ir/example_output/stroke_event_biomech_volley.json")
TONE_PROFILE = {"role": "coach"}

# Exact original text, as it read before this session's fix (rag/prompt.py,
# build_tone_instruction's coach branch) -- for the "before" run only.
ORIGINAL_COACH_TONE = (
    "TONE: the person reading this is a tennis COACH, not the player themselves -- use "
    "precise, technical biomechanical language throughout (exact joint angles, kinetic-chain "
    "terminology, percentile framing where relevant) rather than simplified player-facing "
    "phrasing. Do not soften or simplify terminology for a coach audience."
)


def make_original_tone_fn():
    real_fn = prompt_module.build_tone_instruction

    def _fn(tone_profile):
        role = (tone_profile or {}).get("role")
        if role == "coach":
            return ORIGINAL_COACH_TONE
        return real_fn(tone_profile)
    return _fn


def run_once(label):
    with open(STROKE_EVENT_PATH) as f:
        stroke_event = json.load(f)
    source_records = {("StrokeEvent", stroke_event["event_id"]): stroke_event}
    benchmark = BiomechanicsPercentileBenchmark()
    backend = OllamaBackend(model="qwen3:8b")

    system_prompt, user_prompt = stroke_prompt_module.build_stroke_prompt(
        stroke_event, benchmark, tone_profile=TONE_PROFILE)

    print(f"\n{'=' * 78}\n[{label}]\n{'=' * 78}")
    gated = generate_grounded_response(
        backend, system_prompt, user_prompt, source_records, max_attempts=3, percentile_benchmark=benchmark)

    if gated.success:
        print(f"  -> SUCCESS after {gated.n_attempts} attempt(s), {gated.total_latency_s:.1f}s\n")
        for item in gated.response.items:
            print(f"    [{item.type}] {item.text}")
    else:
        print(f"  -> FAILED: {gated.failure_reason}")


if __name__ == "__main__":
    real_build_tone_instruction = prompt_module.build_tone_instruction
    original_fn = make_original_tone_fn()

    # --- BEFORE: monkeypatch both module namespaces to the pre-fix text ---
    prompt_module.build_tone_instruction = original_fn
    stroke_prompt_module.build_tone_instruction = original_fn
    run_once("BEFORE (original coach-tone instruction)")

    # --- AFTER: restore the real (fixed) function ---
    prompt_module.build_tone_instruction = real_build_tone_instruction
    stroke_prompt_module.build_tone_instruction = real_build_tone_instruction
    run_once("AFTER (fixed coach-tone instruction)")
