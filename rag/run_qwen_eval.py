"""
Runs the shared session-level coaching prompt (rag/prompt.py) through local
Qwen3 8B (Q4_K_M, via Ollama) and reports: whether it loaded/ran within the
RTX 3060's VRAM, the raw output, the parsed CoachingResponse (or parse
failure), the rag/grounding.py verdict, and wall-clock generation latency.

Claude API is intentionally NOT run here -- no ANTHROPIC_API_KEY is
configured in this environment (see rag/DESIGN.md), and the user chose to
proceed Qwen3-only for now rather than block on that. rag/llm.py's
ClaudeBackend exists and implements the identical interface, so re-running
this same prompt through Claude later (once a key is available) requires no
changes to this script beyond swapping the backend.

Usage:
    python rag/run_qwen_eval.py
"""
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend  # noqa: E402
from rag.prompt import build_prompt  # noqa: E402
from rag.schema import CoachingResponse  # noqa: E402
from rag.grounding import check_grounding  # noqa: E402

SESSION_RECORD_PATH = os.path.join(REPO_ROOT, "ir/example_output/session_record_0.json")

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def clean_json_text(raw_text: str) -> str:
    text = _THINK_RE.sub("", raw_text).strip()
    text = _FENCE_RE.sub("", text).strip()
    return text


def query_vram_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
        )
        used, total = (int(x.strip()) for x in out.strip().split(","))
        return used, total
    except Exception as e:
        return None, None


if __name__ == "__main__":
    with open(SESSION_RECORD_PATH) as f:
        session_record = json.load(f)

    print(f"[setup] SessionRecord id={session_record['record_id']}")
    print(f"[setup] explanation: {session_record['explanation']}")

    used_before, total = query_vram_mb()
    print(f"[vram] before generation: {used_before} / {total} MiB used" if used_before is not None
          else "[vram] nvidia-smi query failed")

    system_prompt, user_prompt = build_prompt(session_record)
    print(f"[prompt] system={len(system_prompt)} chars, user={len(user_prompt)} chars")

    backend = OllamaBackend(model="qwen3:8b")
    print(f"[generate] calling {backend.name} ...")
    result = backend.generate(system_prompt, user_prompt)
    print(f"[generate] done in {result.latency_s:.2f}s")

    used_after, _ = query_vram_mb()
    print(f"[vram] after generation: {used_after} / {total} MiB used" if used_after is not None
          else "[vram] nvidia-smi query failed")

    print(f"\n[raw output] ({len(result.raw_text)} chars):\n{result.raw_text}\n")

    cleaned = clean_json_text(result.raw_text)
    out_dir = os.path.join(REPO_ROOT, "rag", "example_output")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "qwen3_raw_output.txt"), "w", encoding="utf-8") as f:
        f.write(result.raw_text)

    try:
        parsed_json = json.loads(cleaned)
        response = CoachingResponse.model_validate(parsed_json)
    except Exception as e:
        print(f"[PARSE FAILURE] could not parse Qwen3 output as a valid CoachingResponse: {e}")
        with open(os.path.join(out_dir, "qwen3_grounding_report.txt"), "w") as f:
            f.write(f"PARSE FAILURE: {e}\n\nRaw output:\n{result.raw_text}")
        sys.exit(1)

    source_records = {("SessionRecord", session_record["record_id"]): session_record}
    report = check_grounding(response, source_records)

    print("\n[grounding report]")
    print(report.summary())

    with open(os.path.join(out_dir, "qwen3_grounding_report.txt"), "w") as f:
        f.write(f"latency_s={result.latency_s:.2f}\n")
        f.write(f"vram_used_mb_before={used_before} vram_used_mb_after={used_after} vram_total_mb={total}\n\n")
        f.write(report.summary())

    print(f"\n[done] latency={result.latency_s:.2f}s, "
          f"grounding_rate={report.grounding_rate * 100:.1f}% "
          f"({report.grounded_pass_count}/{report.grounded_observation_count})")
