"""
Re-runs the exact session that produced the documented thin/generic-response
case (PROJECT_LOG.md, 2026-07-18 entry: `second_serve_ace_decline`, a
retrieval-degraded-but-still-100%-grounded session whose generated text was
bare number restatement with no real coaching interpretation) through the
gated pipeline with the rule-8 prompt fix (rag/prompt.py) applied, so the
before/after can be compared on the actual reproduced case, not a fresh
unrelated example.

Also runs one additional diverse session as a second before/after data
point, per the task's "test against a few real sessions" ask.

Usage:
    python rag/rerun_thin_response_check.py
"""
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from rag.llm import OllamaBackend
from rag.prompt import build_prompt, retrieve_snippets, extract_deviated_feature_names
from rag.gated_pipeline import generate_grounded_response

TARGET_SESSIONS = ["second_serve_ace_decline", "rally_style_shift"]

if __name__ == "__main__":
    backend = OllamaBackend(model="qwen3:8b")

    for target in TARGET_SESSIONS:
        path = os.path.join(REPO_ROOT, f"ir/example_output/session_record_diverse_{target}.json")
        if not os.path.exists(path):
            print(f"[skip] {target}: fixture not found at {path}")
            continue

        with open(path) as f:
            session_record = json.load(f)
        source_records = {("SessionRecord", session_record["record_id"]): session_record}

        expected_features = set(extract_deviated_feature_names(session_record))
        retrieved = retrieve_snippets(session_record)
        retrieved_topics = set(s.topic for s in retrieved)
        overlap = expected_features & retrieved_topics

        print(f"\n{'=' * 78}\n[{target}]\n{'=' * 78}")
        print(f"  explanation: {session_record['explanation']}")
        print(f"  expected features: {sorted(expected_features)}")
        print(f"  retrieved topics:  {sorted(retrieved_topics)}  (recall {len(overlap)}/{len(expected_features)})")

        system_prompt, user_prompt = build_prompt(session_record)
        gated = generate_grounded_response(backend, system_prompt, user_prompt, source_records, max_attempts=3)

        if gated.success:
            print(f"\n  -> SUCCESS after {gated.n_attempts} attempt(s), {gated.total_latency_s:.1f}s\n")
            print(gated.report.summary())
        else:
            print(f"\n  -> FAILED: {gated.failure_reason}")
