"""
LLM backend abstraction for the RAG coaching layer.

Deliberately thin: every backend exposes the same `generate(system_prompt,
user_prompt) -> GenerationResult` signature so rag/prompt.py, rag/grounding.py,
and any future pipeline code never need to know which model actually produced
a response. This exists BECAUSE the model choice is still being evaluated
(see rag/DESIGN.md "LLM choice" -- local Qwen3 8B vs. Claude API) -- nothing
downstream should hard-code assumptions specific to either one (e.g. Ollama's
HTTP response shape, or Anthropic's message-block format) outside this file.

Three backends:
  - OllamaBackend: local inference via Ollama's HTTP API (default
    localhost:11434). Used for Qwen3 8B (Q4_K_M) evaluation.
  - ClaudeBackend: Anthropic Messages API. Written to prove the abstraction
    is real, not just theoretical -- but UNTESTED as of this writing (no
    ANTHROPIC_API_KEY configured in this environment, `anthropic` package not
    installed -- see rag/DESIGN.md). Raises a clear error if used without
    both present, rather than failing silently or faking a response.
  - OpenAICompatibleBackend: hosted inference via any provider implementing
    the OpenAI chat-completions wire format (Fireworks, Together, DeepInfra,
    Groq all speak this). Parameterized by base_url + model rather than
    hard-coded to one provider, since the format is identical across all of
    them -- adopted for Fireworks AI's hosted Qwen3 8B (FP8), see
    PROJECT_LOG.md 2026-08-07 for why Fireworks specifically (only one of
    the 4 evaluated providers that actually hosts this exact model).
"""
import json
import time
import urllib.request
from dataclasses import dataclass


@dataclass
class GenerationResult:
    backend_name: str
    model_name: str
    raw_text: str
    latency_s: float


class OllamaBackend:
    def __init__(self, model="qwen3:8b", host="http://localhost:11434", timeout=300, think: bool | None = None):
        """think (2026-08-26, default None -- byte-identical to every existing
        caller's behavior before this param existed, since Ollama's own default
        for a thinking-capable model like qwen3 applies when the 'think' key is
        omitted from the request entirely): explicitly pass think=False to
        disable Qwen3's reasoning pass. See PROJECT_LOG.md "qwen3:14b thinking-
        mode A/B test" for why -- an 8-paired-trial real comparison (thinking on
        vs. off) across 6 diverse SessionRecords + 2 StrokeEvents found
        think=False both 2.22x faster (mean 64.1s -> 28.9s, faster in 8/8
        pairs) AND more reliable (8/8 fully grounded vs. 6/8 for thinking-on,
        which had one complete grounding failure and one SUSPECT_GENERAL_NOTE
        rejection) -- not a pure speed/quality tradeoff in favor of speed, a
        genuine win on the metric this whole project prioritizes first
        (grounding). Real caveat, not hidden: think=False's output reused
        verbatim boilerplate text across different sessions with the same
        anomaly shape in 7/47 items (0/36 for thinking-on) -- a real, different
        kind of "generic" than the thin/single-instance-vague failure mode
        characterized earlier this session, flagged as a follow-up, not
        blocking this change."""
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.think = think

    @property
    def name(self):
        return f"ollama:{self.model}"

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": "json",  # Ollama-level constraint: valid JSON syntax (not schema-aware)
            # temperature=0.3 (2026-08-27, was a fixed 0.2 previously, never tuned per
            # model until this change): paired A/B on the same 24 real sources found
            # 0.3 gives a real cross-session duplicate-boilerplate-rate reduction
            # (10.5% -> 9.1%, see "Follow-up resolved: templating-rate recheck" in
            # PROJECT_LOG.md) at ZERO grounding cost (24/24 fully-grounded-would-pass-
            # gate, matching the 0.2 baseline exactly) -- unlike 0.5, which was tested
            # first and rejected for regressing grounding (22/24, 2 new SUSPECT_
            # GENERAL_NOTE flags) despite a bigger duplicate-rate drop. Explicitly a
            # modest mitigation, not a fix -- see PROJECT_LOG.md for why the underlying
            # templating tendency remains only partially addressed.
            "options": {"temperature": 0.3},
        }
        if self.think is not None:
            payload["think"] = self.think
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency_s = time.monotonic() - t0
        raw_text = body["message"]["content"]
        return GenerationResult(self.name, self.model, raw_text, latency_s)


class OpenAICompatibleBackend:
    """Hosted inference via any OpenAI-chat-completions-compatible endpoint.
    Not tied to Fireworks specifically -- base_url + model are the only
    provider-specific bits, so this same class covers Together/DeepInfra/Groq
    too if ever needed. UNTESTED as of this writing (no API key configured);
    a small confirmatory re-run against rag/grounding.py is required before
    this is trusted for real traffic, per rag/DESIGN.md."""

    def __init__(self, model, api_key, base_url, timeout=300):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def name(self):
        return f"openai-compatible:{self.model}"

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        if not self.api_key:
            raise RuntimeError(
                "OpenAICompatibleBackend requires an API key (none configured)."
            )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        latency_s = time.monotonic() - t0
        raw_text = body["choices"][0]["message"]["content"]
        return GenerationResult(self.name, self.model, raw_text, latency_s)


class ClaudeBackend:
    """UNTESTED -- see module docstring. Written so the abstraction is proven
    real (both backends implement the exact same interface with no
    Qwen-specific leakage), not to be relied on until a key is configured."""

    def __init__(self, model="claude-sonnet-5", api_key=None):
        self.model = model
        self.api_key = api_key

    @property
    def name(self):
        return f"anthropic:{self.model}"

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "ClaudeBackend requires the `anthropic` package, which is not "
                "installed in this environment (pip install anthropic)."
            ) from e
        if not self.api_key:
            raise RuntimeError(
                "ClaudeBackend requires an API key (no ANTHROPIC_API_KEY was "
                "configured -- see rag/DESIGN.md Step 3)."
            )
        client = anthropic.Anthropic(api_key=self.api_key)
        t0 = time.monotonic()
        resp = client.messages.create(
            model=self.model, max_tokens=2048, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_s = time.monotonic() - t0
        raw_text = "".join(b.text for b in resp.content if b.type == "text")
        return GenerationResult(self.name, self.model, raw_text, latency_s)
