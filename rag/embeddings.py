"""
Local embedding client for semantic KB retrieval (rag/DESIGN.md Step 1).

Previously (see rag/prompt.py git history), "retrieval" meant exact keyword
matching: regex-extract feature names out of a SessionRecord's `explanation`
string and look up a KB snippet whose `topic` equals that name exactly. That
is deterministic but not semantic retrieval -- it cannot rank among several
snippets on the same topic, handle a query that doesn't literally contain a
topic string, or generalize to KB content that isn't keyed 1:1 to a feature
name. This module replaces it with real cosine-similarity ranking over
embeddings.

Model: `all-minilm` served locally via Ollama (already running for the
Qwen3 8B coaching LLM, see rag/llm.py) -- 45MB, 384-dim embeddings, no
Anthropic/API dependency, consistent with the "fully on-device" choice
already made for the generation model. Ollama exposes a dedicated
`/api/embed` endpoint separate from `/api/chat`, used here.
"""
import json
import math
import urllib.request


class OllamaEmbedder:
    def __init__(self, model="all-minilm", host="http://localhost:11434", timeout=60):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Returns one embedding vector per input text, same order."""
        payload = {"model": self.model, "input": texts}
        req = urllib.request.Request(
            f"{self.host}/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["embeddings"]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
