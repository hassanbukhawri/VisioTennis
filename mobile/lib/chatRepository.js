// Stroke chat client (2026-08-08) -- talks to POST /api/v1/history/{id}/chat
// (backend/app.py, see PROJECT_LOG.md's chat grounding architecture entry).
// Deliberately stateless server-side per that design: the client is the only
// place conversation history lives (React state in components/StrokeChat.js,
// lost when the screen unmounts) -- every call resends the turns so far.
import { REQUEST_TIMEOUT_MS } from "./config";
import { apiFetch } from "./api";

// Reuses the same 7-minute budget as analyze-stroke's own upload request
// (mobile/lib/config.js) -- a tier-2 chat turn goes through the identical
// retry-gate (rag/gated_pipeline.py) as the original coaching generation, so
// the same measured worst case applies. Tier 1/3 turns resolve in seconds;
// this timeout only matters for tier 2.
export async function sendChatMessage(token, sessionId, message, conversationHistory) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    return await apiFetch(`/api/v1/history/${sessionId}/chat`, {
      token,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        conversation_history: conversationHistory.map(({ role, text }) => ({ role, text })),
      }),
      signal: controller.signal,
    });
  } finally {
    clearTimeout(timeoutId);
  }
}
