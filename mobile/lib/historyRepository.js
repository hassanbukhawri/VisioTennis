// Session history storage. Phase 2 (2026-08-07): swapped from on-device
// expo-sqlite to backend-persisted history tied to accounts -- exactly the
// swap mvp/DESIGN.md's "swappable seam" comment on this file originally
// called for. Confirmed before touching anything: every screen already
// called only the functions this module exports (listSessions, getSession,
// getRole, setRole), never expo-sqlite directly, so the storage backend
// swap is contained to this file's function BODIES. The one real deviation
// from "zero other changes needed": listSessions/getSession now need the
// caller's auth token (backend-persisted history is per-account), so their
// signatures gained a `token` parameter -- the 2 read call sites
// (history/index.js, history/[sessionId].js) pass `useAuth().token` in.
// addSession() is gone entirely: the backend now persists a history row
// automatically inside POST /api/v1/analyze-stroke itself (see
// backend/app.py), atomically with the analysis, so there's no longer a
// separate "save it" step for the client to call.
//
// getRole/setRole are UNCHANGED and still on-device via expo-sqlite -- the
// role setting is deliberately anonymous, unrelated to accounts (see its
// own comment below), so it has no reason to move to the backend.
import * as SQLite from "expo-sqlite";

import { apiFetch, ApiError } from "./api";

let db = null;

function getDb() {
  if (!db) {
    db = SQLite.openDatabaseSync("visiotennis_history.db");
    db.execSync(`
      CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY NOT NULL,
        value TEXT NOT NULL
      );
    `);
  }
  return db;
}

// Called once from the root layout so the table exists before any screen
// reads/writes -- cheap and idempotent (CREATE TABLE IF NOT EXISTS), safe to
// call more than once.
export function initHistoryDb() {
  getDb();
}

// Phase 1 structured feedback (2026-08-01): a bare role CATEGORY
// ("player"/"coach"), not an identifier -- stored on-device only, never
// sent anywhere except as this plain string alongside a feedback
// submission, never tied to a name or persistent per-tester id. Lets the
// feedback form show the coach-specific question without needing any real
// identity mechanism. Stays on-device even now that accounts exist --
// deliberately not part of the users table (see PROJECT_LOG.md).
export async function getRole() {
  const row = await getDb().getFirstAsync("SELECT value FROM settings WHERE key = 'role'");
  return row ? row.value : "player";
}

export async function setRole(role) {
  await getDb().runAsync(
    "INSERT INTO settings (key, value) VALUES ('role', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
    [role]
  );
}

// Summaries only (id, createdAt, stroke class/confidence, and -- as of the
// Insights charts, 2026-08-08 -- weightTransferRatio) -- enough for both the
// History list and the Insights charts without a second round-trip per
// session. The backend's /api/v1/history already sends the full
// result_json per row (see backend/app.py's list_history), so pulling one
// more field out of it client-side costs nothing extra over the network.
export async function listSessions(token) {
  const rows = await apiFetch("/api/v1/history", { token });
  return rows.map((row) => {
    const result = JSON.parse(row.result_json);
    return {
      id: row.id,
      createdAt: row.created_at,
      strokeClass: result?.stroke_event?.stroke?.predicted_class ?? "unknown",
      confidence: result?.stroke_event?.stroke?.confidence ?? null,
      weightTransferRatio: result?.stroke_event?.biomechanics?.weight_transfer_ratio ?? null,
    };
  });
}

// Full stored result for one session, for the detail screen. Returns null
// if not found (matches the on-device version's "no such id" behavior) --
// any other failure (network, server error) propagates instead of being
// mistaken for "not found."
export async function getSession(id, token) {
  try {
    const row = await apiFetch(`/api/v1/history/${id}`, { token });
    return { id: row.id, createdAt: row.created_at, result: row.result };
  } catch (e) {
    if (e instanceof ApiError && e.status === 404) return null;
    throw e;
  }
}
