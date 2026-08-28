// Small shared fetch helper (Phase 2, 2026-08-07) -- every authenticated
// request needs the same two headers (trial passcode + bearer token), so
// this is centralized here rather than duplicated across AuthContext,
// historyRepository, and processing.js.
import { BACKEND_URL, TRIAL_PASSCODE, TRIAL_PASSCODE_HEADER } from "./config";

export class ApiError extends Error {
  constructor(status, body) {
    super(`API error ${status}`);
    this.status = status;
    this.body = body;
  }
}

// `token` (if given) becomes the Authorization header; any other fetch
// options (method, headers, body) pass through as-is.
export async function apiFetch(path, { token, headers, ...options } = {}) {
  const mergedHeaders = { [TRIAL_PASSCODE_HEADER]: TRIAL_PASSCODE, ...headers };
  if (token) mergedHeaders.Authorization = `Bearer ${token}`;

  const response = await fetch(`${BACKEND_URL}${path}`, { ...options, headers: mergedHeaders });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body);
  }
  if (response.status === 204) return null;
  return response.json();
}

// FastAPI's error body shape differs between a raised HTTPException
// (`detail` is a plain string) and a pydantic validation failure (`detail`
// is a list of {loc, msg, ...}) -- screens showing an error to the user
// need one plain string either way.
export function extractErrorMessage(error) {
  if (!(error instanceof ApiError)) return error?.message || "Something went wrong.";
  const detail = error.body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) return detail[0].msg || "That didn't look right.";
  return "Something went wrong. Please try again.";
}
