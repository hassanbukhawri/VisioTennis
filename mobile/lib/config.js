// Shared constants used across screens -- moved here from the old single-screen
// App.js so Home/Processing/Results can all reference the same values.

export const BACKEND_URL = "https://visiotennis-trial.tail458008.ts.net"; // PERMANENT DEFAULT (2026-08-28) -- stable Tailscale Funnel hostname; public, survives LAN IP changes/DHCP re-leases, works off home Wi-Fi. Re-diagnosed and shipped for real after the 2026-08-21 "RN fetch fails, curl succeeds" scare: that failure never reproduced under a controlled real-device A/B this session (GET /auth/me, POST /auth/signup, PATCH /auth/profile all succeeded over Funnel on a real Pixel 6a, confirmed again from cellular/off-Wi-Fi) -- most likely it was the same port-mismatch/backend-down class of bug as the recurring 502s, never isolated with a controlled test at the time. See PROJECT_LOG.md "Tailscale Funnel -- real diagnosis, shipped as permanent default" (2026-08-28). Use start_visiotennis.ps1 each session to bring backend/Ollama/Funnel up together and confirm this URL is live.
// export const BACKEND_URL = "http://192.168.1.3:8000"; // LAN IP fallback -- same-Wi-Fi-only, needs manual IP refresh after DHCP re-leases (see 2026-08-26 note in git history)
// 7 minutes (2026-08-07, was 5). Real measured worst case (PROJECT_LOG.md):
// the retry-gate's 3-attempt ceiling is ~204s of LLM generation alone, plus
// ~3-10s for pose/classification/biomechanics/prompt-retrieval, plus Funnel
// overhead -- confirmed via a real controlled test (a debug endpoint held
// open for exactly 260s over the live Tailscale Funnel URL, chosen because
// that's the client's abort threshold under the OLD 5-minute budget) that
// the Funnel itself imposes no idle/total timeout shorter than that; the
// measured overhead over a real request is ~5s (TLS handshake), not tunnel-
// specific slowness. So the realistic worst case is ~215-225s, leaving only
// ~75-85s of margin under the old 5-minute (300s) budget -- thin enough that
// a slightly slower-than-typical run (bigger clip, extra Funnel connect
// latency) could abort a request that was still legitimately processing.
// 7 minutes (420s) gives ~195s of margin over the measured worst case
// instead. Raising this does NOT fix (and isn't intended to fix) a
// genuinely dead/dropped connection -- see PROJECT_LOG.md's 2026-08-07
// Funnel-timeout investigation for the one confirmed case of a request
// that succeeded server-side while the client still reported a timeout;
// that was a network-level failure, not a latency-budget shortfall.
export const REQUEST_TIMEOUT_MS = 7 * 60 * 1000;

// Trial-scope gate (2026-07-28): must match TRIAL_PASSCODE in backend/app.py exactly.
// Not real auth -- a single shared value for this bounded trial, sent on every request
// so the backend's tunnel URL isn't wide open to anyone who stumbles on the link.
export const TRIAL_PASSCODE_HEADER = "X-Trial-Passcode";
export const TRIAL_PASSCODE = "vt-trial-8f2c91";

// Video constraints (2026-08-12) -- must match backend/app.py's MAX_UPLOAD_BYTES
// and MAX_RAW_FRAMES exactly (900 frames @30fps ~= 30s). Checked client-side on
// "Pick from Library" (mobile/app/(tabs)/index.js), which has no picker-level cap
// unlike "Record Video"'s videoMaxDuration, so a bad pick is rejected before an
// upload even starts rather than after the backend does the same check. If either
// backend constant changes, check these too -- not auto-derived, no shared source
// of truth between the JS and Python codebases.
export const MAX_PICKED_VIDEO_DURATION_S = 30;
export const MAX_PICKED_VIDEO_BYTES = 100 * 1024 * 1024;
