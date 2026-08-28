// Processing -- pushed from Home with { uri, fileName } params, uploads the
// clip, then either replaces itself with /results (so the back stack from
// Results skips straight to Home, not back to a stuck Processing screen) or
// shows an inline error with a Try Again that returns to Home -- same
// semantics as the old single-screen App.js's error state.
import { useEffect, useMemo, useRef, useState } from "react";
import { useLocalSearchParams, useRouter } from "expo-router";
import { ActivityIndicator, Pressable, StyleSheet, Text, View } from "react-native";

import { BACKEND_URL, REQUEST_TIMEOUT_MS, TRIAL_PASSCODE, TRIAL_PASSCODE_HEADER } from "../lib/config";
import { UPLOAD_ERROR_TITLES, UNKNOWN_ERROR_TITLE } from "../lib/errorMessages";
import { useAuth } from "../lib/AuthContext";
import { useSession } from "../lib/SessionContext";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

// Real per-attempt timing measured against the live backend (PROJECT_LOG.md,
// 2026-08-07): a single retry-gate attempt (rag/gated_pipeline.py) takes
// ~27-40s end to end; a 2nd attempt pushes total latency to ~110-130s. The
// retry count itself is never shown (stays internal, per the same KB-tone
// rewrite that dropped "percentile"/"ratio" jargon from coaching text) --
// only a plain-language cue that something deliberate, not stalled, is
// still happening once a first attempt has clearly finished and a retry is
// underway.
const INITIAL_STATUS_MESSAGE = "Analyzing your stroke -- usually takes 30-40 seconds";
const RETRY_STATUS_MESSAGE = "Double-checking the coaching feedback for accuracy -- this can take a bit longer";
// Comfortably past the ~27-40s single-attempt range but well before a 2nd
// attempt would finish (~110-130s total) -- crossing this means a retry is
// very likely already in progress.
const RETRY_STATUS_THRESHOLD_MS = 47000;

export default function ProcessingScreen() {
  const { uri, fileName, context } = useLocalSearchParams();
  const router = useRouter();
  const { token } = useAuth();
  const { setCurrentResult } = useSession();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  // { title, message } -- kept as two fields (not one string) so each of the
  // 5 hardened failure modes renders as a distinct screen state (Phase C,
  // 2026-08-01), not one undifferentiated error blob.
  const [error, setError] = useState(null);
  const [statusMessage, setStatusMessage] = useState(INITIAL_STATUS_MESSAGE);
  const startedRef = useRef(false);

  useEffect(() => {
    // Guard against React 19/StrictMode-style double-invoke re-running the
    // upload a second time for the same params.
    if (startedRef.current) return;
    startedRef.current = true;
    uploadVideo();
  }, []);

  async function uploadVideo() {
    setError(null);
    setStatusMessage(INITIAL_STATUS_MESSAGE);

    const formData = new FormData();
    formData.append("video", {
      uri,
      name: fileName || "stroke.mp4",
      type: "video/mp4",
    });
    // Session-type context selected on Home (2026-08-12) -- already a JSON
    // string from router params, passed through as-is. Purely descriptive
    // metadata (backend/app.py's session_context field); absent for any
    // session started before this feature existed, which is fine, it's optional.
    if (context) {
      formData.append("session_context", context);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    const retryStatusTimeoutId = setTimeout(() => setStatusMessage(RETRY_STATUS_MESSAGE), RETRY_STATUS_THRESHOLD_MS);

    try {
      // Deliberately NOT setting a Content-Type header here: fetch/FormData
      // auto-generates "multipart/form-data; boundary=..." with the actual
      // boundary token. Setting Content-Type manually to a bare
      // "multipart/form-data" (no boundary) breaks the server's multipart
      // parser -- a real, easy-to-miss gotcha, not something to add "for
      // clarity."
      const response = await fetch(`${BACKEND_URL}/api/v1/analyze-stroke`, {
        method: "POST",
        // Phase 2 (2026-08-07): analyze-stroke now requires auth (history
        // is tied to the account) -- Authorization alongside the existing
        // trial-passcode gate, which is unchanged and still separate.
        headers: { [TRIAL_PASSCODE_HEADER]: TRIAL_PASSCODE, Authorization: `Bearer ${token}` },
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      clearTimeout(retryStatusTimeoutId);

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        // body.detail carries the backend's structured {error_code, message} for each of the
        // 3 upload/content failure modes it can detect (bad_video_file, no_player_detected,
        // low_pose_detection -- see backend/app.py) -- the message is already written to be
        // shown to a user as-is. A missing/unrecognized shape means something crashed outside
        // those handled cases, so fall back to a generic (still non-crashing) error state.
        const detail = body.detail;
        if (detail && typeof detail === "object" && detail.message) {
          throw { title: UPLOAD_ERROR_TITLES[detail.error_code] || UNKNOWN_ERROR_TITLE, message: detail.message };
        }
        throw {
          title: UNKNOWN_ERROR_TITLE,
          message: (typeof detail === "string" && detail) ||
            `The server had a problem processing this (error ${response.status}). Please try again.`,
        };
      }

      const data = await response.json();
      // No separate "save to history" call needed -- the backend persists
      // this result automatically, tied to the authenticated user, inside
      // analyze-stroke itself (see backend/app.py + PROJECT_LOG.md).
      setCurrentResult(data);
      router.replace("/results");
    } catch (err) {
      clearTimeout(timeoutId);
      clearTimeout(retryStatusTimeoutId);
      // Distinguish failure types so the user sees something actionable instead of a raw
      // JS error string -- see PROJECT_LOG.md (2026-07-20 hardening pass, Phase C 2026-08-01)
      // for what each of these actually looks like when triggered for real (killed backend,
      // timeout, bad upload, etc.).
      if (err && err.title && err.message) {
        setError(err);
      } else if (err.name === "AbortError") {
        setError({
          title: "This is taking a while",
          message: "The request timed out and was cancelled. Check your connection and try again, " +
            "or record a shorter clip.",
        });
      } else if (err instanceof TypeError) {
        // React Native's fetch throws a TypeError ("Network request failed") for any
        // connection-level failure -- backend down, wrong IP, phone not on the same Wi-Fi, etc.
        // There's no way to tell those apart from here, so the message covers all of them and
        // says plainly that it's not a problem with the video, per the roadmap's instruction to
        // say so when the fix isn't on the user's side.
        setError({
          title: "Couldn't reach the server",
          message: "Make sure your phone has a working internet connection and the backend service " +
            "is running. This isn't a problem with your video.",
        });
      } else {
        setError({ title: UNKNOWN_ERROR_TITLE, message: err.message || "Something unexpected went wrong." });
      }
    }
  }

  if (error) {
    return (
      <View style={styles.centered}>
        <View style={styles.errorBox}>
          <Text style={styles.errorTitle}>{error.title}</Text>
          <Text style={styles.errorText}>{error.message}</Text>
          <Pressable
            onPress={() => router.replace("/(tabs)")}
            style={({ pressed }) => [styles.btnPrimary, pressed && styles.btnPressed]}
          >
            <Text style={styles.btnPrimaryText}>Try Again</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.centered}>
      <ActivityIndicator size="large" color={colors.navy} />
      <Text style={styles.note}>{statusMessage}</Text>
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  centered: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: spacing.lg,
  },
  note: { marginTop: spacing.sm, fontSize: fontSizes.small, color: colors.textSoft },
  errorBox: {
    width: "100%",
    backgroundColor: colors.errorTint,
    borderRadius: radii.md,
    padding: spacing.lg,
    alignItems: "center",
  },
  errorTitle: {
    fontFamily: fonts.serif,
    fontSize: fontSizes.body,
    color: colors.error,
    textAlign: "center",
    marginBottom: spacing.xs,
  },
  errorText: {
    fontSize: fontSizes.small,
    color: colors.text,
    textAlign: "center",
    lineHeight: 19,
    marginBottom: spacing.md,
  },
  btnPrimary: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    paddingHorizontal: spacing.lg,
    alignSelf: "stretch",
  },
  btnPrimaryText: {
    color: colors.onPrimary,
    fontSize: fontSizes.small,
    fontWeight: "600",
    textAlign: "center",
  },
  btnPressed: { opacity: 0.75 },
});
