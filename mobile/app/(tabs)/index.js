// Home/Capture -- session-type selection (2026-08-12) followed by record or
// pick a stroke clip, then hand off to the Processing screen. Video URI/
// filename are small serializable strings, so they're passed via router
// params (unlike the full analysis result later, which goes through
// SessionContext -- see mvp/DESIGN.md's Phase B proposal). The selected
// session context is passed the same way, as a JSON string -- it's metadata
// sent alongside the video (backend/app.py's session_context form field),
// not something the capture flow itself needs to branch on.
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "expo-router";
import { Alert, Pressable, StyleSheet, Text, View } from "react-native";
import * as ImagePicker from "expo-image-picker";

import { MAX_PICKED_VIDEO_BYTES, MAX_PICKED_VIDEO_DURATION_S } from "../../lib/config";
import { useAuth } from "../../lib/AuthContext";
import { listSessions } from "../../lib/historyRepository";
import { fonts, fontSizes, radii, spacing, useTheme } from "../../lib/theme";

// Matches ir/schema.py's STROKE_CLASSES exactly.
const STROKE_TYPES = [
  { value: "forehand", label: "Forehand" },
  { value: "backhand", label: "Backhand" },
  { value: "serve", label: "Serve" },
  { value: "volley", label: "Volley" },
  { value: "smash", label: "Smash" },
];

function contextSummary(context) {
  if (context.mode === "match") return "Friendly Match (singles)";
  if (context.practiceType === "roadmap") return "Practice · Beginner → Intermediate → Advanced";
  if (context.practiceType === "stroke" && context.strokeType) {
    const label = STROKE_TYPES.find((s) => s.value === context.strokeType)?.label;
    return `Practice · ${label}`;
  }
  return "";
}

// Mirrors backend/app.py's MAX_UPLOAD_BYTES/MAX_RAW_FRAMES rejection wording
// (see lib/config.js) so a client-side and server-side rejection for the same
// underlying reason read the same way to the user. Returns null if the asset
// is within bounds, else a { title, message } pair for Alert.alert -- checked
// against both duration and fileSize being present, since expo-image-picker
// documents both as optional/nullable (not guaranteed on every platform); a
// missing value skips that specific check rather than blocking the pick, and
// the backend's own limits remain the real backstop either way.
function checkVideoConstraints(asset) {
  if (typeof asset.duration === "number" && asset.duration / 1000 > MAX_PICKED_VIDEO_DURATION_S) {
    return {
      title: "Video too long",
      message: `This clip is over ${MAX_PICKED_VIDEO_DURATION_S} seconds. Pick a short clip of just one ` +
        `stroke, not a full rally or match.`,
    };
  }
  if (typeof asset.fileSize === "number" && asset.fileSize > MAX_PICKED_VIDEO_BYTES) {
    return {
      title: "Video too large",
      message: `This clip is over ${MAX_PICKED_VIDEO_BYTES / (1024 * 1024)}MB. Pick a short clip of just ` +
        `one stroke, not a long recording.`,
    };
  }
  return null;
}

// De-genericizing pass (2026-08-25, approved mockup "De-Genericizing Pass" --
// section 01): the stroke picker's "last practiced" marker uses real history
// data, not a fabricated placeholder date -- see listSessions() below.
function relativeTime(dateStr) {
  const days = Math.floor((Date.now() - new Date(dateStr).getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "1d ago";
  if (days < 7) return `${days}d ago`;
  const weeks = Math.floor(days / 7);
  if (weeks < 5) return `${weeks}w ago`;
  return `${Math.floor(days / 30)}mo ago`;
}

export default function HomeScreen() {
  const router = useRouter();
  // System-adaptive theme (2026-08-20, task 2): styles are built per-render from
  // the live OS scheme instead of a module-scope StyleSheet.create, since the
  // static form can't react to a mid-session OS appearance change.
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const { token } = useAuth();
  // "type" -> "practice_mode" (if practice) -> "stroke" (if practice+by-stroke) -> "capture"
  const [step, setStep] = useState("type");
  const [context, setContext] = useState({ mode: null, practiceType: null, strokeType: null });
  // Fetched once on mount, not gated on reaching the stroke-picker step --
  // cheap (same call History already makes) and avoids a loading flicker the
  // first time a player reaches that step.
  const [sessions, setSessions] = useState([]);
  useEffect(() => {
    if (token) listSessions(token).then(setSessions).catch(() => {});
  }, [token]);

  const lastPracticedByStroke = useMemo(() => {
    const map = {};
    for (const s of sessions) {
      if (!map[s.strokeClass] || new Date(s.createdAt) > new Date(map[s.strokeClass])) {
        map[s.strokeClass] = s.createdAt;
      }
    }
    return map;
  }, [sessions]);
  const mostRecentStroke = useMemo(() => {
    let best = null;
    let bestDate = null;
    for (const [strokeClass, createdAt] of Object.entries(lastPracticedByStroke)) {
      if (!bestDate || new Date(createdAt) > bestDate) {
        best = strokeClass;
        bestDate = new Date(createdAt);
      }
    }
    return best;
  }, [lastPracticedByStroke]);

  function chooseMatch() {
    setContext({ mode: "match", practiceType: null, strokeType: null });
    setStep("capture");
  }

  function choosePractice() {
    setContext({ mode: "practice", practiceType: null, strokeType: null });
    setStep("practice_mode");
  }

  function chooseRoadmap() {
    setContext((c) => ({ ...c, practiceType: "roadmap", strokeType: null }));
    setStep("capture");
  }

  function chooseByStroke() {
    setContext((c) => ({ ...c, practiceType: "stroke", strokeType: null }));
    setStep("stroke");
  }

  function chooseStrokeType(strokeType) {
    setContext((c) => ({ ...c, strokeType }));
    setStep("capture");
  }

  function changeSelection() {
    setStep("type");
    setContext({ mode: null, practiceType: null, strokeType: null });
  }

  async function recordVideo() {
    const { status: permStatus } = await ImagePicker.requestCameraPermissionsAsync();
    if (permStatus !== "granted") {
      Alert.alert("Camera permission required", "Please allow camera access to record a stroke.");
      return;
    }
    const pickerResult = await ImagePicker.launchCameraAsync({
      mediaTypes: ["videos"],
      videoMaxDuration: 10,
    });
    if (!pickerResult.canceled && pickerResult.assets?.length) {
      const asset = pickerResult.assets[0];
      // Duration is already capped by videoMaxDuration above -- only the size
      // check applies here, kept for defense-in-depth (a high-res/high-bitrate
      // 10s recording could still be large) rather than assuming it never will.
      const violation = checkVideoConstraints({ fileSize: asset.fileSize });
      if (violation) {
        Alert.alert(violation.title, violation.message);
        return;
      }
      router.push({
        pathname: "/processing",
        params: { uri: asset.uri, fileName: asset.fileName || "", context: JSON.stringify(context) },
      });
    }
  }

  async function pickVideo() {
    const { status: permStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (permStatus !== "granted") {
      Alert.alert("Library permission required", "Please allow photo library access to pick a video.");
      return;
    }
    const pickerResult = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ["videos"],
    });
    if (!pickerResult.canceled && pickerResult.assets?.length) {
      const asset = pickerResult.assets[0];
      // Unlike Record Video, the library picker has no built-in duration cap --
      // this is the only client-side gate for a long/large picked file, so it
      // rejects before an upload even starts rather than after the backend
      // does the identical check post-upload.
      const violation = checkVideoConstraints(asset);
      if (violation) {
        Alert.alert(violation.title, violation.message);
        return;
      }
      router.push({
        pathname: "/processing",
        params: { uri: asset.uri, fileName: asset.fileName || "", context: JSON.stringify(context) },
      });
    }
  }

  if (step === "type") {
    // De-genericizing pass (approved mockup, section 01): Practice is the
    // hero action (real weight, filled block), Friendly Match is a quiet
    // secondary link -- not a twin pill of equal size/weight, since the two
    // aren't used equally often.
    return (
      <View style={[styles.container, styles.heroContainer]}>
        <View>
          <Text style={styles.wordmark}>VisioTennis</Text>
          <Text style={styles.heroHeadline}>What are you working on?</Text>
        </View>
        <View>
          <Pressable
            onPress={choosePractice}
            style={({ pressed }) => [styles.heroCard, pressed && styles.btnPressed]}
          >
            <Text style={styles.heroCardEyebrow}>Most common</Text>
            <Text style={styles.heroCardLabel}>Practice</Text>
          </Pressable>
          <Pressable
            onPress={chooseMatch}
            style={({ pressed }) => [styles.secondaryLink, pressed && styles.btnPressed]}
          >
            <Text style={styles.secondaryLinkText}>or log a Friendly Match {"→"}</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  if (step === "practice_mode") {
    return (
      <View style={styles.container}>
        <Pressable onPress={changeSelection} style={styles.backRow}>
          <Text style={styles.backText}>{"< Back"}</Text>
        </Pressable>
        <Text style={styles.breadcrumb}>Practice</Text>
        <Text style={styles.title}>Practice</Text>
        <Text style={styles.subtitle}>Work on one stroke, or follow the roadmap</Text>
        <View style={styles.buttonGroup}>
          <Pressable
            onPress={chooseByStroke}
            style={({ pressed }) => [styles.bigOption, pressed && styles.btnPressed]}
          >
            <Text style={styles.bigOptionText}>By Stroke</Text>
          </Pressable>
          <Pressable
            onPress={chooseRoadmap}
            style={({ pressed }) => [styles.bigOption, styles.bigOptionOutline, pressed && styles.btnPressed]}
          >
            <Text style={styles.bigOptionOutlineText}>Beginner {"→"} Intermediate {"→"} Advanced</Text>
          </Pressable>
        </View>
      </View>
    );
  }

  if (step === "stroke") {
    // De-genericizing pass (approved mockup, section 01): the 5 identical
    // cards with a blank placeholder swatch become typography-led rows --
    // no icon standing in for real stroke art. The most recently practiced
    // stroke (real listSessions data, not fabricated) gets a marker instead
    // of every row being interchangeable.
    return (
      <View style={styles.container}>
        <Pressable onPress={() => setStep("practice_mode")} style={styles.backRow}>
          <Text style={styles.backText}>{"< Back"}</Text>
        </Pressable>
        <Text style={styles.breadcrumb}>Practice · By Stroke</Text>
        <Text style={[styles.title, styles.titleLeft]}>Which stroke?</Text>
        <Text style={[styles.subtitle, styles.subtitleLeft]}>Pick the one you want feedback on</Text>
        <View style={styles.strokeList}>
          {STROKE_TYPES.map((s) => {
            const last = lastPracticedByStroke[s.value];
            const isRecent = s.value === mostRecentStroke;
            return (
              <Pressable
                key={s.value}
                onPress={() => chooseStrokeType(s.value)}
                style={({ pressed }) => [
                  styles.strokeRow,
                  isRecent && styles.strokeRowRecent,
                  pressed && styles.btnPressed,
                ]}
              >
                <Text style={styles.strokeRowText}>{s.label}</Text>
                {last ? (
                  <Text style={styles.strokeRowMeta}>last: {relativeTime(last)}</Text>
                ) : (
                  <Text style={styles.strokeRowArrow}>{"→"}</Text>
                )}
              </Pressable>
            );
          })}
        </View>
      </View>
    );
  }

  // step === "capture"
  return (
    <View style={styles.container}>
      <Text style={styles.title}>VisioTennis</Text>
      <Pressable onPress={changeSelection} style={styles.selectionRow}>
        <Text style={styles.selectionText}>{contextSummary(context)}</Text>
        <Text style={styles.selectionChange}>Change</Text>
      </Pressable>
      <Text style={styles.subtitle}>Record or pick a short clip of ONE stroke (a few seconds)</Text>

      <View style={styles.buttonGroup}>
        <Pressable
          onPress={recordVideo}
          style={({ pressed }) => [styles.btnPrimary, pressed && styles.btnPressed]}
        >
          <Text style={styles.btnPrimaryText}>Record Video</Text>
        </Pressable>
        <Pressable
          onPress={pickVideo}
          style={({ pressed }) => [styles.btnSecondary, pressed && styles.btnPressed]}
        >
          <Text style={styles.btnSecondaryText}>Pick from Library</Text>
        </Pressable>
      </View>
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { flex: 1, backgroundColor: colors.bg, paddingTop: spacing.xxl, paddingHorizontal: spacing.lg },
  title: {
    fontFamily: fonts.serif,
    fontSize: fontSizes.display,
    color: colors.navyInk,
    textAlign: "center",
  },
  subtitle: {
    fontSize: fontSizes.small,
    color: colors.textSoft,
    textAlign: "center",
    lineHeight: 19,
    marginTop: spacing.xs,
    marginBottom: spacing.lg,
  },
  backRow: { marginBottom: spacing.lg },
  backText: { color: colors.textSoft, fontSize: fontSizes.small },
  // Reused typographic breadcrumb (De-genericizing pass, section 03) --
  // same recipe as cardEyebrow elsewhere in the app (Results/History/
  // Settings), standing in for a numbered-step indicator instead of
  // inventing a new widget. No fixed "step X of Y": this flow branches
  // (match / roadmap / by-stroke all have different total step counts), so
  // a literal fraction would sometimes be wrong -- a contextual breadcrumb
  // stays honest at every branch.
  breadcrumb: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.xs,
  },
  buttonGroup: { gap: spacing.sm },
  bigOption: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 14,
  },
  bigOptionText: { color: colors.onPrimary, fontSize: fontSizes.body, fontWeight: "600", textAlign: "center" },
  bigOptionOutline: { backgroundColor: "transparent", borderWidth: 1.5, borderColor: colors.navy },
  bigOptionOutlineText: { color: colors.navy, fontSize: fontSizes.body, fontWeight: "600", textAlign: "center" },
  // Hero/secondary restructure of the "type" step (De-genericizing pass,
  // section 01) -- Practice and Friendly Match are not used equally often,
  // so they no longer look like an equal choice.
  heroContainer: { justifyContent: "space-between", paddingBottom: spacing.xl },
  wordmark: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
  },
  heroHeadline: {
    fontFamily: fonts.serif,
    fontSize: 30,
    color: colors.navyInk,
    lineHeight: 36,
    marginTop: spacing.md,
  },
  heroCard: {
    backgroundColor: colors.navy,
    borderRadius: radii.lg,
    padding: spacing.lg,
  },
  heroCardEyebrow: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.border, // muted-on-navy, matches the app's existing dark-surface caption pattern
  },
  heroCardLabel: { fontFamily: fonts.serif, fontSize: 26, color: colors.onPrimary, marginTop: spacing.xs },
  secondaryLink: { alignSelf: "flex-end", marginTop: spacing.md },
  secondaryLinkText: { fontSize: fontSizes.small, color: colors.textSoft, textDecorationLine: "underline" },
  // Typography-led stroke list (De-genericizing pass, section 01) --
  // replaces the 5-card grid + blank placeholder swatch.
  titleLeft: { textAlign: "left" },
  subtitleLeft: { textAlign: "left" },
  strokeList: { marginTop: spacing.xs },
  strokeRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "baseline",
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  strokeRowRecent: {
    borderBottomColor: colors.navy,
    borderLeftWidth: 3,
    borderLeftColor: colors.navy,
    paddingLeft: spacing.sm,
  },
  strokeRowText: { fontFamily: fonts.serif, fontSize: 19, color: colors.navyInk },
  strokeRowMeta: {
    fontSize: fontSizes.caption,
    textTransform: "uppercase",
    letterSpacing: 0.4,
    color: colors.textFaint,
  },
  strokeRowArrow: { fontSize: fontSizes.small, color: colors.textSoft },
  selectionRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    marginTop: spacing.md,
  },
  selectionText: { fontSize: fontSizes.small, color: colors.text, flexShrink: 1, marginRight: spacing.sm },
  selectionChange: { fontSize: fontSizes.small, fontWeight: "600", color: colors.navy },
  btnPrimary: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    paddingHorizontal: spacing.md,
  },
  btnPrimaryText: {
    color: colors.onPrimary,
    fontSize: fontSizes.small,
    fontWeight: "600",
    textAlign: "center",
  },
  btnSecondary: {
    backgroundColor: "transparent",
    borderWidth: 1.5,
    borderColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    paddingHorizontal: spacing.md,
  },
  btnSecondaryText: {
    color: colors.navy,
    fontSize: fontSizes.small,
    fontWeight: "600",
    textAlign: "center",
  },
  btnPressed: { opacity: 0.75 },
});
