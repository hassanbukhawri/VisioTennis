// Settings -- Phase B placeholder plus the Phase 1 role selector
// (2026-08-01). BACKEND_URL and the trial passcode are still hardcoded in
// lib/config.js; the only real setting so far is the feedback-form role.
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "expo-router";
import { ScrollView, StyleSheet, Text, TextInput, TouchableOpacity, View } from "react-native";

import { DEFAULT_COACH_NAME } from "../../components/StrokeChat";
import { useAuth } from "../../lib/AuthContext";
import { getRole, setRole } from "../../lib/historyRepository";
import { fonts, fontSizes, radii, spacing, useTheme } from "../../lib/theme";

const SKILL_LEVEL_LABELS = { beginner: "Beginner", intermediate: "Intermediate", advanced: "Advanced" };
const CERTIFICATION_LABELS = {
  club_coach: "Club coach", national: "National", itf_certified: "ITF-certified", other: "Other",
};

export default function SettingsScreen() {
  const router = useRouter();
  const { user, logout, updateCoachName } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [role, setRoleState] = useState(null);
  const [coachNameDraft, setCoachNameDraft] = useState(user?.coach_name ?? "");
  const [coachNameError, setCoachNameError] = useState(null);

  useEffect(() => {
    getRole().then(setRoleState);
  }, []);

  // Keeps the draft in sync if `user` changes from elsewhere (e.g. a fresh
  // login) -- but only while the field isn't mid-edit-and-saved locally,
  // same reasoning as every other "server response is the source of truth"
  // field in this app (AuthContext.js).
  useEffect(() => {
    setCoachNameDraft(user?.coach_name ?? "");
  }, [user?.coach_name]);

  async function choose(next) {
    setRoleState(next);
    await setRole(next);
  }

  async function saveCoachName() {
    const trimmed = coachNameDraft.trim();
    if (trimmed === (user?.coach_name ?? "")) return; // unchanged -- no-op, no request
    try {
      setCoachNameError(null);
      await updateCoachName(trimmed || null);
    } catch (e) {
      setCoachNameError("Couldn't save -- check your connection and try again.");
    }
  }

  return (
    <ScrollView style={styles.screen} contentContainerStyle={styles.container}>
      <Text style={styles.cardEyebrow}>Account</Text>
      <View style={styles.accountRow}>
        <Text style={styles.accountEmail}>{user?.email}</Text>
        <TouchableOpacity onPress={logout} style={styles.logoutBtn}>
          <Text style={styles.logoutBtnText}>Log Out</Text>
        </TouchableOpacity>
      </View>

      <Text style={[styles.cardEyebrow, styles.sectionSpacing]}>Coaching profile</Text>
      <View style={styles.accountRow}>
        <View style={styles.profileSummary}>
          {user?.role === "player" ? (
            <Text style={styles.accountEmail}>
              Player · {SKILL_LEVEL_LABELS[user.skill_level] || user.skill_level}
            </Text>
          ) : user?.role === "coach" ? (
            <Text style={styles.accountEmail}>
              Coach · {user.years_coaching} {user.years_coaching === 1 ? "year" : "years"} ·{" "}
              {CERTIFICATION_LABELS[user.certification_level] || user.certification_level}
            </Text>
          ) : (
            <Text style={styles.accountEmail}>Not set up yet</Text>
          )}
        </View>
        <TouchableOpacity onPress={() => router.push("/profile-setup")} style={styles.logoutBtn}>
          <Text style={styles.logoutBtnText}>{user?.role ? "Edit" : "Set up"}</Text>
        </TouchableOpacity>
      </View>
      <Text style={styles.note}>
        Tunes the language in your generated coaching feedback -- technical for coaches, scaled to
        experience for players. Tied to your account, not this device.
      </Text>

      <Text style={styles.coachNameLabel}>Coach name</Text>
      <TextInput
        style={styles.coachNameInput}
        value={coachNameDraft}
        onChangeText={setCoachNameDraft}
        onBlur={saveCoachName}
        placeholder={DEFAULT_COACH_NAME}
        placeholderTextColor={colors.textFaint}
        maxLength={40}
        returnKeyType="done"
        onSubmitEditing={saveCoachName}
      />
      {coachNameError ? <Text style={styles.coachNameError}>{coachNameError}</Text> : null}
      <Text style={styles.note}>
        Shown in StrokeChat instead of "{DEFAULT_COACH_NAME}." Leave blank to keep the default.
      </Text>

      <Text style={[styles.cardEyebrow, styles.sectionSpacing]}>Role</Text>
      <Text style={styles.label}>I am reviewing as:</Text>
      <View style={styles.row}>
        <TouchableOpacity
          style={[styles.option, role === "player" && styles.optionSelected]}
          onPress={() => choose("player")}
        >
          <Text style={[styles.optionText, role === "player" && styles.optionTextSelected]}>Player</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.option, role === "coach" && styles.optionSelected]}
          onPress={() => choose("coach")}
        >
          <Text style={[styles.optionText, role === "coach" && styles.optionTextSelected]}>Coach</Text>
        </TouchableOpacity>
      </View>
      <Text style={styles.note}>
        This only changes which feedback questions you see after a session -- it's stored on
        this device only, not linked to any account or identity.
      </Text>

      <Text style={[styles.cardEyebrow, styles.sectionSpacing]}>Privacy &amp; data</Text>
      <View style={styles.placeholderCard}>
        <View style={styles.placeholderTag}>
          <Text style={styles.placeholderTagText}>PLACEHOLDER -- NOT FINAL COPY</Text>
        </View>
        <Text style={styles.placeholderText}>
          During this trial, clips you record or upload are sent to the analysis server to produce
          your stroke report, and analysis results are stored on this device. [Insert approved
          language here on: how long clips are retained on the server, whether they're used for
          anything beyond producing your report, and how to request deletion.]
        </Text>
        <Text style={styles.placeholderText}>
          [Insert approved language here on: who can access trial data, and contact details for
          data questions.]
        </Text>
        <Text style={styles.placeholderFootnote}>
          This section is a structural placeholder only. It will be replaced with reviewed,
          approved consent and data-handling copy before any wider release.
        </Text>
      </View>
    </ScrollView>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  container: { padding: spacing.lg },
  cardEyebrow: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.sm,
  },
  sectionSpacing: { marginTop: spacing.xl },
  accountRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  accountEmail: { fontSize: fontSizes.small, color: colors.text, flexShrink: 1, marginRight: spacing.sm },
  profileSummary: { flexShrink: 1, marginRight: spacing.sm },
  logoutBtn: {
    paddingVertical: 8,
    paddingHorizontal: spacing.md,
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.error,
  },
  logoutBtnText: { color: colors.error, fontSize: fontSizes.small, fontWeight: "600" },
  label: { fontFamily: fonts.serif, fontSize: fontSizes.h1, color: colors.navyInk, marginBottom: spacing.md },
  row: { flexDirection: "row", gap: spacing.sm },
  option: {
    paddingVertical: 10,
    paddingHorizontal: spacing.lg,
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.navy,
  },
  optionSelected: { backgroundColor: colors.navy },
  optionText: { color: colors.navy, fontSize: fontSizes.small, fontWeight: "600" },
  optionTextSelected: { color: colors.onPrimary },
  note: { marginTop: spacing.md, color: colors.textSoft, fontSize: fontSizes.small, lineHeight: 19 },
  coachNameLabel: {
    marginTop: spacing.md,
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.sm,
  },
  coachNameInput: {
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    backgroundColor: colors.surface,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    fontSize: fontSizes.small,
    color: colors.text,
  },
  coachNameError: { marginTop: spacing.xs, color: colors.error, fontSize: fontSizes.caption },
  placeholderCard: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    borderWidth: 1.5,
    borderStyle: "dashed",
    borderColor: colors.borderStrong,
    padding: spacing.md,
  },
  placeholderTag: {
    alignSelf: "flex-start",
    backgroundColor: colors.caveatTint,
    borderRadius: radii.pill,
    paddingVertical: 2,
    paddingHorizontal: spacing.sm,
    marginBottom: spacing.sm,
  },
  placeholderTagText: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.4,
    color: colors.caveat,
  },
  placeholderText: { fontSize: fontSizes.small, color: colors.text, lineHeight: 19, marginBottom: spacing.sm },
  placeholderFootnote: { fontSize: fontSizes.caption, color: colors.textFaint, lineHeight: 16, marginTop: spacing.xs },
});
