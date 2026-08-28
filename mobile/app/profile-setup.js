// Coaching-tone profile setup (2026-08-10) -- asks role first, then the
// role-appropriate follow-up, matching backend/auth.py's ProfileUpdateRequest
// exactly (a player always has skill_level and never years_coaching/
// certification_level, a coach is the reverse). Used two ways: pushed
// straight after signup (mandatory first run, no way back since there's no
// screen behind it in the stack -- see app/signup.js), and pushed again
// from Settings to edit an existing profile (back button works normally
// there since real history exists).
//
// NOT the same "role" as the pre-existing on-device player/coach toggle in
// (tabs)/settings.js (see that file's own comment, and backend/db.py's User
// model docstring) -- this one is the real account field that actually
// changes generated coaching language; that one only picks which feedback
// question shows after a session.
import { useMemo, useState } from "react";
import { useRouter } from "expo-router";
import { Pressable, StyleSheet, Text, TextInput, View } from "react-native";

import { extractErrorMessage } from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

const SKILL_LEVELS = [
  { value: "beginner", label: "Beginner" },
  { value: "intermediate", label: "Intermediate" },
  { value: "advanced", label: "Advanced" },
];

const CERTIFICATION_LEVELS = [
  { value: "club_coach", label: "Club coach" },
  { value: "national", label: "National" },
  { value: "itf_certified", label: "ITF-certified" },
  { value: "other", label: "Other" },
];

function OptionRow({ options, value, onChange, styles }) {
  return (
    <View style={styles.optionRow}>
      {options.map((opt) => (
        <Pressable
          key={opt.value}
          onPress={() => onChange(opt.value)}
          style={[styles.option, value === opt.value && styles.optionSelected]}
        >
          <Text style={[styles.optionText, value === opt.value && styles.optionTextSelected]}>{opt.label}</Text>
        </Pressable>
      ))}
    </View>
  );
}

export default function ProfileSetupScreen() {
  const router = useRouter();
  const { user, updateProfile } = useAuth();
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [step, setStep] = useState(1);
  const [role, setRole] = useState(user?.role ?? null);
  const [skillLevel, setSkillLevel] = useState(user?.skill_level ?? null);
  const [yearsCoaching, setYearsCoaching] = useState(user?.years_coaching != null ? String(user.years_coaching) : "");
  const [certificationLevel, setCertificationLevel] = useState(user?.certification_level ?? null);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  function chooseRole(next) {
    setRole(next);
    setStep(2);
  }

  const yearsNumber = parseInt(yearsCoaching, 10);
  const canSubmit =
    !submitting &&
    (role === "player"
      ? skillLevel != null
      : role === "coach"
      ? Number.isInteger(yearsNumber) && yearsNumber >= 0 && certificationLevel != null
      : false);

  async function handleSubmit() {
    setError(null);
    setSubmitting(true);
    try {
      await updateProfile(
        role === "player"
          ? { role: "player", skill_level: skillLevel }
          : { role: "coach", years_coaching: yearsNumber, certification_level: certificationLevel }
      );
      router.replace("/(tabs)");
    } catch (e) {
      setError(extractErrorMessage(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <View style={styles.screen}>
      {step === 1 ? (
        <View style={styles.container}>
          <Text style={styles.title}>Set up your profile</Text>
          <Text style={styles.subtitle}>
            This tunes the language in your coaching feedback -- technical for coaches, scaled to
            experience for players.
          </Text>
          <Text style={styles.label}>I am a...</Text>
          <Pressable
            onPress={() => chooseRole("player")}
            style={({ pressed }) => [styles.bigOption, pressed && styles.btnPressed]}
          >
            <Text style={styles.bigOptionText}>Player</Text>
          </Pressable>
          <Pressable
            onPress={() => chooseRole("coach")}
            style={({ pressed }) => [styles.bigOption, styles.bigOptionOutline, pressed && styles.btnPressed]}
          >
            <Text style={styles.bigOptionOutlineText}>Coach</Text>
          </Pressable>
        </View>
      ) : (
        <View style={styles.container}>
          <Pressable onPress={() => setStep(1)} style={styles.backRow}>
            <Text style={styles.backText}>{"< Back"}</Text>
          </Pressable>
          {/* De-genericizing pass (section 03, approved): a fixed 2-step
              flow, unlike Home's branching capture flow -- "Step 2 of 2" is
              honest here, not a guess. Same eyebrow recipe as elsewhere,
              standing in for a numbered-circle indicator. */}
          <Text style={styles.breadcrumb}>Step 2 of 2</Text>

          {role === "player" ? (
            <>
              <Text style={styles.title}>What's your skill level?</Text>
              <Text style={styles.subtitle}>Coaching feedback will be scaled to match.</Text>
              <OptionRow options={SKILL_LEVELS} value={skillLevel} onChange={setSkillLevel} styles={styles} />
            </>
          ) : (
            <>
              <Text style={styles.title}>Your coaching background</Text>
              <Text style={styles.subtitle}>Coach feedback always uses technical language, regardless of experience.</Text>

              <Text style={styles.label}>Years coaching</Text>
              <TextInput
                style={styles.input}
                value={yearsCoaching}
                onChangeText={setYearsCoaching}
                keyboardType="number-pad"
                placeholder="e.g. 5"
                placeholderTextColor={colors.textFaint}
              />

              <Text style={[styles.label, styles.sectionSpacing]}>Certification level</Text>
              <OptionRow options={CERTIFICATION_LEVELS} value={certificationLevel} onChange={setCertificationLevel} styles={styles} />
            </>
          )}

          {error ? <Text style={styles.error}>{error}</Text> : null}

          <Pressable
            onPress={handleSubmit}
            disabled={!canSubmit}
            style={({ pressed }) => [styles.btnPrimary, (pressed || !canSubmit) && styles.btnPressed]}
          >
            <Text style={styles.btnPrimaryText}>{submitting ? "Saving..." : "Save"}</Text>
          </Pressable>
        </View>
      )}
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  container: { flexGrow: 1, justifyContent: "center", padding: spacing.lg },
  title: {
    fontFamily: fonts.serif, fontSize: fontSizes.h1, color: colors.navyInk,
    textAlign: "center", marginBottom: spacing.xs,
  },
  subtitle: {
    fontSize: fontSizes.small, color: colors.textSoft, textAlign: "center",
    marginBottom: spacing.xl, lineHeight: 19,
  },
  label: { fontSize: fontSizes.small, fontWeight: "600", color: colors.textSoft, marginBottom: spacing.sm },
  sectionSpacing: { marginTop: spacing.lg },
  breadcrumb: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.md,
  },
  bigOption: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 14,
    marginBottom: spacing.sm,
  },
  bigOptionText: { color: colors.onPrimary, fontSize: fontSizes.body, fontWeight: "600", textAlign: "center" },
  bigOptionOutline: { backgroundColor: "transparent", borderWidth: 1.5, borderColor: colors.navy },
  bigOptionOutlineText: { color: colors.navy, fontSize: fontSizes.body, fontWeight: "600", textAlign: "center" },
  backRow: { marginBottom: spacing.lg },
  backText: { color: colors.textSoft, fontSize: fontSizes.small },
  optionRow: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm, marginBottom: spacing.md },
  option: {
    paddingVertical: 10,
    paddingHorizontal: spacing.md,
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.navy,
  },
  optionSelected: { backgroundColor: colors.navy },
  optionText: { color: colors.navy, fontSize: fontSizes.small, fontWeight: "600" },
  optionTextSelected: { color: colors.onPrimary },
  input: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    paddingVertical: 12,
    paddingHorizontal: spacing.md,
    fontSize: fontSizes.body,
    color: colors.text,
    marginBottom: spacing.md,
  },
  error: { color: colors.error, fontSize: fontSizes.small, marginTop: spacing.sm, marginBottom: spacing.sm },
  btnPrimary: {
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    paddingHorizontal: spacing.lg,
    marginTop: spacing.md,
  },
  btnPressed: { opacity: 0.6 },
  btnPrimaryText: { color: colors.onPrimary, fontSize: fontSizes.small, fontWeight: "600", textAlign: "center" },
});
