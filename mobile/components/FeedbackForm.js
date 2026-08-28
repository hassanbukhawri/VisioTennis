// Structured feedback (Phase 1, 2026-08-01) -- shown once, on the live Results
// screen only (not from History), right after a session completes. Two tracks
// kept separate per the roadmap: reliability (was the shot right?) and
// usefulness (was the coaching useful?), plus a one-tap actual-class follow-up
// when the prediction was wrong/partly wrong -- real ground truth on real phone
// footage, which the project has none of otherwise.
import { useEffect, useMemo, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TextInput, TouchableOpacity, View } from "react-native";

import { BACKEND_URL, TRIAL_PASSCODE, TRIAL_PASSCODE_HEADER } from "../lib/config";
import { getRole } from "../lib/historyRepository";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

const STROKE_CLASSES = ["forehand", "backhand", "serve", "volley", "smash"];

function ChoiceRow({ options, selected, onSelect, styles }) {
  return (
    <View style={styles.row}>
      {options.map(({ value, label }) => (
        <TouchableOpacity
          key={value}
          style={[styles.chip, selected === value && styles.chipSelected]}
          onPress={() => onSelect(value)}
        >
          <Text style={[styles.chipText, selected === value && styles.chipTextSelected]}>{label}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

export default function FeedbackForm({ result }) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  const [role, setRole] = useState(null);
  const [reliability, setReliability] = useState(null);
  const [actualClass, setActualClass] = useState(null);
  const [usefulnessRating, setUsefulnessRating] = useState(null);
  const [usefulnessText, setUsefulnessText] = useState("");
  const [coachAnswer, setCoachAnswer] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    getRole().then(setRole);
  }, []);

  const predictedClass = result.stroke_event.stroke.predicted_class;
  const needsActualClass = reliability === "no" || reliability === "partly";
  const canSubmit =
    reliability !== null &&
    usefulnessRating !== null &&
    (!needsActualClass || actualClass !== null);

  async function handleSubmit() {
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(`${BACKEND_URL}/api/v1/feedback`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          [TRIAL_PASSCODE_HEADER]: TRIAL_PASSCODE,
        },
        body: JSON.stringify({
          event_id: result.stroke_event.event_id,
          reliability,
          actual_class: needsActualClass ? actualClass : null,
          usefulness_rating: usefulnessRating,
          usefulness_text: usefulnessText || null,
          coach_would_say_to_student: role === "coach" ? coachAnswer : null,
          role,
        }),
      });
      if (!response.ok) throw new Error("submit failed");
      setSubmitted(true);
    } catch {
      setError("Couldn't send feedback. Check your connection and try again.");
    } finally {
      setSubmitting(false);
    }
  }

  if (submitted) {
    return (
      <View style={styles.container}>
        <View style={styles.thanksPill}>
          <Text style={styles.thanks}>Thanks -- feedback recorded.</Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.sectionTitle}>How did this go?</Text>

      <Text style={styles.question}>We said {predictedClass}. Was that right?</Text>
      <ChoiceRow
        options={[
          { value: "yes", label: "Yes" },
          { value: "no", label: "No" },
          { value: "partly", label: "Partly" },
        ]}
        selected={reliability}
        onSelect={setReliability}
        styles={styles}
      />

      {needsActualClass ? (
        <>
          <Text style={styles.question}>What was it actually?</Text>
          <ChoiceRow
            options={STROKE_CLASSES.map((c) => ({ value: c, label: c }))}
            selected={actualClass}
            onSelect={setActualClass}
            styles={styles}
          />
        </>
      ) : null}

      <Text style={styles.question}>Was the coaching feedback useful?</Text>
      <ChoiceRow
        options={[1, 2, 3, 4, 5].map((n) => ({ value: n, label: String(n) }))}
        selected={usefulnessRating}
        onSelect={setUsefulnessRating}
        styles={styles}
      />
      <TextInput
        style={styles.textInput}
        placeholder="Anything else? (optional)"
        multiline
        value={usefulnessText}
        onChangeText={setUsefulnessText}
      />

      {role === "coach" ? (
        <>
          <Text style={styles.question}>Would you say this to a student?</Text>
          <ChoiceRow
            options={[
              { value: true, label: "Yes" },
              { value: false, label: "No" },
            ]}
            selected={coachAnswer}
            onSelect={setCoachAnswer}
            styles={styles}
          />
        </>
      ) : null}

      {error ? <Text style={styles.errorText}>{error}</Text> : null}

      <TouchableOpacity
        style={[styles.submitButton, !canSubmit && styles.submitButtonDisabled]}
        disabled={!canSubmit || submitting}
        onPress={handleSubmit}
      >
        {submitting ? <ActivityIndicator color={colors.onPrimary} /> : <Text style={styles.submitText}>Submit</Text>}
      </TouchableOpacity>
    </View>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  container: { marginTop: spacing.lg, paddingTop: spacing.md, borderTopWidth: 1, borderTopColor: colors.border },
  sectionTitle: { fontFamily: fonts.serif, fontSize: fontSizes.h2, color: colors.navyInk, marginBottom: spacing.sm },
  question: { fontSize: fontSizes.small, color: colors.text, marginTop: spacing.md, marginBottom: spacing.xs },
  row: { flexDirection: "row", flexWrap: "wrap", gap: spacing.sm },
  chip: {
    paddingVertical: 6,
    paddingHorizontal: spacing.md,
    borderRadius: radii.pill,
    borderWidth: 1.5,
    borderColor: colors.navy,
  },
  chipSelected: { backgroundColor: colors.navy },
  chipText: { color: colors.navy, fontSize: fontSizes.small },
  chipTextSelected: { color: colors.onPrimary },
  textInput: {
    marginTop: spacing.sm,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.sm,
    backgroundColor: colors.surface,
    padding: spacing.sm,
    minHeight: 44,
    fontSize: fontSizes.small,
    color: colors.text,
    textAlignVertical: "top",
  },
  errorText: { color: colors.error, marginTop: spacing.sm, fontSize: fontSizes.small },
  submitButton: {
    marginTop: spacing.lg,
    backgroundColor: colors.navy,
    borderRadius: radii.pill,
    paddingVertical: 13,
    alignItems: "center",
  },
  submitButtonDisabled: { backgroundColor: colors.borderStrong },
  submitText: { color: colors.onPrimary, fontSize: fontSizes.small, fontWeight: "600" },
  thanksPill: {
    alignSelf: "flex-start",
    backgroundColor: colors.tealTint,
    borderRadius: radii.pill,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  thanks: { color: colors.tealDeep, fontSize: fontSizes.small, fontWeight: "600" },
});
