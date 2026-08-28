// Shared rendering for one analyze-stroke result -- extracted from the old
// single-screen App.js so both the live Results screen and a History detail
// screen (viewing a past, on-device-stored session) can render the exact same
// way without duplicating this JSX.
import { useEffect, useMemo } from "react";
import { ScrollView, StyleSheet, Text, View } from "react-native";

import { COACHING_UNAVAILABLE_MESSAGE } from "../lib/errorMessages";
import { fonts, fontSizes, radii, spacing, useTheme } from "../lib/theme";

// Confidence signalling (Phase 1, 2026-08-01) -- both provisional placeholders, not
// rigorously calibrated (no confidence-vs-accuracy curve exists for this model, only
// aggregate test macro-F1=0.834). Meant to be revisited once real trial "was this
// correct?" feedback exists to actually calibrate against -- see PROJECT_LOG.md.
export const CONFIDENCE_CAVEAT_THRESHOLD = 0.60; // below this: flag regardless of runner-up gap
const CONFIDENCE_GAP_THRESHOLD = 0.20; // top1-top2 gap below this: flag and name the
// runner-up class, independent of absolute confidence -- specifically to catch the
// documented volley/forehand confusability (training/stroke_classification/README.md),
// which can produce a confidently wrong answer that a pure confidence threshold misses.
const POSE_DETECTION_CAVEAT_THRESHOLD = 0.90; // backend already hard-rejects below 0.80
// (Phase C's low_pose_detection error) -- this is a softer caveat for the 80-90% band
// that passes the hard gate but is still below the 98-100% clean-footage baseline.

function confidenceCaveat(stroke) {
  const sorted = Object.entries(stroke.class_probabilities).sort((a, b) => b[1] - a[1]);
  const [, top1Prob] = sorted[0];
  const [runnerUpClass, top2Prob] = sorted[1];
  const gap = top1Prob - top2Prob;
  const lowConfidence = stroke.confidence < CONFIDENCE_CAVEAT_THRESHOLD;
  const closeCall = gap < CONFIDENCE_GAP_THRESHOLD;
  if (!lowConfidence && !closeCall) return null;

  const pct = (stroke.confidence * 100).toFixed(0);
  const callOut = closeCall ? ` -- it could also be a ${runnerUpClass}.` : ".";
  return `Lower confidence on this one (${pct}%)${callOut} For a clearer read next time: ` +
    "film closer to the player with their full swing visible in frame.";
}

function poseCaveat(diagnostics) {
  if (diagnostics.pose_detection_rate >= POSE_DETECTION_CAVEAT_THRESHOLD) return null;
  const pct = (diagnostics.pose_detection_rate * 100).toFixed(0);
  return `Pose tracking was inconsistent on this clip (${pct}% of frames). For a cleaner ` +
    "result next time: get closer to the player, use brighter or more even lighting, and " +
    "make sure only one person is in frame.";
}

function citationLabel(citation) {
  const field = citation.ir_field_path.replace("biomechanics_percentile.", "").replace("biomechanics.", "");
  const suffix = citation.ir_field_path.startsWith("biomechanics_percentile.") ? "th percentile" : "";
  return `${field}: ${citation.cited_value}${suffix}`;
}

export default function ResultView({ result, children }) {
  const { colors } = useTheme();
  const styles = useMemo(() => makeStyles(colors), [colors]);
  useEffect(() => {
    if (!result.coaching.success) {
      console.log("[ResultView] coaching generation failed:", result.coaching.failure_reason);
    }
  }, [result]);

  const confCaveat = confidenceCaveat(result.stroke_event.stroke);
  const poseCaveatText = poseCaveat(result.diagnostics);
  // Teal signals a genuinely clean call (design system: teal is reserved for a narrow,
  // meaningful positive signal). A confidence pill next to a caveat saying "treat this
  // number with suspicion" would contradict itself if it stayed teal -- so the pill
  // switches to the same amber used for the caveat box itself when one is showing.
  const pillIsCaveat = confCaveat !== null;
  // Same signal as poseCaveatText below, surfaced prominently near the top too
  // (2026-08-12) rather than only in the Diagnostics section at the bottom --
  // not a new signal, just more visible framing of pose_detection_rate, which
  // already existed. Same threshold/color logic as the confidence pill above,
  // for visual consistency.
  const qualityIsCaveat = poseCaveatText !== null;
  const qualityPct = (result.diagnostics.pose_detection_rate * 100).toFixed(0);

  // Pull-quote hero item + demoted secondary list (De-genericizing pass,
  // section 02, approved mockup) -- the first coaching item is not more
  // "correct" than the others, but it does arrive first for a reason (the
  // backend already orders by significance), so the layout is allowed to
  // say what the data already implies. Every item keeps its full citation --
  // only the type scale/spacing shrinks for items after the first, nothing
  // is dropped, unlike the mockup's illustrative simplification.

  return (
    <ScrollView style={styles.resultBox} contentContainerStyle={styles.resultContent}>
      <Text style={styles.cardEyebrow}>Predicted stroke</Text>
      <Text style={styles.heroStroke}>{result.stroke_event.stroke.predicted_class}</Text>
      <View style={[styles.pill, styles.heroConfidencePill, pillIsCaveat && styles.pillCaveat]}>
        <Text style={[styles.pillText, pillIsCaveat && styles.pillTextCaveat]}>
          {(result.stroke_event.stroke.confidence * 100).toFixed(0)}% confidence
        </Text>
      </View>
      {confCaveat ? (
        <View style={styles.caveatBox}>
          <Text style={styles.caveatText}>{confCaveat}</Text>
        </View>
      ) : null}

      <View style={[styles.pill, styles.qualityPill, qualityIsCaveat && styles.pillCaveat]}>
        <Text style={[styles.pillText, qualityIsCaveat && styles.pillTextCaveat]}>
          Tracking quality: {qualityPct}%
        </Text>
      </View>

      <Text style={[styles.cardEyebrow, styles.sectionSpacing]}>Coaching feedback</Text>
      {result.coaching.success ? (
        <View style={styles.card}>
          <View style={styles.pullQuote}>
            <Text style={styles.pullQuoteText}>{result.coaching.items[0].text}</Text>
            {result.coaching.items[0].citation ? (
              <Text style={styles.pullQuoteCitation}>{citationLabel(result.coaching.items[0].citation)}</Text>
            ) : null}
          </View>
          {result.coaching.items.length > 1 ? (
            <View style={styles.secondaryItems}>
              {result.coaching.items.slice(1).map((item, i) => (
                <View key={i + 1} style={i > 0 ? styles.secondaryItemDivider : null}>
                  <Text style={styles.secondaryItemText}>{item.text}</Text>
                  {item.citation ? (
                    <Text style={styles.secondaryItemCitation}>{citationLabel(item.citation)}</Text>
                  ) : null}
                </View>
              ))}
            </View>
          ) : null}
        </View>
      ) : (
        // Phase C (2026-08-01): the raw `failure_reason` (e.g. a Python exception repr from an
        // unreachable Ollama backend) is developer-facing, not something to show a tester --
        // logged for debugging instead of rendered. Deliberately does NOT imply anything is
        // wrong with the tester's video: stroke/biomechanics above are already valid regardless
        // of whether coaching generation succeeded.
        <View style={[styles.card, styles.errorCard]}>
          <Text style={styles.errorCardText}>{COACHING_UNAVAILABLE_MESSAGE}</Text>
        </View>
      )}

      <Text style={[styles.cardEyebrow, styles.sectionSpacing]}>Diagnostics</Text>
      <Text style={styles.diagnosticsText}>
        Pose detection: {(result.diagnostics.pose_detection_rate * 100).toFixed(0)}% ·{" "}
        {result.diagnostics.total_processing_time_s.toFixed(1)}s
      </Text>
      {poseCaveatText ? (
        <View style={styles.caveatBox}>
          <Text style={styles.caveatText}>{poseCaveatText}</Text>
        </View>
      ) : null}
      {children}
    </ScrollView>
  );
}

const makeStyles = (colors) => StyleSheet.create({
  resultBox: { flex: 1 },
  resultContent: { paddingBottom: spacing.xxl },
  cardEyebrow: {
    fontSize: fontSizes.caption,
    fontWeight: "600",
    letterSpacing: 0.6,
    textTransform: "uppercase",
    color: colors.textFaint,
    marginBottom: spacing.xs,
  },
  sectionSpacing: { marginTop: spacing.lg },
  // Hero stroke name (De-genericizing pass, section 02) -- the actual
  // headline of this screen, previously sized the same as a card eyebrow's
  // sibling heading. Confidence moves to a small pill below it instead of
  // sharing a row at equal weight.
  heroStroke: {
    fontFamily: fonts.serif,
    fontSize: 40,
    lineHeight: 44,
    color: colors.navyInk,
    textTransform: "capitalize",
  },
  pill: {
    paddingVertical: 3,
    paddingHorizontal: spacing.sm,
    borderRadius: radii.pill,
    backgroundColor: colors.tealTint,
  },
  pillText: { fontSize: fontSizes.caption, fontWeight: "600", color: colors.tealDeep },
  heroConfidencePill: { alignSelf: "flex-start", marginTop: spacing.xs },
  qualityPill: { alignSelf: "flex-start", marginTop: spacing.sm },
  pillCaveat: { backgroundColor: colors.caveatTint },
  pillTextCaveat: { color: colors.caveat },
  card: {
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radii.md,
    padding: spacing.md,
  },
  // Pull-quote hero coaching item (De-genericizing pass, section 02) --
  // the highest-value insight (backend already orders items by
  // significance) gets real typographic weight instead of reading at the
  // same volume as items 2 and 3.
  pullQuote: { borderLeftWidth: 3, borderLeftColor: colors.navy, paddingLeft: spacing.md },
  pullQuoteText: {
    fontFamily: fonts.serif,
    fontStyle: "italic",
    fontSize: fontSizes.h2,
    color: colors.navyInk,
    lineHeight: 25,
  },
  pullQuoteCitation: { fontSize: fontSizes.caption, color: colors.textFaint, marginTop: spacing.xs },
  // Demoted secondary items -- same information as before (full citation,
  // nothing dropped), just smaller and quieter than the pull-quote above.
  secondaryItems: { marginTop: spacing.md, paddingTop: spacing.md, borderTopWidth: 1, borderTopColor: colors.border },
  secondaryItemDivider: { marginTop: spacing.sm, paddingTop: spacing.sm, borderTopWidth: 1, borderTopColor: colors.border },
  secondaryItemText: { fontSize: fontSizes.caption, color: colors.textSoft, lineHeight: 17 },
  secondaryItemCitation: { fontSize: fontSizes.caption, color: colors.textFaint, marginTop: 2 },
  errorCard: { backgroundColor: colors.errorTint, borderColor: colors.errorTint },
  errorCardText: { fontSize: fontSizes.small, color: colors.text, lineHeight: 19 },
  caveatBox: {
    backgroundColor: colors.caveatTint,
    borderRadius: radii.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.sm,
    marginTop: spacing.sm,
  },
  caveatText: { fontSize: fontSizes.small, color: colors.caveat, lineHeight: 18 },
  diagnosticsText: { fontSize: fontSizes.small, color: colors.textSoft },
});
